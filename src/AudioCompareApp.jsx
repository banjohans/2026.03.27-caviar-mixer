import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  memo,
} from "react";
import {
  Upload,
  Play,
  Pause,
  Volume2,
  VolumeX,
  Trash2,
  Headphones,
  Repeat,
  SkipBack,
  SkipForward,
  Lock,
  Unlock,
  X,
  BarChart3,
  GripVertical,
  ChevronUp,
  ChevronDown,
  Activity,
  GitCompareArrows,
} from "lucide-react";

/* ── helpers ── */

function fmt(seconds) {
  if (!Number.isFinite(seconds)) return "0:00.0";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const t = Math.floor((seconds % 1) * 10);
  return `${m}:${String(s).padStart(2, "0")}.${t}`;
}

function parseFmt(str) {
  const s = str.trim();
  const m = s.match(/^(\d+):(\d{1,2})\.(\d+)$/);
  if (m) return Number(m[1]) * 60 + Number(m[2]) + Number(`0.${m[3]}`);
  const n = parseFloat(s);
  return Number.isFinite(n) ? Math.max(0, n) : null;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

function computePeaks(buf, n = 1200) {
  const d = buf.getChannelData(0);
  const spp = Math.floor(d.length / n);
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let mx = 0;
    const s = i * spp;
    const e = Math.min(s + spp, d.length);
    for (let j = s; j < e; j++) {
      const a = Math.abs(d[j]);
      if (a > mx) mx = a;
    }
    p[i] = mx;
  }
  return p;
}

function MiniWave({ id, peaks, color, muted, progress }) {
  if (!peaks || peaks.length === 0) return null;
  const n = 120;
  const step = Math.max(1, Math.floor(peaks.length / n));
  const pts = [];
  for (let i = 0; i < n; i++) {
    let mx = 0;
    const s = i * step;
    for (let j = s; j < s + step && j < peaks.length; j++) {
      if (peaks[j] > mx) mx = peaks[j];
    }
    pts.push(mx);
  }
  const vw = 200;
  const vh = 60;
  const cy = vh / 2;
  let d = `M 0 ${cy}`;
  for (let i = 0; i < pts.length; i++) {
    const x = (i / (pts.length - 1)) * vw;
    const h = pts[i] * cy * 0.85;
    d += ` L ${x.toFixed(1)} ${(cy - h).toFixed(1)}`;
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const x = (i / (pts.length - 1)) * vw;
    const h = pts[i] * cy * 0.85;
    d += ` L ${x.toFixed(1)} ${(cy + h).toFixed(1)}`;
  }
  d += " Z";
  const px = (progress || 0) * vw;
  const clipId = `wave-clip-${id}`;
  return (
    <svg
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      className="absolute inset-0 w-full h-full pointer-events-none"
    >
      <defs>
        <clipPath id={clipId}>
          <rect x="0" y="0" width={px.toFixed(1)} height={vh} />
        </clipPath>
      </defs>
      <path d={d} fill={muted ? "#94a3b8" : color} opacity={0.1} />
      <path
        d={d}
        fill={muted ? "#94a3b8" : color}
        opacity={0.25}
        clipPath={`url(#${clipId})`}
      />
      {progress > 0 && progress < 1 && (
        <line
          x1={px.toFixed(1)}
          y1="0"
          x2={px.toFixed(1)}
          y2={vh}
          stroke={muted ? "#94a3b8" : color}
          strokeWidth="1"
          opacity={0.5}
        />
      )}
    </svg>
  );
}

function computeRms(buf, from, to) {
  const d = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s = Math.floor(from * sr);
  const e = Math.min(Math.floor(to * sr), d.length);
  if (e <= s) return { rms: 0, peak: 0 };
  let sum = 0,
    peak = 0;
  for (let i = s; i < e; i++) {
    const v = Math.abs(d[i]);
    sum += d[i] * d[i];
    if (v > peak) peak = v;
  }
  return { rms: Math.sqrt(sum / (e - s)), peak };
}

function dbStr(linear) {
  if (linear <= 0) return "-∞ dB";
  return (20 * Math.log10(linear)).toFixed(1) + " dB";
}

/* ── advanced analysis helpers ── */

/*
 * BPM detection via spectral-flux onset strength + autocorrelation
 * Based on Ellis 2007 "Beat Tracking by Dynamic Programming" (the approach
 * used by librosa.beat.beat_track) adapted for vanilla JS.
 *
 * Key design choices:
 *  1. Spectral flux onset function — captures rhythmic changes in vocals,
 *     drums, guitars equally (unlike energy-only which needs a kick drum).
 *  2. Silence gating — frames below a noise floor are zeroed so intros,
 *     outros and gaps don't dilute the autocorrelation.
 *  3. Perceptual tempo prior (log-normal centered ~120 BPM) — biases
 *     toward the musically most likely octave, reducing octave errors.
 *  4. Harmonic summation of the autocorrelation — true beat periods
 *     show peaks at 2× and ½× the period; noise doesn't.
 */
function detectBPM(buf, from = 0, to) {
  const ch = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s0 = Math.floor(from * sr);
  const s1 = to != null ? Math.min(Math.floor(to * sr), ch.length) : ch.length;
  const len = s1 - s0;
  if (len < sr * 4) return { bpm: null, confidence: 0 };

  // ── 1. Short-time spectral flux (onset strength) ──────────────
  // ~23 ms frames (~43 Hz hop rate) with 50 % overlap, matching librosa defaults
  const fftN = 1024;
  const hop = 512;
  const nFrames = Math.floor((len - fftN) / hop) + 1;
  if (nFrames < 100) return { bpm: null, confidence: 0 };

  const hann = new Float32Array(fftN);
  for (let i = 0; i < fftN; i++) hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftN - 1)));

  const nBins = fftN / 2 + 1;
  let prevMag = null;
  const osf = new Float32Array(nFrames); // onset strength function
  const frameRms = new Float32Array(nFrames);

  for (let f = 0; f < nFrames; f++) {
    const base = s0 + f * hop;
    // windowed FFT (reuse fftInPlace from the codebase)
    const re = new Float32Array(fftN);
    const im = new Float32Array(fftN);
    for (let i = 0; i < fftN; i++) re[i] = (ch[base + i] || 0) * hann[i];
    fftInPlace(re, im);

    const mag = new Float32Array(nBins);
    let rmsSq = 0;
    for (let i = 0; i < nBins; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
      rmsSq += mag[i] * mag[i];
    }
    frameRms[f] = Math.sqrt(rmsSq / nBins);

    if (prevMag) {
      // half-wave rectified spectral flux — only positive (louder) changes
      let flux = 0;
      for (let i = 0; i < nBins; i++) {
        const diff = mag[i] - prevMag[i];
        if (diff > 0) flux += diff;
      }
      osf[f] = flux;
    }
    prevMag = mag;
  }

  // ── 2. Silence gating ─────────────────────────────────────────
  // Find noise floor: median RMS of sorted frames, then gate at −40 dB below peak RMS
  let peakRms = 0;
  for (let i = 0; i < nFrames; i++) if (frameRms[i] > peakRms) peakRms = frameRms[i];
  const gateThresh = peakRms * 0.01; // −40 dB
  for (let i = 0; i < nFrames; i++) {
    if (frameRms[i] < gateThresh) osf[i] = 0;
  }

  // ── 3. Normalise onset strength (subtract local mean, à la librosa) ──
  const meanWin = Math.floor(sr / hop); // ~1 second window
  const osfN = new Float32Array(nFrames);
  let rSum = 0;
  for (let i = 0; i < nFrames; i++) {
    rSum += osf[i];
    if (i >= meanWin) rSum -= osf[i - meanWin];
    const localMean = rSum / Math.min(i + 1, meanWin);
    osfN[i] = Math.max(0, osf[i] - localMean);
  }

  // ── 4. Autocorrelation of onset strength ──────────────────────
  // BPM range 60–200 → periods in frames
  const osfRate = sr / hop; // frames per second
  const minLag = Math.floor(osfRate * (60 / 200)); // 200 BPM
  const maxLag = Math.ceil(osfRate * (60 / 60));   // 60 BPM
  const corrLen = maxLag - minLag + 1;
  if (corrLen < 2) return { bpm: null, confidence: 0 };

  const acLen = nFrames - maxLag;
  if (acLen < 100) return { bpm: null, confidence: 0 };

  const ac = new Float32Array(corrLen);
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < acLen; i++) sum += osfN[i] * osfN[i + lag];
    ac[lag - minLag] = sum;
  }

  // ── 5. Perceptual tempo prior (log-normal, µ=120 BPM, σ=0.9 octaves) ──
  // This biases towards musically common tempi and resolves octave ambiguity
  const prior = new Float32Array(corrLen);
  for (let lag = minLag; lag <= maxLag; lag++) {
    const bpm = (osfRate * 60) / lag;
    const logRatio = Math.log2(bpm / 120);
    prior[lag - minLag] = Math.exp(-0.5 * (logRatio / 0.9) * (logRatio / 0.9));
  }

  // ── 6. Harmonic summation + prior weighting ───────────────────
  const scored = new Float32Array(corrLen);
  let acMax = -Infinity;
  for (let i = 0; i < corrLen; i++) if (ac[i] > acMax) acMax = ac[i];
  if (acMax <= 0) return { bpm: null, confidence: 0 };

  for (let lag = minLag; lag <= maxLag; lag++) {
    const idx = lag - minLag;
    let s = ac[idx] / acMax;

    // Check harmonics: 2×lag (half-tempo) and lag/2 (double-tempo)
    const i2 = lag * 2 - minLag;
    if (i2 >= 0 && i2 < corrLen) s += (ac[i2] / acMax) * 0.5;
    const iH = Math.round(lag / 2) - minLag;
    if (iH >= 0 && iH < corrLen) s += (ac[iH] / acMax) * 0.4;
    // 3× lag (triplet / bar level)
    const i3 = lag * 3 - minLag;
    if (i3 >= 0 && i3 < corrLen) s += (ac[i3] / acMax) * 0.15;

    scored[idx] = s * prior[idx];
  }

  // ── 7. Find best tempo ────────────────────────────────────────
  let bestIdx = 0;
  for (let i = 1; i < corrLen; i++) {
    if (scored[i] > scored[bestIdx]) bestIdx = i;
  }
  const bestLag = bestIdx + minLag;

  // Parabolic interpolation for sub-frame precision
  let refinedLag = bestLag;
  if (bestIdx > 0 && bestIdx < corrLen - 1) {
    const y0 = ac[bestIdx - 1], y1 = ac[bestIdx], y2 = ac[bestIdx + 1];
    const denom = 2 * (2 * y1 - y0 - y2);
    if (Math.abs(denom) > 1e-10) refinedLag = bestLag + (y0 - y2) / denom;
  }

  let bpm = (osfRate * 60) / refinedLag;
  // Octave normalize to 80–160
  while (bpm > 160) bpm /= 2;
  while (bpm < 80) bpm *= 2;

  // ── 8. Confidence from peak prominence ────────────────────────
  let acMean = 0;
  for (let i = 0; i < corrLen; i++) acMean += ac[i];
  acMean /= corrLen;
  const prominence = acMean > 0 ? (ac[bestIdx] - acMean) / acMean : 0;
  const confidence = Math.max(0, Math.min(1, prominence / 4));

  return {
    bpm: Math.round(bpm * 10) / 10,
    confidence,
  };
}

function computeSpectrum(buf, from = 0, to, fftSize = 4096) {
  const d = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s = Math.floor(from * sr);
  const e = to ? Math.min(Math.floor(to * sr), d.length) : d.length;
  if (e - s < fftSize) return null;

  // average multiple FFT frames
  const hop = Math.floor(fftSize / 2);
  const nFrames = Math.floor((e - s - fftSize) / hop) + 1;
  const bins = fftSize / 2;
  const avg = new Float64Array(bins);

  for (let f = 0; f < nFrames; f++) {
    const offset = s + f * hop;
    // apply Hann window + DFT magnitude
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
      re[i] = (d[offset + i] || 0) * w;
    }
    // simple radix-2 FFT (in-place)
    fftInPlace(re, im);
    for (let i = 0; i < bins; i++) {
      avg[i] += Math.sqrt(re[i] * re[i] + im[i] * im[i]) / nFrames;
    }
  }

  // spectral centroid & bandwidth
  let sumMag = 0,
    sumFreqMag = 0,
    sumBwMag = 0;
  const binHz = sr / fftSize;
  for (let i = 1; i < bins; i++) {
    const freq = i * binHz;
    sumMag += avg[i];
    sumFreqMag += freq * avg[i];
  }
  const centroid = sumMag > 0 ? sumFreqMag / sumMag : 0;
  for (let i = 1; i < bins; i++) {
    const freq = i * binHz;
    sumBwMag += (freq - centroid) ** 2 * avg[i];
  }
  const bandwidth = sumMag > 0 ? Math.sqrt(sumBwMag / sumMag) : 0;

  return { spectrum: avg, centroid, bandwidth, binHz, bins };
}

function fftInPlace(re, im) {
  const n = re.length;
  // bit-reversal
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang),
      wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1,
        curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const tRe = curRe * re[i + j + len / 2] - curIm * im[i + j + len / 2];
        const tIm = curRe * im[i + j + len / 2] + curIm * re[i + j + len / 2];
        re[i + j + len / 2] = re[i + j] - tRe;
        im[i + j + len / 2] = im[i + j] - tIm;
        re[i + j] += tRe;
        im[i + j] += tIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }
}

function spectralSimilarity(specA, specB) {
  if (!specA || !specB) return null;
  const len = Math.min(specA.length, specB.length);
  let dot = 0,
    magA = 0,
    magB = 0;
  for (let i = 1; i < len; i++) {
    dot += specA[i] * specB[i];
    magA += specA[i] * specA[i];
    magB += specB[i] * specB[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Compute a spectrogram (2D array of dB magnitude frames) for rendering.
 * Returns { frames: Float32Array[], nBins, binHz, maxDb, minDb }
 * Each frame[i] is a Float32Array of dB values for frequency bins.
 */
function computeSpectrogram(buf, from = 0, to, fftSize = 2048) {
  const d = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s = Math.floor(from * sr);
  const e = to != null ? Math.min(Math.floor(to * sr), d.length) : d.length;
  const hop = Math.floor(fftSize / 2);
  const nFrames = Math.floor((e - s - fftSize) / hop) + 1;
  if (nFrames < 2) return null;
  const nBins = fftSize / 2;
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));

  const frames = [];
  let globalMax = -Infinity, globalMin = Infinity;
  for (let f = 0; f < nFrames; f++) {
    const base = s + f * hop;
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) re[i] = (d[base + i] || 0) * hann[i];
    fftInPlace(re, im);
    const dbFrame = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) {
      const mag = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
      const db = 20 * Math.log10(Math.max(mag, 1e-10));
      dbFrame[i] = db;
      if (db > globalMax) globalMax = db;
      if (db < globalMin) globalMin = db;
    }
    frames.push(dbFrame);
  }
  return { frames, nBins, binHz: sr / fftSize, maxDb: globalMax, minDb: Math.max(globalMin, globalMax - 90) };
}

function computeDynamicRange(buf, from = 0, to) {
  const d = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s = Math.floor(from * sr);
  const e = to ? Math.min(Math.floor(to * sr), d.length) : d.length;
  if (e <= s) return 0;
  let sum = 0,
    peak = 0;
  for (let i = s; i < e; i++) {
    sum += d[i] * d[i];
    const v = Math.abs(d[i]);
    if (v > peak) peak = v;
  }
  const rms = Math.sqrt(sum / (e - s));
  return rms > 0 ? peak / rms : 0;
}

function crossCorrelation(bufA, bufB, fromA, toA, fromB, toB, invertB = false) {
  const dA = bufA.getChannelData(0),
    dB = bufB.getChannelData(0);
  const srA = bufA.sampleRate,
    srB = bufB.sampleRate;
  const sA = Math.floor(fromA * srA),
    eA = Math.min(Math.floor(toA * srA), dA.length);
  const sB = Math.floor(fromB * srB),
    eB = Math.min(Math.floor(toB * srB), dB.length);
  const len = Math.min(eA - sA, eB - sB, srA * 10); // max 10s
  if (len <= 0) return 0;
  const sign = invertB ? -1 : 1;
  // downsample to ~4000 Hz for speed
  const ds = Math.max(1, Math.floor(srA / 4000));
  const n = Math.floor(len / ds);
  let dot = 0,
    magA = 0,
    magB = 0;
  for (let i = 0; i < n; i++) {
    const a = dA[sA + i * ds] || 0;
    const b = (dB[sB + i * ds] || 0) * sign;
    dot += a * b;
    magA += a * a;
    magB += b * b;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}

function analyzeTrackFull(buf, from, to) {
  // BPM uses the FULL track (needs many windows for reliable detection)
  const bpmResult = detectBPM(buf, 0, buf.duration);
  // spectral / level analysis uses the selected segment
  const spec = computeSpectrum(buf, from, to);
  const dynRange = computeDynamicRange(buf, from, to);
  const rmsData = computeRms(buf, from || 0, to || buf.duration);
  const crestFactorDb = rmsData.rms > 0 ? 20 * Math.log10(rmsData.peak / rmsData.rms) : 0;
  return {
    bpm: bpmResult.bpm,
    bpmConfidence: bpmResult.confidence,
    centroid: spec?.centroid || 0,
    bandwidth: spec?.bandwidth || 0,
    spectrum: spec?.spectrum || null,
    dynRange,
    rms: rmsData.rms,
    peak: rmsData.peak,
    crestFactorDb,
  };
}

function pctStr(v) {
  return (v * 100).toFixed(1) + "%";
}

const TRACK_COLORS = [
  "#e8453c",
  "#f5a623",
  "#c9590a",
  "#d4463a",
  "#e09520",
  "#b73d2a",
  "#d97b1a",
  "#a83828",
  "#cc6b10",
  "#f0892d",
];

/* ── static waveform drawing ── */

function drawWave(
  canvas,
  peaks,
  duration,
  totalDur,
  segStart,
  segEnd,
  color,
  skipZones,
) {
  if (!canvas) return;
  const ctr = canvas.parentElement;
  if (!ctr) return;
  const r = ctr.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = r.width,
    h = 80;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const td = totalDur || duration;
  ctx.fillStyle = "#fdf6f0";
  ctx.fillRect(0, 0, w, h);
  if (!peaks || !peaks.length || td <= 0) return;

  const tw = (duration / td) * w;
  const cy = h / 2,
    amp = h / 2 - 4;

  // segment highlight (per-track)
  if (segEnd > segStart) {
    const ss = (segStart / td) * w,
      se = (segEnd / td) * w;
    ctx.fillStyle = "rgba(245,166,35,0.15)";
    ctx.fillRect(ss, 0, se - ss, h);
    ctx.fillStyle = "rgba(0,0,0,0.04)";
    ctx.fillRect(0, 0, ss, h);
    ctx.fillRect(se, 0, w - se, h);
  }

  // bars
  ctx.fillStyle = color || "#64748b";
  const bw = Math.max(1, tw / peaks.length - 0.5);
  for (let i = 0; i < peaks.length; i++) {
    const x = (i / peaks.length) * tw;
    const hh = peaks[i] * amp;
    ctx.fillRect(x, cy - hh, bw, hh * 2 || 1);
  }

  // skip zones (red striped)
  if (skipZones && skipZones.length > 0) {
    skipZones.forEach((z) => {
      const zs = (z.start / td) * w,
        ze = (z.end / td) * w;
      // semi-transparent red fill
      ctx.fillStyle = "rgba(232,69,60,0.15)";
      ctx.fillRect(zs, 0, ze - zs, h);
      // diagonal stripes
      ctx.save();
      ctx.beginPath();
      ctx.rect(zs, 0, ze - zs, h);
      ctx.clip();
      ctx.strokeStyle = "rgba(232,69,60,0.3)";
      ctx.lineWidth = 1;
      for (let sx = zs - h; sx < ze + h; sx += 8) {
        ctx.beginPath();
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx + h, h);
        ctx.stroke();
      }
      ctx.restore();
      // boundary lines
      ctx.strokeStyle = "#e8453c";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 2]);
      [zs, ze].forEach((lx) => {
        ctx.beginPath();
        ctx.moveTo(lx, 0);
        ctx.lineTo(lx, h);
        ctx.stroke();
      });
      ctx.setLineDash([]);
    });
  }

  // segment boundary lines
  if (segEnd > segStart) {
    const ss = (segStart / td) * w,
      se = (segEnd / td) * w;
    ctx.strokeStyle = "#c9590a";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    [ss, se].forEach((lx) => {
      ctx.beginPath();
      ctx.moveTo(lx, 0);
      ctx.lineTo(lx, h);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }
}

/* ── Spectrogram overlay for compare view ── */

function drawSpectrogram(canvas, sgData, colorFn) {
  if (!canvas || !sgData) return;
  const ctx = canvas.getContext("2d");
  const { frames, nBins, maxDb, minDb } = sgData;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const nF = frames.length;
  const colW = w / nF;
  const range = maxDb - minDb || 1;
  // draw bottom-up: low frequencies at bottom
  for (let f = 0; f < nF; f++) {
    const frame = frames[f];
    // only draw up to ~12 kHz for visual clarity
    const maxBin = Math.min(nBins, Math.floor(nBins * 0.6));
    const rowH = h / maxBin;
    for (let b = 0; b < maxBin; b++) {
      const norm = Math.max(0, Math.min(1, (frame[b] - minDb) / range));
      const [r, g, bl] = colorFn(norm);
      ctx.fillStyle = `rgb(${r},${g},${bl})`;
      ctx.fillRect(f * colW, h - (b + 1) * rowH, Math.ceil(colW), Math.ceil(rowH));
    }
  }
}

// warm color map (track A — orange/gold)
function colorMapA(t) {
  const r = Math.floor(10 + t * 235);
  const g = Math.floor(5 + t * 140);
  const b = Math.floor(0 + t * 30);
  return [r, g, b];
}
// cool color map (track B — teal/blue)
function colorMapB(t) {
  const r = Math.floor(0 + t * 50);
  const g = Math.floor(10 + t * 170);
  const b = Math.floor(20 + t * 215);
  return [r, g, b];
}

const SpectrogramOverlay = memo(function SpectrogramOverlay({ bufA, bufB, fromA, toA, fromB, toB }) {
  const canvasARef = useRef(null);
  const canvasBRef = useRef(null);
  const containerRef = useRef(null);
  const [fade, setFade] = useState(0.5);
  const sgARef = useRef(null);
  const sgBRef = useRef(null);
  const [drawGen, setDrawGen] = useState(0);

  // 2D rectangle selection: { x0, y0, x1, y1 } all 0–1 fractions
  const [selRect, setSelRect] = useState(null);
  const draggingRef = useRef(false);
  const startPtRef = useRef({ x: 0, y: 0 });
  const [regionMetrics, setRegionMetrics] = useState(null);
  const [selDone, setSelDone] = useState(0); // bumped on pointer-up to trigger similarity calc
  const [regionPhaseInvert, setRegionPhaseInvert] = useState(false);

  // Audio playback state
  const [playing, setPlaying] = useState(false);
  const playCtxRef = useRef(null); // { srcA, srcB, gainA, gainB, ctx }
  const playProgressRef = useRef(0);
  const [playProgress, setPlayProgress] = useState(0);
  const rafRef = useRef(null);
  const playStartRef = useRef(0);
  const playDurRef = useRef(0);

  // Max frequency rendered (fraction of Nyquist)
  const MAX_FREQ_FRAC = 0.6; // ~12 kHz at 44.1 kHz

  useEffect(() => {
    if (!bufA || !bufB) return;
    const maxFrames = 256;
    const durA = toA - fromA, durB = toB - fromB;
    const fftA = Math.min(2048, Math.pow(2, Math.ceil(Math.log2(bufA.sampleRate * durA / maxFrames * 2))));
    const fftB = Math.min(2048, Math.pow(2, Math.ceil(Math.log2(bufB.sampleRate * durB / maxFrames * 2))));
    sgARef.current = computeSpectrogram(bufA, fromA, toA, Math.max(512, fftA));
    sgBRef.current = computeSpectrogram(bufB, fromB, toB, Math.max(512, fftB));
    setDrawGen((g) => g + 1);
    setSelRect(null);
    setRegionMetrics(null);
    stopPlayback();
  }, [bufA, bufB, fromA, toA, fromB, toB]);

  useEffect(() => {
    if (!sgARef.current || !sgBRef.current) return;
    drawSpectrogram(canvasARef.current, sgARef.current, colorMapA);
    drawSpectrogram(canvasBRef.current, sgBRef.current, colorMapB);
  }, [drawGen]);

  // Helper: get x,y fraction from pointer event
  const getXY = useCallback((e) => {
    const rect = containerRef.current.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: Math.max(0, Math.min(1, (cx - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (cy - rect.top) / rect.height)),
    };
  }, []);

  const onPointerDown = useCallback((e) => {
    e.preventDefault();
    draggingRef.current = true;
    const pt = getXY(e);
    startPtRef.current = pt;
    setSelRect({ x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y });
    setRegionMetrics(null);
    stopPlayback();
  }, [getXY]);

  const onPointerMove = useCallback((e) => {
    if (!draggingRef.current) return;
    e.preventDefault();
    const pt = getXY(e);
    setSelRect({ x0: startPtRef.current.x, y0: startPtRef.current.y, x1: pt.x, y1: pt.y });
  }, [getXY]);

  const onPointerUp = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setSelDone((n) => n + 1);
  }, []);

  useEffect(() => {
    const up = () => { if (draggingRef.current) { draggingRef.current = false; setSelDone((n) => n + 1); } };
    window.addEventListener("mouseup", up);
    window.addEventListener("touchend", up);
    return () => { window.removeEventListener("mouseup", up); window.removeEventListener("touchend", up); };
  }, []);

  // Derived normalized rect
  const normRect = useMemo(() => {
    if (!selRect) return null;
    const left = Math.min(selRect.x0, selRect.x1);
    const right = Math.max(selRect.x0, selRect.x1);
    const top = Math.min(selRect.y0, selRect.y1);
    const bottom = Math.max(selRect.y0, selRect.y1);
    if (right - left < 0.015 && bottom - top < 0.03) return null;
    return { left, right, top, bottom };
  }, [selRect]);

  // Convert Y fraction (0=top=high freq, 1=bottom=low freq) to Hz
  const yToHz = useCallback((yFrac) => {
    const nyquist = (bufA?.sampleRate || 44100) / 2;
    const maxHz = nyquist * MAX_FREQ_FRAC;
    return maxHz * (1 - yFrac);
  }, [bufA]);

  // Compute region similarity (now uses 2D: time range + freq-limited spectrum)
  useEffect(() => {
    if (!normRect || !bufA || !bufB) return;
    const { left, right, top, bottom } = normRect;
    const rFromA = fromA + left * (toA - fromA);
    const rToA = fromA + right * (toA - fromA);
    const rFromB = fromB + left * (toB - fromB);
    const rToB = fromB + right * (toB - fromB);
    const hiHz = yToHz(top);
    const loHz = yToHz(bottom);
    const specA = computeSpectrum(bufA, rFromA, rToA);
    const specB = computeSpectrum(bufB, rFromB, rToB);
    // Limit similarity to selected frequency bins
    let sim = null;
    if (specA && specB) {
      const binHz = specA.binHz;
      const loBin = Math.max(1, Math.floor(loHz / binHz));
      const hiBin = Math.min(specA.bins, Math.ceil(hiHz / binHz));
      if (hiBin > loBin) {
        const sliceA = specA.spectrum.slice(loBin, hiBin);
        const sliceB = specB.spectrum.slice(loBin, hiBin);
        sim = spectralSimilarity(sliceA, sliceB);
      }
    }
    const corr = crossCorrelation(bufA, bufB, rFromA, rToA, rFromB, rToB, regionPhaseInvert);
    const durSec = Math.max(rToA - rFromA, rToB - rFromB);
    setRegionMetrics({ specSim: sim, corr, duration: durSec, loHz, hiHz });
  }, [selDone, bufA, bufB, fromA, toA, fromB, toB, regionPhaseInvert]);

  // ── Audio playback helpers ──
  function stopPlayback() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    const pc = playCtxRef.current;
    if (pc) {
      try { pc.srcA.stop(); } catch (_) {}
      try { pc.srcB.stop(); } catch (_) {}
      try { pc.ctx.close(); } catch (_) {}
      playCtxRef.current = null;
    }
    setPlaying(false);
    setPlayProgress(0);
  }

  function startPlayback() {
    if (!normRect || !bufA || !bufB) return;
    stopPlayback();
    const { left, right, top, bottom } = normRect;
    const rFromA = fromA + left * (toA - fromA);
    const rToA = fromA + right * (toA - fromA);
    const rFromB = fromB + left * (toB - fromB);
    const rToB = fromB + right * (toB - fromB);
    const hiHz = yToHz(top);
    const loHz = yToHz(bottom);
    const dur = Math.max(rToA - rFromA, rToB - rFromB);
    if (dur < 0.05) return;

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    ctx.resume();

    // Create source A
    const srcA = ctx.createBufferSource();
    srcA.buffer = bufA;
    const gainA = ctx.createGain();
    // Create source B
    const srcB = ctx.createBufferSource();
    srcB.buffer = bufB;
    const gainB = ctx.createGain();

    // Bandpass filters for frequency selection
    const makeFilter = () => {
      const center = Math.sqrt(Math.max(loHz, 20) * Math.min(hiHz, 20000));
      const bw = Math.min(hiHz, 20000) - Math.max(loHz, 20);
      if (bw > 18000) return null; // nearly full range, skip filter
      const lo = ctx.createBiquadFilter();
      lo.type = "highpass";
      lo.frequency.value = Math.max(loHz, 20);
      lo.Q.value = 0.707;
      const hi = ctx.createBiquadFilter();
      hi.type = "lowpass";
      hi.frequency.value = Math.min(hiHz, 20000);
      hi.Q.value = 0.707;
      return { lo, hi };
    };

    const filterA = makeFilter();
    const filterB = makeFilter();

    // Wire A
    if (filterA) {
      srcA.connect(filterA.lo);
      filterA.lo.connect(filterA.hi);
      filterA.hi.connect(gainA);
    } else {
      srcA.connect(gainA);
    }
    gainA.connect(ctx.destination);

    // Wire B
    if (filterB) {
      srcB.connect(filterB.lo);
      filterB.lo.connect(filterB.hi);
      filterB.hi.connect(gainB);
    } else {
      srcB.connect(gainB);
    }
    gainB.connect(ctx.destination);

    // Apply crossfade
    gainA.gain.value = 1 - fade;
    gainB.gain.value = fade;

    srcA.start(0, rFromA, rToA - rFromA);
    srcB.start(0, rFromB, rToB - rFromB);

    playCtxRef.current = { srcA, srcB, gainA, gainB, ctx, rFromA, rToA, rFromB, rToB, filterA, filterB };
    playStartRef.current = ctx.currentTime;
    playDurRef.current = dur;
    setPlaying(true);

    // On ended, restart for looping
    const scheduleLoop = () => {
      if (!playCtxRef.current || playCtxRef.current.ctx !== ctx) return;
      // Create new sources for next loop iteration
      const newSrcA = ctx.createBufferSource();
      newSrcA.buffer = bufA;
      const newSrcB = ctx.createBufferSource();
      newSrcB.buffer = bufB;
      if (filterA) { newSrcA.connect(filterA.lo); } else { newSrcA.connect(gainA); }
      if (filterB) { newSrcB.connect(filterB.lo); } else { newSrcB.connect(gainB); }
      newSrcA.start(0, rFromA, rToA - rFromA);
      newSrcB.start(0, rFromB, rToB - rFromB);
      playCtxRef.current.srcA = newSrcA;
      playCtxRef.current.srcB = newSrcB;
      playStartRef.current = ctx.currentTime;
      newSrcA.onended = scheduleLoop;
    };
    srcA.onended = scheduleLoop;

    // Animate progress (loops)
    const animate = () => {
      if (!playCtxRef.current || playCtxRef.current.ctx !== ctx) return;
      const elapsed = ctx.currentTime - playStartRef.current;
      const p = dur > 0 ? (elapsed % dur) / dur : 0;
      setPlayProgress(p);
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
  }

  // Update gains when fade changes during playback
  useEffect(() => {
    const pc = playCtxRef.current;
    if (pc) {
      pc.gainA.gain.value = 1 - fade;
      pc.gainB.gain.value = fade;
    }
  }, [fade]);

  // Cleanup on unmount
  useEffect(() => () => stopPlayback(), []);

  if (!drawGen) {
    return (
      <div className="flex items-center gap-2 text-[#9a6a40] text-sm py-4">
        <span className="animate-spin inline-block w-4 h-4 border-2 border-[#d4a87a] border-t-transparent rounded-full" />
        Computing spectrogram…
      </div>
    );
  }

  // Rect CSS helpers
  const rLeft = normRect ? normRect.left * 100 : 0;
  const rTop = normRect ? normRect.top * 100 : 0;
  const rW = normRect ? (normRect.right - normRect.left) * 100 : 0;
  const rH = normRect ? (normRect.bottom - normRect.top) * 100 : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-[#5a2a00]">Spectrogram (waterfall)</div>
        <div className="text-[9px] text-[#9a6a40] italic">Drag a rectangle to select time + frequency</div>
      </div>
      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden bg-[#1a0a00] cursor-crosshair select-none touch-none"
        style={{ height: 200 }}
        onMouseDown={onPointerDown}
        onMouseMove={onPointerMove}
        onMouseUp={onPointerUp}
        onTouchStart={onPointerDown}
        onTouchMove={onPointerMove}
        onTouchEnd={onPointerUp}
      >
        <canvas
          ref={canvasARef}
          width={600}
          height={200}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ opacity: 1 - fade }}
        />
        <canvas
          ref={canvasBRef}
          width={600}
          height={200}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ opacity: fade }}
        />
        {/* 2D Selection rectangle */}
        {normRect && rW > 0.3 && (
          <>
            {/* Dim outside selection — four rects forming an L-shape mask */}
            <div className="absolute left-0 top-0 bg-black/50 pointer-events-none" style={{ width: `${rLeft}%`, height: "100%" }} />
            <div className="absolute right-0 top-0 bg-black/50 pointer-events-none" style={{ width: `${100 - rLeft - rW}%`, height: "100%" }} />
            <div className="absolute bg-black/50 pointer-events-none" style={{ left: `${rLeft}%`, width: `${rW}%`, top: 0, height: `${rTop}%` }} />
            <div className="absolute bg-black/50 pointer-events-none" style={{ left: `${rLeft}%`, width: `${rW}%`, top: `${rTop + rH}%`, height: `${100 - rTop - rH}%` }} />
            {/* Selection border */}
            <div
              className="absolute pointer-events-none border-2 border-[#f5a623] rounded-sm"
              style={{ left: `${rLeft}%`, top: `${rTop}%`, width: `${rW}%`, height: `${rH}%` }}
            />
            {/* Playback progress line inside selection */}
            {playing && playProgress > 0 && (
              <div
                className="absolute top-0 pointer-events-none bg-white/70"
                style={{
                  left: `${rLeft + playProgress * rW}%`,
                  top: `${rTop}%`,
                  width: 2,
                  height: `${rH}%`,
                }}
              />
            )}
            {/* Labels */}
            <div
              className="absolute pointer-events-none text-[8px] font-mono text-[#f5a623] bg-black/70 rounded px-0.5 leading-tight"
              style={{ left: `${rLeft}%`, top: `${rTop}%`, transform: "translate(2px, 2px)" }}
            >
              <div>{yToHz(normRect.top).toFixed(0)} Hz</div>
              <div>{((normRect.right - normRect.left) * Math.max(toA - fromA, toB - fromB)).toFixed(1)}s</div>
            </div>
            <div
              className="absolute pointer-events-none text-[8px] font-mono text-[#f5a623] bg-black/70 rounded px-0.5"
              style={{ left: `${rLeft}%`, top: `${rTop + rH}%`, transform: "translate(2px, -100%)" }}
            >
              {yToHz(normRect.bottom).toFixed(0)} Hz
            </div>
          </>
        )}
        {/* frequency axis labels */}
        <div className="absolute left-1 top-0 bottom-0 flex flex-col justify-between text-[8px] text-white/50 pointer-events-none py-1">
          <span>12k</span>
          <span>6k</span>
          <span>2k</span>
          <span>0 Hz</span>
        </div>
        {/* crossfade position indicator */}
        <div className="absolute bottom-1 right-2 text-[9px] text-white/60 pointer-events-none">
          {fade < 0.3 ? "← Track A" : fade > 0.7 ? "Track B →" : "Overlap"}
        </div>
      </div>
      {/* Play controls — always reserve space to prevent layout shift */}
      <div className={`flex items-center gap-2 min-h-[28px] ${normRect ? "visible" : "invisible"}`}>
        <button
          onClick={() => playing ? stopPlayback() : startPlayback()}
          className="flex items-center gap-1.5 text-[11px] font-semibold px-3 py-1 rounded-full bg-[#f5a623]/20 text-[#f5a623] hover:bg-[#f5a623]/40 transition-colors"
        >
          {playing ? <Pause size={13} /> : <Play size={13} />}
          {playing ? "Stop" : "Play selection"}
        </button>
        {playing && (
          <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className="h-full rounded-full bg-[#f5a623] transition-[width] duration-75" style={{ width: `${playProgress * 100}%` }} />
          </div>
        )}
        {regionMetrics && (
          <span className="text-[9px] text-[#9a6a40] ml-auto">
            {regionMetrics.duration.toFixed(1)}s · {regionMetrics.loHz.toFixed(0)}–{regionMetrics.hiHz.toFixed(0)} Hz
          </span>
        )}
        <button
          onClick={() => { setSelRect(null); setRegionMetrics(null); stopPlayback(); }}
          className="text-[9px] text-[#9a6a40] hover:text-[#f5a623]"
          title="Clear selection"
        >
          <X size={14} />
        </button>
      </div>
      {/* Crossfade slider — also controls audio A/B */}
      <div className="flex items-center gap-3">
        <span className="text-[10px] font-semibold text-[#c97a30] w-8 text-right">A</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={fade}
          onChange={(e) => setFade(Number(e.target.value))}
          className="flex-1 accent-[#f5a623]"
          style={{ height: 4 }}
        />
        <span className="text-[10px] font-semibold text-[#3a9aba] w-8">B</span>
      </div>
      <div className="flex justify-between text-[9px] text-[#9a6a40]">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 rounded-sm" style={{background: "linear-gradient(to right, #0a0500, #f5a623)"}} /> Track A</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-2 rounded-sm" style={{background: "linear-gradient(to right, #001420, #32aad7)"}} /> Track B</span>
      </div>
      {/* Region results — stable height placeholder */}
      <div className={`rounded-xl border p-3 space-y-2 transition-opacity ${
        regionMetrics && normRect
          ? "bg-[#3a1a00]/80 border-[#f5a623]/30 opacity-100"
          : "bg-transparent border-transparent opacity-0 pointer-events-none"
      }`} style={{ minHeight: 72 }}>
        {regionMetrics && (
          <>
            {regionMetrics.specSim != null && (
            <div>
              <div className="flex justify-between text-[11px] text-[#ffe0b2] mb-0.5">
                <span>Spectral Similarity ({regionMetrics.loHz.toFixed(0)}–{regionMetrics.hiHz.toFixed(0)} Hz)</span>
                <span className="font-bold">{(regionMetrics.specSim * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-[#f5a623] to-[#e8453c]" style={{ width: `${regionMetrics.specSim * 100}%` }} />
              </div>
            </div>
          )}
          {regionMetrics.corr != null && (
            <div>
              <div className="flex justify-between text-[11px] text-[#ffe0b2] mb-0.5">
                <span className="flex items-center gap-1.5">
                  Waveform Correlation
                  <button
                    onClick={() => setRegionPhaseInvert((v) => !v)}
                    className={`text-[9px] px-1.5 py-0.5 rounded-full border transition-colors ${
                      regionPhaseInvert
                        ? "bg-[#f5a623]/30 border-[#f5a623] text-[#f5a623]"
                        : "bg-transparent border-[#9a6a40]/40 text-[#9a6a40] hover:border-[#f5a623]/60"
                    }`}
                    title="Invert phase of Track B before comparing (useful if one track is phase-inverted)"
                  >
                    Ø Invert B
                  </button>
                </span>
                <span className="font-bold">{(regionMetrics.corr * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-[#d4a87a] to-[#f5a623]" style={{ width: `${regionMetrics.corr * 100}%` }} />
              </div>
            </div>
          )}
          </>
        )}
      </div>
    </div>
  );
});

/* ── Waveform component ── */

const WaveformTrack = memo(function WaveformTrack({
  track,
  maxDuration,
  segmentLength,
  playhead,
  singlePlayhead,
  isSolo,
  onSeek,
  onSetSegStart,
  onSetSegmentByDrag,
  onSkipSegment,
  onAddSkipZone,
  onRemoveSkipZone,
  onEditSkipZone,
  punchState,
  onPunchIn,
  onPunchOut,
  onToggleMute,
  onToggleSolo,
  onSetVolume,
  onSetPan,
  onRemove,
  onPlaySingle,
  singlePlaying,
  color,
  analysis,
  fullAnalysis,
  analysisExpanded,
  onToggleAnalysis,
}) {
  const canvasRef = useRef(null);
  const dragRef = useRef(null);
  const dur = track.duration || 0;
  const td = maxDuration || dur;
  const segS = track.segStart || 0;
  const segE = Math.min(segS + (segmentLength || 0), dur);
  // adjust playhead to skip over skip zones visually
  let trackPos = segS + playhead;
  const sortedZones = (track.skipZones || [])
    .slice()
    .sort((a, b) => a.start - b.start);
  for (const z of sortedZones) {
    if (trackPos >= z.start) {
      trackPos += z.end - z.start;
    } else {
      break;
    }
  }
  trackPos = clamp(trackPos, 0, dur);
  const phPct = td > 0 ? (trackPos / td) * 100 : 0;
  const segSPct = td > 0 ? (segS / td) * 100 : 0;
  const skipZones = track.skipZones || [];
  const [punchMode, setPunchMode] = useState("segment");
  const isPunchActive = punchState && punchState.trackId === track.id;
  const punchInTime = isPunchActive ? punchState.inTime : null;
  const punchInPct =
    punchInTime != null && td > 0 ? (punchInTime / td) * 100 : null;

  // single-track playhead (cyan)
  const singlePos =
    singlePlaying && singlePlayhead != null
      ? clamp(singlePlayhead, 0, dur)
      : null;
  const singlePct = singlePos != null && td > 0 ? (singlePos / td) * 100 : null;

  useEffect(() => {
    drawWave(
      canvasRef.current,
      track.peaks,
      dur,
      maxDuration,
      segS,
      segE,
      color,
      skipZones,
    );
  }, [track.peaks, dur, maxDuration, segS, segE, color, skipZones]);

  useEffect(() => {
    const fn = () =>
      drawWave(
        canvasRef.current,
        track.peaks,
        dur,
        maxDuration,
        segS,
        segE,
        color,
        skipZones,
      );
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, [track.peaks, dur, maxDuration, segS, segE, color, skipZones]);

  const posFromEvent = (e) => {
    if (!canvasRef.current || td <= 0) return 0;
    const r = canvasRef.current.getBoundingClientRect();
    return clamp(((e.clientX - r.left) / r.width) * td, 0, dur);
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    const isAlt = e.altKey;
    const t = posFromEvent(e);
    dragRef.current = { start: t, moved: false, alt: isAlt };
    const onMove = (me) => {
      if (!dragRef.current) return;
      dragRef.current.moved = true;
    };
    const onUp = (ue) => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (!dragRef.current) return;
      const endT = posFromEvent(ue);
      const dragged =
        dragRef.current.moved && Math.abs(endT - dragRef.current.start) > 0.05;
      if (dragRef.current.alt && dragged) {
        // Alt+drag — create skip zone
        const s = Math.min(dragRef.current.start, endT);
        const en = Math.max(dragRef.current.start, endT);
        onAddSkipZone(track.id, clamp(s, 0, dur), clamp(en, 0, dur));
      } else if (dragged) {
        // Normal drag — set segment start + new global length
        const s = Math.min(dragRef.current.start, endT);
        const len = Math.abs(endT - dragRef.current.start);
        onSetSegmentByDrag(track.id, clamp(s, 0, dur), len);
      } else {
        // Click — set segment start (keep current length)
        onSetSegStart(track.id, clamp(dragRef.current.start, 0, dur));
      }
      dragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div
      className={`rounded-2xl border border-[#d4a87a]/40 bg-white/80 p-3 shadow-sm ${isSolo ? "ring-2 ring-[#e8453c]" : ""} ${track.muted ? "opacity-50" : ""}`}
    >
      {/* header */}
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <div
          className="w-3 h-3 rounded-full shrink-0"
          style={{ background: color }}
        />
        <span className="font-medium text-sm truncate max-w-[200px]">
          {track.name}
        </span>
        <span className="text-xs text-[#9a6a40]">{fmt(dur)}</span>

        <div className="flex-1" />

        <div className="flex items-center gap-0.5 rounded-lg border px-1 py-0.5">
          <button
            onClick={() => onSkipSegment(track.id, -1)}
            className="p-1 rounded hover:bg-[#f5e6d0] text-[#9a6a40] hover:text-[#3a1a00]"
            title="Previous segment"
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => onPlaySingle(track.id)}
            className="p-1 rounded hover:bg-[#f5e6d0]"
            title="Play this track solo"
          >
            {singlePlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={() => onSkipSegment(track.id, 1)}
            className="p-1 rounded hover:bg-[#f5e6d0] text-[#9a6a40] hover:text-[#3a1a00]"
            title="Next segment"
          >
            <SkipForward size={14} />
          </button>
        </div>
        {/* Punch In / Out */}
        <div className="flex items-center gap-0.5 rounded-lg border px-1 py-0.5">
          <button
            onClick={() => setPunchMode("segment")}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === "segment" ? "bg-[#f5a623]/20 text-[#c9590a]" : "text-[#9a6a40] hover:text-[#3a1a00]"}`}
            title="Punch-modus: Segment"
          >
            SEG
          </button>
          <button
            onClick={() => setPunchMode("skip")}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === "skip" ? "bg-red-100 text-red-700" : "text-[#9a6a40] hover:text-[#3a1a00]"}`}
            title="Punch mode: Skip zone"
          >
            SKIP
          </button>
          <div className="w-px h-4 bg-[#d4a87a]/40 mx-0.5" />
          <button
            onClick={() => onPunchIn(track.id, punchMode)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${isPunchActive ? "bg-[#e8453c] text-white" : "text-[#9a6a40] hover:bg-[#f5e6d0]"}`}
            title="Punch In — mark start point"
          >
            IN
          </button>
          <button
            onClick={() => onPunchOut(track.id)}
            disabled={!isPunchActive}
            className="px-1.5 py-0.5 rounded text-[10px] font-bold text-[#9a6a40] hover:bg-[#f5e6d0] disabled:opacity-30"
            title="Punch Out — mark end point"
          >
            OUT
          </button>
        </div>
        <button
          onClick={() => onToggleSolo(track.id)}
          className={`p-1.5 rounded-lg text-xs font-bold ${isSolo ? "bg-[#e8453c] text-white" : "hover:bg-[#f5e6d0] text-[#9a6a40]"}`}
          title="Solo"
        >
          <Headphones size={14} />
        </button>
        <button
          onClick={() => onToggleMute(track.id)}
          className={`p-1.5 rounded-lg ${track.muted ? "bg-red-100 text-red-600" : "hover:bg-[#f5e6d0] text-[#9a6a40]"}`}
          title="Mute"
        >
          {track.muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
        </button>

        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={track.volume}
          onChange={(e) => onSetVolume(track.id, e.target.value)}
          className="w-20 accent-[#c9590a]"
          style={{ height: "4px" }}
          title={`Volume: ${Math.round(track.volume * 100)}%`}
        />

        <div
          className="flex items-center gap-0.5 rounded-lg border px-1.5 py-1"
          title={`Pan: ${(track.pan || 0) > 0 ? `R ${Math.round((track.pan || 0) * 100)}%` : (track.pan || 0) < 0 ? `L ${Math.round(Math.abs(track.pan || 0) * 100)}%` : "Center"}`}
        >
          <span className="text-[9px] font-semibold text-[#9a6a40]">L</span>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={track.pan || 0}
            onChange={(e) => onSetPan(track.id, e.target.value)}
            onDoubleClick={() => onSetPan(track.id, 0)}
            className="w-14 accent-[#f5a623]"
            style={{ height: "4px" }}
          />
          <span className="text-[9px] font-semibold text-[#9a6a40]">R</span>
        </div>

        <button
          onClick={() => onRemove(track.id)}
          className="p-1.5 rounded-lg hover:bg-red-50 text-[#9a6a40] hover:text-red-500"
          title="Remove"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* waveform */}
      <div
        className="relative w-full rounded-lg border border-[#d4a87a]/40 bg-[#fdf6f0]"
        style={{ overflow: "visible" }}
      >
        <canvas
          ref={canvasRef}
          className="block w-full cursor-crosshair rounded-lg"
          style={{ height: "80px" }}
          onMouseDown={handleMouseDown}
        />
        {/* playhead (red = global, cyan = single) */}
        <div
          className="absolute top-0 bottom-0 pointer-events-none"
          style={{ left: `${phPct}%`, zIndex: 10 }}
        >
          <div
            style={{ width: "2px", height: "100%", background: "#e8453c" }}
          />
        </div>
        <div
          className="absolute pointer-events-none"
          style={{
            left: `${phPct}%`,
            bottom: "-16px",
            transform: "translateX(-50%)",
            zIndex: 11,
          }}
        >
          <span className="rounded bg-[#e8453c] px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
            {fmt(trackPos)}
          </span>
        </div>
        {/* single-track playhead (cyan) */}
        {singlePct != null && (
          <>
            <div
              className="absolute top-0 bottom-0 pointer-events-none"
              style={{ left: `${singlePct}%`, zIndex: 12 }}
            >
              <div
                style={{ width: "2px", height: "100%", background: "#f0892d" }}
              />
            </div>
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${singlePct}%`,
                top: "-16px",
                transform: "translateX(-50%)",
                zIndex: 13,
              }}
            >
              <span className="rounded bg-[#f0892d] px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
                ♪ {fmt(singlePos)}
              </span>
            </div>
          </>
        )}
        {/* segment start marker */}
        {segS > 0 && (
          <>
            <div
              className="absolute top-0 bottom-0 pointer-events-none"
              style={{ left: `${segSPct}%`, zIndex: 8 }}
            >
              <div
                style={{
                  width: "2px",
                  height: "100%",
                  background: "#c9590a",
                }}
              />
            </div>
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${segSPct}%`,
                top: "-14px",
                transform: "translateX(-50%)",
                zIndex: 11,
              }}
            >
              <span className="rounded bg-[#c9590a] px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
                ▶ {fmt(segS)}
              </span>
            </div>
          </>
        )}
        {/* punch-in marker (pink) */}
        {punchInPct != null && (
          <>
            <div
              className="absolute top-0 bottom-0 pointer-events-none"
              style={{ left: `${punchInPct}%`, zIndex: 14 }}
            >
              <div
                style={{
                  width: "2px",
                  height: "100%",
                  background: "#e8453c",
                  opacity: 0.8,
                }}
              />
            </div>
            <div
              className="absolute pointer-events-none"
              style={{
                left: `${punchInPct}%`,
                bottom: "-16px",
                transform: "translateX(-50%)",
                zIndex: 15,
              }}
            >
              <span className="rounded bg-[#e8453c] px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
                IN {fmt(punchInTime)}
              </span>
            </div>
          </>
        )}
      </div>

      {/* analysis + segment info */}
      <div className="mt-3 flex items-center gap-3 text-[11px] text-[#9a6a40] flex-wrap">
        {segS > 0 && (
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-[#c9590a]" />
            Segment: {fmt(segS)} – {fmt(segE)}
            <button
              onClick={() => onSetSegStart(track.id, 0)}
              className="ml-1 text-[#c9590a] hover:text-[#a83828] underline"
            >
              reset
            </button>
          </span>
        )}
        {analysis && (
          <>
            <span title="RMS level in segment">RMS: {dbStr(analysis.rms)}</span>
            <span title="Peak level in segment">
              Peak: {dbStr(analysis.peak)}
            </span>
          </>
        )}
        <span className="text-[#b89070] ml-auto">
          Drag = segment · Alt+drag = skip zone · Punch IN/OUT = precise points
        </span>
        <button
          onClick={() => onToggleAnalysis(track.id)}
          className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#5a2a00]/10 hover:bg-[#5a2a00]/20 text-[#7a4a20] font-medium transition-colors min-w-[2.5rem] min-h-[2rem] justify-center"
          title="Analyze track"
        >
          <Activity size={12} />
          Analyse
          {analysisExpanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
        </button>
      </div>

      {/* ── Full analysis panel ── */}
      {analysisExpanded && (
        <div className="mt-2 rounded-xl bg-[#faf0e6] border border-[#d4a87a]/40 p-3 text-[11px] text-[#5a2a00]">
          {fullAnalysis ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-2">
              <div>
                <span className="font-semibold text-[#9a6a40]">BPM</span>
                <div className="text-base font-bold text-[#3a1a00]">
                  {fullAnalysis.bpm > 0 ? fullAnalysis.bpm.toFixed(1) : "N/A"}
                  {fullAnalysis.bpmConfidence > 0 && (
                    <span className="ml-1 text-[10px] font-normal text-[#9a6a40]">
                      ({(fullAnalysis.bpmConfidence * 100).toFixed(0)}% confident)
                    </span>
                  )}
                </div>
              </div>
              <div>
                <span className="font-semibold text-[#9a6a40]">Spectral Centroid</span>
                <div className="text-base font-bold text-[#3a1a00]">
                  {fullAnalysis.centroid > 0 ? `${fullAnalysis.centroid.toFixed(0)} Hz` : "N/A"}
                </div>
              </div>
              <div>
                <span className="font-semibold text-[#9a6a40]">Bandwidth</span>
                <div className="text-base font-bold text-[#3a1a00]">
                  {fullAnalysis.bandwidth > 0 ? `${fullAnalysis.bandwidth.toFixed(0)} Hz` : "N/A"}
                </div>
              </div>
              <div>
                <span className="font-semibold text-[#9a6a40]">RMS</span>
                <div className="text-base font-bold text-[#3a1a00]">{dbStr(fullAnalysis.rms)}</div>
              </div>
              <div>
                <span className="font-semibold text-[#9a6a40]">Peak</span>
                <div className="text-base font-bold text-[#3a1a00]">{dbStr(fullAnalysis.peak)}</div>
              </div>
              <div>
                <span className="font-semibold text-[#9a6a40]">Crest Factor</span>
                <div className="text-base font-bold text-[#3a1a00]">
                  {fullAnalysis.crestFactorDb > 0 ? `${fullAnalysis.crestFactorDb.toFixed(1)} dB` : "N/A"}
                </div>
              </div>
              {fullAnalysis.dynRange != null && (
                <div className="col-span-2 sm:col-span-3">
                  <span className="font-semibold text-[#9a6a40]">Dynamic Range</span>
                  <div className="mt-1 h-2 rounded-full bg-[#d4a87a]/20 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-[#f5a623] to-[#e8453c]"
                      style={{ width: `${Math.min(100, (fullAnalysis.dynRange / 30) * 100)}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-[#9a6a40] mt-0.5">
                    {fullAnalysis.dynRange.toFixed(1)} dB
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-[#9a6a40]">
              <span className="animate-spin inline-block w-3 h-3 border-2 border-[#d4a87a] border-t-transparent rounded-full" />
              Analyzing…
            </div>
          )}
        </div>
      )}

      {/* skip zones list */}
      {skipZones.length > 0 && (
        <div className="mt-2 space-y-1">
          {skipZones.map((z, zi) => (
            <div
              key={zi}
              className="inline-flex items-center gap-1.5 rounded-lg bg-red-50 border border-red-200 px-2 py-1 text-[10px] text-red-700 mr-1.5"
            >
              <span className="font-semibold">Skip:</span>
              <input
                type="text"
                defaultValue={fmt(z.start)}
                onBlur={(e) => {
                  const v = parseFmt(e.target.value);
                  if (v != null && v !== z.start)
                    onEditSkipZone(track.id, zi, v, z.end);
                  else e.target.value = fmt(z.start);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.target.blur();
                }}
                className="w-14 rounded border border-red-300 bg-white px-1 py-0.5 text-center font-mono text-[10px] text-red-800 focus:outline-none focus:ring-1 focus:ring-red-400"
                title="Start point (m:ss.t or seconds)"
              />
              <span>–</span>
              <input
                type="text"
                defaultValue={fmt(z.end)}
                onBlur={(e) => {
                  const v = parseFmt(e.target.value);
                  if (v != null && v !== z.end)
                    onEditSkipZone(track.id, zi, z.start, v);
                  else e.target.value = fmt(z.end);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.target.blur();
                }}
                className="w-14 rounded border border-red-300 bg-white px-1 py-0.5 text-center font-mono text-[10px] text-red-800 focus:outline-none focus:ring-1 focus:ring-red-400"
                title="End point (m:ss.t or seconds)"
              />
              <button
                onClick={() => onRemoveSkipZone(track.id, zi)}
                className="ml-0.5 rounded hover:bg-red-100 p-0.5"
                title="Remove skip zone"
              >
                <X size={10} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

/* ── Main App ── */

export default function AudioCompareApp() {
  const [tracks, setTracks] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const [segmentLength, setSegmentLength] = useState(0);
  const [loopRegion, setLoopRegion] = useState(true);
  const [mixPos, setMixPos] = useState({ x: 0, y: 0 });
  const [showCrossfader, setShowCrossfader] = useState(false);
  const [showABFader, setShowABFader] = useState(false);
  const [abMode, setAbMode] = useState(false);
  const [abTrackA, setAbTrackA] = useState(0);
  const [abTrackB, setAbTrackB] = useState(1);
  const [abValue, setAbValue] = useState(0.5);
  const [soloId, setSoloId] = useState(new Set());
  const [singlePlayId, setSinglePlayId] = useState(null);
  const [singlePlayhead, setSinglePlayhead] = useState(null);
  const [analyses, setAnalyses] = useState({});
  const [fullAnalyses, setFullAnalyses] = useState({});
  const [expandedAnalysis, setExpandedAnalysis] = useState(new Set());
  const [comparing, setComparing] = useState(false);
  const [compareA, setCompareA] = useState(null);
  const [compareB, setCompareB] = useState(null);
  const [punchState, setPunchState] = useState(null);
  const [dragId, setDragId] = useState(null);

  const rafRef = useRef(null);
  const startTimeRef = useRef(0);
  const startOffsetRef = useRef(0);
  const fileInputRef = useRef(null);
  const audioCtxRef = useRef(null);
  const isPlayingRef = useRef(false);
  const singleRafRef = useRef(null);
  const singleTrackRef = useRef(null);
  const decodedBuffers = useRef({});
  const pannerNodes = useRef({});
  const gainNodes = useRef({});
  const [gainMatch, setGainMatch] = useState(false);
  const segmentLengthRef = useRef(0);
  const loopRegionRef = useRef(true);
  const tracksRef = useRef([]);

  const maxDuration = useMemo(() => {
    if (!tracks.length) return 0;
    return Math.max(...tracks.map((t) => t.duration || 0));
  }, [tracks]);

  /* segment length: default to max duration, clamp if needed */
  useEffect(() => {
    if (maxDuration > 0)
      setSegmentLength((p) =>
        p <= 0 ? maxDuration : clamp(p, 0.1, maxDuration),
      );
  }, [maxDuration]);

  /* keep refs in sync */
  useEffect(() => {
    segmentLengthRef.current = segmentLength;
  }, [segmentLength]);
  useEffect(() => {
    loopRegionRef.current = loopRegion;
  }, [loopRegion]);
  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  /* compute polygon vertex positions for all tracks */
  const polyVertices = useMemo(() => {
    const n = tracks.length;
    if (n === 0) return [];
    if (n === 1) return [{ x: 0, y: 0 }];
    if (n === 2)
      return [
        { x: -1, y: 0 },
        { x: 1, y: 0 },
      ];
    return Array.from({ length: n }, (_, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
      return { x: Math.cos(angle), y: Math.sin(angle) };
    });
  }, [tracks.length]);

  /* compute mix weights from mixPos — muted tracks get 0 */
  const mixWeights = useMemo(() => {
    const n = tracks.length;
    if (n === 0) return [];
    if (n === 1) return [tracks[0].muted ? 0 : 1];
    const dists = polyVertices.map((v) => {
      const dx = mixPos.x - v.x,
        dy = mixPos.y - v.y;
      return Math.sqrt(dx * dx + dy * dy);
    });
    const eps = 0.001;
    /* zero out muted distances so they don't participate */
    const activeDists = dists.map((d, i) => (tracks[i].muted ? Infinity : d));
    const snap = activeDists.findIndex((d) => d < eps);
    if (snap >= 0) return activeDists.map((_, i) => (i === snap ? 1 : 0));
    const inv = activeDists.map((d) => (d === Infinity ? 0 : 1 / (d * d)));
    const sum = inv.reduce((a, b) => a + b, 0);
    if (sum === 0) return tracks.map(() => 0);
    return inv.map((w) => w / sum);
  }, [tracks, polyVertices, mixPos]);

  /* gain matching compensation */
  const gainCompensation = useMemo(() => {
    if (!gainMatch) return {};
    const withRms = tracks.filter((t) => t.rms > 0);
    if (withRms.length < 2) return {};
    const targetRms = withRms.reduce((s, t) => s + t.rms, 0) / withRms.length;
    const comp = {};
    tracks.forEach((t) => {
      if (t.rms > 0) {
        comp[t.id] = clamp(targetRms / t.rms, 0.1, 10);
      } else {
        comp[t.id] = 1;
      }
    });
    return comp;
  }, [tracks, gainMatch]);

  /* apply gain compensation via GainNodes */
  /* volume + mute + solo + crossfader logic — all through GainNode for iOS compatibility */
  useEffect(() => {
    tracks.forEach((track, i) => {
      const node = gainNodes.current[track.id];
      if (!node) return;
      const comp = gainMatch ? (gainCompensation[track.id] ?? 1) : 1;
      let v;
      if (soloId.size > 0 && !soloId.has(track.id)) {
        v = 0;
      } else {
        v = track.muted ? 0 : track.volume;
        if (tracks.length >= 2) {
          if (abMode) {
            if (i === abTrackA) v *= 1 - abValue;
            else if (i === abTrackB) v *= abValue;
            else v = 0;
          } else if (showCrossfader) {
            v *= mixWeights[i] ?? 0;
          }
        }
      }
      node.gain.value = clamp(v, 0, 1) * comp;
    });
  }, [
    tracks,
    mixWeights,
    soloId,
    abMode,
    abTrackA,
    abTrackB,
    abValue,
    showCrossfader,
    gainMatch,
    gainCompensation,
  ]);

  /* apply pan */
  useEffect(() => {
    tracks.forEach((t) => {
      const panner = pannerNodes.current[t.id];
      if (panner) panner.pan.value = clamp(t.pan || 0, -1, 1);
    });
  }, [tracks]);

  /* cleanup */
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      cancelAnimationFrame(singleRafRef.current);
    };
  }, []);

  /* analyze per-track segment */
  useEffect(() => {
    const result = {};
    tracks.forEach((t) => {
      const buf = decodedBuffers.current[t.id];
      if (buf) {
        const ss = t.segStart || 0;
        const se = Math.min(ss + segmentLength, t.duration);
        result[t.id] = computeRms(buf, ss, se);
      }
    });
    setAnalyses(result);
    setFullAnalyses({});
  }, [tracks, segmentLength]);

  /* lazy full analysis when panel expanded */
  useEffect(() => {
    expandedAnalysis.forEach((id) => {
      if (fullAnalyses[id]) return;
      const buf = decodedBuffers.current[id];
      if (!buf) return;
      const t = tracks.find((tr) => tr.id === id);
      if (!t) return;
      const from = t.segStart || 0;
      const to = Math.min(from + segmentLength, t.duration || buf.duration);
      // run async-ish via setTimeout so UI isn't blocked
      setTimeout(() => {
        const result = analyzeTrackFull(buf, from, to);
        setFullAnalyses((prev) => ({ ...prev, [id]: result }));
      }, 0);
    });
  }, [expandedAnalysis, tracks, segmentLength, fullAnalyses]);

  /* ── animation ── */
  const updatePlayhead = () => {
    if (!isPlayingRef.current) return;
    const sl = segmentLengthRef.current;
    const elapsed =
      startOffsetRef.current +
      (performance.now() - startTimeRef.current) / 1000;
    if (elapsed >= sl) {
      if (loopRegionRef.current) {
        restartPlayAll();
        return;
      }
      stopPlayback();
      setPlayhead(sl);
      return;
    }
    // skip past skip zones for each playing track
    tracksRef.current.forEach((t) => {
      if (!t.audio || t.audio.paused) return;
      const ct = t.audio.currentTime;
      const zones = (t.skipZones || [])
        .slice()
        .sort((a, b) => a.start - b.start);
      for (const z of zones) {
        if (ct >= z.start && ct < z.end) {
          try {
            t.audio.currentTime = z.end;
          } catch (_) {}
          break;
        }
      }
    });
    setPlayhead(elapsed);
    rafRef.current = requestAnimationFrame(updatePlayhead);
  };

  const syncTimes = (time) => {
    tracksRef.current.forEach((t) => {
      if (!t.audio) return;
      try {
        const ss = t.segStart || 0;
        let target = ss + time;
        // skip past any skip zones
        const zones = (t.skipZones || [])
          .slice()
          .sort((a, b) => a.start - b.start);
        for (const z of zones) {
          if (target >= z.start && target < z.end) {
            target = z.end;
          }
        }
        t.audio.currentTime = clamp(target, 0, t.duration || 0);
      } catch (_) {}
    });
  };

  const stopTimers = () => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  };

  /* internal: stop without resetting single-play (avoids re-entrance) */
  const stopPlayback = () => {
    stopTimers();
    tracksRef.current.forEach((t) => t.audio?.pause());
    setIsPlaying(false);
    isPlayingRef.current = false;
  };

  /* internal: restart from 0 (called from rAF, avoids stale closure) */
  const restartPlayAll = () => {
    stopTimers();
    const sl = segmentLengthRef.current;
    const safe = 0;
    syncTimes(safe);
    if (audioCtxRef.current?.state === "suspended")
      audioCtxRef.current.resume();
    tracksRef.current.forEach((t) => {
      if (!t.audio) return;
      try {
        t.audio.play();
      } catch (_) {}
    });
    setPlayhead(safe);
    startOffsetRef.current = safe;
    startTimeRef.current = performance.now();
    rafRef.current = requestAnimationFrame(updatePlayhead);
  };

  const playAll = async (from = playhead) => {
    if (!tracksRef.current.length) return;
    setSinglePlayId(null);
    stopSingleAnim();
    setSinglePlayhead(null);
    const sl = segmentLengthRef.current;
    const safe = clamp(from, 0, Math.max(0, sl - 0.01));
    stopTimers();
    syncTimes(safe);
    if (audioCtxRef.current?.state === "suspended")
      await audioCtxRef.current.resume();
    for (const t of tracksRef.current) {
      if (!t.audio) continue;
      try {
        await t.audio.play();
      } catch (_) {}
    }
    setIsPlaying(true);
    isPlayingRef.current = true;
    setPlayhead(safe);
    startOffsetRef.current = safe;
    startTimeRef.current = performance.now();
    rafRef.current = requestAnimationFrame(updatePlayhead);
  };

  const stopSingleAnim = () => {
    cancelAnimationFrame(singleRafRef.current);
    singleTrackRef.current = null;
  };

  const pauseAll = () => {
    stopTimers();
    stopSingleAnim();
    tracksRef.current.forEach((t) => t.audio?.pause());
    setIsPlaying(false);
    isPlayingRef.current = false;
    setSinglePlayId(null);
    setSinglePlayhead(null);
  };

  /* individual play with dedicated playhead */
  const updateSinglePlayhead = () => {
    const t = singleTrackRef.current;
    if (!t?.audio) return;
    const ct = t.audio.currentTime;
    // auto-skip past skip zones
    const zones = (t.skipZones || []).slice().sort((a, b) => a.start - b.start);
    for (const z of zones) {
      if (ct >= z.start && ct < z.end) {
        t.audio.currentTime = z.end;
        break;
      }
    }
    setSinglePlayhead(t.audio.currentTime);
    if (!t.audio.paused) {
      singleRafRef.current = requestAnimationFrame(updateSinglePlayhead);
    } else {
      setSinglePlayhead(null);
      setSinglePlayId(null);
    }
  };

  const playSingle = useCallback(
    (id) => {
      if (singlePlayId === id) {
        pauseAll();
        return;
      }
      pauseAll();
      const t = tracks.find((t) => t.id === id);
      if (!t?.audio) return;
      if (audioCtxRef.current?.state === "suspended")
        audioCtxRef.current.resume();
      t.audio.currentTime = t.segStart || 0;
      t.audio.play();
      setSinglePlayId(id);
      singleTrackRef.current = t;
      singleRafRef.current = requestAnimationFrame(updateSinglePlayhead);
    },
    [tracks, singlePlayId],
  );

  /* ── file handling ── */
  const handleFiles = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const newTracks = files.map((file, idx) => {
      const url = URL.createObjectURL(file);
      const audio = new Audio(url);
      audio.preload = "metadata";
      return {
        id: `${file.name}-${file.size}-${Date.now()}-${idx}`,
        name: file.name,
        file,
        url,
        audio,
        duration: 0,
        muted: false,
        volume: 1,
        pan: 0,
        segStart: 0,
        skipZones: [],
        peaks: null,
      };
    });
    newTracks.forEach((t) => {
      t.audio.addEventListener("loadedmetadata", () => {
        setTracks((prev) =>
          prev.map((p) =>
            p.id === t.id
              ? {
                  ...p,
                  duration: Number.isFinite(t.audio.duration)
                    ? t.audio.duration
                    : 0,
                }
              : p,
          ),
        );
      });
      // Set up Web Audio graph for panning
      const setupPanner = () => {
        if (!audioCtxRef.current)
          audioCtxRef.current = new (
            window.AudioContext || window.webkitAudioContext
          )();
        const ctx = audioCtxRef.current;
        try {
          const source = ctx.createMediaElementSource(t.audio);
          const gain = ctx.createGain();
          const panner = ctx.createStereoPanner();
          source.connect(gain);
          gain.connect(panner);
          panner.connect(ctx.destination);
          gainNodes.current[t.id] = gain;
          pannerNodes.current[t.id] = panner;
        } catch (_) {}
      };
      setupPanner();

      t.file.arrayBuffer().then((buf) => {
        audioCtxRef.current
          .decodeAudioData(buf.slice(0))
          .then((decoded) => {
            decodedBuffers.current[t.id] = decoded;
            const rmsData = computeRms(decoded, 0, decoded.duration);
            setTracks((prev) =>
              prev.map((p) =>
                p.id === t.id
                  ? { ...p, peaks: computePeaks(decoded), rms: rmsData.rms }
                  : p,
              ),
            );
          })
          .catch(() => {});
      });
    });
    setTracks((prev) => [...prev, ...newTracks]);
  };

  const removeTrack = useCallback((id) => {
    setTracks((prev) => {
      const f = prev.find((t) => t.id === id);
      if (f?.audio) f.audio.pause();
      if (f?.url) URL.revokeObjectURL(f.url);
      delete decodedBuffers.current[id];
      delete pannerNodes.current[id];
      delete gainNodes.current[id];
      return prev.filter((t) => t.id !== id);
    });
  }, []);

  const toggleAnalysisExpand = useCallback(
    (id) => {
      setExpandedAnalysis((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    [],
  );

  const toggleMute = useCallback((id) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, muted: !t.muted } : t)),
    );
  }, []);

  const toggleSolo = useCallback((id) => {
    setSoloId((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const setVolume = useCallback((id, v) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, volume: Number(v) } : t)),
    );
  }, []);

  const setPan = useCallback((id, v) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, pan: Number(v) } : t)),
    );
  }, []);

  const setSegStart = useCallback((id, v) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, segStart: Number(v) } : t)),
    );
  }, []);

  const setSegmentByDrag = useCallback((id, start, len) => {
    setSegmentLength(len);
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, segStart: Number(start) } : t)),
    );
  }, []);

  const skipSegment = useCallback(
    (id, dir) => {
      setTracks((prev) =>
        prev.map((t) => {
          if (t.id !== id) return t;
          const newStart = (t.segStart || 0) + dir * segmentLength;
          return {
            ...t,
            segStart: clamp(
              newStart,
              0,
              Math.max(0, (t.duration || 0) - segmentLength),
            ),
          };
        }),
      );
    },
    [segmentLength],
  );

  const addSkipZone = useCallback((id, start, end) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const zones = [...(t.skipZones || []), { start, end }].sort(
          (a, b) => a.start - b.start,
        );
        return { ...t, skipZones: zones };
      }),
    );
  }, []);

  const removeSkipZone = useCallback((id, index) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const zones = (t.skipZones || []).filter((_, i) => i !== index);
        return { ...t, skipZones: zones };
      }),
    );
  }, []);

  const editSkipZone = useCallback((id, index, newStart, newEnd) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const zones = (t.skipZones || [])
          .map((z, i) =>
            i === index
              ? {
                  start: Math.min(newStart, newEnd),
                  end: Math.max(newStart, newEnd),
                }
              : z,
          )
          .sort((a, b) => a.start - b.start);
        return { ...t, skipZones: zones };
      }),
    );
  }, []);

  const punchIn = useCallback(
    (id, mode) => {
      const t = tracks.find((tr) => tr.id === id);
      const pos =
        t?.audio && !t.audio.paused
          ? t.audio.currentTime
          : (t?.segStart || 0) + playhead;
      setPunchState({ trackId: id, mode, inTime: pos });
    },
    [tracks, playhead],
  );

  const punchOut = useCallback(
    (id) => {
      if (!punchState || punchState.trackId !== id) return;
      const t = tracks.find((tr) => tr.id === id);
      const pos =
        t?.audio && !t.audio.paused
          ? t.audio.currentTime
          : (t?.segStart || 0) + playhead;
      const s = Math.min(punchState.inTime, pos);
      const e = Math.max(punchState.inTime, pos);
      if (e - s < 0.01) {
        setPunchState(null);
        return;
      }
      if (punchState.mode === "segment") {
        setSegmentLength(e - s);
        setTracks((prev) =>
          prev.map((tr) => (tr.id === id ? { ...tr, segStart: s } : tr)),
        );
      } else {
        setTracks((prev) =>
          prev.map((tr) => {
            if (tr.id !== id) return tr;
            const zones = [...(tr.skipZones || []), { start: s, end: e }].sort(
              (a, b) => a.start - b.start,
            );
            return { ...tr, skipZones: zones };
          }),
        );
      }
      setPunchState(null);
    },
    [punchState, tracks, playhead],
  );

  const seekTo = useCallback(
    (time) => {
      const target = clamp(time, 0, segmentLength || maxDuration || 0);
      setPlayhead(target);
      syncTimes(target);
      if (isPlaying) playAll(target);
    },
    [segmentLength, maxDuration, isPlaying],
  );

  /* ── render ── */
  return (
    <div className="min-h-screen bg-[#fdf6f0] text-[#3a1a00]">
      {/* ─ Transport bar ─ */}
      <div className="sticky top-0 z-30 bg-gradient-to-r from-[#3a1500]/95 to-[#5a2a08]/95 backdrop-blur border-b border-[#7a3a10]/40 shadow-lg">
        {/* Row 1: logo, title, main controls */}
        <div className="mx-auto max-w-7xl px-4 pt-2.5 pb-1 flex items-center gap-3 flex-wrap">
          <svg viewBox="0 0 64 64" className="w-9 h-9 shrink-0 -mr-1">
            <defs>
              <radialGradient id="gr" cx="35%" cy="30%" r="50%">
                <stop offset="0%" stopColor="#ff9a7b" />
                <stop offset="60%" stopColor="#e8453c" />
                <stop offset="100%" stopColor="#b22a1a" />
              </radialGradient>
              <radialGradient id="go1" cx="35%" cy="30%" r="50%">
                <stop offset="0%" stopColor="#ffcf5e" />
                <stop offset="60%" stopColor="#f5a623" />
                <stop offset="100%" stopColor="#c97b0a" />
              </radialGradient>
              <radialGradient id="go2" cx="35%" cy="30%" r="50%">
                <stop offset="0%" stopColor="#ffe08a" />
                <stop offset="60%" stopColor="#f0892d" />
                <stop offset="100%" stopColor="#bf5e0a" />
              </radialGradient>
            </defs>
            <circle
              cx="32"
              cy="18"
              r="15"
              fill="url(#go2)"
              stroke="#4a1a00"
              strokeWidth="2"
            />
            <ellipse cx="28" cy="13" rx="4" ry="3" fill="white" opacity="0.6" />
            <circle
              cx="19"
              cy="42"
              r="15"
              fill="url(#gr)"
              stroke="#4a1a00"
              strokeWidth="2"
            />
            <ellipse cx="15" cy="37" rx="4" ry="3" fill="white" opacity="0.6" />
            <circle
              cx="45"
              cy="42"
              r="15"
              fill="url(#go1)"
              stroke="#4a1a00"
              strokeWidth="2"
            />
            <ellipse cx="41" cy="37" rx="4" ry="3" fill="white" opacity="0.6" />
          </svg>
          <div className="mr-2">
            <h1 className="text-lg font-bold tracking-tight leading-tight text-[#ffe0b2]">
              Caviar Mixer
            </h1>
            <p className="text-[10px] font-medium text-[#c09060] leading-none">
              mix and compare audio
            </p>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-[#f5a623] px-3 py-1.5 text-xs font-medium text-[#3a1500] hover:bg-[#f0892d]"
          >
            <Upload size={14} /> Add files
          </button>
          <div className="h-6 w-px bg-[#7a3a10]/40" />
          <button
            onClick={() => (isPlaying ? pauseAll() : playAll(0))}
            disabled={!tracks.length}
            className="inline-flex items-center gap-1.5 rounded-xl border border-[#7a3a10]/40 px-3 py-1.5 text-xs font-medium text-[#ffe0b2] hover:bg-[#5a2a08] disabled:opacity-40"
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? "Pause all" : "Play all"}
          </button>
          <button
            onClick={() => seekTo(0)}
            disabled={!tracks.length}
            className="p-1.5 rounded-lg text-[#c09060] hover:bg-[#5a2a08] disabled:opacity-40"
            title="To start"
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => setLoopRegion(!loopRegion)}
            className={`p-1.5 rounded-lg ${loopRegion ? "bg-[#f5a623]/20 text-[#f5a623]" : "hover:bg-[#5a2a08] text-[#c09060]"}`}
            title="Loop region"
          >
            <Repeat size={14} />
          </button>
          {tracks.length >= 2 && (
            <button
              onClick={() => {
                setAbMode(false);
                setShowCrossfader(true);
                setShowABFader(false);
              }}
              className="inline-flex items-center gap-1.5 rounded-xl border border-[#e8453c]/40 bg-[#e8453c]/15 px-3 py-1.5 text-xs font-medium text-[#ff9a7b] hover:bg-[#e8453c]/25"
            >
              <BarChart3 size={14} /> Geometric
            </button>
          )}
          {tracks.length >= 2 && (
            <button
              onClick={() => {
                setAbMode(true);
                setShowABFader(true);
                setShowCrossfader(false);
              }}
              className="inline-flex items-center gap-1.5 rounded-xl border border-[#f5a623]/40 bg-[#f5a623]/15 px-3 py-1.5 text-xs font-medium text-[#ffcf5e] hover:bg-[#f5a623]/25"
            >
              <Headphones size={14} /> A/B Fader
            </button>
          )}
          {tracks.length >= 2 && (
            <button
              onClick={() => setGainMatch((p) => !p)}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium ${
                gainMatch
                  ? "border-[#f5a623]/50 bg-[#f5a623]/20 text-[#ffcf5e]"
                  : "border-[#7a3a10]/40 hover:bg-[#5a2a08] text-[#c09060]"
              }`}
              title="Level match all tracks to equal loudness"
            >
              <Volume2 size={14} /> {gainMatch ? "Matched" : "Gain Match"}
            </button>
          )}
          {tracks.length >= 2 && (
            <button
              onClick={() => {
                setCompareA(tracks[0]?.id || null);
                setCompareB(tracks[1]?.id || null);
                setComparing(true);
              }}
              className="inline-flex items-center gap-1.5 rounded-xl border border-[#d4a87a]/40 bg-[#d4a87a]/15 px-3 py-1.5 text-xs font-medium text-[#ffe0b2] hover:bg-[#d4a87a]/25"
            >
              <GitCompareArrows size={14} /> Compare
            </button>
          )}
        </div>
        {/* Row 2: segment length + playhead */}
        <div className="mx-auto max-w-7xl px-4 pb-2 flex items-center gap-4 flex-wrap">
          {/* segment length */}
          <div className="flex items-center gap-2 text-[11px] text-[#c09060]">
            <Lock size={12} className="text-[#f5a623]" />
            <span>Segment:</span>
            <input
              type="range"
              min="0.1"
              max={maxDuration || 1}
              step="0.01"
              value={segmentLength}
              onChange={(e) => setSegmentLength(Number(e.target.value))}
              className="w-28 accent-[#f5a623]"
              style={{ height: "4px" }}
            />
            <span className="tabular-nums font-medium text-[#ffe0b2]">
              {fmt(segmentLength)}
            </span>
          </div>
          <div className="h-4 w-px bg-[#7a3a10]/40" />
          {/* playhead */}
          <div className="flex items-center gap-1.5 text-[11px] text-[#c09060]">
            <span className="font-medium text-[#e8453c] tabular-nums">
              {fmt(playhead)}
            </span>
            <span className="text-[#7a3a10]">/</span>
            <span className="tabular-nums text-[#ffe0b2]">
              {fmt(segmentLength)}
            </span>
            <input
              type="range"
              min="0"
              max={segmentLength || 1}
              step="0.01"
              value={playhead}
              onChange={(e) => seekTo(Number(e.target.value))}
              className="w-32 accent-[#e8453c]"
              style={{ height: "4px" }}
            />
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*,.wav,.mp3,.ogg,.flac,.aac,.m4a,.wma,.aiff,.mp4,.webm,.mov,.mkv"
            multiple
            className="hidden"
            onChange={handleFiles}
          />
        </div>
      </div>

      <div className="mx-auto max-w-7xl p-4 space-y-3">
        {/* ─ Tracks ─ */}
        {tracks.length === 0 && (
          <div className="rounded-2xl border border-dashed border-[#d4a87a] bg-white/80 p-12 text-center text-sm text-[#9a6a40]">
            No audio or video files loaded yet. Click "Add files" to get
            started. Video files will have their audio extracted automatically.
          </div>
        )}

        {tracks.map((track, i) => (
          <div
            key={track.id}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
            }}
            onDrop={() => {
              if (dragId != null && dragId !== track.id) {
                setTracks((prev) => {
                  const arr = [...prev];
                  const fromIdx = arr.findIndex((t) => t.id === dragId);
                  const toIdx = arr.findIndex((t) => t.id === track.id);
                  if (fromIdx < 0 || toIdx < 0) return prev;
                  const [moved] = arr.splice(fromIdx, 1);
                  arr.splice(toIdx, 0, moved);
                  return arr;
                });
              }
              setDragId(null);
            }}
            onDragEnd={() => setDragId(null)}
            className={`transition-opacity ${dragId === track.id ? "opacity-40" : ""}`}
          >
            <div className="flex items-start gap-1">
              <div
                draggable
                onDragStart={(e) => {
                  setDragId(track.id);
                  e.dataTransfer.effectAllowed = "move";
                }}
                className="pt-4 cursor-grab active:cursor-grabbing text-[#d4a87a] hover:text-[#9a6a40]"
              >
                <GripVertical size={16} />
              </div>
              <div className="flex-1 min-w-0">
                <WaveformTrack
                  track={track}
                  index={i}
                  maxDuration={maxDuration}
                  segmentLength={segmentLength}
                  playhead={playhead}
                  singlePlayhead={
                    singlePlayId === track.id ? singlePlayhead : null
                  }
                  isSolo={soloId.has(track.id)}
                  onSeek={seekTo}
                  onSetSegStart={setSegStart}
                  onSetSegmentByDrag={setSegmentByDrag}
                  onSkipSegment={skipSegment}
                  onAddSkipZone={addSkipZone}
                  onRemoveSkipZone={removeSkipZone}
                  onEditSkipZone={editSkipZone}
                  punchState={punchState}
                  onPunchIn={punchIn}
                  onPunchOut={punchOut}
                  onToggleMute={toggleMute}
                  onToggleSolo={toggleSolo}
                  onSetVolume={setVolume}
                  onSetPan={setPan}
                  onRemove={removeTrack}
                  onPlaySingle={playSingle}
                  singlePlaying={singlePlayId === track.id}
                  color={TRACK_COLORS[i % TRACK_COLORS.length]}
                  analysis={analyses[track.id]}
                  fullAnalysis={fullAnalyses[track.id]}
                  analysisExpanded={expandedAnalysis.has(track.id)}
                  onToggleAnalysis={toggleAnalysisExpand}
                />
              </div>
            </div>
          </div>
        ))}

        {/* ─ Geometric Crossfader Modal ─ */}
        {showCrossfader && tracks.length >= 2 && (
          <div
            className="fixed inset-x-0 top-[3.5rem] bottom-0 z-40 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={(e) => {
              if (e.target === e.currentTarget) setShowCrossfader(false);
            }}
          >
            <div className="bg-[#fdf6f0] rounded-3xl shadow-2xl w-[96vw] max-w-[1100px] max-h-[calc(96vh-3.5rem)] overflow-auto p-4 sm:p-6 relative">
              {/* header */}
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h2 className="text-xl font-bold text-[#3a1a00]">
                    Geometric Crossfader
                  </h2>
                  <p className="text-sm text-[#9a6a40] mt-0.5">
                    Drag the point to mix between tracks
                  </p>
                </div>
                <button
                  onClick={() => setShowCrossfader(false)}
                  className="p-2 rounded-xl hover:bg-[#f5e6d0] text-[#9a6a40]"
                >
                  <X size={20} />
                </button>
              </div>

              {/* ── playback progress bar ── */}
              {(() => {
                const maxDur = Math.max(
                  ...tracks.map((t) => t.duration || 0),
                  1,
                );
                const total = segmentLength > 0 ? segmentLength : maxDur;
                const pct =
                  total > 0 ? Math.min((playhead / total) * 100, 100) : 0;
                const seekFromPointer = (ev, bar) => {
                  const rect = bar.getBoundingClientRect();
                  const x = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
                  seekTo(x * total);
                };
                return (
                  <div className="mb-5">
                    <div className="flex items-center justify-between text-xs text-[#9a6a40] mb-1">
                      <span>{playhead.toFixed(1)}s</span>
                      <span>{isPlaying ? "Playing" : "Stopped"}</span>
                      <span>{total.toFixed(1)}s</span>
                    </div>
                    <div
                      className="h-3 rounded-full bg-[#e8d5c0] overflow-hidden cursor-pointer relative select-none"
                      onPointerDown={(e) => {
                        const bar = e.currentTarget;
                        bar.setPointerCapture(e.pointerId);
                        seekFromPointer(e, bar);
                        const move = (ev) => seekFromPointer(ev, bar);
                        const up = () => {
                          bar.removeEventListener("pointermove", move);
                          bar.removeEventListener("pointerup", up);
                        };
                        bar.addEventListener("pointermove", move);
                        bar.addEventListener("pointerup", up);
                      }}
                    >
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#e8453c] to-[#f5a623] pointer-events-none"
                        style={{ width: `${pct}%` }}
                      />
                      {/* drag handle */}
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-white border-2 border-[#e8453c] shadow pointer-events-none"
                        style={{ left: `calc(${pct}% - 8px)` }}
                      />
                    </div>
                  </div>
                );
              })()}

              <div className="flex gap-4 sm:gap-6 flex-wrap items-start justify-center">
                {/* polygon canvas */}
                <svg
                  width="540"
                  height="540"
                  viewBox="-1.5 -1.5 3 3"
                  style={{
                    touchAction: "none",
                    maxWidth: "min(540px, 88vw)",
                    maxHeight: "min(540px, 88vw)",
                  }}
                  className="border border-[#d4a87a]/40 rounded-2xl bg-gradient-to-br from-[#fdf6f0] to-[#f5e6d0] cursor-crosshair select-none shrink-0 w-full h-auto"
                  onPointerDown={(e) => {
                    e.preventDefault();
                    e.currentTarget.setPointerCapture(e.pointerId);
                    const svg = e.currentTarget;
                    const pt = svg.createSVGPoint();
                    const move = (ev) => {
                      ev.preventDefault();
                      pt.x = ev.clientX;
                      pt.y = ev.clientY;
                      const svgP = pt.matrixTransform(
                        svg.getScreenCTM().inverse(),
                      );
                      const r = tracks.length === 2 ? 1.05 : 1.2;
                      const x = clamp(svgP.x, -r, r);
                      const y = clamp(svgP.y, -r, r);
                      setMixPos({ x, y });
                    };
                    move(e);
                    const up = () => {
                      window.removeEventListener("pointermove", move);
                      window.removeEventListener("pointerup", up);
                    };
                    window.addEventListener("pointermove", move, {
                      passive: false,
                    });
                    window.addEventListener("pointerup", up);
                  }}
                >
                  <defs>
                    {/* radial glows for each track color */}
                    {tracks.map((_, i) => {
                      const c = TRACK_COLORS[i % TRACK_COLORS.length];
                      return (
                        <radialGradient
                          key={`rg-${i}`}
                          id={`vtxGlow-${i}`}
                          cx="50%"
                          cy="50%"
                          r="50%"
                        >
                          <stop offset="0%" stopColor={c} stopOpacity="0.7" />
                          <stop offset="60%" stopColor={c} stopOpacity="0.2" />
                          <stop offset="100%" stopColor={c} stopOpacity="0" />
                        </radialGradient>
                      );
                    })}
                  </defs>
                  {/* subtle grid circles */}
                  {[0.33, 0.66, 1].map((r) => (
                    <circle
                      key={r}
                      cx="0"
                      cy="0"
                      r={r}
                      fill="none"
                      stroke="#d4a87a"
                      strokeWidth="0.01"
                      strokeDasharray="0.04 0.04"
                    />
                  ))}
                  {/* polygon fill */}
                  {polyVertices.length >= 3 && (
                    <polygon
                      points={polyVertices
                        .map((v) => `${v.x},${v.y}`)
                        .join(" ")}
                      fill="#f5e6d0"
                      stroke="#d4a87a"
                      strokeWidth="0.03"
                    />
                  )}
                  {/* colored wedge fills showing weight distribution */}
                  {polyVertices.length >= 3 &&
                    polyVertices.map((v, i) => {
                      const w = mixWeights[i] ?? 0;
                      if (w <= 0.01) return null;
                      const c = TRACK_COLORS[i % TRACK_COLORS.length];
                      const n = polyVertices.length;
                      const prev = polyVertices[(i - 1 + n) % n];
                      const next = polyVertices[(i + 1) % n];
                      const midPx = (v.x + prev.x) / 2,
                        midPy = (v.y + prev.y) / 2;
                      const midNx = (v.x + next.x) / 2,
                        midNy = (v.y + next.y) / 2;
                      return (
                        <polygon
                          key={`wedge-${i}`}
                          points={`0,0 ${midPx},${midPy} ${v.x},${v.y} ${midNx},${midNy}`}
                          fill={c}
                          opacity={w * 0.35}
                        />
                      );
                    })}
                  {/* line for 2 tracks */}
                  {polyVertices.length === 2 && (
                    <>
                      <line
                        x1={polyVertices[0].x}
                        y1={polyVertices[0].y}
                        x2={polyVertices[1].x}
                        y2={polyVertices[1].y}
                        stroke="#d4a87a"
                        strokeWidth="0.03"
                      />
                      {/* colored halves for 2-track mode */}
                      {polyVertices.map((v, i) => {
                        const w = mixWeights[i] ?? 0;
                        if (w <= 0.01) return null;
                        const c = TRACK_COLORS[i % TRACK_COLORS.length];
                        return (
                          <rect
                            key={`half-${i}`}
                            x={i === 0 ? -1.5 : 0}
                            y={-1.5}
                            width={1.5}
                            height={3}
                            fill={c}
                            opacity={w * 0.2}
                          />
                        );
                      })}
                    </>
                  )}
                  {/* glow halos at vertices proportional to weight */}
                  {polyVertices.map((v, i) => {
                    const w = mixWeights[i] ?? 0;
                    if (w <= 0.02 || tracks[i]?.muted) return null;
                    return (
                      <circle
                        key={`glow-${i}`}
                        cx={v.x}
                        cy={v.y}
                        r={0.15 + w * 0.35}
                        fill={`url(#vtxGlow-${i})`}
                      />
                    );
                  })}
                  {/* thick weight beams from cursor to each active vertex */}
                  {polyVertices.map((v, i) => {
                    const w = mixWeights[i] ?? 0;
                    if (w <= 0) return null;
                    const c = TRACK_COLORS[i % TRACK_COLORS.length];
                    return (
                      <line
                        key={`wl-${i}`}
                        x1={mixPos.x}
                        y1={mixPos.y}
                        x2={v.x}
                        y2={v.y}
                        stroke={c}
                        strokeWidth={0.02 + w * 0.14}
                        opacity={0.4 + w * 0.6}
                        strokeLinecap="round"
                      />
                    );
                  })}
                  {/* lines from center to vertices */}
                  {polyVertices.length >= 3 &&
                    polyVertices.map((v, i) => (
                      <line
                        key={i}
                        x1="0"
                        y1="0"
                        x2={v.x}
                        y2={v.y}
                        stroke="#d4a87a"
                        strokeWidth="0.015"
                      />
                    ))}
                  {/* vertex dots + labels */}
                  {polyVertices.map((v, i) => {
                    const c = TRACK_COLORS[i % TRACK_COLORS.length];
                    const muted = tracks[i]?.muted;
                    const w = mixWeights[i] ?? 0;
                    const n = tracks.length;
                    const labelR = n === 2 ? 1.28 : 1.32;
                    const angle =
                      n === 2
                        ? i === 0
                          ? Math.PI
                          : 0
                        : -Math.PI / 2 + (2 * Math.PI * i) / n;
                    const lx = Math.cos(angle) * labelR;
                    const ly = Math.sin(angle) * labelR;
                    const dotR = 0.1 + w * 0.1;
                    return (
                      <g key={i} opacity={muted ? 0.3 : 1}>
                        {/* outer pulse ring */}
                        {!muted && w > 0.05 && (
                          <circle
                            cx={v.x}
                            cy={v.y}
                            r={dotR + 0.04 + w * 0.06}
                            fill="none"
                            stroke={c}
                            strokeWidth="0.02"
                            opacity={w * 0.6}
                          />
                        )}
                        <circle
                          cx={v.x}
                          cy={v.y}
                          r={dotR}
                          fill={muted ? "#94a3b8" : c}
                          stroke="white"
                          strokeWidth="0.03"
                        />
                        {muted && (
                          <line
                            x1={v.x - 0.08}
                            y1={v.y - 0.08}
                            x2={v.x + 0.08}
                            y2={v.y + 0.08}
                            stroke="#ef4444"
                            strokeWidth="0.025"
                          />
                        )}
                        <text
                          x={lx}
                          y={ly}
                          textAnchor="middle"
                          dominantBaseline="central"
                          fontSize="0.17"
                          fill={muted ? "#94a3b8" : c}
                          fontWeight="700"
                        >
                          {i + 1}
                        </text>
                        {!muted && w > 0.01 && (
                          <text
                            x={lx}
                            y={ly + (ly > 0 ? 0.13 : -0.13)}
                            textAnchor="middle"
                            dominantBaseline="central"
                            fontSize="0.10"
                            fill={c}
                            fontFamily="monospace"
                            fontWeight="700"
                          >
                            {(w * 100).toFixed(0)}%
                          </text>
                        )}
                      </g>
                    );
                  })}
                  {/* center dot */}
                  <circle cx="0" cy="0" r="0.04" fill="#d4a87a" />
                  {/* mix position indicator */}
                  <circle
                    cx={mixPos.x}
                    cy={mixPos.y}
                    r="0.18"
                    fill="#fdf6f0"
                    stroke="#3a1a00"
                    strokeWidth="0.05"
                    style={{
                      filter: "drop-shadow(0 0.03px 0.08px rgba(0,0,0,0.35))",
                    }}
                  />
                  <circle cx={mixPos.x} cy={mixPos.y} r="0.08" fill="#3a1a00" />
                </svg>

                {/* track list with controls */}
                <div className="flex-1 min-w-[260px] max-h-[50vh] sm:max-h-[540px] overflow-y-auto space-y-1">
                  {tracks.map((t, i) => {
                    const w = mixWeights[i] ?? 0;
                    const c = TRACK_COLORS[i % TRACK_COLORS.length];
                    const muted = t.muted;
                    return (
                      <div
                        key={t.id}
                        className={`relative rounded-xl border overflow-hidden ${muted ? "opacity-50 border-[#d4a87a]/40" : "border-[#d4a87a]/40"}`}
                        style={{
                          borderLeftWidth: "4px",
                          borderLeftColor: muted ? "#94a3b8" : c,
                        }}
                      >
                        {/* waveform background */}
                        <MiniWave
                          id={t.id}
                          peaks={t.peaks}
                          color={c}
                          muted={muted}
                          progress={t.duration ? playhead / t.duration : 0}
                        />
                        {/* mix weight background fill */}
                        <div
                          className="absolute inset-y-0 left-0 transition-all duration-150 pointer-events-none rounded-r-xl"
                          style={{
                            width: `${(w * 100).toFixed(0)}%`,
                            background: muted
                              ? "#f5e6d0"
                              : `linear-gradient(90deg, ${c}50 0%, ${c}30 50%, ${c}10 100%)`,
                          }}
                        />
                        {/* weight percentage bar at bottom */}
                        {!muted && w > 0.01 && (
                          <div
                            className="absolute bottom-0 left-0 h-1 transition-all duration-150 pointer-events-none"
                            style={{
                              width: `${(w * 100).toFixed(0)}%`,
                              background: c,
                              opacity: 0.7,
                            }}
                          />
                        )}
                        <div className="relative px-2.5 py-1.5 space-y-1">
                          {/* row 1: number + name + reorder + mute */}
                          <div className="flex items-center gap-2">
                            <span
                              className="text-sm font-extrabold rounded-lg px-2 py-0.5 shrink-0 min-w-[1.75rem] text-center"
                              style={{
                                background: muted ? "#e8d5c0" : `${c}22`,
                                color: muted ? "#94a3b8" : c,
                              }}
                            >
                              {i + 1}
                            </span>
                            <span
                              className={`text-sm flex-1 truncate ${muted ? "line-through text-[#9a6a40]" : "text-[#3a1a00] font-medium"}`}
                              title={t.name}
                            >
                              {t.name}
                            </span>
                            <span
                              className={`font-mono text-xs shrink-0 ${muted ? "text-[#b89070]" : "text-[#5a3520]"}`}
                            >
                              {muted ? "MUTE" : `${(w * 100).toFixed(0)}%`}
                            </span>
                            {/* reorder buttons */}
                            <button
                              onClick={() => {
                                if (i === 0) return;
                                setTracks((prev) => {
                                  const arr = [...prev];
                                  [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
                                  return arr;
                                });
                              }}
                              disabled={i === 0}
                              className="p-0.5 rounded hover:bg-[#f5e6d0] disabled:opacity-20 text-[#9a6a40]"
                              title="Move up"
                            >
                              <ChevronUp size={14} />
                            </button>
                            <button
                              onClick={() => {
                                if (i === tracks.length - 1) return;
                                setTracks((prev) => {
                                  const arr = [...prev];
                                  [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]];
                                  return arr;
                                });
                              }}
                              disabled={i === tracks.length - 1}
                              className="p-0.5 rounded hover:bg-[#f5e6d0] disabled:opacity-20 text-[#9a6a40]"
                              title="Move down"
                            >
                              <ChevronDown size={14} />
                            </button>
                            <button
                              onClick={() =>
                                setSoloId((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(t.id)) next.delete(t.id);
                                  else next.add(t.id);
                                  return next;
                                })
                              }
                              className={`p-2 rounded-lg text-xs font-bold min-w-[2.5rem] min-h-[2.5rem] flex items-center justify-center ${soloId.has(t.id) ? "bg-[#e8453c] text-white" : "hover:bg-[#f5e6d0] text-[#9a6a40]"}`}
                            >
                              <Headphones size={16} />
                            </button>
                            <button
                              onClick={() => toggleMute(t.id)}
                              className={`p-2 rounded-lg min-w-[2.5rem] min-h-[2.5rem] flex items-center justify-center ${muted ? "bg-red-100 text-red-500" : "hover:bg-[#f5e6d0] text-[#9a6a40]"}`}
                            >
                              {muted ? (
                                <VolumeX size={16} />
                              ) : (
                                <Volume2 size={16} />
                              )}
                            </button>
                          </div>
                          {/* row 2: volume + pan sliders */}
                          <div className="flex items-center gap-2 text-xs text-[#9a6a40]">
                            <Volume2
                              size={12}
                              className="shrink-0 text-[#d4a87a]"
                            />
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.01"
                              value={t.volume}
                              onChange={(e) =>
                                setVolume(t.id, Number(e.target.value))
                              }
                              className="flex-1 min-h-[2rem]"
                            />
                            <span className="w-8 text-right font-mono text-[10px] text-[#9a6a40]">
                              {Math.round(t.volume * 100)}
                            </span>
                            <span className="text-[10px] text-[#d4a87a] ml-1">
                              L
                            </span>
                            <input
                              type="range"
                              min="-1"
                              max="1"
                              step="0.01"
                              value={t.pan || 0}
                              onChange={(e) =>
                                setPan(t.id, Number(e.target.value))
                              }
                              className="w-16 sm:w-20 min-h-[2rem]"
                            />
                            <span className="text-[10px] text-[#d4a87a]">
                              R
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  <div className="pt-3 flex items-center justify-between">
                    <button
                      onClick={() => setMixPos({ x: 0, y: 0 })}
                      className="rounded-lg border border-[#d4a87a]/40 px-4 py-1.5 text-xs font-medium text-[#3a1a00] hover:bg-[#f5e6d0] transition-colors"
                    >
                      Reset to center
                    </button>
                    <button
                      onClick={() => {
                        setAbMode(true);
                        setShowCrossfader(false);
                        setShowABFader(true);
                      }}
                      className="inline-flex items-center gap-1.5 text-xs text-[#f5a623] hover:text-[#e8453c] font-medium"
                    >
                      <Headphones size={14} /> Switch to A/B Fader
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Compare Modal ── */}
        {comparing && tracks.length >= 2 && (
          <div
            className="fixed inset-x-0 top-[3.5rem] bottom-0 z-40 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={(e) => {
              if (e.target === e.currentTarget) setComparing(false);
            }}
          >
            <div className="bg-[#fdf6f0] rounded-3xl shadow-2xl w-[96vw] max-w-[800px] max-h-[calc(96vh-3.5rem)] overflow-auto p-4 sm:p-6 relative">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-[#3a1a00] flex items-center gap-2">
                  <GitCompareArrows size={20} /> Compare Tracks
                </h2>
                <button
                  onClick={() => setComparing(false)}
                  className="rounded-full p-1 hover:bg-[#5a2a00]/10 text-[#9a6a40]"
                >
                  <X size={18} />
                </button>
              </div>

              {/* Track selectors */}
              <div className="flex flex-wrap gap-4 mb-5">
                <label className="flex items-center gap-2 text-sm text-[#5a2a00]">
                  <span className="font-semibold">Track A:</span>
                  <select
                    value={compareA || ""}
                    onChange={(e) => setCompareA(e.target.value)}
                    className="rounded-lg border border-[#d4a87a] bg-white px-2 py-1 text-sm text-[#3a1a00]"
                  >
                    {tracks.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-sm text-[#5a2a00]">
                  <span className="font-semibold">Track B:</span>
                  <select
                    value={compareB || ""}
                    onChange={(e) => setCompareB(e.target.value)}
                    className="rounded-lg border border-[#d4a87a] bg-white px-2 py-1 text-sm text-[#3a1a00]"
                  >
                    {tracks.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {/* Comparison results */}
              {(() => {
                const bufA = decodedBuffers.current[compareA];
                const bufB = decodedBuffers.current[compareB];
                if (!bufA || !bufB || compareA === compareB) {
                  return (
                    <p className="text-sm text-[#9a6a40] italic">
                      {compareA === compareB
                        ? "Select two different tracks to compare."
                        : "Waiting for decoded audio…"}
                    </p>
                  );
                }
                const tA = tracks.find((t) => t.id === compareA);
                const tB = tracks.find((t) => t.id === compareB);
                if (!tA || !tB) return null;
                const fromA = tA.segStart || 0;
                const toA = Math.min(fromA + segmentLength, tA.duration);
                const fromB = tB.segStart || 0;
                const toB = Math.min(fromB + segmentLength, tB.duration);
                const aFull = fullAnalyses[compareA] || analyzeTrackFull(bufA, fromA, toA);
                const bFull = fullAnalyses[compareB] || analyzeTrackFull(bufB, fromB, toB);
                const specSim = spectralSimilarity(
                  aFull.spectrum,
                  bFull.spectrum,
                );
                const corrNormal = crossCorrelation(bufA, bufB, fromA, toA, fromB, toB, false);
                const corrInverted = crossCorrelation(bufA, bufB, fromA, toA, fromB, toB, true);
                const isPhaseInverted = corrInverted > corrNormal + 0.05;
                const corr = Math.max(corrNormal, corrInverted);

                // natural-language conclusion
                const conclusions = [];
                const rmsAdb = 20 * Math.log10(Math.max(aFull.rms, 1e-10));
                const rmsBdb = 20 * Math.log10(Math.max(bFull.rms, 1e-10));
                const rmsDiff = Math.abs(rmsAdb - rmsBdb);
                if (rmsDiff > 3)
                  conclusions.push(
                    `${rmsAdb > rmsBdb ? tA.name : tB.name} is noticeably louder (${rmsDiff.toFixed(1)} dB).`,
                  );
                else if (rmsDiff > 1)
                  conclusions.push(
                    `Small volume difference (${rmsDiff.toFixed(1)} dB).`,
                  );
                else
                  conclusions.push("Tracks are approximately the same volume.");

                if (specSim != null) {
                  if (specSim > 0.95) conclusions.push("Nearly identical tonal color.");
                  else if (specSim > 0.8) conclusions.push("Relatively similar tonal color.");
                  else if (specSim > 0.5) conclusions.push("Noticeably different tonal color.");
                  else conclusions.push("Very different tonal color.");
                }

                if (corr != null) {
                  if (corr > 0.9) conclusions.push("Tracks are very similar (high correlation).");
                  else if (corr > 0.6) conclusions.push("Tracks share many similarities.");
                  else if (corr > 0.3) conclusions.push("Tracks share some commonality.");
                  else conclusions.push("Tracks are relatively different in waveform.");
                  if (isPhaseInverted) conclusions.push("Note: Track B appears phase-inverted relative to Track A.");
                }

                if (aFull.bpm > 0 && bFull.bpm > 0) {
                  const bpmDiff = Math.abs(aFull.bpm - bFull.bpm);
                  if (bpmDiff < 1) conclusions.push("Same tempo.");
                  else if (bpmDiff < 5) conclusions.push(`Nearly the same tempo (${bpmDiff.toFixed(1)} BPM difference).`);
                  else conclusions.push(`Different tempo (${aFull.bpm.toFixed(1)} vs ${bFull.bpm.toFixed(1)} BPM).`);
                }

                return (
                  <div className="space-y-4">
                    {/* Metrics grid */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="rounded-xl bg-[#faf0e6] border border-[#d4a87a]/30 p-3">
                        <h3 className="text-xs font-semibold text-[#9a6a40] mb-2 truncate">{tA.name}</h3>
                        <div className="space-y-1 text-[11px] text-[#5a2a00]">
                          <div>BPM: <b>{aFull.bpm > 0 ? aFull.bpm.toFixed(1) : "N/A"}</b>{aFull.bpmConfidence > 0 && <span className="text-[#9a6a40]"> ({(aFull.bpmConfidence * 100).toFixed(0)}%)</span>}</div>
                          <div>RMS: <b>{dbStr(aFull.rms)}</b></div>
                          <div>Peak: <b>{dbStr(aFull.peak)}</b></div>
                          <div>Centroid: <b>{aFull.centroid.toFixed(0)} Hz</b></div>
                          <div>Crest: <b>{aFull.crestFactorDb.toFixed(1)} dB</b></div>
                        </div>
                      </div>
                      <div className="rounded-xl bg-[#faf0e6] border border-[#d4a87a]/30 p-3">
                        <h3 className="text-xs font-semibold text-[#9a6a40] mb-2 truncate">{tB.name}</h3>
                        <div className="space-y-1 text-[11px] text-[#5a2a00]">
                          <div>BPM: <b>{bFull.bpm > 0 ? bFull.bpm.toFixed(1) : "N/A"}</b>{bFull.bpmConfidence > 0 && <span className="text-[#9a6a40]"> ({(bFull.bpmConfidence * 100).toFixed(0)}%)</span>}</div>
                          <div>RMS: <b>{dbStr(bFull.rms)}</b></div>
                          <div>Peak: <b>{dbStr(bFull.peak)}</b></div>
                          <div>Centroid: <b>{bFull.centroid.toFixed(0)} Hz</b></div>
                          <div>Crest: <b>{bFull.crestFactorDb.toFixed(1)} dB</b></div>
                        </div>
                      </div>
                    </div>

                    {/* Similarity bars */}
                    <div className="space-y-2">
                      {specSim != null && (
                        <div>
                          <div className="flex justify-between text-[11px] text-[#5a2a00] mb-0.5">
                            <span>Spectral Similarity</span>
                            <span className="font-bold">{(specSim * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-[#d4a87a]/20 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-[#f5a623] to-[#e8453c]"
                              style={{ width: `${specSim * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {corr != null && (
                        <div>
                          <div className="flex justify-between text-[11px] text-[#5a2a00] mb-0.5">
                            <span className="flex items-center gap-1.5">
                              Waveform Correlation
                              {isPhaseInverted && (
                                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-[#e8453c]/15 text-[#e8453c] border border-[#e8453c]/30" title="Track B appears to be phase-inverted relative to Track A">
                                  Ø Phase inverted
                                </span>
                              )}
                            </span>
                            <span className="font-bold">{(corr * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-[#d4a87a]/20 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-[#d4a87a] to-[#f5a623]"
                              style={{ width: `${corr * 100}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Spectrogram waterfall overlay */}
                    <SpectrogramOverlay
                      bufA={bufA}
                      bufB={bufB}
                      fromA={fromA}
                      toA={toA}
                      fromB={fromB}
                      toB={toB}
                    />

                    {/* Natural language conclusion */}
                    <div className="rounded-xl bg-[#3a1a00] p-4 text-sm text-[#ffe0b2] leading-relaxed">
                      <span className="font-bold text-[#f5a623]">Summary: </span>
                      {conclusions.join(" ")}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {/* ─ A/B Crossfader Modal ─ */}
        {showABFader && tracks.length >= 2 && (
          <div
            className="fixed inset-x-0 top-[3.5rem] bottom-0 z-40 flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={(e) => {
              if (e.target === e.currentTarget) setShowABFader(false);
            }}
          >
            <div className="bg-[#fdf6f0] rounded-3xl shadow-2xl w-[96vw] max-w-[700px] max-h-[calc(96vh-3.5rem)] overflow-auto p-6 relative">
              {/* header */}
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-xl font-bold text-[#3a1a00]">
                    A/B Crossfader
                  </h2>
                  <p className="text-sm text-[#9a6a40] mt-0.5">
                    Classic two-track crossfade
                  </p>
                </div>
                <button
                  onClick={() => setShowABFader(false)}
                  className="p-2 rounded-xl hover:bg-[#f5e6d0] text-[#9a6a40]"
                >
                  <X size={20} />
                </button>
              </div>

              {/* track selectors */}
              <div className="flex items-center gap-4 mb-6">
                <div className="flex-1">
                  <label className="text-xs font-medium text-[#9a6a40] mb-1 block">
                    Track A
                  </label>
                  <select
                    value={abTrackA}
                    onChange={(e) => setAbTrackA(Number(e.target.value))}
                    className="w-full rounded-xl border border-[#d4a87a]/40 bg-white px-3 py-2 text-sm font-medium text-[#3a1a00]"
                    style={{
                      borderLeftWidth: "4px",
                      borderLeftColor:
                        TRACK_COLORS[abTrackA % TRACK_COLORS.length],
                    }}
                  >
                    {tracks.map((t, i) => (
                      <option key={t.id} value={i}>
                        {i + 1}. {t.name}
                      </option>
                    ))}
                  </select>
                </div>
                <span className="text-[#d4a87a] font-bold text-lg mt-4">
                  vs
                </span>
                <div className="flex-1">
                  <label className="text-xs font-medium text-[#9a6a40] mb-1 block">
                    Track B
                  </label>
                  <select
                    value={abTrackB}
                    onChange={(e) => setAbTrackB(Number(e.target.value))}
                    className="w-full rounded-xl border border-[#d4a87a]/40 bg-white px-3 py-2 text-sm font-medium text-[#3a1a00]"
                    style={{
                      borderLeftWidth: "4px",
                      borderLeftColor:
                        TRACK_COLORS[abTrackB % TRACK_COLORS.length],
                    }}
                  >
                    {tracks.map((t, i) => (
                      <option key={t.id} value={i}>
                        {i + 1}. {t.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* crossfader slider */}
              <div className="rounded-2xl border border-[#d4a87a]/40 bg-gradient-to-br from-[#fdf6f0] to-[#f5e6d0] p-5 mb-4">
                <div className="flex items-end justify-between mb-3">
                  <div className="text-center">
                    <div
                      className="text-2xl font-bold"
                      style={{
                        color: TRACK_COLORS[abTrackA % TRACK_COLORS.length],
                      }}
                    >
                      {((1 - abValue) * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-[#9a6a40] font-medium truncate max-w-[180px]">
                      {tracks[abTrackA]?.name ?? "A"}
                    </div>
                  </div>
                  <div className="text-center">
                    <div
                      className="text-2xl font-bold"
                      style={{
                        color: TRACK_COLORS[abTrackB % TRACK_COLORS.length],
                      }}
                    >
                      {(abValue * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-[#9a6a40] font-medium truncate max-w-[180px]">
                      {tracks[abTrackB]?.name ?? "B"}
                    </div>
                  </div>
                </div>
                {/* visual bar */}
                <div className="h-6 rounded-full overflow-hidden flex mb-3">
                  <div
                    className="h-full transition-all duration-75"
                    style={{
                      width: `${((1 - abValue) * 100).toFixed(0)}%`,
                      background: TRACK_COLORS[abTrackA % TRACK_COLORS.length],
                    }}
                  />
                  <div
                    className="h-full transition-all duration-75"
                    style={{
                      width: `${(abValue * 100).toFixed(0)}%`,
                      background: TRACK_COLORS[abTrackB % TRACK_COLORS.length],
                    }}
                  />
                </div>
                {/* slider */}
                <div className="flex items-center gap-3">
                  <span
                    className="text-xs font-bold"
                    style={{
                      color: TRACK_COLORS[abTrackA % TRACK_COLORS.length],
                    }}
                  >
                    A
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.005"
                    value={abValue}
                    onChange={(e) => setAbValue(Number(e.target.value))}
                    className="flex-1 h-3 cursor-pointer"
                  />
                  <span
                    className="text-xs font-bold"
                    style={{
                      color: TRACK_COLORS[abTrackB % TRACK_COLORS.length],
                    }}
                  >
                    B
                  </span>
                </div>
                <div className="flex justify-center gap-2 mt-4">
                  <button
                    onClick={() => setAbValue(0)}
                    className="rounded-lg border border-[#d4a87a]/40 px-3 py-1 text-xs font-medium text-[#3a1a00] hover:bg-white transition-colors"
                  >
                    All A
                  </button>
                  <button
                    onClick={() => setAbValue(0.5)}
                    className="rounded-lg border border-[#d4a87a]/40 px-3 py-1 text-xs font-medium text-[#3a1a00] hover:bg-white transition-colors"
                  >
                    50/50
                  </button>
                  <button
                    onClick={() => setAbValue(1)}
                    className="rounded-lg border border-[#d4a87a]/40 px-3 py-1 text-xs font-medium text-[#3a1a00] hover:bg-white transition-colors"
                  >
                    All B
                  </button>
                </div>
              </div>

              {/* switch to geometric */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => {
                    setAbMode(false);
                    setShowABFader(false);
                    setShowCrossfader(true);
                  }}
                  className="inline-flex items-center gap-1.5 text-xs text-[#e8453c] hover:text-[#b22a1a] font-medium"
                >
                  <BarChart3 size={14} /> Switch to Geometric Crossfader
                </button>
                <button
                  onClick={() => {
                    setAbMode(false);
                    setShowABFader(false);
                  }}
                  className="text-xs text-[#9a6a40] hover:text-[#3a1a00]"
                >
                  Disable A/B mode
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      <footer className="border-t border-[#d4a87a]/30 bg-[#3a1500]/90 backdrop-blur text-center py-4 text-xs text-[#c09060]">
        Built by{" "}
        <a
          href="https://github.com/banjohans"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-[#ffe0b2] hover:text-white"
        >
          Hans Martin Sognefest Austestad
        </a>{" "}
        ·{" "}
        <a
          href="https://github.com/banjohans"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-[#ffe0b2]"
        >
          github.com/banjohans
        </a>
      </footer>
    </div>
  );
}
