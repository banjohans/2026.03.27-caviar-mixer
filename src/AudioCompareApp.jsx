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

function detectBPM(buf, from = 0, to) {
  const d = buf.getChannelData(0);
  const sr = buf.sampleRate;
  const s = Math.floor(from * sr);
  const e = to ? Math.min(Math.floor(to * sr), d.length) : d.length;
  if (e - s < sr * 2) return null; // need at least 2 seconds

  // energy onset detection — divide into ~10ms windows
  const wLen = Math.floor(sr * 0.01);
  const nWin = Math.floor((e - s) / wLen);
  const energy = new Float32Array(nWin);
  for (let i = 0; i < nWin; i++) {
    let sum = 0;
    const ws = s + i * wLen;
    for (let j = 0; j < wLen; j++) {
      const v = d[ws + j] || 0;
      sum += v * v;
    }
    energy[i] = sum / wLen;
  }

  // onset detection (difference)
  const onset = new Float32Array(nWin);
  for (let i = 1; i < nWin; i++) {
    onset[i] = Math.max(0, energy[i] - energy[i - 1]);
  }

  // autocorrelation in BPM range 60-200
  const minLag = Math.floor((60 / 200) * (1000 / 10)); // ~30 windows
  const maxLag = Math.floor((60 / 60) * (1000 / 10)); // ~100 windows
  let bestLag = minLag,
    bestCorr = -1;
  for (let lag = minLag; lag <= Math.min(maxLag, nWin / 2); lag++) {
    let corr = 0;
    const n = Math.min(nWin - lag, 500);
    for (let i = 0; i < n; i++) {
      corr += onset[i] * onset[i + lag];
    }
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }
  return Math.round(60 / (bestLag * 0.01));
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

function crossCorrelation(bufA, bufB, fromA, toA, fromB, toB) {
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
  // downsample to ~4000 Hz for speed
  const ds = Math.max(1, Math.floor(srA / 4000));
  const n = Math.floor(len / ds);
  let dot = 0,
    magA = 0,
    magB = 0;
  for (let i = 0; i < n; i++) {
    const a = dA[sA + i * ds] || 0;
    const b = dB[sB + i * ds] || 0;
    dot += a * b;
    magA += a * a;
    magB += b * b;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}

function analyzeTrackFull(buf, from, to) {
  const bpm = detectBPM(buf, from, to);
  const spec = computeSpectrum(buf, from, to);
  const dynRange = computeDynamicRange(buf, from, to);
  return {
    bpm,
    centroid: spec?.centroid || 0,
    bandwidth: spec?.bandwidth || 0,
    spectrum: spec?.spectrum || null,
    dynRange,
  };
}

function pctStr(v) {
  return (v * 100).toFixed(1) + "%";
}

const TRACK_COLORS = [
  "#3b82f6",
  "#8b5cf6",
  "#06b6d4",
  "#f59e0b",
  "#ef4444",
  "#10b981",
  "#ec4899",
  "#6366f1",
  "#14b8a6",
  "#f97316",
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
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, w, h);
  if (!peaks || !peaks.length || td <= 0) return;

  const tw = (duration / td) * w;
  const cy = h / 2,
    amp = h / 2 - 4;

  // segment highlight (per-track)
  if (segEnd > segStart) {
    const ss = (segStart / td) * w,
      se = (segEnd / td) * w;
    ctx.fillStyle = "rgba(16,185,129,0.12)";
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
      ctx.fillStyle = "rgba(239,68,68,0.15)";
      ctx.fillRect(zs, 0, ze - zs, h);
      // diagonal stripes
      ctx.save();
      ctx.beginPath();
      ctx.rect(zs, 0, ze - zs, h);
      ctx.clip();
      ctx.strokeStyle = "rgba(239,68,68,0.3)";
      ctx.lineWidth = 1;
      for (let sx = zs - h; sx < ze + h; sx += 8) {
        ctx.beginPath();
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx + h, h);
        ctx.stroke();
      }
      ctx.restore();
      // boundary lines
      ctx.strokeStyle = "#ef4444";
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
    ctx.strokeStyle = "#10b981";
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

/* ── Waveform component ── */

const WaveformTrack = memo(function WaveformTrack({
  track,
  maxDuration,
  segmentLength,
  playhead,
  singlePlayhead,
  isA,
  isB,
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
      className={`rounded-2xl border bg-white p-3 shadow-sm ${isSolo ? "ring-2 ring-blue-400" : ""} ${track.muted ? "opacity-50" : ""}`}
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
        <span className="text-xs text-slate-400">{fmt(dur)}</span>

        {isA && (
          <span className="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700">
            A
          </span>
        )}
        {isB && (
          <span className="rounded bg-violet-100 px-1.5 py-0.5 text-[10px] font-semibold text-violet-700">
            B
          </span>
        )}

        <div className="flex-1" />

        <div className="flex items-center gap-0.5 rounded-lg border px-1 py-0.5">
          <button
            onClick={() => onSkipSegment(track.id, -1)}
            className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-800"
            title="Previous segment"
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => onPlaySingle(track.id)}
            className="p-1 rounded hover:bg-slate-100"
            title="Play this track solo"
          >
            {singlePlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={() => onSkipSegment(track.id, 1)}
            className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-800"
            title="Next segment"
          >
            <SkipForward size={14} />
          </button>
        </div>
        {/* Punch In / Out */}
        <div className="flex items-center gap-0.5 rounded-lg border px-1 py-0.5">
          <button
            onClick={() => setPunchMode("segment")}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === "segment" ? "bg-emerald-100 text-emerald-700" : "text-slate-400 hover:text-slate-600"}`}
            title="Punch-modus: Segment"
          >
            SEG
          </button>
          <button
            onClick={() => setPunchMode("skip")}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === "skip" ? "bg-red-100 text-red-700" : "text-slate-400 hover:text-slate-600"}`}
            title="Punch mode: Skip zone"
          >
            SKIP
          </button>
          <div className="w-px h-4 bg-slate-200 mx-0.5" />
          <button
            onClick={() => onPunchIn(track.id, punchMode)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${isPunchActive ? "bg-pink-500 text-white" : "text-slate-500 hover:bg-slate-100"}`}
            title="Punch In — mark start point"
          >
            IN
          </button>
          <button
            onClick={() => onPunchOut(track.id)}
            disabled={!isPunchActive}
            className="px-1.5 py-0.5 rounded text-[10px] font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30"
            title="Punch Out — mark end point"
          >
            OUT
          </button>
        </div>
        <button
          onClick={() => onToggleSolo(track.id)}
          className={`p-1.5 rounded-lg text-xs font-bold ${isSolo ? "bg-blue-500 text-white" : "hover:bg-slate-100 text-slate-500"}`}
          title="Solo"
        >
          <Headphones size={14} />
        </button>
        <button
          onClick={() => onToggleMute(track.id)}
          className={`p-1.5 rounded-lg ${track.muted ? "bg-red-100 text-red-600" : "hover:bg-slate-100 text-slate-500"}`}
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
          className="w-20 accent-slate-600"
          style={{ height: "4px" }}
          title={`Volume: ${Math.round(track.volume * 100)}%`}
        />

        <div
          className="flex items-center gap-0.5 rounded-lg border px-1.5 py-1"
          title={`Pan: ${(track.pan || 0) > 0 ? `R ${Math.round((track.pan || 0) * 100)}%` : (track.pan || 0) < 0 ? `L ${Math.round(Math.abs(track.pan || 0) * 100)}%` : "Center"}`}
        >
          <span className="text-[9px] font-semibold text-slate-400">L</span>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={track.pan || 0}
            onChange={(e) => onSetPan(track.id, e.target.value)}
            onDoubleClick={() => onSetPan(track.id, 0)}
            className="w-14 accent-cyan-500"
            style={{ height: "4px" }}
          />
          <span className="text-[9px] font-semibold text-slate-400">R</span>
        </div>

        <button
          onClick={() => onRemove(track.id)}
          className="p-1.5 rounded-lg hover:bg-red-50 text-slate-400 hover:text-red-500"
          title="Remove"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* waveform */}
      <div
        className="relative w-full rounded-lg border bg-slate-50"
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
            style={{ width: "2px", height: "100%", background: "#ef4444" }}
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
          <span className="rounded bg-red-500 px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
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
                style={{ width: "2px", height: "100%", background: "#06b6d4" }}
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
              <span className="rounded bg-cyan-500 px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
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
                  background: "#f59e0b",
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
              <span className="rounded bg-amber-500 px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
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
                  background: "#ec4899",
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
              <span className="rounded bg-pink-500 px-1 py-px text-[9px] font-bold text-white whitespace-nowrap">
                IN {fmt(punchInTime)}
              </span>
            </div>
          </>
        )}
      </div>

      {/* analysis + segment info */}
      <div className="mt-3 flex items-center gap-3 text-[11px] text-slate-500 flex-wrap">
        {segS > 0 && (
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-amber-500" />
            Segment: {fmt(segS)} – {fmt(segE)}
            <button
              onClick={() => onSetSegStart(track.id, 0)}
              className="ml-1 text-amber-600 hover:text-amber-800 underline"
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
        <span className="text-slate-400 ml-auto">
          Drag = segment · Alt+drag = skip zone · Punch IN/OUT = precise points
        </span>
      </div>

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
  const [crossfadeA, setCrossfadeA] = useState(0);
  const [crossfadeB, setCrossfadeB] = useState(1);
  const [crossfadeValue, setCrossfadeValue] = useState(0.5);
  const [soloId, setSoloId] = useState(null);
  const [singlePlayId, setSinglePlayId] = useState(null);
  const [singlePlayhead, setSinglePlayhead] = useState(null);
  const [analyses, setAnalyses] = useState({});
  const [fullAnalyses, setFullAnalyses] = useState({});
  const [comparing, setComparing] = useState(false);
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

  /* volume logic */
  useEffect(() => {
    tracks.forEach((track, i) => {
      if (!track.audio) return;
      if (soloId && track.id !== soloId) {
        track.audio.volume = 0;
        return;
      }
      let v = track.muted ? 0 : track.volume;
      const a = Number(crossfadeA),
        b = Number(crossfadeB);
      if (a !== b) {
        if (i === a) v *= 1 - crossfadeValue;
        else if (i === b) v *= crossfadeValue;
      }
      track.audio.volume = clamp(v, 0, 1);
    });
  }, [tracks, crossfadeA, crossfadeB, crossfadeValue, soloId]);

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
  }, [tracks, segmentLength]);

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
      t.audio.currentTime = t.segStart || 0;
      t.audio.volume = t.muted ? 0 : t.volume;
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
          const panner = ctx.createStereoPanner();
          source.connect(panner);
          panner.connect(ctx.destination);
          pannerNodes.current[t.id] = panner;
        } catch (_) {}
      };
      setupPanner();

      t.file.arrayBuffer().then((buf) => {
        audioCtxRef.current
          .decodeAudioData(buf.slice(0))
          .then((decoded) => {
            decodedBuffers.current[t.id] = decoded;
            setTracks((prev) =>
              prev.map((p) =>
                p.id === t.id ? { ...p, peaks: computePeaks(decoded) } : p,
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
      return prev.filter((t) => t.id !== id);
    });
  }, []);

  const toggleMute = useCallback((id) => {
    setTracks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, muted: !t.muted } : t)),
    );
  }, []);

  const toggleSolo = useCallback((id) => {
    setSoloId((prev) => (prev === id ? null : id));
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
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* ─ Transport bar ─ */}
      <div className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b shadow-sm">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-3 flex-wrap">
          <h1 className="text-lg font-bold tracking-tight mr-2">
            Audio Compare
          </h1>
          <h4 className="text-sm font-medium text-slate-500">
            by Hans Martin Sognefest Austestad
          </h4>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
          >
            <Upload size={14} /> Add files
          </button>
          <div className="h-6 w-px bg-slate-200" />
          <button
            onClick={() => (isPlaying ? pauseAll() : playAll(0))}
            disabled={!tracks.length}
            className="inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium hover:bg-slate-100 disabled:opacity-40"
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? "Pause all" : "Play all"}
          </button>
          <button
            onClick={() => seekTo(0)}
            disabled={!tracks.length}
            className="p-1.5 rounded-lg hover:bg-slate-100 disabled:opacity-40"
            title="To start"
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => setLoopRegion(!loopRegion)}
            className={`p-1.5 rounded-lg ${loopRegion ? "bg-emerald-100 text-emerald-700" : "hover:bg-slate-100 text-slate-500"}`}
            title="Loop region"
          >
            <Repeat size={14} />
          </button>
          <div className="h-6 w-px bg-slate-200" />
          {/* segment length */}
          <div className="flex items-center gap-2 text-[11px] text-slate-500">
            <Lock size={12} className="text-emerald-600" />
            <span>Segment length:</span>
            <input
              type="range"
              min="0.1"
              max={maxDuration || 1}
              step="0.01"
              value={segmentLength}
              onChange={(e) => setSegmentLength(Number(e.target.value))}
              className="w-24"
              style={{ height: "4px" }}
            />
            <span className="tabular-nums font-medium">
              {fmt(segmentLength)}
            </span>
          </div>
          <div className="h-6 w-px bg-slate-200" />
          {/* playhead */}
          <div className="flex items-center gap-1 text-[11px] text-slate-500">
            <span className="font-medium text-red-500 tabular-nums">
              {fmt(playhead)}
            </span>
            <span className="text-slate-300">/</span>
            <span className="tabular-nums">{fmt(segmentLength)}</span>
            <input
              type="range"
              min="0"
              max={segmentLength || 1}
              step="0.01"
              value={playhead}
              onChange={(e) => seekTo(Number(e.target.value))}
              className="w-24"
              style={{ height: "4px" }}
            />
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            multiple
            className="hidden"
            onChange={handleFiles}
          />
        </div>
      </div>

      <div className="mx-auto max-w-7xl p-4 space-y-3">
        {/* ─ Tracks ─ */}
        {tracks.length === 0 && (
          <div className="rounded-2xl border border-dashed bg-white p-12 text-center text-sm text-slate-400">
            No audio files loaded yet. Click "Add files" to get started.
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
                className="pt-4 cursor-grab active:cursor-grabbing text-slate-300 hover:text-slate-500"
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
                  isA={Number(crossfadeA) === i}
                  isB={Number(crossfadeB) === i}
                  isSolo={soloId === track.id}
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
                />
              </div>
            </div>
          </div>
        ))}

        {/* ─ Crossfader ─ */}
        {tracks.length >= 2 && (
          <div className="rounded-2xl border bg-white p-4 shadow-sm">
            <div className="flex items-center gap-4 flex-wrap">
              <h3 className="text-sm font-semibold">A/B Crossfader</h3>

              <select
                value={crossfadeA}
                onChange={(e) => setCrossfadeA(Number(e.target.value))}
                className="rounded-lg border px-2 py-1 text-xs"
              >
                {tracks.map((t, i) => (
                  <option key={t.id} value={i}>
                    {t.name}
                  </option>
                ))}
              </select>

              <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                <span className="text-[10px] text-blue-600 font-semibold">
                  A
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={crossfadeValue}
                  onChange={(e) => setCrossfadeValue(Number(e.target.value))}
                  className="w-full h-1.5"
                />
                <span className="text-[10px] text-violet-600 font-semibold">
                  B
                </span>
              </div>

              <select
                value={crossfadeB}
                onChange={(e) => setCrossfadeB(Number(e.target.value))}
                className="rounded-lg border px-2 py-1 text-xs"
              >
                {tracks.map((t, i) => (
                  <option key={t.id} value={i}>
                    {t.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* ─ Analysis summary ─ */}
        {tracks.length > 0 && Object.keys(analyses).length > 0 && (
          <div className="rounded-2xl border bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold mb-3">
              Analysis — segment ({fmt(segmentLength)})
            </h3>
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${Math.min(tracks.length, 4)}, 1fr)`,
              }}
            >
              {tracks.map((t, i) => {
                const a = analyses[t.id];
                if (!a) return null;
                const c = TRACK_COLORS[i % TRACK_COLORS.length];
                return (
                  <div
                    key={t.id}
                    className="rounded-xl border p-3 text-xs space-y-1"
                  >
                    <div className="flex items-center gap-1.5 font-medium">
                      <div
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ background: c }}
                      />
                      <span className="truncate">{t.name}</span>
                    </div>
                    <div className="flex justify-between text-slate-500">
                      <span>RMS</span>
                      <span className="font-mono">{dbStr(a.rms)}</span>
                    </div>
                    <div className="flex justify-between text-slate-500">
                      <span>Peak</span>
                      <span className="font-mono">{dbStr(a.peak)}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(a.rms * 200, 100)}%`,
                          background: c,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {/* ─ Comparison Analysis ─ */}
        {tracks.length >= 2 && (
          <div className="rounded-2xl border bg-white p-4 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <BarChart3 size={16} className="text-indigo-500" />
              <h3 className="text-sm font-semibold">Similarity Analysis</h3>
              <button
                onClick={() => {
                  setComparing(true);
                  const results = {};
                  tracks.forEach((t) => {
                    const buf = decodedBuffers.current[t.id];
                    if (!buf) return;
                    const ss = t.segStart || 0;
                    const se = Math.min(ss + segmentLength, t.duration);
                    results[t.id] = analyzeTrackFull(buf, ss, se);
                  });
                  setFullAnalyses(results);
                  setComparing(false);
                }}
                disabled={comparing}
                className="inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium hover:bg-slate-100 disabled:opacity-40"
              >
                {comparing ? "Analyzing..." : "Run analysis"}
              </button>
              {Object.keys(fullAnalyses).length > 0 && (
                <span className="text-[10px] text-slate-400">
                  Segment: {fmt(segmentLength)}
                </span>
              )}
            </div>

            {Object.keys(fullAnalyses).length > 0 && (
              <>
                {/* per-track stats */}
                <div
                  className="grid gap-2 mb-4"
                  style={{
                    gridTemplateColumns: `repeat(${Math.min(tracks.length, 4)}, 1fr)`,
                  }}
                >
                  {tracks.map((t, i) => {
                    const fa = fullAnalyses[t.id];
                    if (!fa) return null;
                    const c = TRACK_COLORS[i % TRACK_COLORS.length];
                    return (
                      <div
                        key={t.id}
                        className="rounded-xl border p-3 text-xs space-y-1.5"
                      >
                        <div className="flex items-center gap-1.5 font-medium">
                          <div
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ background: c }}
                          />
                          <span className="truncate">{t.name}</span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Tempo</span>
                          <span className="font-mono">
                            {fa.bpm ? `${fa.bpm} BPM` : "—"}
                          </span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Brightness</span>
                          <span className="font-mono">
                            {fa.centroid > 0
                              ? `${Math.round(fa.centroid)} Hz`
                              : "—"}
                          </span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Bandwidth</span>
                          <span className="font-mono">
                            {fa.bandwidth > 0
                              ? `${Math.round(fa.bandwidth)} Hz`
                              : "—"}
                          </span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Crest factor</span>
                          <span className="font-mono">
                            {fa.dynRange > 0
                              ? `${fa.dynRange.toFixed(1)} (${(20 * Math.log10(fa.dynRange)).toFixed(1)} dB)`
                              : "—"}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* pairwise comparison table */}
                <h4 className="text-xs font-semibold text-slate-600 mb-2">
                  Pairwise Comparison
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-[11px] border-collapse">
                    <thead>
                      <tr className="text-left text-slate-400">
                        <th className="p-1.5">Track A vs B</th>
                        <th className="p-1.5 text-center">Spectral</th>
                        <th className="p-1.5 text-center">Waveform</th>
                        <th className="p-1.5 text-center">Tempo</th>
                        <th className="p-1.5 text-center">Brightness</th>
                        <th className="p-1.5 text-center">Overall</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tracks.flatMap((tA, iA) =>
                        tracks.slice(iA + 1).map((tB, jB) => {
                          const iB = iA + 1 + jB;
                          const faA = fullAnalyses[tA.id],
                            faB = fullAnalyses[tB.id];
                          if (!faA || !faB) return null;
                          const specSim =
                            faA.spectrum && faB.spectrum
                              ? spectralSimilarity(faA.spectrum, faB.spectrum)
                              : null;
                          const bufA = decodedBuffers.current[tA.id],
                            bufB = decodedBuffers.current[tB.id];
                          const ssA = tA.segStart || 0,
                            ssB = tB.segStart || 0;
                          const seA = Math.min(
                              ssA + segmentLength,
                              tA.duration,
                            ),
                            seB = Math.min(ssB + segmentLength, tB.duration);
                          const waveSim =
                            bufA && bufB
                              ? crossCorrelation(bufA, bufB, ssA, seA, ssB, seB)
                              : null;
                          const bpmA = faA.bpm,
                            bpmB = faB.bpm;
                          const tempoSim =
                            bpmA && bpmB
                              ? 1 - Math.abs(bpmA - bpmB) / Math.max(bpmA, bpmB)
                              : null;
                          const maxC = Math.max(faA.centroid, faB.centroid);
                          const brightSim =
                            maxC > 0
                              ? 1 - Math.abs(faA.centroid - faB.centroid) / maxC
                              : null;
                          const parts = [
                            specSim,
                            waveSim,
                            tempoSim,
                            brightSim,
                          ].filter((v) => v != null);
                          const overall =
                            parts.length > 0
                              ? parts.reduce((a, b) => a + b, 0) / parts.length
                              : null;
                          const simColor = (v) =>
                            v == null
                              ? ""
                              : v > 0.85
                                ? "text-emerald-600 font-bold"
                                : v > 0.6
                                  ? "text-amber-600"
                                  : "text-red-500";
                          return (
                            <tr
                              key={`${tA.id}-${tB.id}`}
                              className="border-t border-slate-100 hover:bg-slate-50"
                            >
                              <td className="p-1.5">
                                <span
                                  className="inline-block w-2 h-2 rounded-full mr-1"
                                  style={{
                                    background:
                                      TRACK_COLORS[iA % TRACK_COLORS.length],
                                  }}
                                />
                                <span className="font-medium">{tA.name}</span>
                                <span className="text-slate-300 mx-1">vs</span>
                                <span
                                  className="inline-block w-2 h-2 rounded-full mr-1"
                                  style={{
                                    background:
                                      TRACK_COLORS[iB % TRACK_COLORS.length],
                                  }}
                                />
                                <span className="font-medium">{tB.name}</span>
                              </td>
                              <td
                                className={`p-1.5 text-center font-mono ${simColor(specSim)}`}
                              >
                                {specSim != null ? pctStr(specSim) : "—"}
                              </td>
                              <td
                                className={`p-1.5 text-center font-mono ${simColor(waveSim != null ? Math.abs(waveSim) : null)}`}
                              >
                                {waveSim != null
                                  ? pctStr(Math.abs(waveSim))
                                  : "—"}
                              </td>
                              <td
                                className={`p-1.5 text-center font-mono ${simColor(tempoSim)}`}
                              >
                                {tempoSim != null ? pctStr(tempoSim) : "—"}
                              </td>
                              <td
                                className={`p-1.5 text-center font-mono ${simColor(brightSim)}`}
                              >
                                {brightSim != null ? pctStr(brightSim) : "—"}
                              </td>
                              <td
                                className={`p-1.5 text-center font-mono ${simColor(overall)}`}
                              >
                                {overall != null ? pctStr(overall) : "—"}
                              </td>
                            </tr>
                          );
                        }),
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="mt-2 text-[10px] text-slate-400 space-y-0.5">
                  <p>
                    <strong>Spectral:</strong> Frequency profile similarity
                    (cosine similarity). <strong>Waveform:</strong>{" "}
                    Cross-correlation of audio. <strong>Tempo:</strong> BPM
                    similarity. <strong>Brightness:</strong> Spectral centroid
                    similarity.
                  </p>
                  <p className="flex items-center gap-2">
                    <span className="text-emerald-600 font-bold">
                      &gt;85% = high
                    </span>{" "}
                    <span className="text-amber-600">&gt;60% = medium</span>{" "}
                    <span className="text-red-500">&lt;60% = low</span>
                  </p>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
