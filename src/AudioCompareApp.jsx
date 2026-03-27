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
} from "lucide-react";

/* ── helpers ── */

function fmt(seconds) {
  if (!Number.isFinite(seconds)) return "0:00.0";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const t = Math.floor((seconds % 1) * 10);
  return `${m}:${String(s).padStart(2, "0")}.${t}`;
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

function drawWave(canvas, peaks, duration, totalDur, segStart, segEnd, color, skipZones) {
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
  const sortedZones = (track.skipZones || []).slice().sort((a, b) => a.start - b.start);
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
  const [punchMode, setPunchMode] = useState('segment');
  const isPunchActive = punchState && punchState.trackId === track.id;
  const punchInTime = isPunchActive ? punchState.inTime : null;
  const punchInPct = punchInTime != null && td > 0 ? (punchInTime / td) * 100 : null;

  // single-track playhead (cyan)
  const singlePos = singlePlaying && singlePlayhead != null ? clamp(singlePlayhead, 0, dur) : null;
  const singlePct = singlePos != null && td > 0 ? (singlePos / td) * 100 : null;

  useEffect(() => {
    drawWave(canvasRef.current, track.peaks, dur, maxDuration, segS, segE, color, skipZones);
  }, [track.peaks, dur, maxDuration, segS, segE, color, skipZones]);

  useEffect(() => {
    const fn = () =>
      drawWave(canvasRef.current, track.peaks, dur, maxDuration, segS, segE, color, skipZones);
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
      const dragged = dragRef.current.moved && Math.abs(endT - dragRef.current.start) > 0.05;
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
            title="Førre segment"
          >
            <SkipBack size={14} />
          </button>
          <button
            onClick={() => onPlaySingle(track.id)}
            className="p-1 rounded hover:bg-slate-100"
            title="Spel dette sporet åleine"
          >
            {singlePlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={() => onSkipSegment(track.id, 1)}
            className="p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-800"
            title="Neste segment"
          >
            <SkipForward size={14} />
          </button>
        </div>
        {/* Punch In / Out */}
        <div className="flex items-center gap-0.5 rounded-lg border px-1 py-0.5">
          <button
            onClick={() => setPunchMode('segment')}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === 'segment' ? 'bg-emerald-100 text-emerald-700' : 'text-slate-400 hover:text-slate-600'}`}
            title="Punch-modus: Segment"
          >SEG</button>
          <button
            onClick={() => setPunchMode('skip')}
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${punchMode === 'skip' ? 'bg-red-100 text-red-700' : 'text-slate-400 hover:text-slate-600'}`}
            title="Punch-modus: Hopp over"
          >SKIP</button>
          <div className="w-px h-4 bg-slate-200 mx-0.5" />
          <button
            onClick={() => onPunchIn(track.id, punchMode)}
            className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${isPunchActive ? 'bg-pink-500 text-white' : 'text-slate-500 hover:bg-slate-100'}`}
            title="Punch In — marker startpunktet"
          >IN</button>
          <button
            onClick={() => onPunchOut(track.id)}
            disabled={!isPunchActive}
            className="px-1.5 py-0.5 rounded text-[10px] font-bold text-slate-500 hover:bg-slate-100 disabled:opacity-30"
            title="Punch Out — marker sluttpunktet"
          >UT</button>
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
          title={`Volum: ${Math.round(track.volume * 100)}%`}
        />

        <div className="flex items-center gap-0.5 rounded-lg border px-1.5 py-1" title={`Pan: ${(track.pan || 0) > 0 ? `R ${Math.round((track.pan || 0) * 100)}%` : (track.pan || 0) < 0 ? `L ${Math.round(Math.abs(track.pan || 0) * 100)}%` : 'Senter'}`}>
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
          title="Fjern"
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
              nullstill
            </button>
          </span>
        )}
        {analysis && (
          <>
            <span title="RMS-nivå i segmentet">
              RMS: {dbStr(analysis.rms)}
            </span>
            <span title="Peak-nivå i segmentet">
              Peak: {dbStr(analysis.peak)}
            </span>
          </>
        )}
        <span className="text-slate-400 ml-auto">
          Dra = segment · Alt+dra = hopp over-sone · Punch IN/UT = nøyaktige punkt
        </span>
      </div>

      {/* skip zones list */}
      {skipZones.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {skipZones.map((z, zi) => (
            <span
              key={zi}
              className="inline-flex items-center gap-1 rounded-lg bg-red-50 border border-red-200 px-2 py-0.5 text-[10px] text-red-700"
            >
              <span className="font-semibold">Hopp over:</span>
              {fmt(z.start)} – {fmt(z.end)}
              <button
                onClick={() => onRemoveSkipZone(track.id, zi)}
                className="ml-0.5 rounded hover:bg-red-100 p-0.5"
                title="Fjern hopp over-sone"
              >
                <X size={10} />
              </button>
            </span>
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
  const [punchState, setPunchState] = useState(null);

  const rafRef = useRef(null);
  const startTimeRef = useRef(0);
  const startOffsetRef = useRef(0);
  const endedTimeoutRef = useRef(null);
  const fileInputRef = useRef(null);
  const audioCtxRef = useRef(null);
  const isPlayingRef = useRef(false);
  const singleRafRef = useRef(null);
  const singleTrackRef = useRef(null);
  const decodedBuffers = useRef({});
  const pannerNodes = useRef({});

  const maxDuration = useMemo(() => {
    if (!tracks.length) return 0;
    return Math.max(...tracks.map((t) => t.duration || 0));
  }, [tracks]);

  /* segment length: default to max duration, clamp if needed */
  useEffect(() => {
    if (maxDuration > 0)
      setSegmentLength((p) => p <= 0 ? maxDuration : clamp(p, 0.1, maxDuration));
  }, [maxDuration]);

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
      clearTimeout(endedTimeoutRef.current);
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
    const elapsed =
      startOffsetRef.current +
      (performance.now() - startTimeRef.current) / 1000;
    if (elapsed >= segmentLength) {
      if (loopRegion) {
        playAll(0);
        return;
      }
      pauseAll();
      setPlayhead(segmentLength);
      return;
    }
    // skip past skip zones for each playing track
    tracks.forEach((t) => {
      if (!t.audio || t.audio.paused) return;
      const ct = t.audio.currentTime;
      const zones = (t.skipZones || []).slice().sort((a, b) => a.start - b.start);
      for (const z of zones) {
        if (ct >= z.start && ct < z.end) {
          try { t.audio.currentTime = z.end; } catch (_) {}
          break;
        }
      }
    });
    setPlayhead(elapsed);
    rafRef.current = requestAnimationFrame(updatePlayhead);
  };

  const syncTimes = (time) => {
    tracks.forEach((t) => {
      if (!t.audio) return;
      try {
        const ss = t.segStart || 0;
        let target = ss + time;
        // skip past any skip zones
        const zones = (t.skipZones || []).slice().sort((a, b) => a.start - b.start);
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
    clearTimeout(endedTimeoutRef.current);
  };

  const playAll = async (from = playhead) => {
    if (!tracks.length) return;
    setSinglePlayId(null);
    stopSingleAnim();
    setSinglePlayhead(null);
    const safe = clamp(from, 0, Math.max(0, segmentLength - 0.01));
    stopTimers();
    syncTimes(safe);
    for (const t of tracks) {
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
    const ms = Math.max(0, (segmentLength - safe) * 1000);
    endedTimeoutRef.current = setTimeout(() => {
      if (loopRegion) playAll(0);
      else {
        pauseAll();
        setPlayhead(segmentLength);
      }
    }, ms + 30);
  };

  const stopSingleAnim = () => {
    cancelAnimationFrame(singleRafRef.current);
    singleTrackRef.current = null;
  };

  const pauseAll = () => {
    stopTimers();
    stopSingleAnim();
    tracks.forEach((t) => t.audio?.pause());
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
      prev.map((t) =>
        t.id === id ? { ...t, segStart: Number(v) } : t,
      ),
    );
  }, []);

  const setSegmentByDrag = useCallback((id, start, len) => {
    setSegmentLength(len);
    setTracks((prev) =>
      prev.map((t) =>
        t.id === id ? { ...t, segStart: Number(start) } : t,
      ),
    );
  }, []);

  const skipSegment = useCallback((id, dir) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const newStart = (t.segStart || 0) + dir * segmentLength;
        return { ...t, segStart: clamp(newStart, 0, Math.max(0, (t.duration || 0) - segmentLength)) };
      }),
    );
  }, [segmentLength]);

  const addSkipZone = useCallback((id, start, end) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const zones = [...(t.skipZones || []), { start, end }].sort((a, b) => a.start - b.start);
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

  const punchIn = useCallback((id, mode) => {
    const t = tracks.find((tr) => tr.id === id);
    const pos = t?.audio && !t.audio.paused ? t.audio.currentTime : (t?.segStart || 0) + playhead;
    setPunchState({ trackId: id, mode, inTime: pos });
  }, [tracks, playhead]);

  const punchOut = useCallback((id) => {
    if (!punchState || punchState.trackId !== id) return;
    const t = tracks.find((tr) => tr.id === id);
    const pos = t?.audio && !t.audio.paused ? t.audio.currentTime : (t?.segStart || 0) + playhead;
    const s = Math.min(punchState.inTime, pos);
    const e = Math.max(punchState.inTime, pos);
    if (e - s < 0.01) { setPunchState(null); return; }
    if (punchState.mode === 'segment') {
      setSegmentLength(e - s);
      setTracks((prev) => prev.map((tr) => tr.id === id ? { ...tr, segStart: s } : tr));
    } else {
      setTracks((prev) => prev.map((tr) => {
        if (tr.id !== id) return tr;
        const zones = [...(tr.skipZones || []), { start: s, end: e }].sort((a, b) => a.start - b.start);
        return { ...tr, skipZones: zones };
      }));
    }
    setPunchState(null);
  }, [punchState, tracks, playhead]);

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

          <button
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700"
          >
            <Upload size={14} /> Legg til filer
          </button>

          <div className="h-6 w-px bg-slate-200" />

          <button
            onClick={() =>
              isPlaying ? pauseAll() : playAll(0)
            }
            disabled={!tracks.length}
            className="inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium hover:bg-slate-100 disabled:opacity-40"
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? "Pause alle" : "Spel alle"}
          </button>

          <button
            onClick={() => seekTo(0)}
            disabled={!tracks.length}
            className="p-1.5 rounded-lg hover:bg-slate-100 disabled:opacity-40"
            title="Til start"
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
            <span>Segmentlengde:</span>
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
            <span className="tabular-nums font-medium">{fmt(segmentLength)}</span>
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
            Ingen lydfiler lasta inn enno. Klikk «Legg til filer» for å starte.
          </div>
        )}

        {tracks.map((track, i) => (
          <WaveformTrack
            key={track.id}
            track={track}
            index={i}
            maxDuration={maxDuration}
            segmentLength={segmentLength}
            playhead={playhead}
            singlePlayhead={singlePlayId === track.id ? singlePlayhead : null}
            isA={Number(crossfadeA) === i}
            isB={Number(crossfadeB) === i}
            isSolo={soloId === track.id}
            onSeek={seekTo}
            onSetSegStart={setSegStart}
            onSetSegmentByDrag={setSegmentByDrag}
            onSkipSegment={skipSegment}
            onAddSkipZone={addSkipZone}
            onRemoveSkipZone={removeSkipZone}
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
              Analyse — segment ({fmt(segmentLength)})
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
      </div>
    </div>
  );
}
