# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
"""Write the golden vectors for sonification-core from the reference implementation.

The oracle has three parts, and nothing here calls the Rust code.

1. Code of the project's earlier Python toolkit, `lifesound.py`, copied: the
   instruments (`modal_chime`, `fm_bell`, `karplus_strong`, `soft_voice`,
   `heartbeat` and the helpers they call) as synthesis-theory's oracle copies
   them, with signal-core's two documented changes (a pure-Python SplitMix64,
   copied from signal-core's vectors/generate.py, in place of numpy's random
   generator, and ln 1000 in `exp_decay`); synthesis-theory's plain-loop
   reference for `karplus_strong_tuned`; `pan`, `mix_into` and `write_wav`; and
   `audify`, with two documented changes: audio that would not vary (a constant
   series, or interpolated samples that are all equal) is an error, and the
   mean is removed from the samples less the first one (the same in exact
   arithmetic, and exact differences for a series that barely varies).
2. Exact references for the mapping grammar: Fractions for the time map, linear
   maps, octave shifts and folds, heard frequencies and scale quantisation, and
   mpmath at 50 digits for exponential maps and equal-tempered pitches.
3. Independent Python versions of what is new in the crate, written from its
   documentation: `sonify`, `render`, the manifest writer (every manifest is
   validated against schema/manifest.schema.json with jsonschema before it is
   written), the onset detector, the matching of `verify`, and `verify_wav`
   (the WAV file decoded with Python's `wave`, and compared with a new render).

Run with numpy, mpmath and jsonschema installed:

    python vectors/generate.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import wave
from fractions import Fraction as F
from pathlib import Path

import jsonschema
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = json.loads((ROOT / "schema" / "manifest.schema.json").read_text(encoding="utf-8"))
VERSION = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M)[1]
SR = 44100
CLOSE = {"rel": 1e-12, "abs": 1e-12}
MAPS = {"rel": 1e-14, "abs": 0}
mp.mp.dps = 50
jsonschema.Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def write(name: str, description: str, cases: list[dict], tolerance: dict = CLOSE, compact: bool = False) -> None:
    doc = {"schema": "golden-vectors@1", "description": description, "tolerance": tolerance, "cases": cases}
    text = json.dumps(doc, separators=(",", ":")) if compact else json.dumps(doc, indent=1, ensure_ascii=False)
    (HERE / f"{name}.json").write_text(text + "\n", encoding="utf-8", newline="\n")


# signal-core's random number generator, ported (copied from signal-core's vectors/generate.py) --------

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    """SplitMix64 (Steele, Lea and Flood 2014), with the numpy Generator method lifesound.py calls."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        return (self.next_u64() >> 11) * 2.0**-53  # exact: 53 bits times a power of two

    def uniform(self, low: float, high: float, size) -> np.ndarray:
        shape = (size,) if isinstance(size, int) else tuple(size)
        count = math.prod(shape)
        return np.array([low + (high - low) * self.random() for _ in range(count)], dtype=float).reshape(shape)


class _Random:
    """Stands in for `np.random` inside the copied lifesound.py code."""

    default_rng = SplitMix64


rng_module = _Random()


# The reference implementation: lifesound.py ------------------------------------


def n_samples(dur: float, sr: int = SR) -> int:
    return int(round(dur * sr))


def time_axis(dur: float, sr: int = SR) -> np.ndarray:
    return np.arange(n_samples(dur, sr)) / sr


def osc(freq, dur: float, sr: int = SR, phase: float = 0.0, shape: str = "sine") -> np.ndarray:
    n = n_samples(dur, sr)
    f = np.broadcast_to(np.asarray(freq, dtype=float), (n,))
    ph = 2 * np.pi * np.cumsum(f) / sr + phase
    if shape == "sine":
        return np.sin(ph)
    if shape == "triangle":
        return 2 / np.pi * np.arcsin(np.sin(ph))
    raise ValueError(shape)


def additive(f0, dur: float, partial_amps, sr: int = SR, ratios=None) -> np.ndarray:
    ratios = ratios if ratios is not None else np.arange(1, len(partial_amps) + 1)
    out = np.zeros(n_samples(dur, sr))
    for r, a in zip(ratios, partial_amps, strict=False):
        out += np.asarray(a) * osc(np.asarray(f0) * r, dur, sr)
    return out


def adsr(n: int, sr: int = SR, a=0.01, d=0.1, s=0.7, r=0.2) -> np.ndarray:
    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    ns = max(n - na - nd - nr, 0)
    env = np.concatenate(
        [
            np.linspace(0, 1, na, endpoint=False),
            np.linspace(1, s, nd, endpoint=False),
            np.full(ns, s),
            np.linspace(s, 0, nr),
        ]
    )
    return np.pad(env, (0, max(n - len(env), 0)))[:n]


def exp_decay(n: int, t60: float, sr: int = SR) -> np.ndarray:
    return np.exp(-math.log(1000) * np.arange(n) / (sr * t60))  # lifesound.py: 6.907755


BAR_MODES = (1.0, 2.756, 5.404, 8.933)


def modal_chime(freq: float, dur: float = 3.0, brightness: float = 1.0, sr: int = SR) -> np.ndarray:
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k, ratio in enumerate(BAR_MODES):
        amp = (0.55**k) * (brightness**k)
        t60 = dur / (1 + 1.6 * k)  # higher modes die away faster
        out += amp * np.sin(2 * np.pi * freq * ratio * t) * exp_decay(n, t60, sr)
    strike = np.exp(-t / 0.002) * 0.15  # tiny noise-free "strike" transient
    out[: len(strike)] += strike * np.sin(2 * np.pi * freq * 6.0 * t)
    return out / (np.max(np.abs(out)) + 1e-12)


def fm_bell(fc: float, dur: float = 6.0, ratio: float = 1.4, index: float = 10.0, sr: int = SR) -> np.ndarray:
    n = n_samples(dur, sr)
    t = np.arange(n) / sr
    env = exp_decay(n, dur, sr)
    idx = index * env  # brightness decays with loudness, like a real bell
    y = env * np.sin(2 * np.pi * fc * t + idx * np.sin(2 * np.pi * fc * ratio * t))
    return y * adsr(n, sr, a=0.002, d=0.0, s=1.0, r=0.05)


def karplus_strong(
    freq: float,
    dur: float,
    sr: int = SR,
    decay: float = 0.996,
    seed: int = 0,
    pluck_brightness_hz: float | None = 2000.0,
) -> np.ndarray:
    rng = rng_module.default_rng(seed)
    period = max(int(round(sr / freq)), 2)
    n = n_samples(dur, sr)
    out = np.zeros(n + period + 1)
    burst = rng.uniform(-1, 1, period + 1)
    if pluck_brightness_hz:
        burst = box_lowpass(burst, pluck_brightness_hz, sr, passes=2)
    out[: period + 1] = burst - burst.mean()
    pos = period + 1
    while pos < len(out):
        end = min(pos + period, len(out))
        seg = np.arange(pos, end)
        out[seg] = decay * 0.5 * (out[seg - period] + out[seg - period - 1])
        pos = end
    return out[:n] / (np.max(np.abs(out[:n])) + 1e-12)


def soft_voice(freq: float, dur: float, sr: int = SR, vibrato_hz: float = 5.0, vibrato_cents: float = 12) -> np.ndarray:
    t = time_axis(dur, sr)
    f = freq * 2 ** (vibrato_cents / 1200 * np.sin(2 * np.pi * vibrato_hz * t) * np.minimum(t / 0.4, 1))
    y = additive(f, dur, [1.0, 0.25, 0.08, 0.03], sr)
    return y * adsr(len(y), sr, a=0.08, d=0.2, s=0.8, r=min(0.4, dur / 3))


def heartbeat(bpm: float, dur: float, sr: int = SR, f0: float = 48.0) -> np.ndarray:
    out = np.zeros(n_samples(dur, sr))
    thump_n = n_samples(0.18, sr)
    tt = np.arange(thump_n) / sr
    thump = np.sin(2 * np.pi * f0 * tt * (1 - 0.35 * tt / 0.18)) * np.exp(-tt / 0.045)
    beat = 60.0 / bpm
    s2_delay = min(0.30, beat * 0.4)
    t = 0.0
    while t < dur:
        for offset, amp in ((0.0, 1.0), (s2_delay, 0.6)):
            i = n_samples(t + offset, sr)
            if i + thump_n < len(out):
                out[i : i + thump_n] += amp * thump
        t += beat
    return out


def box_lowpass(x: np.ndarray, cutoff_hz: float, sr: int = SR, passes: int = 3) -> np.ndarray:
    h = max(int(0.44 * sr / cutoff_hz) // 2, 1)
    L = 2 * h + 1
    y = x.astype(float)
    for _ in range(passes):
        padded = np.pad(y, [(h + 1, h)] + [(0, 0)] * (y.ndim - 1), mode="edge")
        c = np.cumsum(padded, axis=0)
        y = (c[L:] - c[:-L]) / L
    return y


def pan(x: np.ndarray, position: float) -> np.ndarray:
    a = (position + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


def mix_into(buf: np.ndarray, sig: np.ndarray, start_s: float, gain: float = 1.0, sr: int = SR) -> None:
    i = n_samples(start_s, sr)
    if i >= len(buf):
        return
    m = min(len(sig), len(buf) - i)
    buf[i : i + m] += gain * sig[:m]


def write_wav(path, x: np.ndarray, sr: int = SR, seed: int = 3) -> None:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    rng = rng_module.default_rng(seed)
    dither = (rng.uniform(-0.5, 0.5, x.shape) + rng.uniform(-0.5, 0.5, x.shape)) / 32768
    pcm = np.clip(np.round((x + dither) * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def audify(values: np.ndarray, src_dt: float, speedup: float, sr: int = SR) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if np.all(v == v[0]):  # the one change: audio that would not vary is an error
        raise ValueError("a constant series has nothing to audify")
    t_src = np.arange(len(v)) * src_dt
    n_out = max(int(t_src[-1] / speedup * sr), 2)
    y = np.interp(np.arange(n_out) / sr * speedup, t_src, v)
    y = y - y[0]  # the second change: the mean of y - y[0], exact differences for close samples
    if np.all(y == 0):  # (the first change) samples that miss every change in the series
        raise ValueError("the audio's samples do not vary: nothing to audify")
    y -= y.mean()
    return y / (np.max(np.abs(y)) + 1e-12)


# References beyond lifesound.py: synthesis-theory's tuned string -----------------------------


def karplus_strong_tuned(
    freq: float,
    dur: float,
    sr: int = SR,
    decay: float = 0.996,
    seed: int = 0,
    pluck_brightness_hz: float | None = 2000.0,
) -> np.ndarray:
    """lifesound.py's string with a first-order allpass of delay delta in the loop, as a plain loop."""
    rng = rng_module.default_rng(seed)
    d = sr / freq
    period = math.floor(d - 0.5 - 0.1)
    delta = d - 0.5 - period
    c = (1 - delta) / (1 + delta)
    n = n_samples(dur, sr)
    burst = rng.uniform(-1, 1, period + 1)
    if pluck_brightness_hz:
        burst = box_lowpass(burst, pluck_brightness_hz, sr, passes=2)
    out = list(burst - burst.mean())
    k = decay * 0.5
    v1 = y1 = 0.0
    for i in range(period + 1, n):
        v = k * (out[i - period] + out[i - period - 1])
        y = c * v + v1 - c * y1
        v1, y1 = v, y
        out.append(y)
    x = np.array(out[:n])
    return x / (np.max(np.abs(x)) + 1e-12)


# Exact references for the mapping grammar ----------------------------------------------------


def mpf(q) -> mp.mpf:
    return mp.mpf(q.numerator) / q.denominator if isinstance(q, F) else mp.mpf(q)


def to_audio(time: dict, real_s: float) -> float:
    return float(F(time.get("offset_s", 0.0)) + F(real_s) / F(time["speedup"]))


def to_real(time: dict, audio_s: float) -> float:
    return float((F(audio_s) - F(time.get("offset_s", 0.0))) * F(time["speedup"]))


def exact_value(pm: dict, x: float):
    """The map's output in exact arithmetic: a Fraction (constant, linear) or a 50-digit mpf (exponential)."""
    if pm["kind"] == "constant":
        return F(pm["value"])
    t = (F(x) - F(pm["x_lo"])) / (F(pm["x_hi"]) - F(pm["x_lo"]))
    if pm.get("clamp", False):
        t = min(max(t, F(0)), F(1))
    lo, hi = F(pm["out_lo"]), F(pm["out_hi"])
    if pm["kind"] == "linear":
        return lo + t * (hi - lo)
    assert pm["kind"] == "exponential" and lo > 0 and hi > 0
    return mpf(lo) * (mpf(hi) / mpf(lo)) ** mpf(t)


def map_value(pm: dict, x: float) -> float:
    return float(exact_value(pm, x))


def whole(x) -> int:
    """A whole number written as an int or a whole float (JSON Schema's integer)."""
    assert float(x).is_integer()
    return int(x)


def quantise_exact(scale: dict, m: F) -> int:
    """The nearest note whose pitch class above the root is in the scale; exact halves go up."""
    root, members = whole(scale["root"]), {whole(c) for c in scale["pitch_classes"]}
    assert 0 <= root < 12 and members and members <= set(range(12))

    def member(n: int) -> bool:
        return (n - root) % 12 in members

    below, above = math.floor(m), math.ceil(m)
    while not member(below):
        below -= 1
    while not member(above):
        above += 1
    return below if m - below < above - m else above


def quantise(scale: dict, midi: float) -> int:
    return quantise_exact(scale, F(midi))


def map_pitch(pm: dict, x: float) -> float:
    kind = pm["kind"]
    if kind == "hz":
        return float(exact_value(pm["map"], x))
    if kind == "midi":
        m = exact_value(pm["map"], x)
        if pm.get("scale"):
            m = quantise_exact(pm["scale"], m if isinstance(m, F) else F(float(m)))
        return float(440 * mp.mpf(2) ** ((mpf(F(m) if isinstance(m, int) else m) - 69) / 12))
    if kind == "rhythm":
        return float(F(x) * F(pm.get("speedup", 1.0)) * F(2) ** whole(pm.get("octaves", 0)))
    assert kind == "fold"
    f, low = F(x) * F(pm.get("speedup", 1.0)), F(pm["low_hz"])
    while f >= 2 * low:
        f /= 2
    while f < low:
        f *= 2
    return float(f)


def heard_hz(period_s: float, speedup: float) -> float:
    return float(F(speedup) / F(period_s))


# The crate's new parts, written independently from its documentation ----------------------------

INSTRUMENT_DEFAULTS = {
    "modal_chime": {"brightness": 1.0},
    "fm_bell": {"ratio": 1.4, "index": 10.0},
    "karplus_strong": {"decay": 0.996, "pluck_brightness_hz": 2000.0},
    "karplus_strong_tuned": {"decay": 0.996, "pluck_brightness_hz": 2000.0},
    "soft_voice": {"vibrato_hz": 5.0, "vibrato_cents": 12.0},
    "heartbeat": {"bpm": 60.0},
}


def full_instrument(i: dict) -> dict:
    return {"kind": i["kind"], **INSTRUMENT_DEFAULTS[i["kind"]], **i}


def full_param(p: dict) -> dict:
    return dict(p) if p["kind"] == "constant" else {"clamp": False, **p}


def full_pitch(p: dict) -> dict:
    if p["kind"] in ("hz", "midi"):
        out = {**p, "map": full_param(p["map"])}
        if p["kind"] == "midi":
            scale = p.get("scale")
            out["scale"] = scale and {
                "root": whole(scale["root"]),
                "pitch_classes": list(map(whole, scale["pitch_classes"])),
            }
        return out
    if p["kind"] == "rhythm":
        return {"speedup": 1.0, **p, "octaves": whole(p.get("octaves", 0))}
    return {"speedup": 1.0, **p}


def full_mapping(m: dict) -> dict:
    """The mapping with every default filled in, as the crate writes it."""
    zero = {"kind": "constant", "value": 0.0}
    return {
        "time": {"offset_s": 0.0, **m["time"]},
        "pitch": full_pitch(m["pitch"]),
        "gain_db": full_param(m.get("gain_db", zero)),
        "pan": full_param(m.get("pan", zero)),
        "duration_s": full_param(m["duration_s"]),
        "instrument": full_instrument(m["instrument"]),
    }


def sonify(mapping: dict, times: list, values: list, claim: str | None = None) -> list[dict]:
    m = full_mapping(mapping)
    events = []
    for t, v in zip(times, values, strict=True):
        e = {"real_s": t, "value": v}
        if claim is not None:
            e["claim"] = claim
        e |= {
            "audio_s": to_audio(m["time"], t),
            "hz": map_pitch(m["pitch"], v),
            "gain_db": map_value(m["gain_db"], v),
            "pan": map_value(m["pan"], v),
            "dur_s": map_value(m["duration_s"], v),
            "instrument": m["instrument"],
        }
        assert e["audio_s"] >= 0 and -1 <= e["pan"] <= 1 and e["dur_s"] > 0 and e["hz"] > 0
        events.append(e)
    return events


def note(i: dict, hz: float, dur: float, sr: int, seed: int) -> np.ndarray:
    i = full_instrument(i)
    kind = i["kind"]
    if kind == "modal_chime":
        return modal_chime(hz, dur, i["brightness"], sr)
    if kind == "fm_bell":
        return fm_bell(hz, dur, i["ratio"], i["index"], sr)
    if kind in ("karplus_strong", "karplus_strong_tuned"):
        string = karplus_strong if kind == "karplus_strong" else karplus_strong_tuned
        return string(hz, dur, sr, i["decay"], seed, i["pluck_brightness_hz"])
    if kind == "soft_voice":
        return soft_voice(hz, dur, sr, i["vibrato_hz"], i["vibrato_cents"])
    return heartbeat(i["bpm"], dur, sr, hz)


def render_frames(events: list[dict], sr: int, seed: int) -> np.ndarray:
    """The events mixed into an (n, 2) stereo buffer that ends where the last note ends."""
    frames = max((n_samples(e["audio_s"], sr) + n_samples(e["dur_s"], sr) for e in events), default=0)
    buf = np.zeros((frames, 2))
    for i, e in enumerate(events):
        x = note(e["instrument"], e["hz"], e["dur_s"], sr, (seed + i + 1) & MASK)
        mix_into(buf, pan(x, e["pan"]), e["audio_s"], 10 ** (e["gain_db"] / 20), sr)
    return buf


def render(events: list[dict], sr: int, seed: int) -> list[float]:
    return render_frames(events, sr, seed).reshape(-1).tolist()


def peak(frames: np.ndarray) -> float:
    return float(np.max(np.abs(frames), initial=0.0))


def wav_bytes(frames: np.ndarray, sr: int, seed: int) -> bytes:
    f = io.BytesIO()
    write_wav(f, frames.reshape(-1, 2), sr, seed)
    return f.getvalue()


def write_manifest(info: dict, series: list[dict], sr: int, seed: int) -> dict:
    layers = []
    for s in series:
        layer = {"name": s["name"]}
        if s.get("claim") is not None:
            assert any(src["claim"] == s["claim"] for src in info["sources"])
            layer["claim"] = s["claim"]
        layer |= {
            "mapping": full_mapping(s["mapping"]),
            "events": sonify(s["mapping"], s["times"], s["values"], s.get("claim")),
        }
        layers.append(layer)
    events = [e for layer in layers for e in layer["events"]]
    frames = render_frames(events, sr, seed)
    manifest = {
        "schema": "sonification-manifest/0.1-draft",
        "title": info["title"],
        "creator": info["creator"],
        "licence": info["licence"],
        "software": {"name": "sonification-core", "version": VERSION},
        "sources": info.get("sources", []),
        "layers": layers,
        "render": {"sample_rate": sr, "seed": seed, "instruments": distinct_instruments(events)},
        "audio": {
            "encoding": "wav-pcm16",
            "channels": 2,
            "frames": len(frames),
            "peak": peak(frames),
            "sha256": hashlib.sha256(wav_bytes(frames, sr, seed)).hexdigest(),
        },
    }
    VALIDATOR.validate(manifest)
    assert json.loads(json.dumps(manifest)) == manifest
    return manifest


def detect_onsets(stereo, sr: int, hop_s, window_s, threshold_db, min_gap_s, floor_db) -> list:
    x = np.asarray(stereo, dtype=float).reshape(-1, 2)
    hop, window, gap = n_samples(hop_s, sr), n_samples(window_s, sr), n_samples(min_gap_s, sr)
    assert hop >= 1 and window >= 1
    total = np.concatenate([[0.0], np.cumsum(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1])]).tolist()
    frames, floor = len(x), window * 10 ** (floor_db / 10)

    def ratio(s: int) -> float:
        after = total[min(s + window, frames)] - total[s]
        before = total[s] - total[max(s - window, 0)]
        return max(after, floor) / max(before, floor)

    rise = [ratio(s) for s in range(0, frames, hop)]
    threshold, reach = 10 ** (threshold_db / 10), max(gap // hop, 1)
    onsets = []
    for m, r in enumerate(rise):
        if (
            r >= threshold
            and all(v <= r for v in rise[max(m - reach, 0) : m])
            and all(v < r for v in rise[m + 1 : m + reach + 1])
        ):
            s = best = m * hop
            for f in range(max(s - hop + 1, 0), min(s + hop, frames)):
                if ratio(f) >= ratio(best):
                    best = f
            onsets.append(best / sr)
    return onsets


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(b))


def distinct_instruments(events: list[dict]) -> list[dict]:
    """Every distinct instrument of the events (defaults filled in), in order of first use."""
    out = []
    for e in events:
        if full_instrument(e["instrument"]) not in out:
            out.append(full_instrument(e["instrument"]))
    return out


# The crate's text rules, by code point: control characters (Unicode Cc), and blank ones (Cc and
# White_Space). Python's str.isspace is not used: it counts U+001C to U+001F as white space.
CONTROL = [(0x00, 0x1F), (0x7F, 0x9F)]
BLANK = CONTROL + [(0x20, 0x20), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029)]
BLANK += [(0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def among(ranges: list, c: str) -> bool:
    return any(lo <= ord(c) <= hi for lo, hi in ranges)


def well_formed(manifest: dict) -> bool:
    """write_manifest's rules for the title, creator, licence and sources."""
    for t in (manifest["title"], manifest["creator"], manifest["licence"]):
        if any(among(CONTROL, c) for c in t) or all(among(BLANK, c) for c in t):
            return False
    for s in manifest["sources"]:
        doi = s.get("doi")
        if ":" not in s["claim"] or any(among(BLANK, c) for c in s["claim"]):
            return False
        if doi is not None and not (doi.startswith("10.") and "/" in doi and not any(among(BLANK, c) for c in doi)):
            return False
    return True


def reproduces(manifest: dict) -> bool:
    numbers = ("real_s", "value", "audio_s", "hz", "gain_db", "pan", "dur_s")
    cited = {s["claim"] for s in manifest["sources"]}
    events = [e for layer in manifest["layers"] for e in layer["events"]]
    if not well_formed(manifest):
        return False
    if (manifest["audio"]["encoding"], manifest["audio"]["channels"]) != ("wav-pcm16", 2):
        return False
    if [full_instrument(i) for i in manifest["render"]["instruments"]] != distinct_instruments(events):
        return False
    for layer in manifest["layers"]:
        if layer.get("claim") is not None and layer["claim"] not in cited:
            return False
        times, values = [e["real_s"] for e in layer["events"]], [e["value"] for e in layer["events"]]
        again = sonify(layer["mapping"], times, values, layer.get("claim"))
        for a, b in zip(again, layer["events"], strict=True):
            if a.get("claim") != b.get("claim") or a["instrument"] != full_instrument(b["instrument"]):
                return False
            if not all(close(a[k], b[k]) for k in numbers):
                return False
    return True


def inner_onsets(e: dict, sr: int) -> list[float]:
    """A heartbeat note's dubs and later beats, in audio seconds, where lifesound.heartbeat places its thumps."""
    if e["instrument"]["kind"] != "heartbeat":
        return []
    beat = 60.0 / e["instrument"]["bpm"]
    s2_delay, n, thump_n, t, out = min(0.30, beat * 0.4), n_samples(e["dur_s"], sr), n_samples(0.18, sr), 0.0, []
    while t < e["dur_s"]:
        for at in (t, t + s2_delay):
            if at > 0 and n_samples(at, sr) + thump_n < n:
                out.append(e["audio_s"] + at)
        t += beat
    return out


def verify(manifest: dict, stereo, tolerance_s, *params) -> dict:
    sr = manifest["render"]["sample_rate"]
    onsets = detect_onsets(stereo, sr, *params)
    layers = manifest["layers"]
    order = sorted(
        ((e["audio_s"], li, ei) for li, layer in enumerate(layers) for ei, e in enumerate(layer["events"])),
        key=lambda r: r[0],
    )
    taken, matched, missing, first = [False] * len(onsets), [], [], 0
    for a, li, ei in order:
        while first < len(onsets) and onsets[first] < a - tolerance_s:
            first += 1
        best = None
        for j in range(first, len(onsets)):
            if onsets[j] > a + tolerance_s:
                break
            if not taken[j] and (best is None or abs(onsets[j] - a) < abs(onsets[best] - a)):
                best = j
        if best is None:
            missing.append({"layer": li, "event": ei, "audio_s": a})
            continue
        taken[best] = True
        e, d = layers[li]["events"][ei], onsets[best]
        recovered = to_real(layers[li]["mapping"]["time"], d)
        matched.append(
            {
                "layer": li,
                "event": ei,
                "audio_s": a,
                "detected_s": d,
                "audio_error_s": d - a,
                "real_s": e["real_s"],
                "recovered_real_s": recovered,
                "real_error_s": recovered - e["real_s"],
            }
        )
    beats = [b for layer in layers for e in layer["events"] for b in inner_onsets(e, sr)]
    inside, extra = [], []
    for s, t in zip(onsets, taken, strict=True):
        if not t:
            (inside if any(abs(s - b) <= tolerance_s for b in beats) else extra).append(s)
    ok = reproduces(manifest)
    frames_ok = len(stereo) // 2 == manifest["audio"]["frames"]
    return {
        "tolerance_s": tolerance_s,
        "events_reproduce": ok,
        "frames_match": frames_ok,
        "wav_reproduces": None,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "within_note": inside,
        "max_audio_error_s": max((abs(m["audio_error_s"]) for m in matched), default=0.0),
        "max_real_error_s": max((abs(m["real_error_s"]) for m in matched), default=0.0),
        "pass": ok and frames_ok and not missing and not extra,
    }


def verify_wav(manifest: dict, wav: list[int], tolerance_s, *params) -> dict:
    data = bytes(wav)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        assert w.getframerate() == manifest["render"]["sample_rate"]
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    report = verify(manifest, (pcm / 32767.0).tolist(), tolerance_s, *params)
    sr, seed = manifest["render"]["sample_rate"], manifest["render"]["seed"]
    events = [e for layer in manifest["layers"] for e in layer["events"]]
    frames = render_frames(events, sr, seed)
    same = hashlib.sha256(data).hexdigest() == manifest["audio"]["sha256"]
    same = same and close(peak(frames), manifest["audio"]["peak"]) and wav_bytes(frames, sr, seed) == data
    return report | {"wav_reproduces": same, "pass": report["pass"] and same}


# Adapters from the crate's API to the reference, and the cases ---------------------------------

ORACLE = {
    "to_audio": to_audio,
    "to_real": to_real,
    "map_value": map_value,
    "map_pitch": map_pitch,
    "quantise": quantise,
    "heard_hz": heard_hz,
    "audify": lambda *a: audify(*a).tolist(),
    "sonify": sonify,
    "render": render,
    "write_manifest": write_manifest,
    "detect_onsets": detect_onsets,
    "verify": verify,
    "verify_wav": verify_wav,
}


def case(claim: str, fn: str, *args, note: str | None = None, tolerance: dict | None = None) -> dict:
    """A case whose expected value comes from the oracle."""
    c = {"claim": claim, "fn": fn, "args": list(args), "expected": ORACLE[fn](*args)}
    if tolerance is not None:
        c["tolerance"] = tolerance
    if note:
        c["note"] = note
    return c


def error(fn: str, *args, note: str) -> dict:
    return {"claim": "sonify:INPUT-VALIDATION", "fn": fn, "args": list(args), "error": True, "note": note}


def linear(x_lo, x_hi, out_lo, out_hi, clamp=None) -> dict:
    m = {"kind": "linear", "x_lo": x_lo, "x_hi": x_hi, "out_lo": out_lo, "out_hi": out_hi}
    return m if clamp is None else m | {"clamp": clamp}


def exponential(x_lo, x_hi, out_lo, out_hi, clamp=None) -> dict:
    return linear(x_lo, x_hi, out_lo, out_hi, clamp) | {"kind": "exponential"}


def const(value) -> dict:
    return {"kind": "constant", "value": value}


E_PENTATONIC = {"root": 4, "pitch_classes": [0, 2, 4, 7, 9]}
# The E6 pentatonic notes of the earlier Genesis piece: E major pentatonic, MIDI 88 to 104.
GENESIS_E6 = {"kind": "midi", "map": linear(0.0, 1.0, 88.0, 104.0, True), "scale": E_PENTATONIC}


def maps() -> list[dict]:
    m, q, a = "sonify:MAPS-MATCH-REFERENCE", "sonify:SCALE-QUANTISE", "sonify:AUDIFY-MATCHES-REFERENCE"
    t600 = {"speedup": 600.0, "offset_s": 2.0}
    cases = [
        case(m, "to_audio", t600, 1234.5, note="2 + 1234.5 / 600"),
        case(m, "to_audio", {"speedup": 180.0}, 600.0, note="a 10-minute span sped up 180 times lasts 10/3 s"),
        case(m, "to_audio", {"speedup": 0.25, "offset_s": 10.0}, -2.0, note="k < 1 slows time down"),
        case(m, "to_real", t600, 4.0575),
        case(m, "to_real", {"speedup": 180.0}, 10.0 / 3.0),
        case(m, "to_real", {"speedup": 1e6, "offset_s": 0.5}, 0.25, note="before the offset: a negative real time"),
        case(m, "map_value", const(-12.0), 3.0),
        case(m, "map_value", linear(0.0, 10.0, -1.0, 1.0), 2.5),
        case(m, "map_value", linear(0.0, 10.0, -1.0, 1.0), 10.0, note="t = 1 gives out_hi exactly"),
        case(m, "map_value", linear(10.0, 0.0, -30.0, -6.0), 12.5, note="x_hi < x_lo; past the end, unclamped"),
        case(m, "map_value", linear(10.0, 0.0, -30.0, -6.0, True), 12.5, note="clamped to out_lo"),
        case(m, "map_value", linear(0.0, 1.0, 0.1, 0.3), 0.7),
        case(m, "map_value", exponential(0.0, 3.0, 110.0, 880.0), 1.0, note="equal steps, equal ratios: 220 Hz"),
        case(m, "map_value", exponential(0.0, 3.0, 110.0, 880.0), 2.2),
        case(m, "map_value", exponential(0.0, 3.0, 110.0, 880.0), -1.5, note="unclamped below x_lo"),
        case(m, "map_value", exponential(0.0, 3.0, 110.0, 880.0, True), 7.0, note="clamped to out_hi"),
        case(m, "map_value", exponential(-1.0, 1.0, 1.0, 0.001), 0.25, note="falling outputs"),
        case(m, "map_pitch", {"kind": "hz", "map": exponential(0.0, 1.0, 200.0, 800.0)}, 0.5),
        case(m, "map_pitch", {"kind": "midi", "map": linear(0.0, 1.0, 60.0, 72.0)}, 0.75, note="MIDI 69: 440 Hz"),
        case(m, "map_pitch", {"kind": "midi", "map": linear(0.0, 1.0, 60.0, 72.0)}, 0.3, note="MIDI 63.6, unquantised"),
        case(m, "map_pitch", GENESIS_E6, 0.0, note="E6, MIDI 88"),
        case(m, "map_pitch", GENESIS_E6, 0.0625, note="MIDI 89, a tie between 88 and 90: up to F#6"),
        case(m, "map_pitch", GENESIS_E6, 0.4, note="MIDI 94.4: the nearest scale note is B6 (95)"),
        case(m, "map_pitch", GENESIS_E6, 2.0, note="clamped to MIDI 104, G#7"),
        case(m, "map_pitch", {"kind": "rhythm", "octaves": 4}, 12.5, note="a 12.5 Hz beat heard 4 octaves up"),
        case(m, "map_pitch", {"kind": "rhythm", "speedup": 60000.0}, 1 / 600, note="a 10-minute rhythm at x60000"),
        case(m, "map_pitch", {"kind": "rhythm", "speedup": 3.0, "octaves": -2}, 100.0),
        case(m, "map_pitch", {"kind": "rhythm", "octaves": 4.0}, 10.0, note="a whole float is a whole number"),
        case(m, "map_pitch", {"kind": "fold", "low_hz": 220.0}, 1 / 600, note="18 octaves up: 32768/75 Hz"),
        case(m, "map_pitch", {"kind": "fold", "low_hz": 220.0}, 1 / 86400, note="a day, 25 octaves up"),
        case(m, "map_pitch", {"kind": "fold", "speedup": 0.5, "low_hz": 100.0}, 1000.0),
        case(q, "quantise", E_PENTATONIC, 88.0),
        case(q, "quantise", E_PENTATONIC, 89.0, note="halfway between 88 and 90: up"),
        case(q, "quantise", E_PENTATONIC, 93.49),
        case(q, "quantise", E_PENTATONIC, 93.5, note="halfway between 92 and 95: up"),
        case(q, "quantise", E_PENTATONIC, -1.2),
        case(q, "quantise", {"root": 0, "pitch_classes": [0]}, 66.0, note="a tritone from both Cs: up"),
        case(q, "quantise", {"root": 9, "pitch_classes": [0, 3, 5, 7, 10]}, 70.7, note="A minor pentatonic"),
        case(q, "quantise", {"root": 4.0, "pitch_classes": [0.0, 2, 4, 7, 9]}, 89.0, note="whole floats"),
        case(a, "heard_hz", 600.0, 60000.0, note="a 10-minute rhythm audified 60000 times faster: 100 Hz"),
        case(a, "heard_hz", 0.25, 600.0),
        case(a, "heard_hz", 1260.0, 6000.0, note="21 minutes at x6000"),
    ]
    for c in cases:
        if c["fn"] != "quantise":
            c.setdefault("tolerance", MAPS)
    return cases


def audification() -> list[dict]:
    a = "sonify:AUDIFY-MATCHES-REFERENCE"
    # A synthetic spike train: a 0.5-unit rise and an exponential fall every 6 s, sampled every 0.25 s.
    t = np.arange(40) * 0.25
    spikes = [float(round(np.exp(-((tt % 6.0) / 0.8)) * (tt % 6.0 < 4.0), 6)) for tt in t]
    ramp = [0.0, 1.0, 3.0, 2.0, -1.0, 0.5]
    big = 68648.6989340538  # a large constant, then one step of one unit in its last place
    step = [big] * 4 + [float(np.nextafter(big, np.inf))]
    cases = [
        case(a, "audify", spikes, 0.25, 10.0, 400, note="39 s of data at x10: 390 samples"),
        case(a, "audify", spikes, 0.25, 7.3, 300, note="sample times off the data grid"),
        case(a, "audify", ramp, 1.0, 1.0, 3, note="exact data points and the last value held"),
        case(a, "audify", ramp, 1.0, 100.0, 8000, note="max(..., 2): the shortest output"),
        case(a, "audify", step, 1.0, 1.0, 50, note="a step of one ulp at 7e4: the step, not the mean's rounding"),
        case(a, "audify", [0.0, 1e300], 1e-10, 1.0, 4, note="a slope beyond f64: at a point, its own value"),
    ]
    for c in cases:  # every case agrees with exact mean removal of the interpolated samples
        assert np.max(np.abs(np.array(c["expected"]) - audify_exact(*c["args"]))) < 1e-15
    return cases


def audify_exact(values: list, src_dt: float, speedup: float, sr: int) -> np.ndarray:
    """audify with the mean removed and the peak found in exact arithmetic (Fractions)."""
    n_out = max(int((len(values) - 1) * src_dt / speedup * sr), 2)
    y = [F(v) for v in np.interp(np.arange(n_out) / sr * speedup, np.arange(len(values)) * src_dt, values)]
    mean = sum(y) / len(y)
    scale = max(abs(v - mean) for v in y) + F(1e-12)
    return np.array([float((v - mean) / scale) for v in y])


def event_cases() -> list[dict]:
    e = "sonify:EVENTS-MATCH-REFERENCE"
    chimes = {
        "time": {"speedup": 60.0},
        "pitch": GENESIS_E6,
        "gain_db": linear(0.0, 1.0, -24.0, -12.0, True),
        "pan": linear(0.0, 1.0, -1.0, 1.0, True),
        "duration_s": const(2.2),
        "instrument": {"kind": "modal_chime", "brightness": 1.1},
    }
    beats = {
        "time": {"speedup": 1.0, "offset_s": 0.5},
        "pitch": {"kind": "rhythm", "octaves": 4},
        "duration_s": const(0.8),
        "instrument": {"kind": "soft_voice", "vibrato_cents": 8.0},
    }
    folded = {
        "time": {"speedup": 600.0},
        "pitch": {"kind": "fold", "low_hz": 220.0},
        "gain_db": const(-6.0),
        "duration_s": exponential(0.0, 0.002, 4.0, 1.0, True),
        "instrument": {"kind": "fm_bell"},
    }
    hearts = {
        "time": {"speedup": 2.0, "offset_s": 0.25},
        "pitch": {"kind": "hz", "map": exponential(100.0, 200.0, 40.0, 80.0)},
        "duration_s": const(0.5),
        "instrument": {"kind": "heartbeat"},
    }
    return [
        case(e, "sonify", chimes, [0.0, 37.5, 90.0, 133.2], [0.1, 0.0625, 0.8, 1.4], None),
        case(e, "sonify", beats, [0.0, 2.0], [10.0, 19.0], "bio:HUMAN-SPERM-BEAT-VISCOUS", note="beats x16"),
        case(e, "sonify", folded, [0.0, 600.0, 2700.0], [1 / 600, 1 / 900, 1 / 2100], None),
        case(e, "sonify", hearts, [0.0, 1.0, 1.75], [100.0, 150.0, 200.0], None),
    ]


def ev(t, hz, dur, instrument, gain_db=-6.0, pan_pos=0.0, value=0.0) -> dict:
    return {
        "real_s": t,
        "value": value,
        "audio_s": t,
        "hz": hz,
        "gain_db": gain_db,
        "pan": pan_pos,
        "dur_s": dur,
        "instrument": instrument,
    }


def renders() -> list[dict]:
    r = "sonify:RENDER-MATCHES-REFERENCE"
    return [
        case(r, "render", [ev(0.002, 440.0, 0.03, {"kind": "modal_chime", "brightness": 1.0})], 4000, 0),
        case(r, "render", [ev(0.0, 300.0, 0.03, {"kind": "fm_bell", "ratio": 1.4, "index": 10.0}, 0.0, -0.5)], 4000, 0),
        case(
            r,
            "render",
            [ev(0.001, 440.0, 0.03, {"kind": "karplus_strong", "decay": 0.996, "pluck_brightness_hz": 1000.0})],
            4000,
            5,
            note="the string's noise is seeded seed + 0 + 1 = 6",
        ),
        case(
            r,
            "render",
            [ev(0.0, 500.0, 0.03, {"kind": "karplus_strong_tuned", "decay": 0.99, "pluck_brightness_hz": None})],
            4000,
            2**53 - 2,
        ),
        case(r, "render", [ev(0.0, 220.0, 0.04, {"kind": "soft_voice"}, -3.0, 0.7)], 4000, 0),
        case(r, "render", [ev(0.05, 48.0, 0.5, {"kind": "heartbeat", "bpm": 150.0})], 1000, 0, note="lub and dub"),
        case(
            r,
            "render",
            [
                ev(0.0, 600.0, 0.04, {"kind": "modal_chime"}, -12.0, -1.0),
                ev(0.01, 330.0, 0.03, {"kind": "karplus_strong"}, -9.0, 1.0),
                ev(0.0125, 250.0, 0.02, {"kind": "fm_bell", "ratio": 2.0, "index": 3.0}, 0.0, 0.25),
            ],
            4000,
            11,
            note="overlapping notes, panned hard left and right",
        ),
    ]


SPERM = {
    "time": {"speedup": 1.0, "offset_s": 0.1},
    "pitch": {"kind": "rhythm", "octaves": 4},
    "gain_db": const(-9.0),
    "duration_s": const(0.4),
    "instrument": {"kind": "karplus_strong_tuned"},
}
GALLAGHER = "10.1093/humrep/dez056"
SPERM_INFO = {
    "title": "Sperm flagellar beats, four octaves up",
    "creator": "Chethan Dandebala Venkatarama",
    "licence": "CC-BY-4.0",
    "sources": [
        {"claim": "bio:HUMAN-SPERM-BEAT-VISCOUS", "doi": GALLAGHER, "values": [10.0]},
        {"claim": "bio:HUMAN-SPERM-BEAT-DILUTED-SEMEN", "doi": GALLAGHER, "values": [19.0]},
    ],
}
SPERM_SERIES = [
    {
        "name": "viscous medium",
        "mapping": SPERM,
        "times": [0.0],
        "values": [10.0],
        "claim": "bio:HUMAN-SPERM-BEAT-VISCOUS",
    },
    {
        "name": "diluted semen",
        "mapping": SPERM | {"time": {"speedup": 1.0, "offset_s": 0.6}, "pan": const(0.3)},
        "times": [0.0],
        "values": [19.0],
        "claim": "bio:HUMAN-SPERM-BEAT-DILUTED-SEMEN",
    },
]
CHIMES_INFO = {"title": "Synthetic chimes", "creator": "Chethan Dandebala Venkatarama", "licence": "CC0-1.0"}
CHIMES = {
    "time": {"speedup": 100.0, "offset_s": 0.05},
    "pitch": {"kind": "midi", "map": linear(0.0, 1.0, 45.0, 57.0, True), "scale": E_PENTATONIC},
    "gain_db": const(-6.0),
    "pan": linear(0.0, 1.0, -0.6, 0.6),
    "duration_s": const(0.12),
    "instrument": {"kind": "modal_chime", "brightness": 0.8},
}
HEARTS = {
    "time": {"speedup": 100.0, "offset_s": 0.05},
    "pitch": {"kind": "hz", "map": const(55.0)},
    "duration_s": const(0.5),
    "instrument": {"kind": "heartbeat", "bpm": 300.0},
}
CHIME_SERIES = [{"name": "chimes", "mapping": CHIMES, "times": [0.0, 17.0, 34.5, 52.0], "values": [0.2, 0.9, 0.5, 0.0]}]
ROUND_TRIP_SERIES = CHIME_SERIES + [{"name": "hearts", "mapping": HEARTS, "times": [69.0], "values": [0.0]}]


ONE_INFO = CHIMES_INFO | {"sources": [{"claim": "test:ONE-CHIME", "values": [0.5]}]}
ONE_CHIME = [
    {
        "name": "chime",
        "mapping": CHIMES | {"time": {"speedup": 1.0, "offset_s": 0.02}, "pan": const(0.0)},
        "times": [0.0],
        "values": [0.5],
        "claim": "test:ONE-CHIME",
    }
]
# detect_onsets' defaults: hop_s, window_s, threshold_db, min_gap_s and floor_db.
ONSET_DEFAULTS = (0.001, 0.02, 6.0, 0.05, -90.0)


def shifted(series: list[dict], offset_s: float) -> list[dict]:
    """The series heard from offset_s seconds into the audio."""
    return [s | {"mapping": s["mapping"] | {"time": {"speedup": 100.0, "offset_s": offset_s}}} for s in series]


def manifests() -> list[dict]:
    w = "sonify:MANIFEST"
    return [
        case(w, "write_manifest", SPERM_INFO, SPERM_SERIES, 8000, 1, note="two layers from two cited sources"),
        case(w, "write_manifest", CHIMES_INFO, CHIME_SERIES, 4000, 0, note="no sources: synthetic data"),
    ]


def onset_cases() -> list[dict]:
    o = "sonify:ONSETS-MATCH-REFERENCE"
    sr = 1000
    manifest = write_manifest(CHIMES_INFO, ROUND_TRIP_SERIES, sr, 3)
    frames = render_frames([e for layer in manifest["layers"] for e in layer["events"]], sr, 3)
    stereo = [round(v, 6) for v in frames.reshape(-1).tolist()]  # the input, rounded to keep the file small
    late = write_manifest(CHIMES_INFO, shifted(ROUND_TRIP_SERIES, 0.06), sr, 3)  # every note 10 ms late
    # A WAV file whose first note starts 0.2 s in, after dithered silence, and a manifest with
    # the same timing but other pitches, which only the comparison of the files catches.
    quiet_start = write_manifest(CHIMES_INFO, shifted(ROUND_TRIP_SERIES, 0.2), sr, 5)
    events = [e for layer in quiet_start["layers"] for e in layer["events"]]
    wav = list(wav_bytes(render_frames(events, sr, 5), sr, 5))
    low = {"kind": "midi", "map": linear(0.0, 1.0, 33.0, 45.0, True), "scale": E_PENTATONIC}
    other = shifted(ROUND_TRIP_SERIES, 0.2)
    other[0] = other[0] | {"mapping": other[0]["mapping"] | {"pitch": low}}
    lie = write_manifest(CHIMES_INFO, other, sr, 5)
    params = ONSET_DEFAULTS
    loose = {"rel": 1e-12, "abs": 1e-9}
    # One chime from a cited source, and five manifests that misdescribe it: a false peak, a false
    # instrument list, a claim that is not among the sources, a mono encoding and a malformed claim id.
    one = write_manifest(ONE_INFO, ONE_CHIME, sr, 2)
    one_frames = render_frames(one["layers"][0]["events"], sr, 2)
    one_stereo, one_wav = one_frames.reshape(-1).tolist(), list(wav_bytes(one_frames, sr, 2))
    false_peak = json.loads(json.dumps(one))
    false_peak["audio"]["peak"] = 123.0
    false_instruments = json.loads(json.dumps(one))
    false_instruments["render"]["instruments"] = [full_instrument({"kind": "heartbeat"})]
    uncited = json.loads(json.dumps(one))
    uncited["layers"][0]["claim"] = "x:NOT-A-SOURCE"
    for e in uncited["layers"][0]["events"]:
        e["claim"] = "x:NOT-A-SOURCE"
    mono = json.loads(json.dumps(one))
    mono["audio"]["channels"] = 1
    spaced = json.loads(json.dumps(one))  # a claim id write_manifest refuses, in the sources and the layer
    spaced["sources"][0]["claim"] = spaced["layers"][0]["claim"] = "test: ONE-CHIME"
    for e in spaced["layers"][0]["events"]:
        e["claim"] = "test: ONE-CHIME"
    lies = [
        case(o, "verify_wav", one, one_wav, 0.002, *params, tolerance=loose, note="one cited chime: passes"),
        case(o, "verify_wav", false_peak, one_wav, 0.002, *params, tolerance=loose, note="a false peak"),
        case(o, "verify", false_instruments, one_stereo, 0.002, *params, tolerance=loose, note="false instruments"),
        case(o, "verify", uncited, one_stereo, 0.002, *params, tolerance=loose, note="a claim not among the sources"),
        case(o, "verify", mono, one_stereo, 0.002, *params, tolerance=loose, note="channels must be 2"),
        case(o, "verify", spaced, one_stereo, 0.002, *params, tolerance=loose, note="a claim id with a space"),
    ]
    assert lies[0]["expected"]["pass"] and not any(c["expected"]["pass"] for c in lies[1:])
    return lies + [
        case(
            o,
            "detect_onsets",
            stereo,
            sr,
            *params,
            tolerance=loose,
            note="four chimes, then a heartbeat note of two lub-dubs",
        ),
        case(o, "detect_onsets", stereo, sr, 0.004, 0.03, 12.0, 0.1, -60.0, tolerance=loose, note="coarser"),
        case(o, "verify", manifest, stereo, 0.002, *params, tolerance=loose, note="every event found"),
        case(o, "verify", late, stereo, 0.002, *params, tolerance=loose, note="10 ms late: every event missing"),
        case(o, "verify_wav", quiet_start, wav, 0.002, *params, tolerance=loose, note="the WAV file passes"),
        case(o, "verify_wav", lie, wav, 0.002, *params, tolerance=loose, note="other pitches: the files differ"),
    ]


def errors() -> list[dict]:
    t, lin = {"speedup": 2.0}, linear(0.0, 1.0, 0.0, 1.0)
    chime = {
        "time": t,
        "pitch": {"kind": "hz", "map": const(440.0)},
        "duration_s": const(0.1),
        "instrument": {"kind": "modal_chime"},
    }
    good = ev(0.0, 440.0, 0.01, {"kind": "modal_chime"})
    info = {"title": "x", "creator": "y", "licence": "CC0-1.0", "sources": [{"claim": "a:B", "values": []}]}
    series = [{"name": "n", "mapping": chime, "times": [0.0], "values": [0.5]}]
    return [
        error("to_audio", {"speedup": 0.0}, 1.0, note="speedup must be > 0"),
        error("to_audio", {"speedup": -2.0, "offset_s": 1.0}, 1.0, note="a negative speedup"),
        error("to_audio", {"speedup": 0.5}, 1e308, note="an audio time beyond f64"),
        error("to_real", {"speedup": 2.0, "start_s": 1.0}, 1.0, note="an unknown field"),
        error("map_value", linear(1.0, 1.0, 0.0, 1.0), 1.0, note="x_hi must differ from x_lo"),
        error("map_value", exponential(0.0, 1.0, 0.0, 1.0), 0.5, note="out_lo must be > 0"),
        error("map_value", exponential(0.0, 1.0, 1.0, -2.0), 0.5, note="out_hi must be > 0"),
        error("map_value", {"kind": "cubic", "x_lo": 0.0}, 0.5, note="an unknown kind"),
        error("map_pitch", {"kind": "hz", "map": lin}, 0.0, note="0 Hz"),
        error("map_pitch", {"kind": "midi", "map": lin, "scale": {"root": 12, "pitch_classes": [0]}}, 0.5, note="root"),
        error("map_pitch", {"kind": "rhythm"}, -1.0, note="a rhythm's frequency must be > 0"),
        error("map_pitch", {"kind": "rhythm", "octaves": 2000}, 1.0, note="2^2000 is not an f64"),
        error("map_pitch", {"kind": "rhythm", "octaves": 0.5}, 1.0, note="octaves must be whole"),
        error("map_pitch", {"kind": "fold", "low_hz": 0.0}, 1.0, note="low_hz must be > 0"),
        error("quantise", {"root": 0, "pitch_classes": []}, 60.0, note="no pitch classes"),
        error("quantise", {"root": 0, "pitch_classes": [12]}, 60.0, note="a pitch class above 11"),
        error("quantise", {"root": 0, "pitch_classes": [0]}, 1e300, note="beyond 2^52"),
        error("heard_hz", 0.0, 600.0, note="period_s must be > 0"),
        error("heard_hz", 600.0, -1.0, note="speedup must be > 0"),
        error("audify", [0.5, 0.5, 0.5], 1.0, 2.0, 100, note="a constant series"),
        error("audify", [0.0, 0.0], 1.0, 2.0, 100, note="silence is constant"),
        error("audify", [0.0, 0.0, 0.0, 1.0], 1.0, 150.0, 100, note="the samples miss the change: all equal"),
        error("audify", [0.1] * 10 + [0.0], 1.0, 300.0, 100, note="the change comes after the last sample"),
        error("audify", [1.0], 1.0, 2.0, 100, note="at least two values"),
        error("audify", [0.0, 1.0], 0.0, 2.0, 100, note="src_dt must be > 0"),
        error("audify", [0.0, 1.0], 1.0, 0.0, 100, note="speedup must be > 0"),
        error("audify", [0.0, 1.0], 1.0, 2.0, 0, note="sr must be >= 1"),
        error("audify", [0.0, 1.0], 1.0, 2.0, -1, note="a negative sample rate"),
        error("sonify", chime, [0.0, 1.0], [0.5], None, note="times and values differ in length"),
        error("sonify", chime | {"time": {"speedup": 2.0, "offset_s": -1.0}}, [0.0], [0.5], None, note="audio_s < 0"),
        error("sonify", chime | {"pan": const(1.5)}, [0.0], [0.5], None, note="pan beyond 1"),
        error("sonify", chime | {"duration_s": const(0.0)}, [0.0], [0.5], None, note="a note of 0 s"),
        error("sonify", {"time": t}, [0.0], [0.5], None, note="a mapping without pitch"),
        error("sonify", chime | {"pan": const(7.0)}, [], [], None, note="no points, but a constant pan beyond 1"),
        *[
            error("sonify", chime | {"pitch": pitch}, [], [], None, note=f"no points, but {why}")
            for pitch, why in [
                ({"kind": "fold", "low_hz": 1e308}, "low_hz above f64::MAX / 2"),
                ({"kind": "fold", "low_hz": 1e-310}, "a subnormal low_hz"),
                ({"kind": "rhythm", "octaves": 5000}, "an octave count no frequency survives"),
                ({"kind": "rhythm", "speedup": 1e-320, "octaves": -1074}, "every heard frequency shifted to 0"),
                ({"kind": "midi", "map": const(1e6)}, "a constant MIDI note beyond f64"),
            ]
        ],
        error("sonify", chime | {"gain_db": const(1e6)}, [], [], None, note="no points, but a gain beyond f64"),
        error("render", [good | {"audio_s": -0.5}], 8000, 0, note="a note before 0 s"),
        error("render", [good | {"instrument": {"kind": "karplus_strong", "decay": 2.0}}], 8000, 0, note="decay > 1"),
        error("render", [good], 0, 0, note="sr must be >= 1"),
        error("render", [], 0, 0, note="sr must be >= 1, even with no events"),
        error("render", [good], 8000, -1, note="a negative seed"),
        error("render", [good], 8000, 2**53, note="a seed above 2^53 - 1"),
        error("write_manifest", info | {"title": ""}, series, 8000, 0, note="an empty title"),
        error("write_manifest", info | {"title": "\x1c"}, series, 8000, 0, note="a control character (U+001C)"),
        error("write_manifest", info | {"creator": "a\tb"}, series, 8000, 0, note="a tab"),
        error(
            "write_manifest",
            info | {"sources": [{"claim": "a:\x1fB", "values": []}]},
            series,
            8000,
            0,
            note="a claim holding U+001F",
        ),
        error("write_manifest", info, series, 0, 0, note="sr must be >= 1"),
        error("write_manifest", info, [series[0] | {"claim": "c:D"}], 8000, 0, note="a claim that is not a source"),
        error("write_manifest", info, [], 8000, 0, note="nothing to render"),
        error(
            "write_manifest",
            info,
            series + [{"name": "e", "mapping": chime | {"time": {"speedup": -5.0}}, "times": [], "values": []}],
            8000,
            0,
            note="a layer without points still has its mapping checked",
        ),
        error(
            "write_manifest",
            info | {"sources": [{"claim": "a:B", "doi": "doi 10.1/x", "values": []}]},
            series,
            8000,
            0,
            note="DOI",
        ),
        error(
            "write_manifest", info | {"sources": [{"claim": "no colon", "values": []}]}, series, 8000, 0, note="claim"
        ),
        error("detect_onsets", [0.0, 0.0, 0.0], 8000, *ONSET_DEFAULTS, note="not whole frames"),
        error("detect_onsets", [0.0, 0.0], 0, *ONSET_DEFAULTS, note="sr must be >= 1"),
        *[
            error("detect_onsets", [0.0, 0.0], 8000, *params, note=why)
            for params, why in [
                ((1e-6, 0.02, 6.0, 0.05, -90.0), "a hop shorter than a sample"),
                ((1e9, 0.02, 6.0, 0.05, -90.0), "a hop beyond MAX_SAMPLES"),
                ((0.001, 1e9, 6.0, 0.05, -90.0), "a window beyond MAX_SAMPLES"),
                ((0.001, 0.02, 6.0, -0.05, -90.0), "a negative min_gap_s"),
                ((0.001, 0.02, 6.0, 0.05, -4000.0), "a floor of 0"),
            ]
        ],
        error(
            "verify",
            write_manifest(CHIMES_INFO, CHIME_SERIES, 4000, 0),
            [0.0, 0.0],
            -0.001,
            *ONSET_DEFAULTS,
            note="tolerance",
        ),
        error(
            "verify_wav",
            write_manifest(CHIMES_INFO, CHIME_SERIES, 4000, 0),
            list(b"RIFF"),
            0.002,
            *ONSET_DEFAULTS,
            note="not a WAV file",
        ),
    ]


if __name__ == "__main__":
    write(
        "maps",
        "The time, parameter and pitch maps, scale quantisation and heard frequencies against exact arithmetic.",
        maps(),
        MAPS,
    )
    write("audify", "audify against lifesound.audify (numpy.interp, mean removed, peak normalised).", audification())
    write("events", "sonify against the exact maps.", event_cases())
    write("render", "render against lifesound.py's instruments, pan and mix_into.", renders())
    write(
        "manifests",
        "write_manifest against an independent writer: sources, mappings, events and the WAV's SHA-256.",
        manifests(),
    )
    write(
        "onsets",
        "detect_onsets, verify and verify_wav against an independent port of the documented method.",
        onset_cases(),
        compact=True,
    )
    write("errors", "Invalid input is rejected with an error in every binding.", errors())
    print("software version", VERSION)
