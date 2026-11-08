# Sonification core

**Turns measured rhythms into sound that keeps their ratios: a mapping grammar, audification and sonification manifests, in Rust, Python and WebAssembly.**

[![CI](https://github.com/chethandvg/sonification-core/actions/workflows/ci.yml/badge.svg)](https://github.com/chethandvg/sonification-core/actions/workflows/ci.yml)
[![Proofs](https://github.com/chethandvg/sonification-core/actions/workflows/proofs.yml/badge.svg)](https://github.com/chethandvg/sonification-core/actions/workflows/proofs.yml)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)

Part of **[Frequencies of Life](https://chethandvg.github.io/frequencies-of-life/)**, open code for the rhythms of life in which every claim says how it is known: proved, verified, tested, referenced or assumed.

This is the instrument layer of the project, a library for anyone who builds music, visuals or tools from data: you describe how data becomes sound in a small grammar, render it, and get a manifest that traces every note back to the number that made it. Anyone can then check a WAV file against its manifest: that the manifest's notes render to exactly that file, and that the data's timing comes back from the sound alone.

## Claims and evidence

<!-- claims-table:start -->
| Claim | Level | Statement | Evidence |
|---|---|---|---|
| `sonify:TIME-MAP` | P | For a speed-up k > 0, the time map a(t) = offset + t/k (rhythm-units' scalePeriod t k moved by offset) is strictly increasing, keeps every ratio of durations, (a(t₃) − a(t₂))/(a(t₂) − a(t₁)) = (t₃ − t₂)/(t₂ − t₁), and its inverse a ↦ (a − offset)·k undoes it both ways. | [`SonificationCore.time_map`](proofs/SonificationCore/TimeMap.lean) |
| `sonify:OCTAVE-SHIFT` | P | Moving every frequency by the same n octaves, f ↦ f·2ⁿ (pitch-math's octaveTranspose), keeps every frequency ratio and so every interval in cents: cents(f₁·2ⁿ, f₂·2ⁿ) = cents(f₁, f₂) for all real f₁, f₂ and integers n. | [`SonificationCore.octave_shift`](proofs/SonificationCore/Pitch.lean) |
| `sonify:LOG-MAP` | P | For x_lo < x_hi and 0 < f_lo < f_hi, the exponential (log-frequency) map g(x) = f_lo·(f_hi/f_lo)^((x − x_lo)/(x_hi − x_lo)) sends x_lo to f_lo and x_hi to f_hi, is strictly increasing, and g(x + d)/g(x) = (f_hi/f_lo)^(d/(x_hi − x_lo)) does not depend on x: equal steps of the data give equal frequency ratios. | [`SonificationCore.log_map`](proofs/SonificationCore/Maps.lean) |
| `sonify:LINEAR-MAP` | P | For x_lo < x_hi, the linear map y_lo + (x − x_lo)/(x_hi − x_lo)·(y_hi − y_lo) sends x_lo to y_lo and x_hi to y_hi, is strictly increasing when y_lo < y_hi, and with the position clamped to [0, 1] its output always lies between y_lo and y_hi. | [`SonificationCore.linear_map`](proofs/SonificationCore/Maps.lean) |
| `sonify:ONSET-DETECTION-METHOD` | R | The usual way to detect note onsets in music signals is to look for transient regions, for example a sudden burst of energy; methods based on the signal's amplitude envelope are among the onset-detection techniques that Bello et al. 2005 review, categorise and compare. | Bello, Daudet, Abdallah, Duxbury, Davies and Sandler 2005, IEEE Transactions on Speech and Audio Processing 13(5):1035–1047, 'A tutorial on onset detection in music signals' ([doi:10.1109/TSA.2005.851998](https://doi.org/10.1109/TSA.2005.851998)) |
| `sonify:MAPS-MATCH-REFERENCE` | T | to_audio, to_real, map_value (constant, linear and exponential, clamped or not) and map_pitch (hz; midi with or without a scale; rhythm, a frequency sped up and moved by octaves; fold, a frequency folded into an octave) match exact arithmetic (Fractions, and mpmath at 50 digits for powers) within 1e-14 relative on the golden vectors (largest measured difference 2.3e-16 relative); linear and exponential maps return out_hi exactly at the top end. The time map's spans and frequencies go through rhythm-units' scale_period and scale_frequency, MIDI notes through pitch-math's midi_to_hz, octave shifts and folds through pitch-math's octave_transpose and octave_fold. | [`vectors/maps.json`](vectors/maps.json), [`src/map.rs`](src/map.rs) |
| `sonify:SCALE-QUANTISE` | T | quantise returns the nearest MIDI note whose pitch class above the root is in the scale; a note exactly halfway between two scale notes goes to the higher one, as pitch-math's nearest_note rounds halves up. The E6 pentatonic notes of the earlier Genesis piece (MIDI 88, 90, 92, 95, 97, 100, 102 and 104) are E major pentatonic (root 4, pitch classes 0, 2, 4, 7 and 9) with the MIDI map clamped to 88..104. | [`vectors/maps.json`](vectors/maps.json), [`src/map.rs`](src/map.rs) |
| `sonify:AUDIFY-MATCHES-REFERENCE` | T | audify matches lifesound.audify (copied into the oracle: numpy.interp at the speed-up, mean removed, divided by max\|y\| + 1e-12) within 1e-12 on the golden vectors (largest measured difference 3.3e-16), with two changes made in Rust, Python, JavaScript and the oracle. Audio that would not vary is an error instead of lifesound.py's output (all zeros when the mean came out exact, and otherwise the mean's rounding error scaled up): a constant series, silence included, and a series whose interpolated samples are all equal (it varies only between the audio's sample times or after the last of them). And the mean is removed from the samples less the first one, which in exact arithmetic changes nothing (the oracle's outputs agree with mean removal in Fractions within 1e-15, and with lifesound.py's own formula within 2.2e-16 on the five other vectors) but keeps a series near 7e4 that steps by one unit in the last place a step, where lifesound.py's formula gives the step with a large offset (0.61 away). As in numpy.interp, a sample at a data point's own time takes that point's value, so a slope too steep for an f64 is no error there (a golden vector). heard_hz(T, k), rhythm-units' scale_frequency(hz_from_period(T), k), is k/T within 1e-14 relative of the exact quotient: a rhythm of period T audified at speed-up k is heard at k/T Hz. | [`vectors/audify.json`](vectors/audify.json), [`vectors/maps.json`](vectors/maps.json), [`vectors/errors.json`](vectors/errors.json), [`src/audify.rs`](src/audify.rs) |
| `sonify:EVENTS-MATCH-REFERENCE` | T | sonify gives one event per data point, in order, with audio_s, hz, gain_db, pan and dur_s from the mapping's maps, matching the exact maps within 1e-12 (largest measured difference 1.7e-16 relative); missing JSON fields take the documented defaults (gain 0 dB, centre pan, lifesound.py's instrument parameters), whole numbers may be written as whole floats (4.0), and a note before audio time 0, panned outside [−1, 1], of length 0 or below, or with a gain whose amplitude 10^(gain_db/20) is not finite is an error. sonify first checks every constant of the mapping on its own and the output of every constant map (Mapping::check: the speed-ups, the map ends, the scale, a fold's low_hz within pitch-math's range, an octave count that some frequency heard at the rhythm's speed-up survives, and a constant frequency, gain, pan or length), so such a mapping is an error even for a series with no points; an output that depends on the data is checked as each point is mapped. | [`vectors/events.json`](vectors/events.json), [`src/event.rs`](src/event.rs) |
| `sonify:RENDER-MATCHES-REFERENCE` | T | render matches lifesound.py's instruments (with signal-core's SplitMix64 noise and ln 1000 in exp_decay), pan and mix_into, with each note's gain 10^(gain_db/20), within 1e-12 on the golden vectors (all six instruments, and three overlapping notes panned hard left and right; largest measured difference 5.7e-15): synthesis-theory plays each note, signal-core pans and mixes it and audio-safety-math converts its gain. | [`vectors/render.json`](vectors/render.json) |
| `sonify:RENDER-DETERMINISTIC` | T | The same events, sample rate and seed always give the same WAV bytes and SHA-256 (rendered twice in Rust, Python and JavaScript), and a different seed changes the dither and so the hash; the SHA-256 that an independent writer computes (numpy instruments, Python's wave module, hashlib) equals the crate's exactly in Rust, Python and WebAssembly on the golden vectors. | [`vectors/manifests.json`](vectors/manifests.json), [`tests/roundtrip.rs`](tests/roundtrip.rs), [`bindings/python/tests/test_vectors.py`](bindings/python/tests/test_vectors.py), [`bindings/wasm/tests/vectors.test.mjs`](bindings/wasm/tests/vectors.test.mjs) |
| `sonify:MANIFEST` | T | write_manifest's manifest (format sonification-manifest/0.1-draft: title, creator, the audio's licence, the software name and version, sources with claim ids, DOIs and values, every layer's full mapping and events, render settings with the instruments used, and the frames, peak and SHA-256 of the 16-bit PCM WAV file) matches an independent writer, validates against schema/manifest.schema.json (JSON Schema 2020-12, whose text patterns list the blank characters, Unicode's control characters and white space, by code point, and end in (?![\s\S]), the end of the text both in ECMA-262 and in Python's re, where $ would also match before a final line feed; checked with jsonschema in the oracle and the Python tests, and in the Rust tests by a checker of the schema's own keywords that does not share the crate's character tests; the Python and WebAssembly tests check that a title, claim, DOI or SHA-256 ending in a line feed fails), also when its title, a claim or a DOI holds each control character and each of Unicode's 25 White_Space characters in turn (in the Rust and Python tests), round-trips through JSON unchanged (to_json and from_json both ways in Rust, json in Python), and describes its own events (reproduces_events: they reproduce from its mappings, every layer's claim is one of the sources, its instrument list is the events' own, its audio is wav-pcm16 in 2 channels, and its title, creator, licence and sources have the form write_manifest requires); a series whose claim is not among the sources, a layer without points whose mapping fails Mapping::check, a title, creator or licence that is blank or holds a control character, a claim id or DOI that is malformed or holds a blank character, a piece without events, a sample rate of 0 and a seed above 2^53 − 1 are errors. | [`vectors/manifests.json`](vectors/manifests.json), [`schema/manifest.schema.json`](schema/manifest.schema.json), [`tests/roundtrip.rs`](tests/roundtrip.rs), [`bindings/python/tests/test_vectors.py`](bindings/python/tests/test_vectors.py) |
| `sonify:ONSETS-MATCH-REFERENCE` | T | detect_onsets, verify and verify_wav match an independent Python port of their documented method (running totals of the stereo power l² + r², the energy ratio of the windows after and before each candidate with each energy raised to a power floor, −90 dB by default, if below it, a rise threshold, the latest largest rise within the minimum gap, refinement to the frame; events matched in audio-time order to the nearest free onset within the tolerance, leftover onsets near a heartbeat note's own dubs and later beats counted as within the note, the manifest's description of its events (layer claims among the sources, the instrument list, the encoding and channels, and the form of its title, creator, licence and sources) and the audio's length checked; for verify_wav, the WAV file decoded and compared byte for byte with a new render of the manifest's events, whose peak must be the manifest's) exactly in the onset times and within 1e-12 relative and 1e-9 absolute in the report (largest measured difference 1.4e-14), on twelve cases including a manifest shifted by 10 ms, a WAV file whose first note starts after 0.2 s of dithered silence, a manifest with the same timing but other pitches, and manifests with a false peak, instrument list, claim or channel count, or a claim id with a space. | [`vectors/onsets.json`](vectors/onsets.json), [`src/verify.rs`](src/verify.rs) |
| `sonify:ROUND-TRIP` | T | For renders of all six instruments at 8, 22.05, 44.1 and 48 kHz whose notes start from silence (0.4 s notes 0.47 to 0.69 s apart, gains −18 to −6 dB, speed-up 600, the first note 0.05 or 0.3 s into the audio), verify on the float render and verify_wav on the 16-bit WAV file that write_manifest returns both match every event to an onset within 0.5 ms of its audio time (largest measured 0.38 ms) with no extra onsets, and recover each event's real time within 600 times that (the real-time error is k times the audio error, within rounding); quiet notes after 0.3 s of dithered silence at 8 and 44.1 kHz are matched within 1 ms for the struck, plucked and beating instruments at −50 dB and within 1.5 ms for the soft voice at −30 dB (largest measured 0.55 and 1.3 ms); on the test's own series, struck bars and FM bells that overlap (1.2 s notes 0.19 to 0.29 s apart at −12 dB, 44.1 kHz) are matched within the default 2 ms tolerance (largest measured 1.55 ms), and in 40 seeded random pieces of that kind every FM bell note is too (largest measured 1.33 ms), but for struck bars this is not a bound: 3 of the 40 struck-bar pieces fail, with a bar found 2.16 or 2.36 ms early, or not found, while earlier bars still ring; and the check fails for a manifest whose times are 10 ms off (self-consistent or edited), audio longer than the manifest records, a note the manifest does not list (inside a heartbeat note too), and a manifest whose instrument list, encoding, channel count or claims misdescribe its events, and verify_wav fails for a manifest with the same timing but other pitches, even with the file's SHA-256 copied into it, or with a false peak. | [`tests/roundtrip.rs`](tests/roundtrip.rs), [`bindings/python/tests/test_vectors.py`](bindings/python/tests/test_vectors.py), [`bindings/wasm/tests/vectors.test.mjs`](bindings/wasm/tests/vectors.test.mjs) |
| `sonify:INPUT-VALIDATION` | T | Every function rejects NaN and infinite input, a speed-up of 0 or below, a map whose input ends are equal or whose exponential outputs are not above 0, a pitch of 0 Hz or below, an octave shift outside pitch-math's range, a fold's low_hz outside pitch-math's range, a scale root or pitch class outside 0..11 or an empty scale, an integer field that is not a whole number, an event before audio time 0 or panned outside [−1, 1], a note of length 0 or below, a series to audify whose audio would not vary, times and values of different lengths, a sample rate of 0 (with no events too), a seed above 2^53 − 1, a stereo buffer of odd length, a hop or window shorter than a sample, a hop, window or minimum gap that is negative or beyond MAX_SAMPLES samples, a power floor of 0 or infinity, a negative tolerance, a file that is not a stereo 16-bit PCM WAV file at the manifest's sample rate, a gain whose amplitude is not finite, unknown or missing JSON fields, and the manifest rules of sonify:MANIFEST; the crate's own errors name the argument they are about (sample_rate, hop_s, window_s, min_gap_s, audio_s, dur_s, gain_db, octaves, title and so on; sample_rate also in Instrument::note), errors from rhythm-units, pitch-math, signal-core and synthesis-theory keep their names and messages, and a gain that audio-safety-math's db_to_amplitude cannot turn into a finite amplitude is reported as gain_db. Rust returns SonifyError, Python raises ValueError (OverflowError for a negative or too-large integer argument such as a sample rate, or a seed beyond 64 bits, TypeError for a non-integer one) and JavaScript throws an Error (also for a sample rate that is not a whole number up to 2^32 − 1 or a seed that is not a whole number up to 2^53 − 1). | [`vectors/errors.json`](vectors/errors.json), [`tests/roundtrip.rs`](tests/roundtrip.rs), [`src/lib.rs`](src/lib.rs), [`src/map.rs`](src/map.rs), [`src/audify.rs`](src/audify.rs), [`src/event.rs`](src/event.rs), [`src/verify.rs`](src/verify.rs), [`bindings/python/tests/test_vectors.py`](bindings/python/tests/test_vectors.py), [`bindings/wasm/tests/vectors.test.mjs`](bindings/wasm/tests/vectors.test.mjs) |
| `sonify:IMPLEMENTATIONS-AGREE` | T | The Rust crate, the Python bindings and the WebAssembly bindings all reproduce every golden vector. Every expected output comes from the oracle (lifesound.py's code with signal-core's documented changes and two to audify, exact Fractions and mpmath for the maps, and independent Python versions of sonify, render, the manifest writer, the onset detector, verify and verify_wav), never from the Rust output; the error cases in errors.json are the crate's own input rules. | [`vectors/generate.py`](vectors/generate.py), [`tests/vectors.rs`](tests/vectors.rs), [`bindings/python/tests/test_vectors.py`](bindings/python/tests/test_vectors.py), [`bindings/wasm/tests/vectors.test.mjs`](bindings/wasm/tests/vectors.test.mjs) |
<!-- claims-table:end -->

Levels: **P** machine-checked proof (Lean 4) · **V** verified numerics · **T** tested · **R** referenced (DOI or PMID) · **M** model assumption.

## Use

Five pieces, with the same names in Rust, Python and JavaScript (a Rust method `TimeMap::to_audio` is the function `to_audio(time_map, t)` in the bindings):

- **The mapping grammar.** A `Mapping` holds every constant of a sonification. Its `time` is a `TimeMap`, `audio = offset_s + real / speedup`. Its `pitch` is a `PitchMap`: `hz` (a value mapped straight to hertz), `midi` (a value mapped to a MIDI note, optionally quantised to a `Scale`, then to equal temperament), `rhythm` (a value that is itself a frequency, heard sped up and moved by octaves: `f · speedup · 2^octaves`) or `fold` (a frequency folded into the octave `[low_hz, 2·low_hz)`). Its `gain_db`, `pan` and `duration_s` are `ParamMap`s: `constant`, `linear` or `exponential` (equal data steps give equal ratios, a log-frequency scale), each optionally clamped. Its `instrument` is one of synthesis-theory's six: `modal_chime`, `fm_bell`, `karplus_strong`, `karplus_strong_tuned`, `soft_voice`, `heartbeat`. `Mapping::check` (Rust) checks every constant on its own, and the output of every constant map, before there is any data; an output that depends on the data is checked as each value is mapped.
- **Events and rendering.** `sonify(mapping, times, values, claim)` gives one event per data point (its real time, value, claim id, audio time, frequency, gain, pan, length and instrument); `render(events, settings)` plays them into an interleaved stereo buffer, and `render_wav` writes it as a 16-bit WAV file.
- **Audification.** `audify(values, src_dt, speedup, sample_rate)` plays a measured series directly as a waveform, sped up; a rhythm of period `T` is then heard at `heard_hz(T, speedup)` = `speedup / T` hertz.
- **The manifest.** `write_manifest(info, series, settings)` sonifies and renders a piece and returns its manifest and WAV file (see [The manifest](#the-manifest)).
- **The round trip.** `verify_wav(manifest, wav, tolerance_s, params)` checks a WAV file against its manifest: it finds the onsets in the sound (`detect_onsets`), matches them to the manifest's events and inverts each time map, so you get back the data's real times and the error of each, and it checks that the manifest's events render to exactly that file. `verify(manifest, stereo, …)` does the timing part for a float buffer (see [The round trip](#the-round-trip)).

Rendering is deterministic on one build: the same events, sample rate and seed always give the same samples and the same WAV bytes. Across builds the float samples can differ in their last bits, because WebAssembly has its own `sin`, `cos`, `exp` and `pow` (as native builds on different platforms may too). CI checks every build's renders against the oracle within the golden vectors' 1e-12 tolerance and the SHA-256s of their WAV files against the oracle's exactly, so equal hashes across Rust, Python and JavaScript are checked on the platforms CI runs, not guaranteed (see [How it is verified](#how-it-is-verified)). The pieces reuse the project's foundations rather than rewriting them: time maps go through [rhythm-units](https://github.com/chethandvg/rhythm-units) (`scale_period`, `scale_frequency`), pitches through [pitch-math](https://github.com/chethandvg/pitch-math) (`midi_to_hz`, `octave_transpose`, `octave_fold`), notes through [synthesis-theory](https://github.com/chethandvg/synthesis-theory), panning, mixing and WAV files through [signal-core](https://github.com/chethandvg/signal-core), and gains in dB through [audio-safety-math](https://github.com/chethandvg/audio-safety-math) (`db_to_amplitude`).

The examples below sonify two measured beat rates of human sperm flagella, 10 Hz in a viscous medium and 19 Hz in diluted semen (medians; Gallagher et al. 2019, Human Reproduction 34(7):1173–1185, [doi:10.1093/humrep/dez056](https://doi.org/10.1093/humrep/dez056), recorded in [biological-rhythms-dataset](https://github.com/chethandvg/biological-rhythms-dataset) as `bio:HUMAN-SPERM-BEAT-VISCOUS` and `bio:HUMAN-SPERM-BEAT-DILUTED-SEMEN`). Each rate becomes a plucked note four octaves up, 160 Hz and 304 Hz. Nothing here is a recording: sperm make no sound we could hear; the data is translated into sound.

**Rust**

```toml
[dependencies]
sonification-core = { git = "https://github.com/chethandvg/sonification-core", tag = "v0.1.0" }
```

```rust
use sonification_core::{
    DEFAULT_TOLERANCE_S, Instrument, ManifestInfo, Mapping, OnsetParams, ParamMap, PitchMap,
    RenderSettings, Series, Source, TimeMap, sha256_hex, verify_wav, write_manifest,
};

let beat = |name: &str, claim: &str, hz: f64, offset_s: f64| -> sonification_core::Result<Series> {
    Ok(Series {
        name: name.into(),
        mapping: Mapping {
            time: TimeMap::new(1.0, offset_s)?, // real time, starting offset_s into the audio
            pitch: PitchMap::Rhythm { speedup: 1.0, octaves: 4 },
            gain_db: ParamMap::Constant { value: -9.0 },
            pan: ParamMap::Constant { value: 0.0 },
            duration_s: ParamMap::Constant { value: 0.4 },
            instrument: Instrument::KarplusStrongTuned { decay: 0.996, pluck_brightness_hz: Some(2000.0) },
        },
        times: vec![0.0],
        values: vec![hz],
        claim: Some(claim.into()),
    })
};
let (viscous, diluted) = ("bio:HUMAN-SPERM-BEAT-VISCOUS", "bio:HUMAN-SPERM-BEAT-DILUTED-SEMEN");
let doi = Some("10.1093/humrep/dez056".to_string());
let info = ManifestInfo {
    title: "Two beat rates, four octaves up".into(),
    creator: "Chethan Dandebala Venkatarama".into(),
    licence: "CC-BY-4.0".into(),
    sources: vec![
        Source { claim: viscous.into(), doi: doi.clone(), values: vec![10.0] },
        Source { claim: diluted.into(), doi, values: vec![19.0] },
    ],
};
let series = [beat("viscous medium", viscous, 10.0, 0.1)?, beat("diluted semen", diluted, 19.0, 0.6)?];
let settings = RenderSettings { sample_rate: 44_100, seed: 0 };
let (manifest, wav) = write_manifest(&info, &series, &settings)?;
assert_eq!(manifest.layers[1].events[0].hz, 304.0); // 19 Hz · 2⁴
assert_eq!(manifest.audio.sha256, sha256_hex(&wav)); // 78efc56a…
std::fs::write("beats.json", manifest.to_json()?)?;
std::fs::write("beats.wav", &wav)?;

// The round trip: find the notes in the WAV file, match them to the events, recover the times.
let report = verify_wav(&manifest, &wav, DEFAULT_TOLERANCE_S, &OnsetParams::default())?;
assert!(report.pass && report.wav_reproduces == Some(true));
assert_eq!(report.matched[1].recovered_real_s, 0.0);
```

**Python** (builds the Rust core with maturin; mappings and manifests are plain dicts, signals are numpy arrays)

```bash
pip install "git+https://github.com/chethandvg/sonification-core@v0.1.0#subdirectory=bindings/python"
```

```python
import sonification_core as sc

beat = {
    "time": {"speedup": 1.0, "offset_s": 0.1},
    "pitch": {"kind": "rhythm", "octaves": 4},
    "gain_db": {"kind": "constant", "value": -9.0},
    "duration_s": {"kind": "constant", "value": 0.4},
    "instrument": {"kind": "karplus_strong_tuned"},  # decay 0.996 and a 2000 Hz pluck by default
}
viscous, diluted = "bio:HUMAN-SPERM-BEAT-VISCOUS", "bio:HUMAN-SPERM-BEAT-DILUTED-SEMEN"
info = {
    "title": "Two beat rates, four octaves up",
    "creator": "Chethan Dandebala Venkatarama",
    "licence": "CC-BY-4.0",
    "sources": [
        {"claim": viscous, "doi": "10.1093/humrep/dez056", "values": [10.0]},
        {"claim": diluted, "doi": "10.1093/humrep/dez056", "values": [19.0]},
    ],
}
later = beat | {"time": {"speedup": 1.0, "offset_s": 0.6}}
series = [
    {"name": "viscous medium", "mapping": beat, "times": [0.0], "values": [10.0], "claim": viscous},
    {"name": "diluted semen", "mapping": later, "times": [0.0], "values": [19.0], "claim": diluted},
]
manifest, wav = sc.write_manifest(info, series)  # 44.1 kHz, seed 0
manifest["layers"][1]["events"][0]["hz"]  # 304.0
manifest["audio"]["sha256"] == sc.sha256_hex(wav)  # True: 78efc56a…, the same in Rust and JavaScript

report = sc.verify_wav(manifest, wav)  # the file: its notes, its bytes and its timing
report["pass"], [m["recovered_real_s"] for m in report["matched"]]  # (True, [0.0, 0.0])

sc.map_pitch({"kind": "fold", "low_hz": 220.0}, 1 / 600)  # 436.9066666666667: a 10-minute rhythm, 18 octaves up
sc.heard_hz(600.0, 60000.0)  # 100.0: a 10-minute rhythm audified 60 000 times faster
sc.audify([0.0, 0.0, 0.0], 1.0, 2.0)  # ValueError: values must be a series that varies (...)
```

**JavaScript / WebAssembly** (build steps in [bindings/wasm](bindings/wasm))

```js
import init, { write_manifest, verify_wav } from "./pkg/web/sonification_core_wasm.js";

await init();
// info and series as in the Python example; typed arrays inside them are fine.
const [manifest, wav] = write_manifest(info, series, 44100, 0); // a plain object and a Uint8Array
const report = verify_wav(manifest, wav); // report.pass === true
```

Rendering, hashing and verifying ten seconds of 44.1 kHz notes takes well under a second in Node; the WebAssembly test checks that it finishes in under 10 s.

In Python, mappings, events, manifests and reports are dicts and lists in the JSON shape of the Rust types, numpy arrays inside them are accepted (inside them, a whole float such as `4.0` is fine where a whole number is expected; the `sample_rate` and `seed` arguments must be Python ints), WAV files are `bytes`, and the render settings default to `sample_rate=SR` (44 100) and `seed=0`; `write_manifest` returns `(manifest, wav)`. In JavaScript they are plain objects, WAV files are `Uint8Array`s, `write_manifest` returns `[manifest, wav]`, sample rates must be whole numbers up to 2³² − 1 and seeds whole numbers up to 2⁵³ − 1; trailing settings left `undefined` take the defaults. Seeds are at most 2⁵³ − 1 (`MAX_SEED`) in every language, so a manifest's seed reads back exactly in JavaScript. Errors are `SonifyError` in Rust, `ValueError` in Python (`OverflowError` for a negative or too-large integer argument, `TypeError` for a non-integer one) and `Error` in JavaScript.

### The mapping in JSON

A JSON mapping may leave out `time.offset_s` (0), `gain_db` and `pan` (constant 0), `clamp` (false), `scale` (none), a `rhythm`'s `speedup` (1) and `octaves` (0), and every instrument parameter (`lifesound.py`'s defaults: `brightness` 1, `ratio` 1.4, `index` 10, `decay` 0.996, `pluck_brightness_hz` 2000, `vibrato_hz` 5, `vibrato_cents` 12, `bpm` 60). Unknown fields are errors, and whole numbers (`octaves`, `root`, `pitch_classes`) may be written as whole floats, as JSON Schema's `integer` allows. The manifest always writes every field. `sonify` checks every constant of the mapping first (`Mapping::check`), so a constant that no value could use is an error even for a series with no points: a `fold`'s `low_hz` outside the range pitch-math's `octave_fold` accepts, an octave count that no frequency heard at the rhythm's speed-up survives, or a constant map whose one output is not a playable frequency, gain (one whose amplitude `10^(gain_db/20)` is finite), pan or length. An output that depends on the data is checked point by point: a clamped pan map from 5 to 6 passes the check, and every point then fails.

| Part | Kinds and fields |
|---|---|
| `time` | `speedup` (> 0), `offset_s` |
| `pitch` | `{"kind": "hz", "map"}`, `{"kind": "midi", "map", "scale"}`, `{"kind": "rhythm", "speedup", "octaves"}`, `{"kind": "fold", "speedup", "low_hz"}` |
| `gain_db`, `pan`, `duration_s`, a pitch's `map` | `{"kind": "constant", "value"}`, `{"kind": "linear" or "exponential", "x_lo", "x_hi", "out_lo", "out_hi", "clamp"}` |
| `scale` | `{"root": 0–11, "pitch_classes": [0–11, …]}`; E major pentatonic is root 4, pitch classes 0, 2, 4, 7, 9 |
| `instrument` | `modal_chime` (`brightness`), `fm_bell` (`ratio`, `index`), `karplus_strong` and `karplus_strong_tuned` (`decay`, `pluck_brightness_hz` or `null`), `soft_voice` (`vibrato_hz`, `vibrato_cents`), `heartbeat` (`bpm`; the note's frequency is the thumps' pitch) |

Scale quantisation goes to the nearest note of the scale, and a note exactly halfway between two scale notes goes up, as pitch-math's `nearest_note` rounds halves up. The E6 pentatonic notes of the earlier Genesis piece (MIDI 88, 90, 92, 95, 97, 100, 102, 104) are a `midi` pitch map clamped to 88..104 with the scale E major pentatonic. A note must start at audio time 0 or later and be panned within [−1, 1]. When rendering, event `i` (from 0) plucks its strings with noise seeded `seed + i + 1`, and the WAV file's dither uses `seed`. Nothing is normalised or limited, so loud overlapping notes can exceed ±1, and the 16-bit file clips them. The manifest records the render's peak (`audio.peak`): above 1, lower the gains (notes that sound together add, so keep their combined gain at or below 0 dB).

### Audification

`audify` is `lifesound.audify`: the series is linearly interpolated at the audio's sample times (as `numpy.interp`, which takes a data point's own value at its time, so a slope too steep for an `f64` does no harm there), its mean removed and its peak scaled just below 1. Two things changed, in all three languages:

- **Audio that would not vary is an error**, because it has nothing to audify. That covers a constant series (silence included) and a series whose interpolated samples are all equal, because it varies only between the audio's sample times or after the last of them; lower the speed-up or raise the sample rate. `lifesound.py` returned all zeros for these when their mean came out exact, and otherwise that mean's rounding error, scaled up.
- **The mean is removed from the samples less the first one**, `y − y[0]`, which in exact arithmetic gives the same result. The differences are exact when the samples are close, so a series that varies by a few units in the last place of a large value keeps that variation. `lifesound.py` took the mean of `y` itself: its rounding error is about one unit in the last place, and for a series near 7 × 10⁴ that steps by one unit in the last place, normalising turned that error into a large offset (on the golden vector, `lifesound.py`'s formula is 0.61 away from the step). On the other vectors the two formulas agree within 2.2e-16.

### The manifest

`write_manifest` writes a small JSON format, `sonification-manifest/0.1-draft`, described by [`schema/manifest.schema.json`](schema/manifest.schema.json) (JSON Schema 2020-12). It records the piece's title, creator and the audio's licence (an SPDX identifier); the software's name and version; the data sources (claim ids, DOIs and the values used); every layer's full mapping and every event (its real time, value, claim, audio time, frequency, gain, pan, length and instrument); the render settings (sample rate, seed at most 2⁵³ − 1, and the instruments used); and the WAV file's frames, the render's peak and the SHA-256 of the whole file, header included, so `sha256sum beats.wav` checks it. Every series' claim must be one of the sources. The title, creator and licence must not be blank or hold control characters, and claim ids and DOIs hold no blank characters (control characters or white space). The schema lists those characters by code point, so that no regular-expression dialect's own idea of white space enters, and ends each pattern in `(?![\s\S])`, the end of the text both in ECMA-262 and in Python's `re` (with `$`, Python's `re`, and so jsonschema, would also accept text that ends in a line feed). The tests check that whatever text the crate accepts, the schema accepts too, and that text ending in a line feed fails under jsonschema and JavaScript's `RegExp` alike. `Manifest::to_json` and `from_json` round-trip a manifest unchanged, and `reproduces_events` checks that a manifest describes its own events: they are the mappings applied to their own data, every layer's claim is one of the sources, the instrument list is the events' own, the audio is `wav-pcm16` in 2 channels, and the title, creator, licence and sources have the form `write_manifest` requires.

This is a draft for this library. The open standard is planned for the [`sonification-manifest`](https://github.com/chethandvg/sonification-manifest) repository in Sprint 4, with a CC-BY-4.0 specification and an Apache-2.0 validator.

### The round trip

`detect_onsets` is an energy-based onset detector, the general approach of looking for a sudden burst of energy reviewed by Bello et al. 2005 ([doi:10.1109/TSA.2005.851998](https://doi.org/10.1109/TSA.2005.851998)). Its design is this crate's own:

1. The power of each stereo frame is `l² + r²`, which equal-power panning leaves unchanged.
2. Every `hop_s` (1 ms), it compares the energy of the `window_s` (20 ms) after that time with the energy of the 20 ms before it, from running totals (so silence sums to exactly 0). Each energy is raised to a floor of `floor_db` (−90 dB) per frame if it is below it, so quieter sound counts as silence. The signal counts as silent before its start, so a note at 0 s is found. The floor is set by the 16-bit file's dither: a dithered sample is at most one step (1/32767) from 0, so a frame of the file's silence has a power of at most 2/32767² (−87.3 dB, and −93.3 dB on average), and a window of it at most 1.86 times the floor's energy. That is a rise of 2.7 dB, below the 6 dB threshold, so dithered silence is never an onset, and the dither before a note, 0.47 times the floor on average, seldom moves its onset. Audio with louder background noise needs a higher `floor_db`.
3. A time is an onset when that ratio rises by at least `threshold_db` (6 dB) and is the largest within `min_gap_s` (50 ms) on either side, the latest of equal ratios winning.
4. The onset is refined to the frame of largest ratio within one hop.

`verify` takes the events in order of audio time; each takes the nearest free onset within `tolerance_s` (default 2 ms) or is missing. Onsets left over are extra, except those within the tolerance of a heartbeat note's own later thumps (its dubs and later beats, where synthesis-theory places them), which are reported as `within_note`. It reports each event's audio error and recovered real time (the real-time error is the speed-up times the audio error).

What a pass means:

- `verify` passes when the manifest describes its own events (`events_reproduce`: they reproduce from their mappings, every layer's claim is one of the sources, the instrument list is the events' own, the audio is `wav-pcm16` in 2 channels, and the title, creator, licence and sources have the form `write_manifest` requires), the audio has the manifest's number of frames, no event is missing and no onset is extra. Against the audio, that checks **timing and length only**: a manifest with other pitches, gains or pans but the same timing also passes.
- `verify_wav` adds the file itself: it decodes the WAV file with signal-core's `read_wav`, runs `verify` on its samples, and passes only if the file's SHA-256 is the manifest's and rendering the manifest's events again gives the same bytes and the manifest's peak (`wav_reproduces`). So a pass ties every note's pitch, gain, pan, length and instrument to the file, and recovers its timing from the sound.
- Neither checks what the descriptive fields say (the title, creator, licence, software, and the sources' DOIs and values), only that all but the software have the form `write_manifest` requires. They say where the data came from, which the audio cannot show.

Energy onsets separate only what rises above what is already sounding. The property tests (8 to 48 kHz, on the float render and on the 16-bit WAV file) check these bounds, with the largest errors they measure in brackets: notes that start from silence at −18 to −6 dB are found within 0.5 ms for all six instruments [0.38 ms]; quiet notes after 0.3 s of dithered silence (8 and 44.1 kHz) within 1 ms for the struck, plucked and beating instruments at −50 dB [0.55 ms] and within 1.5 ms for the soft voice at −30 dB [1.3 ms]; and overlapping FM bells (1.2 s notes 0.19 to 0.29 s apart at −12 dB) within the default 2 ms tolerance, on a fixed series and in 40 seeded random pieces [1.33 ms]. A note that fades in is found where the energy before it rises above the floor, so later the quieter it is: at −40 dB some notes of the soft voice, with its 80 ms attack, are found more than 2 ms late, outside the default tolerance (all within 3 ms in the tests [2.75 ms]). Struck bars that overlap at equal gain are not reliably found either: the fixed series passes [1.55 ms], but in 3 of the 40 random pieces a bar is found 2.16 or 2.36 ms early, or not found, while earlier bars still ring. Nor are a soft voice over sustained notes, a quiet note while a loud one still rings, and notes closer than 50 ms to another onset, so leave room between notes and keep slow notes loud enough (or tune `threshold_db`, `min_gap_s` and `floor_db`) when a piece must pass the round trip.

## How it is verified

- **Proofs:** the Lean 4 files in [`proofs/`](proofs) prove the four P-level claims, reusing the Lean libraries of rhythm-units and pitch-math. In [`TimeMap.lean`](proofs/SonificationCore/TimeMap.lean): the time map `t ↦ offset + t/k` (rhythm-units' `scalePeriod` moved by the offset) is strictly increasing for `k > 0`, keeps every ratio of durations, and its inverse undoes it. In [`Pitch.lean`](proofs/SonificationCore/Pitch.lean): moving frequencies by `n` octaves keeps their ratio (pitch-math's own proof, reused) and their interval in cents. In [`Maps.lean`](proofs/SonificationCore/Maps.lean): the linear map hits its ends, rises when its outputs do, and stays in range when clamped; the exponential map hits its ends, rises, and gives the same ratio `g(x + d)/g(x)` for every `x`. CI rebuilds the proofs with Mathlib, rejects `sorry`, and allows only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`).
- **Referenced, not proved:** that a rhythm sped up by `k` is heard `k` times faster, so a period `T` at `k/T` Hz, is rhythm-units' proof (`rhythm:TIME-SCALING-FREQUENCY`), cited rather than proved again; the energy-onset approach is referenced to Bello et al. 2005.
- **Implementation:** one Rust implementation, compiled to Python (PyO3, numpy arrays) and WebAssembly. The same 138 golden vectors in [`vectors/`](vectors) (73 from the oracle, 65 invalid inputs) are checked in all three; CI runs the Rust and Python checks on Linux and Windows and the WebAssembly checks on Linux. Their oracle, [`vectors/generate.py`](vectors/generate.py), never calls the Rust code: it runs `lifesound.py`'s instruments, `pan`, `mix_into`, `write_wav` and `audify` (with signal-core's documented changes and audify's two), exact Fractions and 50-digit mpmath for the maps (and Fractions to check audify's mean removal), and its own Python versions of `sonify`, `render`, the manifest writer (each manifest validated against the schema with jsonschema), the onset detector, `verify` and `verify_wav` (the WAV file decoded with Python's `wave`). The vectors' SHA-256s are computed by that writer with numpy, Python's `wave` and `hashlib`, and the crate matches them exactly. Property tests in [`tests/roundtrip.rs`](tests/roundtrip.rs) render pieces with every instrument at four sample rates and check the round trip on the float render and the WAV file, for loud and quiet notes and for overlapping struck bars and bells (also in 40 seeded random pieces each); determinism; JSON round trips; that manifests shifted by 10 ms, with other pitches, or with a false peak, instrument list, encoding, channel count, claim, claim id or title fail, as do extra audio and unlisted notes; and that written manifests validate against the schema (with a small checker of the schema's own keywords that does not share the crate's character tests), also when their text holds each control character and each of Unicode's 25 White_Space characters. The Python tests validate manifests with jsonschema, which CI installs as a dependency of evidence-kernel (the tests are skipped if it is missing).
- **What the proofs don't cover:** they are about exact real numbers, and only the four maps above are proved. The pitch maps `hz`, `midi` and `fold`, scale quantisation and the code's floating-point rounding are checked by the golden vectors (the maps agree with exact arithmetic to 2.3e-16 relative), not proved. The instruments, the rendering, the manifest and the onset detector are tested, not proved, and the round trip is tested only for the kinds of render described above. Equal SHA-256s across platforms are checked in CI, not guaranteed: they hold because 16-bit quantisation absorbs the last-bit differences of `sin`, `exp` and `pow`. And nothing here says anything about biology or health: the data's meaning comes from its sources, and the sound is a translation of it.

## Planned

- **MIDI file export** of any mapping or piece (developer kit item 3, Sprint 5), so the same events can drive a synthesiser or a DAW; together with a CLAP plugin in the new `rhythm-sequencer-clap` repository.
- **The open manifest standard** in [`sonification-manifest`](https://github.com/chethandvg/sonification-manifest) (Sprint 4): a CC-BY-4.0 specification and an Apache-2.0 validator that this crate's manifests will be checked against.

## Cite

If you use this work, please cite it. GitHub's **Cite this repository** button gives APA and BibTeX from [CITATION.cff](CITATION.cff).

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE)). Copyright notices are in [NOTICE](NOTICE). Audio you render with it is yours to license; the examples here use CC-BY-4.0.
