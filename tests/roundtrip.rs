// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! Property tests: the round trip from data to audio and back (on the float
//! render and on the 16-bit WAV file), determinism, and manifests that survive
//! JSON and validate against the schema.

use serde_json::Value;
use sonification_core::{
    DEFAULT_TOLERANCE_S, Instrument, MAX_SEED, Manifest, ManifestInfo, Mapping, OnsetParams,
    ParamMap, PitchMap, RenderSettings, Scale, Series, Source, TimeMap, Verification, render,
    render_wav, sha256_hex, verify, verify_wav, write_manifest,
};

/// One of each instrument, with `lifesound.py`'s parameters (the heartbeat at 150 bpm,
/// so that a 0.4 s note holds its dub, 0.16 s after the lub).
fn instruments() -> [Instrument; 6] {
    [
        Instrument::ModalChime { brightness: 1.0 },
        Instrument::FmBell {
            ratio: 1.4,
            index: 10.0,
        },
        Instrument::KarplusStrong {
            decay: 0.996,
            pluck_brightness_hz: Some(2000.0),
        },
        Instrument::KarplusStrongTuned {
            decay: 0.996,
            pluck_brightness_hz: None,
        },
        Instrument::SoftVoice {
            vibrato_hz: 5.0,
            vibrato_cents: 12.0,
        },
        Instrument::Heartbeat { bpm: 150.0 },
    ]
}

/// A mapping at speed-up `k`: values 0..1 to MIDI 57..81 quantised to A minor
/// pentatonic, gains from -18 to -6 dB, a gentle pan, and notes of `dur` seconds.
fn mapping(k: f64, offset: f64, dur: f64, instrument: Instrument) -> Mapping {
    let unit = |lo, hi| ParamMap::Linear {
        x_lo: 0.0,
        x_hi: 1.0,
        out_lo: lo,
        out_hi: hi,
        clamp: true,
    };
    Mapping {
        time: TimeMap::new(k, offset).unwrap(),
        pitch: PitchMap::Midi {
            map: unit(57.0, 81.0),
            scale: Some(Scale::new(9, &[0, 3, 5, 7, 10]).unwrap()),
        },
        gain_db: unit(-18.0, -6.0),
        pan: unit(-0.8, 0.8),
        duration_s: ParamMap::Constant { value: dur },
        instrument,
    }
}

/// A synthetic series: real times `step` apart with a jitter that lands notes off the
/// detector's hop grid, and values spread over [0, 1].
fn series(n: usize, step: f64) -> (Vec<f64>, Vec<f64>) {
    let times = (0..n)
        .map(|i| step * i as f64 + 0.37 * step * ((i * 7) % 5) as f64 / 5.0)
        .collect();
    let values = (0..n).map(|i| ((i * 37) % 11) as f64 / 10.0).collect();
    (times, values)
}

fn info() -> ManifestInfo {
    ManifestInfo {
        title: "Round-trip test".into(),
        creator: "Chethan Dandebala Venkatarama".into(),
        licence: "CC-BY-4.0".into(),
        sources: vec![Source {
            claim: "test:SYNTHETIC".into(),
            doi: None,
            values: vec![],
        }],
    }
}

fn layer(name: &str, map: Mapping, times: Vec<f64>, values: Vec<f64>) -> Series {
    Series {
        name: name.into(),
        mapping: map,
        times,
        values,
        claim: Some("test:SYNTHETIC".into()),
    }
}

/// A piece of the given layers at `sr` (seed 7): its manifest, float render and WAV file.
fn layers(series: &[Series], sr: u32) -> (Manifest, Vec<f64>, Vec<u8>) {
    let settings = RenderSettings {
        sample_rate: sr,
        seed: 7,
    };
    let (manifest, wav) = write_manifest(&info(), series, &settings).unwrap();
    let events: Vec<_> = manifest.events().cloned().collect();
    (manifest, render(&events, &settings).unwrap(), wav)
}

fn piece(
    map: Mapping,
    times: Vec<f64>,
    values: Vec<f64>,
    sr: u32,
) -> (Manifest, Vec<f64>, Vec<u8>) {
    layers(&[layer("notes", map, times, values)], sr)
}

fn report(manifest: &Manifest, stereo: &[f64]) -> Verification {
    verify(
        manifest,
        stereo,
        DEFAULT_TOLERANCE_S,
        &OnsetParams::default(),
    )
    .unwrap()
}

/// Checks the round trip on the float render and on the WAV file; returns the
/// WAV file's report. Every event is matched, and the real times come back
/// within k times the audio error, up to rounding.
fn check(manifest: &Manifest, stereo: &[f64], wav: &[u8], k: f64, what: &str) -> Verification {
    let float = report(manifest, stereo);
    let file = verify_wav(manifest, wav, DEFAULT_TOLERANCE_S, &OnsetParams::default()).unwrap();
    for (v, kind) in [(&float, "float"), (&file, "wav")] {
        assert!(
            v.pass,
            "{what} ({kind}): {:?} missing, {:?} extra",
            v.missing, v.extra
        );
        assert!(v.events_reproduce && v.frames_match, "{what} ({kind})");
        assert_eq!(v.matched.len(), manifest.events().count(), "{what}");
        for m in &v.matched {
            assert!(
                (m.real_error_s - k * m.audio_error_s).abs() <= 1e-9 * m.real_s.abs().max(1.0),
                "{what}"
            );
            assert!(
                m.real_error_s.abs() <= k * DEFAULT_TOLERANCE_S * (1.0 + 1e-9),
                "{what}"
            );
        }
    }
    assert_eq!(
        (float.wav_reproduces, file.wav_reproduces),
        (None, Some(true))
    );
    file
}

/// Notes that start from silence are found within a few samples at every rate
/// tested, for every instrument, on the float render and the WAV file, and the
/// real times within k times that.
#[test]
fn round_trip_recovers_every_onset_of_separated_notes() {
    let mut worst: f64 = 0.0;
    for sr in [44_100, 48_000, 22_050, 8000] {
        for instrument in instruments() {
            let k = 600.0;
            let (times, values) = series(12, 360.0); // notes 0.47 to 0.69 s apart, each 0.4 s long
            for offset in [0.05, 0.3] {
                // 0.3 s of dithered silence before the first note in the WAV file
                let (manifest, stereo, wav) = piece(
                    mapping(k, offset, 0.4, instrument),
                    times.clone(),
                    values.clone(),
                    sr,
                );
                let what = format!("{instrument:?} at {sr} Hz, offset {offset}");
                let v = check(&manifest, &stereo, &wav, k, &what);
                worst = worst.max(v.max_audio_error_s);
                worst = worst.max(report(&manifest, &stereo).max_audio_error_s);
                if matches!(instrument, Instrument::Heartbeat { .. }) {
                    assert_eq!(v.within_note.len(), 12, "each lub-dub has one dub");
                }
            }
        }
    }
    eprintln!("separated notes: largest audio error {:.4} ms", 1e3 * worst);
    assert!(worst <= SEPARATED_S, "{worst}");
}

/// The largest audio error allowed for notes that start from silence: 0.5 ms
/// (the largest measured is in the README).
const SEPARATED_S: f64 = 0.0005;

/// Quiet notes that start from silence are found too, on the float render and
/// on the WAV file after 0.3 s of dithered silence: the struck, plucked and
/// beating instruments at -50 dB, and the soft voice at -30 dB (its 80 ms
/// attack is found later the quieter it is; the README gives the limits).
#[test]
fn round_trip_recovers_quiet_notes() {
    let (mut worst, mut worst_voice): (f64, f64) = (0.0, 0.0);
    for sr in [44_100, 8000] {
        for instrument in instruments() {
            let voice = matches!(instrument, Instrument::SoftVoice { .. });
            let gain = if voice { -30.0 } else { -50.0 };
            let (times, values) = series(12, 360.0);
            let mut map = mapping(600.0, 0.3, 0.4, instrument);
            map.gain_db = ParamMap::Constant { value: gain };
            let (manifest, stereo, wav) = piece(map, times, values, sr);
            let what = format!("{instrument:?} at {gain} dB, {sr} Hz");
            let v = check(&manifest, &stereo, &wav, 600.0, &what);
            let error = v
                .max_audio_error_s
                .max(report(&manifest, &stereo).max_audio_error_s);
            let w = if voice { &mut worst_voice } else { &mut worst };
            *w = w.max(error);
        }
    }
    eprintln!(
        "quiet notes: largest audio error {:.4} ms (-50 dB), {:.4} ms (soft voice, -30 dB)",
        1e3 * worst,
        1e3 * worst_voice
    );
    assert!(worst <= QUIET_S && worst_voice <= QUIET_VOICE_S);
}

/// The largest audio errors allowed for quiet notes: 1 ms, and 1.5 ms for the
/// soft voice (the largest measured are in the README).
const QUIET_S: f64 = 0.001;
const QUIET_VOICE_S: f64 = 0.0015;

/// The limit the README gives: a soft voice at -40 dB fades in so quietly that
/// some of its notes are found later than the default 2 ms tolerance (all of
/// them within 3 ms).
#[test]
fn quiet_soft_voices_are_found_late() {
    let mut worst: f64 = 0.0;
    for sr in [44_100, 8000] {
        let (times, values) = series(12, 360.0);
        let mut map = mapping(600.0, 0.3, 0.4, instruments()[4]);
        map.gain_db = ParamMap::Constant { value: -40.0 };
        let (manifest, stereo, wav) = piece(map, times, values, sr);
        let params = OnsetParams::default();
        for v in [
            verify(&manifest, &stereo, 0.01, &params).unwrap(),
            verify_wav(&manifest, &wav, 0.01, &params).unwrap(),
        ] {
            assert!(v.pass, "every note is found within 10 ms");
            worst = worst.max(v.max_audio_error_s);
        }
    }
    eprintln!(
        "soft voice at -40 dB: largest audio error {:.4} ms",
        1e3 * worst
    );
    assert!(worst > DEFAULT_TOLERANCE_S && worst < 0.003, "{worst}");
}

/// On this test's series, struck and bell notes that start while earlier ones
/// still ring (1.2 s notes, 0.19 to 0.29 s apart, all at -12 dB) are found
/// within the default 2 ms tolerance. That is not a bound for struck bars:
/// on other series of this kind a bar is now and then found more than 2 ms
/// early (see the next test). Sustained notes (soft_voice, the slowly decaying
/// strings) and heartbeat dubs within `min_gap_s` of another note are not
/// reliably separated this way; the README says so.
#[test]
fn round_trip_recovers_onsets_of_overlapping_struck_notes() {
    for instrument in &instruments()[..2] {
        let k = 1000.0;
        let (times, values) = series(16, 250.0);
        let mut map = mapping(k, 0.0, 1.2, *instrument);
        map.gain_db = ParamMap::Constant { value: -12.0 };
        let (manifest, stereo, wav) = piece(map, times, values, 44_100);
        let what = format!("overlapping {instrument:?}");
        let v = check(&manifest, &stereo, &wav, k, &what);
        let float = report(&manifest, &stereo);
        eprintln!(
            "{what}: largest audio error {:.4} ms (wav), {:.4} ms (float)",
            1e3 * v.max_audio_error_s,
            1e3 * float.max_audio_error_s
        );
    }
}

/// Random pieces of the kind above (seeded, signal-core's SplitMix64): 16 notes
/// of 1.2 s at -12 dB, 0.19 to 0.29 s apart, with values uniform in [0, 1].
/// Every FM bell note is found within the default tolerance, on the float render
/// and the WAV file. A struck bar that starts while earlier bars still ring is
/// now and then found more than 2 ms early, or not found at all, so for the
/// chime this counts the pieces that fail and prints what happened to each
/// missing note (its error within 10 ms, or not found).
#[test]
fn random_overlapping_pieces() {
    const PIECES: usize = 40;
    let k = 1000.0;
    let params = OnsetParams::default();
    let mut rng = signal_core::SplitMix64::new(2026);
    for instrument in &instruments()[..2] {
        let (mut worst, mut failing) = (0.0_f64, 0);
        for _ in 0..PIECES {
            let mut at = 0.0;
            let times: Vec<f64> = (0..16)
                .map(|_| {
                    let t = at;
                    at += k * rng.uniform(0.19, 0.29);
                    t
                })
                .collect();
            let values = (0..16).map(|_| rng.next_f64()).collect();
            let mut map = mapping(k, 0.0, 1.2, *instrument);
            map.gain_db = ParamMap::Constant { value: -12.0 };
            let (manifest, stereo, wav) = piece(map, times, values, 44_100);
            let float = report(&manifest, &stereo);
            let file = verify_wav(&manifest, &wav, DEFAULT_TOLERANCE_S, &params).unwrap();
            if float.pass && file.pass {
                worst = worst.max(float.max_audio_error_s.max(file.max_audio_error_s));
                continue;
            }
            failing += 1;
            let loose = verify(&manifest, &stereo, 0.01, &params).unwrap();
            let only_wav = file.missing.iter().filter(|r| !float.missing.contains(r));
            for r in float.missing.iter().chain(only_wav) {
                let found = loose.matched.iter().find(|m| m.event == r.event);
                let what = found.map_or("not found within 10 ms".into(), |m| {
                    format!("found {:+.4} ms off", 1e3 * m.audio_error_s)
                });
                eprintln!(
                    "random overlapping {instrument:?}: note {} at {:.4} s {what}",
                    r.event, r.audio_s
                );
            }
        }
        eprintln!(
            "random overlapping {instrument:?}: {failing} of {PIECES} pieces fail at 2 ms; \
             largest audio error in the others {:.4} ms",
            1e3 * worst
        );
        if matches!(instrument, Instrument::FmBell { .. }) {
            assert_eq!(failing, 0, "every bell note within 2 ms");
        } else {
            assert!(failing <= PIECES / 4, "{failing} chime pieces fail");
        }
    }
}

/// A manifest whose times are shifted, though self-consistent, fails against the audio.
#[test]
fn shifted_manifests_fail() {
    let (times, values) = series(10, 300.0);
    let chime = instruments()[0];
    let (manifest, stereo, wav) = piece(
        mapping(600.0, 0.05, 0.4, chime),
        times.clone(),
        values.clone(),
        44_100,
    );
    assert!(check(&manifest, &stereo, &wav, 600.0, "unshifted").pass);
    // The same data heard 10 ms later: the events reproduce, but no onset is where they say.
    let (shifted, _, _) = piece(mapping(600.0, 0.06, 0.4, chime), times, values, 44_100);
    let v = report(&shifted, &stereo);
    assert!(v.events_reproduce && !v.frames_match && !v.pass);
    assert_eq!((v.missing.len(), v.extra.len()), (10, 10));
    // Shifting the stored events alone breaks reproduction as well.
    let mut edited = manifest.clone();
    edited.layers[0].events[3].audio_s += 0.01;
    let v = report(&edited, &stereo);
    assert!(!v.events_reproduce && !v.pass && v.missing.len() == 1);
}

/// A pass ties the file to the manifest: other pitches, extra audio and an
/// unlisted note inside a heartbeat note all fail.
#[test]
fn verification_catches_what_timing_alone_misses() {
    let (times, values) = series(8, 360.0);
    let chime = instruments()[0];
    let (manifest, stereo, wav) = piece(
        mapping(600.0, 0.05, 0.4, chime),
        times.clone(),
        values.clone(),
        22_050,
    );
    // The same timing with every note two octaves lower: the onsets match, the file does not.
    let mut low = mapping(600.0, 0.05, 0.4, chime);
    low.pitch = PitchMap::Midi {
        map: ParamMap::Linear {
            x_lo: 0.0,
            x_hi: 1.0,
            out_lo: 33.0,
            out_hi: 57.0,
            clamp: true,
        },
        scale: None,
    };
    let (mut lie, _, _) = piece(low, times, values, 22_050);
    assert!(report(&lie, &stereo).pass, "timing alone cannot see pitch");
    let v = verify_wav(&lie, &wav, DEFAULT_TOLERANCE_S, &OnsetParams::default()).unwrap();
    assert_eq!((v.wav_reproduces, v.pass), (Some(false), false));
    lie.audio.sha256 = manifest.audio.sha256.clone(); // even with the file's own hash copied in
    let v = verify_wav(&lie, &wav, DEFAULT_TOLERANCE_S, &OnsetParams::default()).unwrap();
    assert_eq!((v.wav_reproduces, v.pass), (Some(false), false));
    // Extra silence after the piece: the onsets match, the length does not.
    let mut longer = stereo.clone();
    longer.resize(stereo.len() + 2 * 22_050, 0.0);
    let v = report(&manifest, &longer);
    assert!(!v.frames_match && !v.pass && v.missing.is_empty() && v.extra.is_empty());
    // A note that the manifest does not list, inside a slow heartbeat's note,
    // away from its beats (0, 0.3, 2, 2.3, 4 and 4.3 s at 30 bpm), is extra.
    let mut heart = mapping(1.0, 0.0, 6.0, Instrument::Heartbeat { bpm: 30.0 });
    heart.pitch = PitchMap::Hz {
        map: ParamMap::Constant { value: 50.0 },
    };
    let beat = || layer("heart", heart.clone(), vec![0.0], vec![0.5]);
    let notes = |t: Vec<f64>| {
        let v = vec![0.5; t.len()];
        layer("notes", mapping(1.0, 0.0, 0.4, chime), t, v)
    };
    let (honest, _, _) = layers(&[beat(), notes(vec![1.0, 5.0])], 22_050);
    let (_, stereo_extra, _) = layers(&[beat(), notes(vec![1.0, 3.3, 5.0])], 22_050);
    let v = report(&honest, &stereo_extra);
    assert!(v.frames_match && !v.pass, "{:?}", v.extra);
    assert_eq!(v.extra.len(), 1);
    assert!((v.extra[0] - 3.3).abs() < 0.002);
    assert_eq!(
        v.within_note.len(),
        5,
        "the heartbeat's dubs and later beats"
    );
    // A file that is not stereo at the manifest's rate is an error, not a fail.
    let mono = signal_core::write_wav(&stereo, 1, 22_050, 0).unwrap();
    let e = verify_wav(
        &manifest,
        &mono,
        DEFAULT_TOLERANCE_S,
        &OnsetParams::default(),
    )
    .unwrap_err();
    assert_eq!(e.name(), "channels");
    let slow = signal_core::write_wav(&stereo, 2, 11_025, 0).unwrap();
    let e = verify_wav(
        &manifest,
        &slow,
        DEFAULT_TOLERANCE_S,
        &OnsetParams::default(),
    )
    .unwrap_err();
    assert_eq!(e.name(), "sample_rate");
}

/// A pass also ties the manifest's description of its events to them: a false
/// peak (which only the file shows), instrument list, encoding, channel count,
/// a claim that is not among the sources, a claim id that write_manifest would
/// refuse, and a title it would refuse all fail.
#[test]
fn verification_catches_false_metadata() {
    let (times, values) = series(4, 360.0);
    let map = mapping(600.0, 0.05, 0.4, instruments()[0]);
    let (manifest, stereo, wav) = piece(map, times, values, 8000);
    let passes = |m: &Manifest| {
        let file = verify_wav(m, &wav, DEFAULT_TOLERANCE_S, &OnsetParams::default()).unwrap();
        (report(m, &stereo).pass, file.pass)
    };
    let lie = |change: fn(&mut Manifest)| {
        let mut m = manifest.clone();
        change(&mut m);
        m
    };
    assert_eq!(passes(&manifest), (true, true));
    assert_eq!(passes(&lie(|m| m.audio.peak = 123.0)), (true, false));
    for false_metadata in [
        lie(|m| m.render.instruments = vec![Instrument::Heartbeat { bpm: 60.0 }]),
        lie(|m| m.render.instruments.clear()),
        lie(|m| m.audio.channels = 1),
        lie(|m| m.audio.encoding = "wav-pcm24".into()),
        lie(|m| {
            let claim = Some("x:NOT-A-SOURCE".to_string());
            m.layers[0].claim.clone_from(&claim);
            for e in &mut m.layers[0].events {
                e.claim.clone_from(&claim);
            }
        }),
        lie(|m| {
            let claim = "test: SYNTHETIC".to_string(); // a source too, but with a space
            m.sources[0].claim.clone_from(&claim);
            m.layers[0].claim = Some(claim.clone());
            for e in &mut m.layers[0].events {
                e.claim = Some(claim.clone());
            }
        }),
        lie(|m| m.title.push('\n')),
    ] {
        assert!(!false_metadata.reproduces_events());
        assert_eq!(passes(&false_metadata), (false, false));
    }
}

/// The same series, mapping and seed give the same WAV bytes and SHA-256; another seed does not.
#[test]
fn rendering_is_deterministic() {
    for instrument in instruments() {
        let (times, values) = series(6, 300.0);
        let map = mapping(600.0, 0.0, 0.3, instrument);
        let series = [Series {
            name: "n".into(),
            mapping: map,
            times,
            values,
            claim: None,
        }];
        let run = |seed| {
            write_manifest(
                &info(),
                &series,
                &RenderSettings {
                    sample_rate: 22_050,
                    seed,
                },
            )
            .unwrap()
        };
        let ((a, wav_a), (b, wav_b), (c, _)) = (run(3), run(3), run(4));
        assert_eq!(wav_a, wav_b);
        assert_eq!(a, b);
        assert_eq!(a.audio.sha256, sha256_hex(&wav_a));
        assert_ne!(
            a.audio.sha256, c.audio.sha256,
            "the dither depends on the seed"
        );
        let events: Vec<_> = a.events().cloned().collect();
        assert_eq!(render_wav(&events, &a.settings()).unwrap(), wav_a);
        assert_eq!(a.audio.frames as usize, (wav_a.len() - 44) / 4);
        let peak = render(&events, &a.settings())
            .unwrap()
            .iter()
            .fold(0.0, |m: f64, x| m.max(x.abs()));
        assert!(a.audio.peak == peak && peak > 0.0 && peak < 1.0);
    }
}

/// Manifests survive JSON unchanged, both ways.
#[test]
fn manifests_round_trip_through_json() {
    let (times, values) = series(5, 300.0);
    let (manifest, _, _) = piece(
        mapping(600.0, 0.0, 0.3, instruments()[2]),
        times,
        values,
        8000,
    );
    let text = manifest.to_json().unwrap();
    let back = Manifest::from_json(&text).unwrap();
    assert_eq!(back, manifest);
    assert_eq!(back.to_json().unwrap(), text);
    assert!(back.reproduces_events());
    assert_eq!(manifest.render.instruments, [instruments()[2]]);
    let wrong = text.replace("sonification-manifest/0.1-draft", "sonification-manifest/9");
    assert_eq!(Manifest::from_json(&wrong).unwrap_err().name(), "schema");
    assert_eq!(Manifest::from_json("{}").unwrap_err().name(), "json");
}

/// The code points the schema's patterns exclude, written out from the schema:
/// control characters (Unicode `Cc`), and blank ones (`Cc` and `White_Space`).
/// The crate decides with `char::is_control` and `char::is_whitespace`; this
/// checker does not, so a difference between the two would fail the tests.
const CONTROL: [(u32, u32); 2] = [(0x00, 0x1f), (0x7f, 0x9f)];
const BLANK: [(u32, u32); 8] = [
    (0x00, 0x20),
    (0x7f, 0xa0),
    (0x1680, 0x1680),
    (0x2000, 0x200a),
    (0x2028, 0x2029),
    (0x202f, 0x202f),
    (0x205f, 0x205f),
    (0x3000, 0x3000),
];

fn among(set: &[(u32, u32)], c: char) -> bool {
    set.iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&u32::from(c)))
}

/// The body of a regular-expression character class of `set`, as the schema writes it.
fn class(set: &[(u32, u32)]) -> String {
    set.iter()
        .map(|&(lo, hi)| {
            if lo == hi {
                format!(r"\u{lo:04x}")
            } else {
                format!(r"\u{lo:04x}-\u{hi:04x}")
            }
        })
        .collect()
}

/// Whether the JSON pattern `p` matches `s`, for the four patterns the schema
/// uses (each ends in `(?![\s\S])`, the end of the whole text); any other
/// pattern stops the test, so that this checker grows with the schema.
fn pattern(p: &str, s: &str) -> bool {
    let (control, blank) = (class(&CONTROL), class(&BLANK));
    let unbroken = !s.chars().any(|c| among(&BLANK, c));
    let end = r"(?![\s\S])";
    if p == format!("^[^{control}]*[^{blank}][^{control}]*{end}") {
        !s.chars().any(|c| among(&CONTROL, c)) && s.chars().any(|c| !among(&BLANK, c))
    } else if p == format!("^[^{blank}]*:[^{blank}]*{end}") {
        unbroken && s.contains(':')
    } else if p == format!(r"^10\.[^{blank}]*/[^{blank}]*{end}") {
        unbroken && s.starts_with("10.") && s[3..].contains('/')
    } else if p == format!("^[0-9a-f]{{64}}{end}") {
        s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    } else {
        panic!("the schema uses a pattern this checker does not read: {p}")
    }
}

/// Every way `x` breaks `schema` (empty when it is valid), for the JSON Schema
/// 2020-12 keywords that schema/manifest.schema.json uses; any other keyword
/// stops the test. The Python tests and the oracle validate with jsonschema;
/// this lets the Rust tests check the manifests the crate writes on their own.
fn violations(root: &Value, schema: &Value, x: &Value, at: &str) -> Vec<String> {
    const KNOWN: [&str; 19] = [
        "$schema",
        "$id",
        "title",
        "description",
        "$defs",
        "$ref",
        "oneOf",
        "const",
        "enum",
        "type",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "pattern",
        "minItems",
        "items",
        "required",
        "properties",
        "additionalProperties",
    ];
    let s = schema.as_object().expect("a schema is an object");
    for key in s.keys() {
        assert!(
            KNOWN.contains(&key.as_str()),
            "unknown schema keyword {key}"
        );
    }
    let mut bad = Vec::new();
    let mut fail = |what: String| bad.push(format!("{at}: {what}"));
    if let Some(r) = s.get("$ref").and_then(Value::as_str) {
        let name = r.strip_prefix("#/$defs/").expect("a $ref into $defs");
        return violations(root, &root["$defs"][name], x, at);
    }
    if let Some(options) = s.get("oneOf").and_then(Value::as_array) {
        let n = options
            .iter()
            .filter(|o| violations(root, o, x, at).is_empty())
            .count();
        if n != 1 {
            fail(format!("matches {n} of the oneOf options"));
        }
    }
    if s.get("const").is_some_and(|c| c != x) {
        fail(format!("is not {}", s["const"]));
    }
    if s.get("enum")
        .and_then(Value::as_array)
        .is_some_and(|e| !e.contains(x))
    {
        fail("is not one of the enum".into());
    }
    if let Some(t) = s.get("type").and_then(Value::as_str) {
        let ok = match t {
            "object" => x.is_object(),
            "array" => x.is_array(),
            "string" => x.is_string(),
            "boolean" => x.is_boolean(),
            "null" => x.is_null(),
            "number" => x.is_number(),
            "integer" => x.as_f64().is_some_and(|v| v.fract() == 0.0),
            _ => panic!("unknown type {t}"),
        };
        if !ok {
            fail(format!("is not of type {t}"));
        }
    }
    if let Some(v) = x.as_f64() {
        let bound = |k: &str| s.get(k).and_then(Value::as_f64);
        if bound("minimum").is_some_and(|m| v < m)
            || bound("maximum").is_some_and(|m| v > m)
            || bound("exclusiveMinimum").is_some_and(|m| v <= m)
        {
            fail(format!("{v} is out of range"));
        }
    }
    if let (Some(text), Some(p)) = (x.as_str(), s.get("pattern").and_then(Value::as_str)) {
        if !pattern(p, text) {
            fail(format!("{text:?} does not match {p}"));
        }
    }
    if let Some(items) = x.as_array() {
        if s.get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|n| (items.len() as u64) < n)
        {
            fail("has too few items".into());
        }
        if let Some(item) = s.get("items") {
            for (i, v) in items.iter().enumerate() {
                bad.extend(violations(root, item, v, &format!("{at}[{i}]")));
            }
        }
    }
    if let Some(object) = x.as_object() {
        for key in s
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if !object.contains_key(key.as_str().unwrap()) {
                bad.push(format!("{at}: lacks {key}"));
            }
        }
        let props = s.get("properties").and_then(Value::as_object);
        for (key, v) in object {
            match props.and_then(|p| p.get(key)) {
                Some(sub) => bad.extend(violations(root, sub, v, &format!("{at}.{key}"))),
                None if s.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    bad.push(format!("{at}: has the unknown field {key}"));
                }
                None => {}
            }
        }
    }
    bad
}

/// The manifests the crate writes, for every instrument and with every kind of
/// map, validate against schema/manifest.schema.json; broken ones do not.
#[test]
fn written_manifests_validate_against_the_schema() {
    let text = include_str!("../schema/manifest.schema.json");
    let schema: Value = serde_json::from_str(text).unwrap();
    let valid = |m: &Value| violations(&schema, &schema, m, "manifest");
    let mut pitch_maps = vec![
        PitchMap::Rhythm {
            speedup: 2.0,
            octaves: -1,
        },
        PitchMap::Fold {
            speedup: 600.0,
            low_hz: 220.0,
        },
        PitchMap::Hz {
            map: ParamMap::Exponential {
                x_lo: 0.0,
                x_hi: 1.0,
                out_lo: 200.0,
                out_hi: 800.0,
                clamp: false,
            },
        },
    ];
    let mut series = Vec::new();
    for (i, instrument) in instruments().into_iter().enumerate() {
        let mut map = mapping(600.0, 0.05 + 0.5 * i as f64, 0.3, instrument);
        if let Some(pitch) = pitch_maps.pop() {
            map.pitch = pitch;
        }
        series.push(layer(
            &format!("layer {i}"),
            map,
            vec![0.0, 60.0],
            vec![0.25, 1.0],
        ));
    }
    let (manifest, _, _) = layers(&series, 8000);
    let mut json = serde_json::to_value(&manifest).unwrap();
    assert_eq!(valid(&json), Vec::<String>::new());
    let seed = MAX_SEED;
    json["render"]["seed"] = seed.into();
    assert!(valid(&json).is_empty());
    json["render"]["seed"] = (seed + 1).into();
    json["layers"][0]["mapping"]["time"]["speedup"] = (-5.0).into();
    json["audio"].as_object_mut().unwrap().remove("peak");
    assert_eq!(valid(&json).len(), 3, "{:?}", valid(&json));
    json["title"] = "Round-trip test\n".into(); // the pattern ends at the end of the text
    assert_eq!(valid(&json).len(), 4, "{:?}", valid(&json));
}

/// Whatever text write_manifest accepts, the schema accepts too: a title, a
/// claim and a DOI holding each control character, each of Unicode's 25
/// White_Space characters and a few other unusual ones, in turn.
#[test]
fn accepted_text_validates_against_the_schema() {
    let schema: Value =
        serde_json::from_str(include_str!("../schema/manifest.schema.json")).unwrap();
    let unusual = [
        0x1680, 0x180e, 0x200b, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff, 0x1f600,
    ];
    let tried: Vec<u32> = (0..=0xa0).chain(0x2000..=0x200a).chain(unusual).collect();
    // char::is_whitespace is Unicode's White_Space: every such character is tried.
    let white: Vec<char> = (0..=0x10ffff)
        .filter_map(char::from_u32)
        .filter(|c| c.is_whitespace())
        .collect();
    assert_eq!(white.len(), 25);
    assert!(white.iter().all(|&c| tried.contains(&u32::from(c))));
    let chars = tried.iter().filter_map(|&u| char::from_u32(u));
    let map = mapping(1.0, 0.0, 0.01, instruments()[0]);
    let settings = RenderSettings {
        sample_rate: 1000,
        seed: 0,
    };
    let (mut accepted, mut rejected) = (0, 0);
    for c in chars {
        let (x, claim, doi) = (format!("x{c}"), format!("a:{c}B"), format!("10.1/{c}x"));
        for (title, claim, doi) in [
            (c.to_string(), "a:B".to_string(), "10.1/x".to_string()),
            (x, "a:B".into(), "10.1/x".into()),
            ("t".into(), claim, "10.1/x".into()),
            ("t".into(), "a:B".into(), doi),
        ] {
            let info = ManifestInfo {
                title,
                creator: "c".into(),
                licence: "CC0-1.0".into(),
                sources: vec![Source {
                    claim: claim.clone(),
                    doi: Some(doi),
                    values: vec![],
                }],
            };
            let series = [Series {
                name: "n".into(),
                mapping: map.clone(),
                times: vec![0.0],
                values: vec![0.5],
                claim: Some(claim),
            }];
            let Ok((manifest, _)) = write_manifest(&info, &series, &settings) else {
                rejected += 1;
                continue;
            };
            accepted += 1;
            let json = serde_json::to_value(&manifest).unwrap();
            let bad = violations(&schema, &schema, &json, "manifest");
            assert!(bad.is_empty(), "U+{:04X}: {bad:?}", u32::from(c));
        }
    }
    eprintln!("unusual text: {accepted} manifests written and valid, {rejected} rejected");
    assert!(accepted > 0 && rejected > 0);
}

/// Values JSON cannot carry (NaN, infinity) and the manifest's own rules.
#[test]
fn invalid_input_is_rejected() {
    let map = mapping(600.0, 0.0, 0.3, instruments()[0]);
    let one = |claim: Option<&str>, times: Vec<f64>| Series {
        name: "n".into(),
        mapping: map.clone(),
        times,
        values: vec![0.5],
        claim: claim.map(Into::into),
    };
    let settings = RenderSettings::default();
    let err = |series: &[Series], info: &ManifestInfo| {
        write_manifest(info, series, &settings).unwrap_err().name()
    };
    assert_eq!(
        err(&[one(Some("other:CLAIM"), vec![0.0])], &info()),
        "claim"
    );
    assert_eq!(err(&[one(None, vec![f64::NAN])], &info()), "real_s");
    assert_eq!(err(&[], &info()), "series");
    // A layer with no points still has its mapping checked.
    let mut empty = one(None, vec![0.0]);
    (empty.times, empty.values) = (vec![], vec![]);
    empty.mapping.time.speedup = -5.0;
    assert_eq!(err(&[one(None, vec![0.0]), empty], &info()), "speedup");
    let mut bad = info();
    bad.licence = " ".into();
    assert_eq!(err(&[one(None, vec![0.0])], &bad), "licence");
    bad = info();
    bad.sources[0].doi = Some("doi:10.1/x".into());
    assert_eq!(err(&[one(None, vec![0.0])], &bad), "doi");
    bad = info();
    bad.sources[0].values = vec![f64::INFINITY];
    assert_eq!(err(&[one(None, vec![0.0])], &bad), "values");
    let big = RenderSettings {
        sample_rate: 8000,
        seed: MAX_SEED + 1,
    };
    let e = write_manifest(&info(), &[one(None, vec![0.0])], &big).unwrap_err();
    assert_eq!(e.name(), "seed");
    let silent = RenderSettings {
        sample_rate: 0,
        seed: 0,
    };
    let e = write_manifest(&info(), &[one(None, vec![0.0])], &silent).unwrap_err();
    assert_eq!(e.name(), "sample_rate");
    assert_eq!(render(&[], &silent).unwrap_err().name(), "sample_rate");
    bad = info();
    bad.title = "\u{1c}".into(); // a control character that Python's re counts as white space
    assert_eq!(err(&[one(None, vec![0.0])], &bad), "title");
    let (manifest, stereo, _) = piece(map, vec![0.0], vec![0.5], 8000);
    let v = |tol: f64, s: &[f64]| {
        verify(&manifest, s, tol, &OnsetParams::default()).map_err(|e| e.name())
    };
    assert_eq!(v(f64::NAN, &stereo).unwrap_err(), "tolerance_s");
    assert_eq!(v(0.002, &stereo[1..]).unwrap_err(), "stereo");
    assert_eq!(v(0.002, &[f64::INFINITY, 0.0]).unwrap_err(), "stereo");
    let e = verify_wav(&manifest, b"RIFF", 0.002, &OnsetParams::default()).unwrap_err();
    assert_eq!(e.name(), "bytes");
    // A note too late or too long to render names its own field.
    let mut late = manifest.layers[0].events[0].clone();
    late.audio_s = 1e9;
    assert_eq!(
        render(&[late.clone()], &settings).unwrap_err().name(),
        "audio_s"
    );
    (late.audio_s, late.dur_s) = (0.0, 1e9);
    assert_eq!(render(&[late], &settings).unwrap_err().name(), "dur_s");
}
