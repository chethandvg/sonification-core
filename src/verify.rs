// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! The round trip: onsets detected in the audio, matched to a manifest's
//! events, and turned back into the data's real times.

use crate::event::Event;
use crate::manifest::{Manifest, close};
use crate::render::{encode, peak, render, sha256_hex};
use crate::{Result, SonifyError, all_finite, finite, frames_of, positive, require};
use serde::{Deserialize, Serialize};
use signal_core::read_wav;

/// The default matching tolerance of [`verify`], 2 ms of audio time: the
/// detector's hop (1 ms) and one sample, with room to spare (see the README).
pub const DEFAULT_TOLERANCE_S: f64 = 0.002;

/// The settings of [`detect_onsets`]. The defaults (1 ms hop, 20 ms windows,
/// a 6 dB rise, 50 ms between onsets, a −90 dB floor) suit renders of this
/// crate's instruments, as float buffers and as 16-bit WAV files.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnsetParams {
    /// The step between candidate onset times, in seconds (`n_samples` of it must be at least 1).
    pub hop_s: f64,
    /// The length of the energy windows before and after each candidate, in seconds.
    pub window_s: f64,
    /// The smallest rise, in dB, from the window before to the window after.
    pub threshold_db: f64,
    /// An onset must be the largest rise within this many seconds on either side
    /// (and at least one hop).
    pub min_gap_s: f64,
    /// The power floor, in dB relative to a full-scale sample: both energies
    /// of the rise ratio are raised to a mean frame power of
    /// `10^(floor_db / 10)` if they are below it, so noise below it counts as
    /// silence.
    ///
    /// The default, −90 dB, is set by 16-bit dither. signal-core writes
    /// silence with TPDF dither, which rounds each sample to at most one step
    /// (1/32767) from 0, so a frame's power is at most `2 / 32767²` (−87.3 dB)
    /// and is `0.5 / 32767²` (−93.3 dB) on average. A window of dithered
    /// silence therefore holds at most `2 / 32767² / 10⁻⁹ = 1.86` times the
    /// floor's energy, a rise of at most 2.7 dB, below the default 6 dB
    /// threshold, so it is never an onset; and the dither before a note almost
    /// always stays below the floor (its energy averages 0.47 times the
    /// floor's), so it seldom moves a note's onset. A floor near −93 dB would
    /// let the dither's ups and downs move onsets; a higher
    /// one finds quiet notes that fade in later (see [`detect_onsets`]).
    /// Audio with louder noise (a recording, say) needs a higher floor.
    pub floor_db: f64,
}

impl Default for OnsetParams {
    fn default() -> Self {
        Self {
            hop_s: 0.001,
            window_s: 0.02,
            threshold_db: 6.0,
            min_gap_s: 0.05,
            floor_db: -90.0,
        }
    }
}

/// The onset times, in seconds, of the interleaved stereo signal `stereo` at
/// `sample_rate` samples per second: the points where its energy rises sharply.
///
/// This is an energy-based detector of the kind reviewed by Bello et al. 2005
/// (`sonify:ONSET-DETECTION-METHOD`), in four steps:
///
/// 1. The power of frame `i` is `p[i] = l[i]² + r[i]²`; with equal-power
///    panning it does not depend on the pan position.
/// 2. At every candidate time `s = m · H` (`H = n_samples(hop_s, sr)`), the
///    energy of the `W = n_samples(window_s, sr)` frames after it,
///    `A = Σ p[s .. s + W]`, and before it, `B = Σ p[s − W .. s]`, each raised
///    to the floor `ε = W · 10^(floor_db / 10)` if below it, give the rise
///    ratio `ρ(s) = max(A, ε) / max(B, ε)`. Energy below the floor counts as
///    silence, so noise whose energy stays below it (16-bit dither, almost
///    always) neither makes an onset nor moves one. The signal counts as silent before its start and after
///    its end (the windows are cut off there), so a note at 0 s is found. The
///    sums are differences of running totals of `p`, so a silent stretch sums
///    to exactly 0.
/// 3. `s` is an onset when `ρ(s) ≥ 10^(threshold_db / 10)` and `ρ(s)` is the
///    largest ratio among the candidates within `G = n_samples(min_gap_s, sr)`
///    frames on either side (`⌊G / H⌋` hops, at least one): at least as large
///    as those before it and strictly larger than those after it, so of equal
///    largest ratios the latest wins (a sound shorter than the window gives
///    equal ratios at every candidate whose after-window holds all of it, and
///    the latest of them is the one nearest its start).
/// 4. The onset is refined to the frame: of the frames within one hop of `s`
///    (`s − H + 1` to `s + H − 1`), the latest whose `ρ` is largest. The onset
///    time is that frame divided by `sr`.
///
/// A note that starts sharply from silence (or over much quieter sound) is
/// found at its first sound. A note that fades in is found where the energy
/// of the window before it rises above the floor, so later the quieter it is.
/// This crate's tests, with the default settings, check these bounds (the
/// largest errors they measure are in the README): every instrument within
/// 0.5 ms at −18 to −6 dB; the struck, plucked and beating instruments within
/// 1 ms at −50 dB; the soft voice (an 80 ms attack) within 1.5 ms at −30 dB,
/// but at −40 dB some of its notes more than 2 ms late, outside the default
/// tolerance of [`verify`] (all within 3 ms). A struck bar that starts while earlier bars
/// still ring is now and then found a few milliseconds early, outside the
/// default tolerance too. Notes closer than
/// `min_gap_s`, a note that rises less than `threshold_db` above what is
/// already sounding (a soft voice over sustained notes, say), and a note whose
/// mean power over a window stays below `10^(threshold_db / 10)` times the
/// floor (about −84 dB by default) are not separated. Noise louder than the
/// floor (a recording, say) needs a higher `floor_db`, or its own start is
/// found as an onset.
pub fn detect_onsets(stereo: &[f64], sample_rate: u32, params: &OnsetParams) -> Result<Vec<f64>> {
    let rate = crate::sample_rate(sample_rate)?;
    require(
        stereo.len() % 2 == 0,
        "stereo",
        stereo.len() as f64,
        "a whole number of stereo frames (an even length)",
    )?;
    all_finite("stereo", stereo)?;
    let sr = sample_rate;
    let hop = frames_of("hop_s", positive("hop_s", params.hop_s)?, sr)?;
    require(hop >= 1, "hop_s", params.hop_s, "at least one sample long")?;
    let window = frames_of("window_s", positive("window_s", params.window_s)?, sr)?;
    require(
        window >= 1,
        "window_s",
        params.window_s,
        "at least one sample long",
    )?;
    let gap = frames_of("min_gap_s", finite("min_gap_s", params.min_gap_s)?, sr)?;
    let threshold = 10f64.powf(finite("threshold_db", params.threshold_db)? / 10.0);
    let floor = window as f64 * 10f64.powf(finite("floor_db", params.floor_db)? / 10.0);
    require(
        floor > 0.0 && floor.is_finite(),
        "floor_db",
        params.floor_db,
        "a floor whose energy over a window is finite and above 0",
    )?;
    let frames = stereo.len() / 2;
    let mut total = Vec::with_capacity(frames + 1);
    let mut acc = 0.0;
    total.push(acc);
    for f in stereo.chunks_exact(2) {
        acc += f[0] * f[0] + f[1] * f[1];
        total.push(acc);
    }
    require(
        acc.is_finite(),
        "stereo",
        acc,
        "small enough that its total energy is finite",
    )?;
    let ratio = |s: usize| {
        let after = total[(s + window).min(frames)] - total[s];
        let before = total[s] - total[s.saturating_sub(window)];
        after.max(floor) / before.max(floor)
    };
    let rise: Vec<f64> = (0..frames).step_by(hop).map(ratio).collect();
    let reach = (gap / hop).max(1);
    Ok((0..rise.len())
        .filter(|&m| {
            let r = rise[m];
            r >= threshold
                && rise[m.saturating_sub(reach)..m].iter().all(|&x| x <= r)
                && rise[m + 1..(m + reach + 1).min(rise.len())]
                    .iter()
                    .all(|&x| x < r)
        })
        .map(|m| {
            // The latest frame of largest ratio within one hop either side.
            let s = m * hop;
            let around = s.saturating_sub(hop - 1)..(s + hop).min(frames);
            let best = around.fold(s, |b, f| if ratio(f) >= ratio(b) { f } else { b });
            best as f64 / rate
        })
        .collect())
}

/// An event of a manifest: its layer, its index in the layer and its audio time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventRef {
    /// The layer's index.
    pub layer: usize,
    /// The event's index in the layer.
    pub event: usize,
    /// When the manifest says the note starts, in audio seconds.
    pub audio_s: f64,
}

/// An event matched to a detected onset, and the real time recovered from it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnsetMatch {
    /// The layer's index.
    pub layer: usize,
    /// The event's index in the layer.
    pub event: usize,
    /// When the manifest says the note starts, in audio seconds.
    pub audio_s: f64,
    /// The detected onset, in audio seconds.
    pub detected_s: f64,
    /// `detected_s − audio_s`.
    pub audio_error_s: f64,
    /// The event's real time.
    pub real_s: f64,
    /// The layer's time map inverted at the detected onset: `to_real(detected_s)`.
    pub recovered_real_s: f64,
    /// `recovered_real_s − real_s`, which is `speedup · audio_error_s`.
    pub real_error_s: f64,
}

/// The result of [`verify`] and [`verify_wav`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verification {
    /// The matching tolerance, in audio seconds.
    pub tolerance_s: f64,
    /// Whether the manifest describes its own events
    /// ([`Manifest::reproduces_events`]): each layer's events are its mapping
    /// applied to their own data, each layer's claim is one of the sources,
    /// the instrument list is the events' instruments, the audio is
    /// `wav-pcm16` in 2 channels, and the title, creator, licence and sources
    /// have the form [`write_manifest`](crate::write_manifest) requires.
    pub events_reproduce: bool,
    /// Whether the audio has as many frames as the manifest records (`audio.frames`).
    pub frames_match: bool,
    /// From [`verify_wav`]: whether the file is the one the manifest records
    /// (its SHA-256 is `audio.sha256`) and the one its events render to (the
    /// events rendered again with the manifest's settings give the same bytes,
    /// and a peak equal to `audio.peak` within `1e-12 · max(1, |peak|)`).
    /// `None` from [`verify`], which has no file.
    pub wav_reproduces: Option<bool>,
    /// Every event matched to an onset, in audio-time order.
    pub matched: Vec<OnsetMatch>,
    /// Every event with no onset within the tolerance.
    pub missing: Vec<EventRef>,
    /// Onsets that match no event and no heartbeat beat, in seconds.
    pub extra: Vec<f64>,
    /// Unmatched onsets within the tolerance of a heartbeat note's own later
    /// thumps (its dubs and later beats), in seconds.
    pub within_note: Vec<f64>,
    /// The largest `|audio_error_s|` (0 if nothing matched).
    pub max_audio_error_s: f64,
    /// The largest `|real_error_s|` (0 if nothing matched).
    pub max_real_error_s: f64,
    /// True when the events reproduce, the frames match, the file (if any)
    /// reproduces, no event is missing and no onset is extra.
    pub pass: bool,
}

/// Checks rendered audio against its manifest: detects the onsets of
/// `stereo` (interleaved, at the manifest's sample rate) with
/// [`detect_onsets`], matches each event to one onset within `tolerance_s`
/// seconds, and recovers each matched event's real time by inverting its
/// layer's time map.
///
/// Events are taken in order of audio time (ties in layer and event order);
/// each takes the nearest onset not yet taken within the tolerance (the
/// earlier on a tie), or is missing. A heartbeat note holds more onsets than
/// its start: an unmatched onset within the tolerance of one of its later
/// thumps (each dub, and the later beats, where synthesis-theory places them)
/// is counted as `within_note`, and any other unmatched onset as extra.
///
/// A pass says that the manifest is self-consistent
/// ([`Verification::events_reproduce`]), that the audio is as long as the
/// manifest records, and that every note starts where the manifest says with
/// no unlisted onset: it checks the timing. It does not compare pitches,
/// gains, pans or samples; [`verify_wav`] adds that, through the WAV file's
/// bytes. Neither checks what the descriptive fields say (the title, creator,
/// licence and software, and the sources' DOIs and values; all but the
/// software are checked for the form `write_manifest` requires): they say
/// where the data came from, which the audio cannot show.
pub fn verify(
    manifest: &Manifest,
    stereo: &[f64],
    tolerance_s: f64,
    params: &OnsetParams,
) -> Result<Verification> {
    let tolerance = require(
        tolerance_s.is_finite() && tolerance_s >= 0.0,
        "tolerance_s",
        tolerance_s,
        "a finite number of at least 0",
    )?;
    let sr = manifest.render.sample_rate;
    let onsets = detect_onsets(stereo, sr, params)?;
    let mut order: Vec<EventRef> = manifest
        .layers
        .iter()
        .enumerate()
        .flat_map(|(layer, l)| {
            l.events.iter().enumerate().map(move |(event, e)| EventRef {
                layer,
                event,
                audio_s: e.audio_s,
            })
        })
        .collect();
    order.sort_by(|a, b| a.audio_s.total_cmp(&b.audio_s)); // stable: ties keep layer and event order
    let mut taken = vec![false; onsets.len()];
    let (mut matched, mut missing) = (Vec::new(), Vec::new());
    let mut first = 0; // the first onset that is not too early for this or any later event
    for r in order {
        while first < onsets.len() && onsets[first] < r.audio_s - tolerance {
            first += 1;
        }
        let nearest = (first..onsets.len())
            .take_while(|&j| onsets[j] <= r.audio_s + tolerance)
            .filter(|&j| !taken[j])
            .min_by(|&x, &y| {
                (onsets[x] - r.audio_s)
                    .abs()
                    .total_cmp(&(onsets[y] - r.audio_s).abs())
            });
        let Some(j) = nearest else {
            missing.push(r);
            continue;
        };
        taken[j] = true;
        let layer = &manifest.layers[r.layer];
        let event = layer
            .events
            .get(r.event)
            .ok_or_else(|| SonifyError::new("event", r.event as f64, "an event of the manifest"))?;
        let recovered = layer.mapping.time.to_real(onsets[j])?;
        matched.push(OnsetMatch {
            layer: r.layer,
            event: r.event,
            audio_s: r.audio_s,
            detected_s: onsets[j],
            audio_error_s: onsets[j] - r.audio_s,
            real_s: event.real_s,
            recovered_real_s: recovered,
            real_error_s: recovered - event.real_s,
        });
    }
    let mut beats: Vec<f64> = manifest
        .events()
        .flat_map(|e| {
            let start = e.audio_s;
            let inner = e.instrument.inner_onsets(e.dur_s, sr);
            inner.into_iter().map(move |t| start + t)
        })
        .collect();
    beats.sort_by(f64::total_cmp);
    let near_beat = |s: f64| {
        let i = beats.partition_point(|&b| b < s - tolerance);
        beats.get(i).is_some_and(|&b| b <= s + tolerance)
    };
    let (within_note, extra): (Vec<f64>, Vec<f64>) = onsets
        .iter()
        .zip(&taken)
        .filter(|&(_, &t)| !t)
        .map(|(&s, _)| s)
        .partition(|&s| near_beat(s));
    let largest =
        |f: fn(&OnsetMatch) -> f64| matched.iter().fold(0.0_f64, |m, x| m.max(f(x).abs()));
    let events_reproduce = manifest.reproduces_events();
    let frames_match = (stereo.len() / 2) as u64 == manifest.audio.frames;
    Ok(Verification {
        tolerance_s: tolerance,
        events_reproduce,
        frames_match,
        wav_reproduces: None,
        max_audio_error_s: largest(|m| m.audio_error_s),
        max_real_error_s: largest(|m| m.real_error_s),
        pass: events_reproduce && frames_match && missing.is_empty() && extra.is_empty(),
        matched,
        missing,
        extra,
        within_note,
    })
}

/// Checks a WAV file against its manifest: [`verify`] on the file's samples
/// (decoded by signal-core's `read_wav`), and whether the file is the one the
/// manifest records and the one its events render to
/// ([`Verification::wav_reproduces`]). A pass therefore ties every note's
/// pitch, gain, pan, length and instrument to the file, byte for byte, and
/// recovers its timing from the sound. `wav` must be a stereo 16-bit PCM WAV
/// file at the manifest's sample rate.
pub fn verify_wav(
    manifest: &Manifest,
    wav: &[u8],
    tolerance_s: f64,
    params: &OnsetParams,
) -> Result<Verification> {
    let file = read_wav(wav)?;
    require(
        file.channels == 2,
        "channels",
        file.channels as f64,
        "2 (a stereo WAV file)",
    )?;
    require(
        file.sample_rate == manifest.render.sample_rate,
        "sample_rate",
        f64::from(file.sample_rate),
        "the manifest's sample rate",
    )?;
    let mut report = verify(manifest, &file.samples, tolerance_s, params)?;
    let events: Vec<Event> = manifest.events().cloned().collect();
    let settings = manifest.settings();
    let reproduces = sha256_hex(wav) == manifest.audio.sha256
        && render(&events, &settings).is_ok_and(|stereo| {
            close(peak(&stereo), manifest.audio.peak)
                && encode(&stereo, &settings).is_ok_and(|again| again == wav)
        });
    report.wav_reproduces = Some(reproduces);
    report.pass &= reproduces;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clicks(at: &[usize], frames: usize) -> Vec<f64> {
        let mut x = vec![0.0; 2 * frames];
        for &i in at {
            for k in 0..50 {
                x[2 * (i + k)] = 0.5 * (0.9_f64).powi(k as i32);
            }
        }
        x
    }

    #[test]
    fn finds_onsets_from_silence_within_a_hop() {
        let params = OnsetParams::default();
        let onsets = detect_onsets(&clicks(&[0, 1000, 3000, 3020], 8000), 8000, &params).unwrap();
        // 3020 is 2.5 ms after 3000: within the 50 ms gap, so one onset; a click at 0 is found.
        assert_eq!(onsets, [0.0, 0.125, 0.375]);
        assert!(detect_onsets(&[0.0; 3], 8000, &params).is_err());
        assert!(
            detect_onsets(
                &[0.0; 4],
                8000,
                &OnsetParams {
                    hop_s: 1e-6,
                    ..params
                }
            )
            .is_err()
        );
        let no_floor = OnsetParams {
            floor_db: -4000.0,
            ..params
        };
        assert_eq!(
            detect_onsets(&[0.0; 4], 8000, &no_floor)
                .unwrap_err()
                .name(),
            "floor_db"
        );
        assert!(
            detect_onsets(&vec![0.0; 2000], 8000, &params)
                .unwrap()
                .is_empty()
        );
        // Every error names its own argument.
        let name = |sr: u32, p: OnsetParams| detect_onsets(&[0.0; 4], sr, &p).unwrap_err().name();
        assert_eq!(name(0, params), "sample_rate");
        for (field, p) in [
            (
                "hop_s",
                OnsetParams {
                    hop_s: 1e9,
                    ..params
                },
            ),
            (
                "window_s",
                OnsetParams {
                    window_s: 1e9,
                    ..params
                },
            ),
            (
                "min_gap_s",
                OnsetParams {
                    min_gap_s: -0.05,
                    ..params
                },
            ),
            (
                "min_gap_s",
                OnsetParams {
                    min_gap_s: 1e9,
                    ..params
                },
            ),
        ] {
            assert_eq!(name(8000, p), field);
        }
    }

    #[test]
    fn dithered_silence_has_no_onsets() {
        // One 16-bit step of noise in a quarter of the samples, as TPDF dither
        // leaves silence on average, and in every sample, the worst case.
        let step = 1.0 / 32767.0;
        let typical: Vec<f64> = (0..16_000)
            .map(|i| match i % 8 {
                0 => step,
                5 => -step,
                _ => 0.0,
            })
            .collect();
        let worst: Vec<f64> = (0..16_000)
            .map(|i| if i % 3 == 0 { step } else { -step })
            .collect();
        for noise in [typical, worst] {
            for sr in [8000, 44_100] {
                let onsets = detect_onsets(&noise, sr, &OnsetParams::default()).unwrap();
                assert!(onsets.is_empty(), "{onsets:?}");
            }
        }
    }
}
