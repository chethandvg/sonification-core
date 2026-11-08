// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! Rendering events with synthesis-theory's instruments into a stereo buffer.

use crate::event::Event;
use crate::{
    MAX_SAMPLES, Result, SR, SonifyError, finite, frames_of, positive, require, sample_rate,
};
use serde::{Deserialize, Serialize};
use signal_core::{mix_into, n_samples, pan, write_wav};

/// The largest seed, 2⁵³ − 1: the largest whole number a JavaScript number
/// holds exactly, so a manifest's seed reads back the same in every language.
pub const MAX_SEED: u64 = (1 << 53) - 1;

/// The length of a heartbeat's thump, in seconds (synthesis-theory's `heartbeat`).
const THUMP_S: f64 = 0.18;

/// `lifesound.py`'s defaults, used when a JSON instrument leaves a parameter out.
mod defaults {
    pub fn one() -> f64 {
        1.0
    }
    pub fn ratio() -> f64 {
        1.4
    }
    pub fn index() -> f64 {
        10.0
    }
    pub fn decay() -> f64 {
        0.996
    }
    pub fn pluck() -> Option<f64> {
        Some(2000.0)
    }
    pub fn vibrato_hz() -> f64 {
        5.0
    }
    pub fn vibrato_cents() -> f64 {
        12.0
    }
    pub fn bpm() -> f64 {
        60.0
    }
    pub fn sample_rate() -> u32 {
        super::SR
    }
}

/// An instrument of synthesis-theory and its parameters. Each note is
/// `note(hz, dur_s, sample_rate, seed)`; missing JSON parameters take
/// `lifesound.py`'s defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Instrument {
    /// A struck bar, `synthesis_theory::modal_chime(hz, dur_s, brightness, sr)`.
    ModalChime {
        /// The strength of the higher modes (default 1).
        #[serde(default = "defaults::one")]
        brightness: f64,
    },
    /// Chowning's FM bell, `synthesis_theory::fm_bell(hz, dur_s, ratio, index, sr)`.
    FmBell {
        /// The modulator-to-carrier frequency ratio (default 1.4).
        #[serde(default = "defaults::ratio")]
        ratio: f64,
        /// The starting modulation index (default 10).
        #[serde(default = "defaults::index")]
        index: f64,
    },
    /// The Karplus–Strong plucked string of `lifesound.py`, `synthesis_theory::karplus_strong`.
    KarplusStrong {
        /// The loop gain, in [0, 1] (default 0.996).
        #[serde(default = "defaults::decay")]
        decay: f64,
        /// The burst's low-pass cutoff, or `null` for a bright pluck (default 2000 Hz).
        #[serde(default = "defaults::pluck")]
        pluck_brightness_hz: Option<f64>,
    },
    /// The Karplus–Strong string tuned with an allpass, `synthesis_theory::karplus_strong_tuned`.
    KarplusStrongTuned {
        /// The loop gain, in [0, 1] (default 0.996).
        #[serde(default = "defaults::decay")]
        decay: f64,
        /// The burst's low-pass cutoff, or `null` for a bright pluck (default 2000 Hz).
        #[serde(default = "defaults::pluck")]
        pluck_brightness_hz: Option<f64>,
    },
    /// A soft voice, `synthesis_theory::soft_voice(hz, dur_s, sr, vibrato_hz, vibrato_cents)`.
    SoftVoice {
        /// The vibrato rate (default 5 Hz).
        #[serde(default = "defaults::vibrato_hz")]
        vibrato_hz: f64,
        /// The vibrato depth (default 12 cents).
        #[serde(default = "defaults::vibrato_cents")]
        vibrato_cents: f64,
    },
    /// A lub-dub heartbeat at `bpm` beats per minute for the note's length,
    /// `synthesis_theory::heartbeat(bpm, dur_s, sr, hz)`: the note's frequency
    /// is the thumps' pitch `f0`. A note holds several onsets (each lub and dub).
    Heartbeat {
        /// Beats per minute within the note (default 60).
        #[serde(default = "defaults::bpm")]
        bpm: f64,
    },
}

impl Instrument {
    /// One mono note of `n_samples(dur_s, sample_rate)` samples at `hz` hertz.
    /// `seed` feeds the Karplus–Strong strings' noise burst; the other
    /// instruments ignore it. The sample rate must be at least 1.
    pub fn note(&self, hz: f64, dur_s: f64, sample_rate: u32, seed: u64) -> Result<Vec<f64>> {
        use synthesis_theory as st;
        crate::sample_rate(sample_rate)?;
        let sr = sample_rate;
        Ok(match *self {
            Self::ModalChime { brightness } => st::modal_chime(hz, dur_s, brightness, sr)?,
            Self::FmBell { ratio, index } => st::fm_bell(hz, dur_s, ratio, index, sr)?,
            Self::KarplusStrong {
                decay,
                pluck_brightness_hz,
            } => st::karplus_strong(hz, dur_s, sr, decay, seed, pluck_brightness_hz)?,
            Self::KarplusStrongTuned {
                decay,
                pluck_brightness_hz,
            } => st::karplus_strong_tuned(hz, dur_s, sr, decay, seed, pluck_brightness_hz)?,
            Self::SoftVoice {
                vibrato_hz,
                vibrato_cents,
            } => st::soft_voice(hz, dur_s, sr, vibrato_hz, vibrato_cents)?,
            Self::Heartbeat { bpm } => st::heartbeat(bpm, dur_s, sr, hz)?,
        })
    }

    /// Checks the parameters that do not depend on the note or the sample
    /// rate, as synthesis-theory checks them: `brightness`, `ratio`, `index`,
    /// `vibrato_hz` and `vibrato_cents` finite, `decay` in [0, 1],
    /// `pluck_brightness_hz` (if given) and `bpm` finite and above 0. The
    /// rest (a string's period, `bpm` at most `60 · sample_rate`) is checked
    /// when a note is played.
    pub fn check(&self) -> Result<()> {
        match *self {
            Self::ModalChime { brightness } => finite("brightness", brightness).map(drop),
            Self::FmBell { ratio, index } => {
                finite("ratio", ratio)?;
                finite("index", index).map(drop)
            }
            Self::KarplusStrong {
                decay,
                pluck_brightness_hz,
            }
            | Self::KarplusStrongTuned {
                decay,
                pluck_brightness_hz,
            } => {
                require((0.0..=1.0).contains(&decay), "decay", decay, "in [0, 1]")?;
                pluck_brightness_hz
                    .map_or(Ok(()), |hz| positive("pluck_brightness_hz", hz).map(drop))
            }
            Self::SoftVoice {
                vibrato_hz,
                vibrato_cents,
            } => {
                finite("vibrato_hz", vibrato_hz)?;
                finite("vibrato_cents", vibrato_cents).map(drop)
            }
            Self::Heartbeat { bpm } => positive("bpm", bpm).map(drop),
        }
    }

    /// The onsets a note of `dur_s` seconds at `sr` hertz holds besides its start, in
    /// seconds after it: for a heartbeat, each dub (`min(0.3, 0.4 · 60/bpm)` s
    /// after its lub) and every later lub and dub (beats `60/bpm` s apart),
    /// each placed where synthesis-theory's `heartbeat` places a thump (one
    /// that would reach the end of the note is left out); none for the other
    /// instruments, or for parameters that cannot be played.
    pub(crate) fn inner_onsets(&self, dur_s: f64, sr: u32) -> Vec<f64> {
        let Self::Heartbeat { bpm } = *self else {
            return Vec::new();
        };
        let (Ok(n), Ok(thump)) = (n_samples(dur_s, sr), n_samples(THUMP_S, sr)) else {
            return Vec::new();
        };
        if !(bpm > 0.0 && bpm <= 60.0 * f64::from(sr)) {
            return Vec::new(); // at most one beat a sample, so the loop below is bounded
        }
        let beat = 60.0 / bpm;
        let dub = (beat * 0.4).min(0.30);
        let mut onsets = Vec::new();
        let mut t = 0.0; // summed beat by beat, as synthesis-theory does
        while t < dur_s {
            for at in [t, t + dub] {
                if at > 0.0 && n_samples(at, sr).is_ok_and(|i| i + thump < n) {
                    onsets.push(at);
                }
            }
            t += beat;
        }
        onsets
    }
}

/// How events are rendered: the sample rate and the seed of every random
/// element (default 44 100 Hz and seed 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderSettings {
    /// Samples per second.
    #[serde(default = "defaults::sample_rate")]
    pub sample_rate: u32,
    /// The seed, at most [`MAX_SEED`] (2⁵³ − 1): event `i` (from 0) of a render
    /// plucks its strings with noise seeded `seed + i + 1`, and the WAV file's
    /// dither uses `seed`.
    #[serde(default)]
    pub seed: u64,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sample_rate: SR,
            seed: 0,
        }
    }
}

impl RenderSettings {
    /// Checks the settings: a sample rate of at least 1 and a seed of at most [`MAX_SEED`].
    pub(crate) fn check(&self) -> Result<()> {
        sample_rate(self.sample_rate)?;
        require(
            self.seed <= MAX_SEED,
            "seed",
            self.seed as f64,
            "at most 2^53 - 1, so that JavaScript reads it exactly",
        )
        .map(drop)
    }
}

/// The amplitude of a note's gain, audio-safety-math's `db_to_amplitude(gain_db)`,
/// which must be finite; an error names `gain_db`.
pub(crate) fn amplitude(gain_db: f64) -> Result<f64> {
    audio_safety_math::db_to_amplitude(gain_db).map_err(|_| {
        SonifyError::new(
            "gain_db",
            gain_db,
            "a finite gain whose amplitude, 10^(gain_db / 20), fits in an f64",
        )
    })
}

/// The largest `|sample|` of a render: the manifest's `audio.peak`.
pub(crate) fn peak(stereo: &[f64]) -> f64 {
    stereo.iter().fold(0.0, |m: f64, x| m.max(x.abs()))
}

/// Renders `events` into an interleaved stereo buffer (`[l0, r0, l1, r1, …]`).
///
/// Each event's note is `instrument.note(hz, dur_s, sample_rate, seed + i + 1)`,
/// panned by signal-core's equal-power `pan` and added with signal-core's
/// `mix_into` at `audio_s` seconds (frame `n_samples(audio_s, sample_rate)`)
/// with the amplitude `db_to_amplitude(gain_db)` from audio-safety-math. The
/// buffer ends where the last note ends. Nothing is normalised or limited, so
/// loud overlapping notes can exceed ±1 (a 16-bit WAV file clips them; choose
/// the gains so that the notes that overlap sum to at most 0 dB). The same
/// events and settings always give the same samples on one build (another
/// build, WebAssembly say, has its own `sin`, `exp` and `pow`, so its samples
/// can differ in the last bits). The sample rate must be at least 1 and the
/// seed at most [`MAX_SEED`], even when there are no events.
pub fn render(events: &[Event], settings: &RenderSettings) -> Result<Vec<f64>> {
    settings.check()?;
    let sr = settings.sample_rate;
    let mut frames = 0;
    for e in events {
        frames =
            frames.max(frames_of("audio_s", e.audio_s, sr)? + frames_of("dur_s", e.dur_s, sr)?);
    }
    require(
        frames <= MAX_SAMPLES / 2,
        "events",
        frames as f64,
        "short enough to render: at most MAX_SAMPLES / 2 stereo frames",
    )?;
    let mut buf = vec![0.0; 2 * frames];
    for (i, e) in events.iter().enumerate() {
        let seed = settings.seed.wrapping_add(i as u64).wrapping_add(1);
        let note = e.instrument.note(e.hz, e.dur_s, sr, seed)?;
        let gain = amplitude(e.gain_db)?;
        mix_into(&mut buf, &pan(&note, e.pan)?, 2, e.audio_s, gain, sr)?;
    }
    Ok(buf)
}

/// [`render`] as a 16-bit PCM stereo WAV file, written by signal-core's
/// `write_wav` with TPDF dither seeded `settings.seed`.
pub fn render_wav(events: &[Event], settings: &RenderSettings) -> Result<Vec<u8>> {
    encode(&render(events, settings)?, settings)
}

/// A rendered stereo buffer as the WAV file [`render_wav`] writes.
pub(crate) fn encode(stereo: &[f64], settings: &RenderSettings) -> Result<Vec<u8>> {
    Ok(write_wav(stereo, 2, settings.sample_rate, settings.seed)?)
}

/// The SHA-256 digest of `bytes`, as 64 lowercase hexadecimal digits (what
/// `sha256sum` prints).
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_known_inputs() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn instruments_read_lifesound_defaults() {
        let i: Instrument = serde_json::from_str(r#"{"kind": "karplus_strong"}"#).unwrap();
        let want = Instrument::KarplusStrong {
            decay: 0.996,
            pluck_brightness_hz: Some(2000.0),
        };
        assert_eq!(i, want);
        let i: Instrument =
            serde_json::from_str(r#"{"kind": "karplus_strong", "pluck_brightness_hz": null}"#)
                .unwrap();
        assert_eq!(
            i,
            Instrument::KarplusStrong {
                decay: 0.996,
                pluck_brightness_hz: None
            }
        );
        assert!(serde_json::from_str::<Instrument>(r#"{"kind": "organ"}"#).is_err());
        assert_eq!(
            RenderSettings::default(),
            serde_json::from_str("{}").unwrap()
        );
    }

    #[test]
    fn empty_renders_are_empty() {
        assert!(render(&[], &RenderSettings::default()).unwrap().is_empty());
        assert_eq!(
            render_wav(&[], &RenderSettings::default()).unwrap().len(),
            44
        );
        let big = RenderSettings {
            sample_rate: 8000,
            seed: MAX_SEED + 1,
        };
        assert_eq!(render(&[], &big).unwrap_err().name(), "seed");
        let silent = RenderSettings {
            sample_rate: 0,
            seed: 0,
        };
        assert_eq!(render(&[], &silent).unwrap_err().name(), "sample_rate");
        let chime = Instrument::ModalChime { brightness: 1.0 };
        let e = chime.note(440.0, 0.1, 0, 0).unwrap_err();
        assert_eq!(e.name(), "sample_rate");
        assert_eq!(amplitude(1e6).unwrap_err().name(), "gain_db");
        assert_eq!(amplitude(0.0).unwrap(), 1.0);
    }

    #[test]
    fn heartbeats_hold_their_dubs_and_later_beats() {
        let heart = Instrument::Heartbeat { bpm: 60.0 };
        // Beats at 0, 1 and 2 s with dubs 0.3 s later; the thump at 2.3 s would reach the end.
        assert_eq!(heart.inner_onsets(2.4, 1000), [0.3, 1.0, 1.3, 2.0]);
        assert!(
            Instrument::ModalChime { brightness: 1.0 }
                .inner_onsets(2.0, 1000)
                .is_empty()
        );
        assert!(
            Instrument::Heartbeat { bpm: -1.0 }
                .inner_onsets(2.0, 1000)
                .is_empty()
        );
        assert!(heart.check().is_ok());
        let bad = Instrument::KarplusStrong {
            decay: 1.5,
            pluck_brightness_hz: None,
        };
        assert_eq!(bad.check().unwrap_err().name(), "decay");
    }
}
