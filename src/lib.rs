// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! # Sonification core
//!
//! Turns measured rhythms into sound that keeps their ratios: a mapping grammar, audification and sonification manifests, in Rust, Python and WebAssembly.
//!
//! Part of [Frequencies of Life](https://chethandvg.github.io/frequencies-of-life/). Each public function corresponds to
//! claims in `claims.yaml`; the README lists what is proved and how.
//!
//! The crate is the instrument layer of the project: data goes in, sound comes
//! out, and every note can be traced back to the number that made it.
//!
//! * **The mapping grammar.** A [`Mapping`] records every constant of a
//!   sonification: a [`TimeMap`] (`audio = offset + real / k`), a [`PitchMap`]
//!   (a value mapped to hertz or to a MIDI note quantised to a [`Scale`], or a
//!   rhythm heard sped up and moved by octaves), [`ParamMap`]s for gain in dB,
//!   pan and note length, and an [`Instrument`]. It serialises to JSON.
//! * **Events and rendering.** [`sonify`] applies a mapping to a data series
//!   and gives one [`Event`] per data point; [`render`] plays the events with
//!   synthesis-theory's instruments, panned and mixed with signal-core, into a
//!   stereo buffer, the same samples for the same events and seed on one
//!   build.
//! * **Audification.** [`audify`] plays a measured series directly as a
//!   waveform, sped up; a rhythm of period `T` is then heard at [`heard_hz`]
//!   `= k / T`.
//! * **The manifest.** [`write_manifest`] renders a piece and describes it in
//!   a small JSON format, `sonification-manifest/0.1-draft`: its sources, the
//!   full mapping, every event and the SHA-256 of the 16-bit WAV file.
//! * **The round trip.** [`verify`] detects the onsets in rendered audio
//!   ([`detect_onsets`]), matches them to the manifest's events and inverts the
//!   time map, so the data's timing is recovered from the sound alone;
//!   [`verify_wav`] does the same for the WAV file and checks that it is,
//!   byte for byte, the file the manifest's events render to.
//!
//! ```
//! use sonification_core::{
//!     Instrument, Mapping, ParamMap, PitchMap, RenderSettings, Scale, TimeMap, render, sonify,
//! };
//!
//! // A synthetic series of five readings an hour apart, sped up 3600 times
//! // (an hour a second), starting half a second into the audio.
//! let mapping = Mapping {
//!     time: TimeMap::new(3600.0, 0.5)?,
//!     // Readings from 0 to 1 become notes from E6 to E7 (MIDI 88 to 100) of E major pentatonic.
//!     pitch: PitchMap::Midi {
//!         map: ParamMap::Linear { x_lo: 0.0, x_hi: 1.0, out_lo: 88.0, out_hi: 100.0, clamp: true },
//!         scale: Some(Scale::major_pentatonic(4)?),
//!     },
//!     gain_db: ParamMap::Constant { value: -12.0 },
//!     pan: ParamMap::Linear { x_lo: 0.0, x_hi: 1.0, out_lo: -0.5, out_hi: 0.5, clamp: true },
//!     duration_s: ParamMap::Constant { value: 0.8 },
//!     instrument: Instrument::ModalChime { brightness: 1.0 },
//! };
//! let times = [0.0, 3600.0, 7200.0, 10800.0, 14400.0];
//! let values = [0.1, 0.5, 0.9, 0.4, 0.75];
//! let events = sonify(&mapping, &times, &values, None)?;
//! assert_eq!(events[1].audio_s, 1.5);
//! // 0.5 maps to MIDI 94 (F#6), not in the scale; the nearest scale note is B6 (95).
//! assert_eq!(events[1].hz, pitch_math::midi_to_hz(95.0)?);
//! let stereo = render(&events, &RenderSettings { sample_rate: 44_100, seed: 0 })?;
//! assert_eq!(stereo.len(), 2 * (198_450 + 35_280)); // the last note starts at 4.5 s and lasts 0.8 s
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Invalid input is a [`SonifyError`], never a panic: a speed-up of 0 or below,
//! NaN or an infinity, a map whose two input ends are equal, an event before
//! audio time 0 or panned outside [−1, 1], a series to audify whose audio
//! would not vary, a manifest event whose claim is not among the sources, a
//! seed above [`MAX_SEED`]. Errors raised by rhythm-units, pitch-math,
//! signal-core and synthesis-theory keep their names and messages; a gain that
//! audio-safety-math cannot turn into a finite amplitude is a `gain_db` error.

use std::fmt;

mod audify;
mod event;
mod manifest;
mod map;
mod render;
mod verify;

pub use audify::{audify, heard_hz};
pub use event::{Event, Mapping, sonify};
pub use manifest::{
    AudioInfo, Layer, MANIFEST_SCHEMA, Manifest, ManifestInfo, RenderInfo, Series, Software,
    Source, write_manifest,
};
pub use map::{ParamMap, PitchMap, Scale, TimeMap};
pub use render::{Instrument, MAX_SEED, RenderSettings, render, render_wav, sha256_hex};
pub use signal_core::{MAX_SAMPLES, SR};
pub use verify::{
    DEFAULT_TOLERANCE_S, EventRef, OnsetMatch, OnsetParams, Verification, detect_onsets, verify,
    verify_wav,
};

/// An argument outside the range a function accepts, or JSON that does not
/// describe the expected value.
#[derive(Debug, Clone, PartialEq)]
pub struct SonifyError {
    name: &'static str,
    value: f64,
    message: String,
}

impl SonifyError {
    fn new(name: &'static str, value: f64, requirement: &str) -> Self {
        Self {
            name,
            value,
            message: format!("{name} must be {requirement}, got {value}"),
        }
    }

    /// An error whose offending value is text, not a number (its `value` is NaN).
    fn text(name: &'static str, requirement: &str, got: &str) -> Self {
        Self {
            name,
            value: f64::NAN,
            message: format!("{name} must be {requirement}, got {got:?}"),
        }
    }

    /// The parameter that was out of range (`"json"` for unreadable JSON).
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The offending value (for a length or a count, the number as an `f64`;
    /// NaN when the value is text).
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl fmt::Display for SonifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SonifyError {}

/// Errors of rhythm-units, pitch-math, signal-core and synthesis-theory keep
/// their parameter name, value and message.
macro_rules! upstream {
    ($($t:ty),*) => {$(
        impl From<$t> for SonifyError {
            fn from(e: $t) -> Self {
                Self { name: e.name(), value: e.value(), message: e.to_string() }
            }
        }
    )*};
}

upstream!(
    rhythm_units::RhythmError,
    pitch_math::PitchError,
    signal_core::SignalError,
    synthesis_theory::SynthError
);

impl From<serde_json::Error> for SonifyError {
    fn from(e: serde_json::Error) -> Self {
        Self {
            name: "json",
            value: f64::NAN,
            message: format!("invalid JSON: {e}"),
        }
    }
}

/// The result type of this crate.
pub type Result<T> = std::result::Result<T, SonifyError>;

fn require(ok: bool, name: &'static str, value: f64, requirement: &str) -> Result<f64> {
    if ok {
        Ok(value)
    } else {
        Err(SonifyError::new(name, value, requirement))
    }
}

fn finite(name: &'static str, x: f64) -> Result<f64> {
    require(x.is_finite(), name, x, "a finite number")
}

fn positive(name: &'static str, x: f64) -> Result<f64> {
    require(
        x.is_finite() && x > 0.0,
        name,
        x,
        "a finite number greater than 0",
    )
}

fn all_finite(name: &'static str, xs: &[f64]) -> Result<()> {
    match xs.iter().find(|x| !x.is_finite()) {
        Some(&bad) => Err(SonifyError::new(name, bad, "finite in every element")),
        None => Ok(()),
    }
}

/// The sample rate `sr` as an `f64`, checked: at least 1 sample per second.
fn sample_rate(sr: u32) -> Result<f64> {
    require(
        sr > 0,
        "sample_rate",
        f64::from(sr),
        "at least 1 sample per second",
    )
}

/// signal-core's `n_samples(seconds, sr)` for the time argument `name`, with the
/// error naming it. `sr` must already be checked ([`sample_rate`]), so an error
/// can only be about the time.
fn frames_of(name: &'static str, seconds: f64, sr: u32) -> Result<usize> {
    signal_core::n_samples(seconds, sr).map_err(|_| {
        SonifyError::new(
            name,
            seconds,
            "a finite, non-negative time of at most MAX_SAMPLES samples",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_name_the_argument_and_keep_upstream_messages() {
        let e = positive("speedup", 0.0).unwrap_err();
        assert_eq!((e.name(), e.value()), ("speedup", 0.0));
        assert_eq!(
            e.to_string(),
            "speedup must be a finite number greater than 0, got 0"
        );
        let e: SonifyError = rhythm_units::scale_frequency(-1.0, 2.0).unwrap_err().into();
        assert_eq!(e.name(), "hz");
        assert_eq!(
            e.to_string(),
            "hz must be a finite number greater than 0, got -1"
        );
        let e: SonifyError = serde_json::from_str::<f64>("x").unwrap_err().into();
        assert_eq!(e.name(), "json");
        assert!(e.value().is_nan());
        let e = SonifyError::text("claim", "a listed source", "x:Y");
        assert_eq!(e.to_string(), "claim must be a listed source, got \"x:Y\"");
        assert_eq!(all_finite("v", &[1.0, f64::NAN]).unwrap_err().name(), "v");
    }
}
