// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! The mapping grammar: the time map, parameter maps, pitch maps and scales.

use crate::{Result, SonifyError, finite, positive, require};
use serde::{Deserialize, Deserializer, Serialize};

fn one() -> f64 {
    1.0
}

/// `x` as an `i64` when it is a whole number below 2⁵³ in size (so that it is exact).
fn to_whole<E: serde::de::Error>(x: f64) -> std::result::Result<i64, E> {
    if x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0 {
        Ok(x as i64)
    } else {
        Err(E::custom(format!("expected a whole number, got {x}")))
    }
}

/// A whole number written as an integer or as a whole float (`4` or `4.0`): JSON
/// Schema's `integer`, and what Python's `json` writes for numpy floats.
fn whole<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<i64, D::Error> {
    to_whole(f64::deserialize(d)?)
}

/// A list of whole numbers, each read as [`whole`] reads one.
fn wholes<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<i64>, D::Error> {
    Vec::<f64>::deserialize(d)?
        .into_iter()
        .map(to_whole)
        .collect()
}

/// The time map `audio = offset_s + real / speedup`: real time sped up by the
/// factor `speedup` (`k > 0`), heard from `offset_s` seconds into the audio.
///
/// It is strictly increasing, keeps every ratio of durations, and
/// [`to_real`](Self::to_real) undoes it (`sonify:TIME-MAP`, proved in Lean). A
/// span of `T` real seconds lasts rhythm-units' `scale_period(T, k) = T / k`
/// audio seconds ([`audio_duration`](Self::audio_duration)), and a rhythm of
/// `f` hertz is heard at its `scale_frequency(f, k) = k · f`
/// ([`audio_frequency`](Self::audio_frequency)).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeMap {
    /// The speed-up `k > 0`: one audio second holds `k` real seconds.
    pub speedup: f64,
    /// The audio time, in seconds, of real time 0 (default 0).
    #[serde(default)]
    pub offset_s: f64,
}

impl TimeMap {
    /// A time map, checked: `speedup` must be finite and above 0, `offset_s` finite.
    pub fn new(speedup: f64, offset_s: f64) -> Result<Self> {
        let map = Self { speedup, offset_s };
        map.speedup()?;
        Ok(map)
    }

    /// The checked speed-up.
    fn speedup(&self) -> Result<f64> {
        finite("offset_s", self.offset_s)?;
        positive("speedup", self.speedup)
    }

    /// Checks the constants, as [`new`](Self::new) does.
    pub(crate) fn check(&self) -> Result<()> {
        self.speedup().map(drop)
    }

    /// The audio time of real time `real_s`: `offset_s + real_s / speedup`.
    pub fn to_audio(&self, real_s: f64) -> Result<f64> {
        let k = self.speedup()?;
        let audio = self.offset_s + finite("real_s", real_s)? / k;
        require(
            audio.is_finite(),
            "real_s",
            real_s,
            "small enough that its audio time is finite",
        )
        .map(|_| audio)
    }

    /// The real time of audio time `audio_s`: `(audio_s − offset_s) · speedup`,
    /// the inverse of [`to_audio`](Self::to_audio).
    pub fn to_real(&self, audio_s: f64) -> Result<f64> {
        let k = self.speedup()?;
        let real = (finite("audio_s", audio_s)? - self.offset_s) * k;
        require(
            real.is_finite(),
            "audio_s",
            audio_s,
            "small enough that its real time is finite",
        )
        .map(|_| real)
    }

    /// How long `real_s > 0` real seconds last in the audio: rhythm-units'
    /// `scale_period(real_s, speedup)`, `real_s / speedup`.
    pub fn audio_duration(&self, real_s: f64) -> Result<f64> {
        Ok(rhythm_units::scale_period(real_s, self.speedup()?)?)
    }

    /// The frequency at which a rhythm of `hz > 0` hertz is heard: rhythm-units'
    /// `scale_frequency(hz, speedup)`, `hz · speedup`.
    pub fn audio_frequency(&self, hz: f64) -> Result<f64> {
        Ok(rhythm_units::scale_frequency(hz, self.speedup()?)?)
    }
}

/// A map from a data value to a sound parameter (a gain in dB, a pan position,
/// a note length in seconds, a MIDI note or a frequency).
///
/// With `t = (x − x_lo) / (x_hi − x_lo)` (clamped to [0, 1] when `clamp` is
/// set), `linear` gives `out_lo + t · (out_hi − out_lo)` (`sonify:LINEAR-MAP`)
/// and `exponential` gives `out_lo · (out_hi / out_lo)ᵗ`, so equal steps of the
/// data give equal ratios of the output: a log-frequency scale
/// (`sonify:LOG-MAP`). Both return `out_hi` exactly at `t = 1` and `out_lo`
/// at `t = 0`. `x_lo` and `x_hi` may be in either order but must differ.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParamMap {
    /// The same output for every value.
    Constant {
        /// The output.
        value: f64,
    },
    /// An affine map from `[x_lo, x_hi]` to `[out_lo, out_hi]`.
    Linear {
        /// The value mapped to `out_lo`.
        x_lo: f64,
        /// The value mapped to `out_hi`.
        x_hi: f64,
        /// The output at `x_lo`.
        out_lo: f64,
        /// The output at `x_hi`.
        out_hi: f64,
        /// Whether values outside `[x_lo, x_hi]` are held at the nearest end (default false).
        #[serde(default)]
        clamp: bool,
    },
    /// A geometric map from `[x_lo, x_hi]` to `[out_lo, out_hi]`; both outputs must be above 0.
    Exponential {
        /// The value mapped to `out_lo`.
        x_lo: f64,
        /// The value mapped to `out_hi`.
        x_hi: f64,
        /// The output at `x_lo`, above 0.
        out_lo: f64,
        /// The output at `x_hi`, above 0.
        out_hi: f64,
        /// Whether values outside `[x_lo, x_hi]` are held at the nearest end (default false).
        #[serde(default)]
        clamp: bool,
    },
}

/// `(x − x_lo) / (x_hi − x_lo)`, clamped to [0, 1] when `clamp` is set.
fn position(x: f64, x_lo: f64, x_hi: f64, clamp: bool) -> Result<f64> {
    let span = finite("x_hi", x_hi)? - finite("x_lo", x_lo)?;
    require(
        span != 0.0 && span.is_finite(),
        "x_hi",
        x_hi,
        "different from x_lo, by a finite amount",
    )?;
    let x = finite("x", x)?;
    let t = (x - x_lo) / span;
    require(t.is_finite(), "x", x, "within range of the map")?;
    Ok(if clamp { t.clamp(0.0, 1.0) } else { t })
}

impl ParamMap {
    /// The output for the data value `x`.
    pub fn map_value(&self, x: f64) -> Result<f64> {
        let out = match *self {
            Self::Constant { value } => finite("value", value)?,
            Self::Linear {
                x_lo,
                x_hi,
                out_lo,
                out_hi,
                clamp,
            } => {
                let t = position(x, x_lo, x_hi, clamp)?;
                let (lo, hi) = (finite("out_lo", out_lo)?, finite("out_hi", out_hi)?);
                let span = hi - lo;
                require(
                    span.is_finite(),
                    "out_hi",
                    hi,
                    "within a finite distance of out_lo",
                )?;
                if t == 1.0 { hi } else { lo + t * span }
            }
            Self::Exponential {
                x_lo,
                x_hi,
                out_lo,
                out_hi,
                clamp,
            } => {
                let t = position(x, x_lo, x_hi, clamp)?;
                let (lo, hi) = (positive("out_lo", out_lo)?, positive("out_hi", out_hi)?);
                let ratio = hi / lo;
                require(
                    ratio.is_finite() && ratio > 0.0,
                    "out_hi",
                    hi,
                    "such that out_hi / out_lo is a finite number above 0",
                )?;
                if t == 1.0 { hi } else { lo * ratio.powf(t) }
            }
        };
        require(
            out.is_finite(),
            "x",
            x,
            "a value whose mapped output is finite",
        )
        .map(|_| out)
    }

    /// Checks the constants, whatever the data: computing the map at `x_lo`
    /// (any value, for a constant) validates every one of them.
    pub(crate) fn check(&self) -> Result<()> {
        let x = match *self {
            Self::Linear { x_lo, .. } | Self::Exponential { x_lo, .. } => x_lo,
            Self::Constant { .. } => 0.0,
        };
        self.map_value(x).map(drop)
    }

    /// The output of a constant map, which is the same for every value.
    pub(crate) fn fixed(&self) -> Option<f64> {
        match *self {
            Self::Constant { value } => Some(value),
            _ => None,
        }
    }
}

/// A map from a data value to a frequency in hertz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PitchMap {
    /// The value mapped straight to hertz (an `exponential` map is a log-frequency scale).
    Hz {
        /// The map from the value to hertz; its output must be above 0.
        map: ParamMap,
    },
    /// The value mapped to a MIDI note number (fractional notes allowed),
    /// quantised to `scale` if one is given, then to equal-tempered hertz with
    /// A4 = MIDI 69 = 440 Hz (`pitch_math::midi_to_hz`).
    Midi {
        /// The map from the value to a MIDI note number.
        map: ParamMap,
        /// The scale the note is quantised to (default none).
        #[serde(default)]
        scale: Option<Scale>,
    },
    /// The value is a frequency (a rhythm, in hertz), heard sped up by
    /// `speedup` and moved by `octaves` octaves: rhythm-units'
    /// `scale_frequency(value, speedup)`, then pitch-math's
    /// `octave_transpose(·, octaves)`, which is `f · speedup · 2^octaves`.
    Rhythm {
        /// The speed-up applied to the rhythm (default 1, its real rate). Use
        /// the time map's speed-up to hear it as the audio's clock plays it.
        #[serde(default = "one")]
        speedup: f64,
        /// Octaves to move the sped-up rhythm by, up if positive (default 0).
        #[serde(default, deserialize_with = "whole")]
        octaves: i64,
    },
    /// The value is a frequency, sped up by `speedup` (rhythm-units'
    /// `scale_frequency`) and folded into the octave `[low_hz, 2·low_hz)` by
    /// pitch-math's `octave_fold`.
    Fold {
        /// The speed-up applied to the rhythm (default 1).
        #[serde(default = "one")]
        speedup: f64,
        /// The bottom of the octave the frequency is folded into.
        low_hz: f64,
    },
}

impl PitchMap {
    /// The frequency, in hertz, for the data value `x`.
    pub fn map_pitch(&self, x: f64) -> Result<f64> {
        match self {
            Self::Hz { map } => positive("hz", map.map_value(x)?),
            Self::Midi { map, scale } => {
                let midi = map.map_value(x)?;
                let midi = match scale {
                    Some(s) => s.quantise(midi)? as f64,
                    None => midi,
                };
                Ok(pitch_math::midi_to_hz(midi)?)
            }
            Self::Rhythm { speedup, octaves } => {
                let heard = rhythm_units::scale_frequency(x, *speedup)?;
                Ok(pitch_math::octave_transpose(heard, *octaves)?)
            }
            Self::Fold { speedup, low_hz } => {
                let heard = rhythm_units::scale_frequency(x, *speedup)?;
                Ok(pitch_math::octave_fold(heard, *low_hz)?.1)
            }
        }
    }

    /// Checks the constants, whatever the data: every map's constants and the
    /// scale; speed-ups above 0; `low_hz` in the range pitch-math's
    /// `octave_fold` accepts; an octave count that some heard frequency
    /// survives (pitch-math's `octave_transpose` of the smallest positive
    /// frequency up, or down of the largest frequency heard at the speed-up);
    /// and, for a constant map, the one frequency it
    /// gives (above 0, and a finite equal-tempered pitch).
    pub(crate) fn check(&self) -> Result<()> {
        match self {
            Self::Hz { map } => map.check()?,
            Self::Midi { map, scale } => {
                map.check()?;
                scale.as_ref().map_or(Ok([true; 12]), Scale::members)?;
            }
            Self::Rhythm { speedup, octaves } => {
                let k = positive("speedup", *speedup)?;
                let extreme = if *octaves >= 0 {
                    f64::from_bits(1) // the smallest positive f64, heard for some value
                } else {
                    // The largest frequency heard: f64::MAX sped up, or a value
                    // near f64::MAX / k heard as f64::MAX when k is above 1.
                    rhythm_units::scale_frequency(f64::MAX, k.min(1.0))?
                };
                pitch_math::octave_transpose(extreme, *octaves).map_err(|_| {
                    SonifyError::new(
                        "octaves",
                        *octaves as f64,
                        "an octave count that some finite frequency survives (pitch-math's octave_transpose)",
                    )
                })?;
            }
            Self::Fold { speedup, low_hz } => {
                positive("speedup", *speedup)?;
                pitch_math::octave_fold(positive("low_hz", *low_hz)?, *low_hz)?;
            }
        }
        match self {
            // A constant map gives every value the same frequency.
            Self::Hz { map } | Self::Midi { map, .. } if map.fixed().is_some() => {
                self.map_pitch(0.0).map(drop)
            }
            _ => Ok(()),
        }
    }
}

/// The largest note number [`Scale::quantise`] accepts in size, 2⁵²: below it
/// every whole note number near the input is exact in an `f64`.
const MAX_NOTE: f64 = 4_503_599_627_370_496.0;

/// A scale: the pitch classes `pitch_classes` (semitones above the root, 0 to
/// 11) in every octave, above the root pitch class `root` (0 = C, 4 = E, 9 = A).
///
/// The E6 pentatonic notes of the earlier Genesis piece, MIDI 88, 90, 92, 95,
/// 97, 100, 102 and 104, are [`Scale::major_pentatonic`]`(4)` (E major
/// pentatonic) with the MIDI map clamped to 88 to 104.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scale {
    /// The root pitch class, 0 (C) to 11 (B).
    #[serde(deserialize_with = "whole")]
    pub root: i64,
    /// The pitch classes of the scale, in semitones above the root (0 to 11).
    #[serde(deserialize_with = "wholes")]
    pub pitch_classes: Vec<i64>,
}

impl Scale {
    /// A scale, checked: `root` and every pitch class must be from 0 to 11,
    /// and there must be at least one pitch class.
    pub fn new(root: i64, pitch_classes: &[i64]) -> Result<Self> {
        let scale = Self {
            root,
            pitch_classes: pitch_classes.to_vec(),
        };
        scale.members()?;
        Ok(scale)
    }

    /// The major pentatonic scale on `root`: pitch classes 0, 2, 4, 7 and 9.
    pub fn major_pentatonic(root: i64) -> Result<Self> {
        Self::new(root, &[0, 2, 4, 7, 9])
    }

    /// Which of the twelve pitch classes above the root are in the scale.
    fn members(&self) -> Result<[bool; 12]> {
        let class = |n: i64| (0..12).contains(&n);
        require(
            class(self.root),
            "root",
            self.root as f64,
            "a pitch class from 0 (C) to 11 (B)",
        )?;
        require(
            !self.pitch_classes.is_empty(),
            "pitch_classes",
            0.0,
            "at least one pitch class",
        )?;
        let mut members = [false; 12];
        for &c in &self.pitch_classes {
            require(
                class(c),
                "pitch_classes",
                c as f64,
                "semitones above the root, from 0 to 11",
            )?;
            members[c as usize] = true;
        }
        Ok(members)
    }

    /// The scale note nearest to the MIDI note number `midi`. A note exactly
    /// halfway between two scale notes goes to the higher one, as pitch-math's
    /// `nearest_note` rounds halves up. `midi` must be finite and at most 2⁵²
    /// in size.
    pub fn quantise(&self, midi: f64) -> Result<i64> {
        let members = self.members()?;
        let m = require(
            midi.is_finite() && midi.abs() <= MAX_NOTE,
            "midi",
            midi,
            "a finite note number of at most 2^52 in size",
        )?;
        let member = |n: i64| members[(n - self.root).rem_euclid(12) as usize];
        // Every octave holds a member, so each search takes at most 11 steps.
        let (mut below, mut above) = (m.floor() as i64, m.ceil() as i64);
        while !member(below) {
            below -= 1;
        }
        while !member(above) {
            above += 1;
        }
        Ok(if m - (below as f64) < (above as f64) - m {
            below
        } else {
            above
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_map_round_trips() {
        let map = TimeMap::new(600.0, 2.0).unwrap();
        assert_eq!(map.to_audio(600.0).unwrap(), 3.0);
        assert_eq!(map.to_real(3.0).unwrap(), 600.0);
        assert_eq!(map.audio_duration(1200.0).unwrap(), 2.0);
        assert_eq!(map.audio_frequency(0.5).unwrap(), 300.0);
        assert!(TimeMap::new(0.0, 0.0).is_err());
        assert_eq!(
            TimeMap::new(0.5, 0.0)
                .unwrap()
                .to_audio(f64::MAX)
                .unwrap_err()
                .name(),
            "real_s"
        );
    }

    #[test]
    fn maps_hit_their_ends_exactly() {
        let lin = ParamMap::Linear {
            x_lo: 0.0,
            x_hi: 1.0,
            out_lo: 0.1,
            out_hi: 0.3,
            clamp: false,
        };
        assert_eq!(lin.map_value(0.0).unwrap(), 0.1);
        assert_eq!(lin.map_value(1.0).unwrap(), 0.3);
        assert!((lin.map_value(2.0).unwrap() - 0.5).abs() < 1e-15);
        let exp = ParamMap::Exponential {
            x_lo: 2.0,
            x_hi: 0.0,
            out_lo: 110.0,
            out_hi: 880.0,
            clamp: true,
        };
        assert_eq!(exp.map_value(0.0).unwrap(), 880.0);
        assert_eq!(exp.map_value(2.0).unwrap(), 110.0);
        assert_eq!(exp.map_value(-5.0).unwrap(), 880.0);
        assert!((exp.map_value(1.0).unwrap() - 311.126_983_722_080_9).abs() < 1e-12);
        let flat = ParamMap::Linear {
            x_lo: 1.0,
            x_hi: 1.0,
            out_lo: 0.0,
            out_hi: 1.0,
            clamp: false,
        };
        assert_eq!(flat.map_value(1.0).unwrap_err().name(), "x_hi");
    }

    #[test]
    fn quantise_goes_to_the_nearest_scale_note_and_halves_up() {
        let e = Scale::major_pentatonic(4).unwrap();
        assert_eq!(e.quantise(88.0).unwrap(), 88);
        assert_eq!(e.quantise(89.0).unwrap(), 90); // a tie between 88 and 90 goes up
        assert_eq!(e.quantise(93.4).unwrap(), 92);
        assert_eq!(e.quantise(93.5).unwrap(), 95); // 93.5 is 1.5 from both 92 and 95
        assert_eq!(e.quantise(-1.2).unwrap(), -1); // MIDI -1 is a B, in E major pentatonic
        let c = Scale::new(0, &[0]).unwrap();
        assert_eq!(c.quantise(66.0).unwrap(), 72); // a tritone from both Cs goes up
        assert_eq!(c.quantise(65.9).unwrap(), 60);
        assert!(Scale::new(12, &[0]).is_err() && Scale::new(0, &[]).is_err());
        assert!(c.quantise(f64::NAN).is_err());
    }

    #[test]
    fn pitch_maps_reuse_rhythm_units_and_pitch_math() {
        let rhythm = PitchMap::Rhythm {
            speedup: 1.0,
            octaves: 4,
        };
        assert_eq!(rhythm.map_pitch(12.5).unwrap(), 200.0);
        let fold = PitchMap::Fold {
            speedup: 1.0,
            low_hz: 220.0,
        };
        assert_eq!(fold.map_pitch(1.0 / 600.0).unwrap(), 32_768.0 / 75.0);
        let midi = PitchMap::Midi {
            map: ParamMap::Constant { value: 69.0 },
            scale: None,
        };
        assert_eq!(midi.map_pitch(0.0).unwrap(), 440.0);
        let hz = PitchMap::Hz {
            map: ParamMap::Constant { value: -1.0 },
        };
        assert_eq!(hz.map_pitch(0.0).unwrap_err().name(), "hz");
        assert_eq!(hz.check().unwrap_err().name(), "hz");
    }

    #[test]
    fn checks_find_maps_that_no_value_can_use() {
        let bad = [
            (
                PitchMap::Fold {
                    speedup: 1.0,
                    low_hz: 1e308,
                },
                "low_hz",
            ),
            (
                PitchMap::Fold {
                    speedup: 1.0,
                    low_hz: 1e-310,
                },
                "low_hz",
            ),
            (
                PitchMap::Rhythm {
                    speedup: 1.0,
                    octaves: 5000,
                },
                "octaves",
            ),
            (
                PitchMap::Rhythm {
                    speedup: 1.0,
                    octaves: -5000,
                },
                "octaves",
            ),
            (
                PitchMap::Midi {
                    map: ParamMap::Constant { value: 1e6 },
                    scale: None,
                },
                "result",
            ),
        ];
        for (map, name) in bad {
            assert_eq!(map.check().unwrap_err().name(), name, "{map:?}");
        }
        // The extreme octave counts pitch-math accepts, each with a frequency that survives.
        for (octaves, hz) in [(1023, 1e-300), (-1074, 1e300)] {
            let map = PitchMap::Rhythm {
                speedup: 1.0,
                octaves,
            };
            assert!(
                map.check().is_ok() && map.map_pitch(hz).is_ok(),
                "{octaves}"
            );
        }
        let ok = PitchMap::Fold {
            speedup: 1.0,
            low_hz: 1e300,
        };
        assert!(ok.check().is_ok());
    }

    #[test]
    fn whole_numbers_may_be_written_as_whole_floats() {
        let rhythm: PitchMap =
            serde_json::from_str(r#"{"kind": "rhythm", "octaves": 4.0}"#).unwrap();
        assert_eq!(rhythm.map_pitch(12.5).unwrap(), 200.0);
        let scale: Scale =
            serde_json::from_str(r#"{"root": 4.0, "pitch_classes": [0, 2.0]}"#).unwrap();
        assert_eq!(scale, Scale::new(4, &[0, 2]).unwrap());
        for bad in [
            r#"{"kind": "rhythm", "octaves": 0.5}"#,
            r#"{"kind": "rhythm", "octaves": 1e300}"#,
        ] {
            assert!(serde_json::from_str::<PitchMap>(bad).is_err(), "{bad}");
        }
    }
}
