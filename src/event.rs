// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! A mapping applied to a data series: one event per data point.

use crate::map::{ParamMap, PitchMap, TimeMap};
use crate::render::{Instrument, amplitude};
use crate::{Result, finite, positive, require};
use serde::{Deserialize, Serialize};

fn zero() -> ParamMap {
    ParamMap::Constant { value: 0.0 }
}

/// Every constant of a sonification: how a data point `(real time, value)`
/// becomes a note. A [`sonification manifest`](crate::Manifest) stores it, so
/// the piece can be reproduced exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mapping {
    /// When each note starts: the data point's real time through the time map.
    pub time: TimeMap,
    /// The note's frequency, from the data value.
    pub pitch: PitchMap,
    /// The note's gain in dB, from the data value (default 0 dB); the render
    /// applies it as audio-safety-math's `db_to_amplitude`, which must be finite.
    #[serde(default = "zero")]
    pub gain_db: ParamMap,
    /// The note's pan position, from the data value (default 0, the centre);
    /// it must land in [−1, 1] (left to right).
    #[serde(default = "zero")]
    pub pan: ParamMap,
    /// The note's length in audio seconds, from the data value; above 0.
    pub duration_s: ParamMap,
    /// The instrument that plays the notes.
    pub instrument: Instrument,
}

/// One note: a data point and the sound parameters the mapping gave it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    /// The data point's real time, in seconds.
    pub real_s: f64,
    /// The data value.
    pub value: f64,
    /// The claim id of the data's source, if given (for example `bio:HUMAN-CA-SPIKE-INTERVAL`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<String>,
    /// When the note starts in the audio, in seconds: `time.to_audio(real_s)`.
    pub audio_s: f64,
    /// The note's frequency in hertz.
    pub hz: f64,
    /// The note's gain in dB.
    pub gain_db: f64,
    /// The note's pan position, −1 (left) to 1 (right).
    pub pan: f64,
    /// The note's length in audio seconds.
    pub dur_s: f64,
    /// The instrument that plays it.
    pub instrument: Instrument,
}

/// A note's pan position, which must lie in [−1, 1].
fn pan_position(pan: f64) -> Result<f64> {
    require((-1.0..=1.0).contains(&pan), "pan", pan, "between -1 and 1")
}

/// A note's gain in dB, whose amplitude must be finite (as the render uses it).
fn gain(gain_db: f64) -> Result<f64> {
    amplitude(gain_db).map(|_| gain_db)
}

impl Mapping {
    /// The event for one data point at real time `real_s` with value `value`.
    ///
    /// The note must start at audio time 0 or later (choose `time.offset_s`
    /// accordingly), be panned within [−1, 1] and have a gain whose amplitude
    /// is finite.
    pub fn event(&self, real_s: f64, value: f64, claim: Option<&str>) -> Result<Event> {
        let value = finite("value", value)?;
        let audio_s = self.time.to_audio(real_s)?;
        require(
            audio_s >= 0.0,
            "audio_s",
            audio_s,
            "at least 0 (choose time.offset_s so that every note starts at or after 0)",
        )?;
        Ok(Event {
            real_s,
            value,
            claim: claim.map(str::to_owned),
            audio_s,
            hz: self.pitch.map_pitch(value)?,
            gain_db: gain(self.gain_db.map_value(value)?)?,
            pan: pan_position(self.pan.map_value(value)?)?,
            dur_s: positive("dur_s", self.duration_s.map_value(value)?)?,
            instrument: self.instrument,
        })
    }

    /// Checks every constant of the mapping on its own, whatever the data: the
    /// time map's speed-up (finite, above 0) and offset (finite); every
    /// parameter map's constants (finite ends that differ, exponential outputs
    /// above 0); the scale; the pitch map's speed-up (above 0), `low_hz` (in
    /// pitch-math's range) and octave count (one that some frequency survives);
    /// the output of every constant map, as a note would use it (a frequency
    /// above 0 and finite, a gain whose amplitude is finite, a pan in [−1, 1],
    /// a length above 0); and the
    /// instrument's parameters ([`Instrument::check`]). [`sonify`] runs it once
    /// per series, so a mapping is checked even for a series with no points.
    ///
    /// An output that depends on the data is checked when each value is
    /// mapped: a clamped pan map from 5 to 6, say, passes this check, and every
    /// point then fails.
    pub fn check(&self) -> Result<()> {
        self.time.check()?;
        self.pitch.check()?;
        self.gain_db.check()?;
        self.gain_db.fixed().map_or(Ok(0.0), gain)?;
        self.pan.check()?;
        self.pan.fixed().map_or(Ok(0.0), pan_position)?;
        self.duration_s.check()?;
        self.duration_s
            .fixed()
            .map_or(Ok(1.0), |d| positive("dur_s", d))?;
        self.instrument.check()
    }
}

/// Applies `mapping` to a data series: one [`Event`] per point
/// `(times[i], values[i])`, in the order given, each carrying `claim` (the
/// claim id of the series' source) if one is given. The mapping is checked
/// first ([`Mapping::check`]), even when the series is empty.
pub fn sonify(
    mapping: &Mapping,
    times: &[f64],
    values: &[f64],
    claim: Option<&str>,
) -> Result<Vec<Event>> {
    require(
        times.len() == values.len(),
        "values",
        values.len() as f64,
        "as long as times",
    )?;
    mapping.check()?;
    times
        .iter()
        .zip(values)
        .map(|(&t, &v)| mapping.event(t, v, claim))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping() -> Mapping {
        Mapping {
            time: TimeMap::new(10.0, 0.0).unwrap(),
            pitch: PitchMap::Rhythm {
                speedup: 1.0,
                octaves: 1,
            },
            gain_db: zero(),
            pan: ParamMap::Linear {
                x_lo: 0.0,
                x_hi: 1.0,
                out_lo: -1.0,
                out_hi: 1.0,
                clamp: false,
            },
            duration_s: ParamMap::Constant { value: 0.5 },
            instrument: Instrument::Heartbeat { bpm: 60.0 },
        }
    }

    #[test]
    fn one_event_per_point() {
        let e = sonify(&mapping(), &[0.0, 10.0], &[0.5, 1.0], Some("x:Y")).unwrap();
        assert_eq!((e[1].audio_s, e[1].hz, e[1].pan), (1.0, 2.0, 1.0));
        assert_eq!(e[0].claim.as_deref(), Some("x:Y"));
        assert_eq!(
            sonify(&mapping(), &[0.0], &[], None).unwrap_err().name(),
            "values"
        );
        assert_eq!(mapping().event(0.0, 1.5, None).unwrap_err().name(), "pan");
        assert_eq!(
            mapping().event(-1.0, 0.5, None).unwrap_err().name(),
            "audio_s"
        );
        assert_eq!(
            mapping().event(0.0, f64::NAN, None).unwrap_err().name(),
            "value"
        );
    }

    #[test]
    fn mappings_are_checked_even_without_points() {
        let bad = |change: fn(&mut Mapping)| {
            let mut m = mapping();
            change(&mut m);
            assert!(m.check().is_err());
            sonify(&m, &[], &[], None).unwrap_err().name()
        };
        assert_eq!(bad(|m| m.time.speedup = -5.0), "speedup");
        assert_eq!(bad(|m| m.pan = ParamMap::Constant { value: 7.0 }), "pan");
        assert_eq!(
            bad(|m| m.gain_db = ParamMap::Constant { value: 1e6 }),
            "gain_db"
        );
        // A speed-up so small that every frequency heard is below 2e-12 Hz,
        // which 1074 octaves down takes to 0.
        assert_eq!(
            bad(|m| m.pitch = PitchMap::Rhythm {
                speedup: 1e-320,
                octaves: -1074
            }),
            "octaves"
        );
        assert_eq!(
            bad(|m| m.duration_s = ParamMap::Constant { value: 0.0 }),
            "dur_s"
        );
        assert_eq!(
            bad(|m| m.gain_db = ParamMap::Exponential {
                x_lo: 0.0,
                x_hi: 1.0,
                out_lo: -1.0,
                out_hi: 1.0,
                clamp: false
            }),
            "out_lo"
        );
        assert_eq!(
            bad(|m| m.pitch = PitchMap::Fold {
                speedup: 1.0,
                low_hz: 0.0
            }),
            "low_hz"
        );
        assert_eq!(
            bad(|m| m.instrument = Instrument::Heartbeat { bpm: 0.0 }),
            "bpm"
        );
        assert!(mapping().check().is_ok());
        assert!(sonify(&mapping(), &[], &[], None).unwrap().is_empty());
    }

    #[test]
    fn mappings_read_their_defaults_from_json() {
        let m: Mapping = serde_json::from_str(
            r#"{"time": {"speedup": 10}, "pitch": {"kind": "rhythm", "octaves": 1},
                "pan": {"kind": "linear", "x_lo": 0, "x_hi": 1, "out_lo": -1, "out_hi": 1},
                "duration_s": {"kind": "constant", "value": 0.5},
                "instrument": {"kind": "heartbeat"}}"#,
        )
        .unwrap();
        assert_eq!(m, mapping());
        let bad = r#"{"time": {"speedup": 10, "extra": 1}}"#;
        assert!(serde_json::from_str::<Mapping>(bad).is_err());
    }
}
