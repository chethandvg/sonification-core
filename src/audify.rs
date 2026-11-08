// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! Audification: a measured series played directly as a waveform, sped up.

use crate::{MAX_SAMPLES, Result, all_finite, positive, require};

/// `lifesound.audify`: plays `values`, sampled every `src_dt` seconds, as a
/// waveform `speedup` times faster than real time, at `sample_rate` samples
/// per second.
///
/// The series is linearly interpolated at the audio's sample times, as
/// `numpy.interp` does (sample `i` sits at real time `(i / sample_rate) ·
/// speedup`; at a point's own time its value is taken, so a slope too steep
/// for an `f64` does no harm there; past the last point the last value holds), over
/// `max(⌊t_last / speedup · sample_rate⌋, 2)` samples, where
/// `t_last = (len − 1) · src_dt`. Its mean is then removed and it is scaled to
/// a peak just below 1 (divided by `max |y| + 10⁻¹²`). The only free choice is
/// the speed-up: a rhythm of period `T` seconds in the data is heard at
/// [`heard_hz`]`(T, speedup) = speedup / T` hertz.
///
/// Two things differ from `lifesound.py`:
///
/// * **Audio that would not vary is an error**, because it has nothing to
///   audify: a constant series (silence, all zeros, included), and a series
///   whose interpolated samples are all equal because it varies only between
///   the audio's sample times or after the last of them (lower the speed-up or
///   raise the sample rate). `lifesound.py` returned all zeros for these when
///   their mean came out exact, and otherwise that mean's rounding error,
///   scaled up.
/// * **The mean is removed from `y − y[0]`**, the samples less the first one,
///   which in exact arithmetic gives the same result. Each difference is exact
///   when the samples are close (Sterbenz's lemma), so a series that varies by
///   a few units in the last place of a large value keeps that variation.
///   `lifesound.py` took the mean of `y` itself, whose rounding error is about
///   one unit in the last place of the values; for a series that varies by
///   about that much, normalising scaled the error up to a large offset.
///
/// `values` must hold at least two finite numbers, `src_dt` and `speedup`
/// must be finite and above 0, the result must fit in [`MAX_SAMPLES`] samples,
/// and the samples' differences and their mean must be finite.
pub fn audify(values: &[f64], src_dt: f64, speedup: f64, sample_rate: u32) -> Result<Vec<f64>> {
    require(
        values.len() >= 2,
        "values",
        values.len() as f64,
        "at least two samples long",
    )?;
    all_finite("values", values)?;
    require(
        values.iter().any(|&v| v != values[0]),
        "values",
        values[0],
        "a series that varies (a constant one, silence included, has nothing to audify)",
    )?; // the cheap, common case of the samples check below
    let (dt, k) = (positive("src_dt", src_dt)?, positive("speedup", speedup)?);
    let rate = crate::sample_rate(sample_rate)?;
    let at = |i: usize| i as f64 * dt; // numpy.arange(len(values)) * src_dt
    let last = values.len() - 1;
    let n = (at(last) / k * rate).trunc().max(2.0);
    require(
        n <= MAX_SAMPLES as f64,
        "speedup",
        k,
        "large enough that the audio is at most MAX_SAMPLES samples long",
    )?;
    let mut y = Vec::with_capacity(n as usize);
    let mut j = 0; // at(j) <= x < at(j + 1): the sample times only grow
    for i in 0..n as usize {
        let x = i as f64 / rate * k;
        if x >= at(last) {
            y.push(values[last]);
            continue;
        }
        while at(j + 1) <= x {
            j += 1;
        }
        if x == at(j) {
            y.push(values[j]); // as numpy.interp, so that an infinite slope is not multiplied by 0
            continue;
        }
        let slope = (values[j + 1] - values[j]) / (at(j + 1) - at(j));
        y.push(slope * (x - at(j)) + values[j]);
    }
    let first = y[0];
    y.iter_mut().for_each(|v| *v -= first); // exact for close samples: see above
    require(
        y.iter().any(|&v| v != 0.0),
        "speedup",
        k,
        "low enough (or the sample rate high enough) that the audio's samples vary: \
         every sample here equals the first, so there is nothing to audify",
    )?;
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter_mut().for_each(|v| *v -= mean);
    let peak = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    require(
        mean.is_finite() && peak.is_finite(),
        "values",
        f64::INFINITY,
        "small enough that the audio's samples, their mean and their spread are finite",
    )?; // (an infinite or NaN sample makes the mean infinite or NaN)
    let scale = peak + 1e-12;
    y.iter_mut().for_each(|v| *v /= scale);
    Ok(y)
}

/// The frequency at which a rhythm of period `period_s` real seconds is heard
/// when time is sped up `speedup` times: `speedup / period_s` hertz, as
/// rhythm-units' `scale_frequency(hz_from_period(period_s), speedup)`
/// (`rhythm:TIME-SCALING-FREQUENCY`). A 10-minute rhythm audified 60 000 times
/// faster is heard at 100 Hz.
pub fn heard_hz(period_s: f64, speedup: f64) -> Result<f64> {
    let hz = rhythm_units::hz_from_period(period_s)?;
    Ok(rhythm_units::scale_frequency(hz, speedup)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audify_interpolates_and_normalises() {
        // A ramp 0, 1, 2, 3 at 1 s steps, sped up 2 times at 4 samples a second: 6 samples.
        let y = audify(&[0.0, 1.0, 2.0, 3.0], 1.0, 2.0, 4).unwrap();
        assert_eq!(y.len(), 6);
        let peak = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(peak < 1.0 && peak > 1.0 - 1e-11);
        assert!(y.iter().sum::<f64>().abs() < 1e-12);
        assert_eq!(heard_hz(600.0, 60_000.0).unwrap(), 100.0);
    }

    #[test]
    fn constant_series_are_errors() {
        for constant in [[0.0; 4], [0.1; 4], [1000.0; 4]] {
            let e = audify(&constant, 1.0, 2.0, 100).unwrap_err();
            assert!(e.to_string().contains("nothing to audify"), "{e}");
        }
        assert!(audify(&[1.0], 1.0, 2.0, 100).is_err());
        assert!(audify(&[0.0, 1.0], 0.0, 2.0, 100).is_err());
        assert!(audify(&[0.0, f64::NAN], 1.0, 2.0, 100).is_err());
        assert_eq!(
            audify(&[0.0, 1.0], 1.0, 2.0, 0).unwrap_err().name(),
            "sample_rate"
        );
        assert_eq!(
            audify(&[0.0, 1.0], 1e9, 1e-9, 44_100).unwrap_err().name(),
            "speedup"
        );
        // Variation only between the audio's sample times, or after the last
        // of them: every sample is equal, so it is an error too.
        let mut late = vec![68_648.698_934_053_8; 20];
        late.push(0.0);
        for (values, k) in [(vec![0.0, 0.0, 0.0, 1.0], 150.0), (late, 300.0)] {
            let e = audify(&values, 1.0, k, 100).unwrap_err();
            assert!(e.to_string().contains("nothing to audify"), "{e}");
            assert_eq!(e.name(), "speedup");
        }
        let wide = audify(&[-1e308, 1e308], 1.0, 1.0, 4).unwrap_err();
        assert_eq!(wide.name(), "values");
    }

    #[test]
    fn a_slope_beyond_f64_is_not_used_at_the_points() {
        // Both slopes, 1e310 per second, overflow, but the first sample falls on
        // the first point, where numpy.interp takes its value (the second is past
        // the last point).
        for (values, src_dt) in [([0.0, 1e300], 1e-10), ([0.0, 1.0], 1e-310)] {
            let y = audify(&values, src_dt, 1.0, 4).unwrap();
            assert!(y.len() == 2 && (y[0] + 1.0).abs() < 1e-11 && (y[1] - 1.0).abs() < 1e-11);
        }
    }

    #[test]
    fn a_step_of_one_ulp_keeps_its_shape() {
        // A large constant and one step of one unit in the last place: the
        // audio is the step, not the rounding error of the mean (full-scale DC
        // before the mean was taken of y - y[0]).
        let c: f64 = 68_648.698_934_053_8;
        let mut values = vec![c; 2000];
        values.push(f64::from_bits(c.to_bits() + 1));
        let y = audify(&values, 1.0, 1.0, 10).unwrap();
        let low = y.iter().filter(|&&v| v < 0.0).count();
        // Only the last few samples, from about 1999.5 s on, round up to the step.
        assert!(low >= y.len() - 10, "{low} of {}", y.len());
        assert!(y.iter().all(|&v| v.abs() < 1.0));
        assert!(y.iter().sum::<f64>().abs() < 1e-9);
        assert!(y[0].abs() < 1e-2 && y[y.len() - 1] > 0.9);
    }
}
