// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! Python bindings for `sonification-core`. Every function keeps the Rust name and
//! argument order (methods become functions that take the map first), so the
//! golden vectors can call it by name. Mappings, events, manifests and reports are
//! plain dicts and lists in the JSON shape of the Rust types (numpy arrays inside
//! them are accepted); signals come back as float64 numpy arrays and WAV files as
//! `bytes`. Invalid input raises `ValueError` (`OverflowError` for a negative or
//! too-large integer argument such as a sample rate, `TypeError` for a
//! non-integer one; a seed above `MAX_SEED`, 2^53 - 1, is a `ValueError`).

use pyo3::prelude::*;

#[pymodule]
mod sonification_core {
    use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArrayLike1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};
    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use sonification_core_rs as imp;
    use std::borrow::Cow;

    /// A signal or list argument: a float64 numpy array, or anything numpy can convert to one.
    type Samples<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;
    /// A signal result.
    type Array<'py> = Bound<'py, PyArray1<f64>>;

    fn ok<T>(r: imp::Result<T>) -> PyResult<T> {
        r.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn array<'py>(py: Python<'py>, r: imp::Result<Vec<f64>>) -> PyResult<Array<'py>> {
        Ok(ok(r)?.into_pyarray(py))
    }

    /// The elements of `x`, borrowed when the array is contiguous.
    fn slice<'a>(x: &'a Samples<'_>) -> Cow<'a, [f64]> {
        x.as_slice()
            .map_or_else(|_| Cow::Owned(x.as_array().to_vec()), Cow::Borrowed)
    }

    /// A JSON-shaped Python value (dicts, lists, numbers, strings, `None`, numpy
    /// arrays and scalars) read as the Rust value `T`, through Python's `json`.
    fn from_py<T: DeserializeOwned>(name: &str, value: &Bound<'_, PyAny>) -> PyResult<T> {
        let py = value.py();
        let kwargs = PyDict::new(py);
        let tolist = py
            .import("operator")?
            .getattr("methodcaller")?
            .call1(("tolist",))?;
        kwargs.set_item("default", tolist)?;
        kwargs.set_item("allow_nan", false)?;
        let text: String = py
            .import("json")?
            .call_method("dumps", (value,), Some(&kwargs))?
            .extract()?;
        serde_json::from_str(&text).map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
    }

    /// A Rust value as plain Python dicts and lists, through Python's `json`.
    fn to_py<'py>(py: Python<'py>, value: &impl Serialize) -> PyResult<Bound<'py, PyAny>> {
        let text =
            serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.import("json")?.call_method1("loads", (text,))
    }

    fn settings(sample_rate: u32, seed: u64) -> imp::RenderSettings {
        imp::RenderSettings { sample_rate, seed }
    }

    fn params(
        hop_s: f64,
        window_s: f64,
        threshold_db: f64,
        min_gap_s: f64,
        floor_db: f64,
    ) -> imp::OnsetParams {
        imp::OnsetParams {
            hop_s,
            window_s,
            threshold_db,
            min_gap_s,
            floor_db,
        }
    }

    /// A WAV file argument: `bytes`, or anything Python can turn into a list of bytes.
    fn bytes_of<'a>(x: &'a Bound<'_, PyAny>) -> PyResult<Cow<'a, [u8]>> {
        match x.cast::<PyBytes>() {
            Ok(b) => Ok(Cow::Borrowed(b.as_bytes())),
            Err(_) => Ok(Cow::Owned(x.extract::<Vec<u8>>()?)),
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add("SR", imp::SR)?;
        m.add("MAX_SAMPLES", imp::MAX_SAMPLES)?;
        m.add("MANIFEST_SCHEMA", imp::MANIFEST_SCHEMA)?;
        m.add("MAX_SEED", imp::MAX_SEED)?;
        m.add("DEFAULT_TOLERANCE_S", imp::DEFAULT_TOLERANCE_S)
    }

    /// The audio time of real time real_s: offset_s + real_s / speedup.
    #[pyfunction]
    fn to_audio(time_map: &Bound<'_, PyAny>, real_s: f64) -> PyResult<f64> {
        let map: imp::TimeMap = from_py("time_map", time_map)?;
        ok(map.to_audio(real_s))
    }

    /// The real time of audio time audio_s: (audio_s - offset_s) * speedup.
    #[pyfunction]
    fn to_real(time_map: &Bound<'_, PyAny>, audio_s: f64) -> PyResult<f64> {
        let map: imp::TimeMap = from_py("time_map", time_map)?;
        ok(map.to_real(audio_s))
    }

    /// A parameter map (constant, linear or exponential) applied to the data value x.
    #[pyfunction]
    fn map_value(param_map: &Bound<'_, PyAny>, x: f64) -> PyResult<f64> {
        let map: imp::ParamMap = from_py("param_map", param_map)?;
        ok(map.map_value(x))
    }

    /// A pitch map (hz, midi, rhythm or fold) applied to the data value x: a frequency in Hz.
    #[pyfunction]
    fn map_pitch(pitch_map: &Bound<'_, PyAny>, x: f64) -> PyResult<f64> {
        let map: imp::PitchMap = from_py("pitch_map", pitch_map)?;
        ok(map.map_pitch(x))
    }

    /// The scale note nearest to the MIDI note number midi; exact halves go to the higher note.
    #[pyfunction]
    fn quantise(scale: &Bound<'_, PyAny>, midi: f64) -> PyResult<i64> {
        let scale: imp::Scale = from_py("scale", scale)?;
        ok(scale.quantise(midi))
    }

    /// The frequency at which a rhythm of period period_s is heard at speed-up speedup: speedup / period_s.
    #[pyfunction]
    fn heard_hz(period_s: f64, speedup: f64) -> PyResult<f64> {
        ok(imp::heard_hz(period_s, speedup))
    }

    /// lifesound.audify: values (every src_dt seconds) played speedup times faster; audio that would not vary raises.
    #[pyfunction]
    #[pyo3(signature = (values, src_dt, speedup, sample_rate = imp::SR))]
    fn audify<'py>(
        py: Python<'py>,
        values: Samples<'py>,
        src_dt: f64,
        speedup: f64,
        sample_rate: u32,
    ) -> PyResult<Array<'py>> {
        array(
            py,
            imp::audify(&slice(&values), src_dt, speedup, sample_rate),
        )
    }

    /// The mapping applied to a data series: one event dict per point (times[i], values[i]).
    #[pyfunction]
    #[pyo3(signature = (mapping, times, values, claim = None))]
    fn sonify<'py>(
        py: Python<'py>,
        mapping: &Bound<'py, PyAny>,
        times: Samples<'py>,
        values: Samples<'py>,
        claim: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mapping: imp::Mapping = from_py("mapping", mapping)?;
        let events = imp::sonify(&mapping, &slice(&times), &slice(&values), claim.as_deref());
        to_py(py, &ok(events)?)
    }

    /// The events rendered into an interleaved stereo buffer (l0, r0, l1, r1, ...).
    #[pyfunction]
    #[pyo3(signature = (events, sample_rate = imp::SR, seed = 0))]
    fn render<'py>(
        py: Python<'py>,
        events: &Bound<'py, PyAny>,
        sample_rate: u32,
        seed: u64,
    ) -> PyResult<Array<'py>> {
        let events: Vec<imp::Event> = from_py("events", events)?;
        array(py, imp::render(&events, &settings(sample_rate, seed)))
    }

    /// The events rendered as a 16-bit PCM stereo WAV file (bytes).
    #[pyfunction]
    #[pyo3(signature = (events, sample_rate = imp::SR, seed = 0))]
    fn render_wav<'py>(
        py: Python<'py>,
        events: &Bound<'py, PyAny>,
        sample_rate: u32,
        seed: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let events: Vec<imp::Event> = from_py("events", events)?;
        let wav = ok(imp::render_wav(&events, &settings(sample_rate, seed)))?;
        Ok(PyBytes::new(py, &wav))
    }

    /// The SHA-256 of data (bytes) as 64 lowercase hexadecimal digits.
    #[pyfunction]
    fn sha256_hex(data: &[u8]) -> String {
        imp::sha256_hex(data)
    }

    /// Sonifies and renders the series and returns (manifest, wav): the manifest dict and the WAV bytes it hashes.
    #[pyfunction]
    #[pyo3(signature = (info, series, sample_rate = imp::SR, seed = 0))]
    fn write_manifest<'py>(
        py: Python<'py>,
        info: &Bound<'py, PyAny>,
        series: &Bound<'py, PyAny>,
        sample_rate: u32,
        seed: u64,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyBytes>)> {
        let info: imp::ManifestInfo = from_py("info", info)?;
        let series: Vec<imp::Series> = from_py("series", series)?;
        let (manifest, wav) = ok(imp::write_manifest(
            &info,
            &series,
            &settings(sample_rate, seed),
        ))?;
        Ok((to_py(py, &manifest)?, PyBytes::new(py, &wav)))
    }

    /// Onset times in seconds of an interleaved stereo signal: sharp rises of its energy.
    #[pyfunction]
    #[pyo3(signature = (
        stereo,
        sample_rate = imp::SR,
        hop_s = imp::OnsetParams::default().hop_s,
        window_s = imp::OnsetParams::default().window_s,
        threshold_db = imp::OnsetParams::default().threshold_db,
        min_gap_s = imp::OnsetParams::default().min_gap_s,
        floor_db = imp::OnsetParams::default().floor_db,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn detect_onsets<'py>(
        py: Python<'py>,
        stereo: Samples<'py>,
        sample_rate: u32,
        hop_s: f64,
        window_s: f64,
        threshold_db: f64,
        min_gap_s: f64,
        floor_db: f64,
    ) -> PyResult<Array<'py>> {
        let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
        array(py, imp::detect_onsets(&slice(&stereo), sample_rate, &p))
    }

    /// Checks rendered stereo audio against its manifest: the timing (onsets matched to events, real times recovered) and the length.
    #[pyfunction]
    #[pyo3(signature = (
        manifest,
        stereo,
        tolerance_s = imp::DEFAULT_TOLERANCE_S,
        hop_s = imp::OnsetParams::default().hop_s,
        window_s = imp::OnsetParams::default().window_s,
        threshold_db = imp::OnsetParams::default().threshold_db,
        min_gap_s = imp::OnsetParams::default().min_gap_s,
        floor_db = imp::OnsetParams::default().floor_db,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn verify<'py>(
        py: Python<'py>,
        manifest: &Bound<'py, PyAny>,
        stereo: Samples<'py>,
        tolerance_s: f64,
        hop_s: f64,
        window_s: f64,
        threshold_db: f64,
        min_gap_s: f64,
        floor_db: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manifest: imp::Manifest = from_py("manifest", manifest)?;
        let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
        let report = ok(imp::verify(&manifest, &slice(&stereo), tolerance_s, &p))?;
        to_py(py, &report)
    }

    /// Checks a WAV file (bytes) against its manifest: verify on its samples, and that it is the file the manifest's events render to.
    #[pyfunction]
    #[pyo3(signature = (
        manifest,
        wav,
        tolerance_s = imp::DEFAULT_TOLERANCE_S,
        hop_s = imp::OnsetParams::default().hop_s,
        window_s = imp::OnsetParams::default().window_s,
        threshold_db = imp::OnsetParams::default().threshold_db,
        min_gap_s = imp::OnsetParams::default().min_gap_s,
        floor_db = imp::OnsetParams::default().floor_db,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn verify_wav<'py>(
        py: Python<'py>,
        manifest: &Bound<'py, PyAny>,
        wav: &Bound<'py, PyAny>,
        tolerance_s: f64,
        hop_s: f64,
        window_s: f64,
        threshold_db: f64,
        min_gap_s: f64,
        floor_db: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manifest: imp::Manifest = from_py("manifest", manifest)?;
        let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
        let report = ok(imp::verify_wav(&manifest, &bytes_of(wav)?, tolerance_s, &p))?;
        to_py(py, &report)
    }
}
