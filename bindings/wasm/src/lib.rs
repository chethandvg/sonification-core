// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! WebAssembly bindings for `sonification-core`. Every function keeps the Rust name
//! (methods become functions that take the map first), so the golden vectors can
//! call it by name; invalid input throws a JS `Error`. Mappings, events,
//! manifests and reports are plain objects in the JSON shape of the Rust types
//! (typed arrays inside them are accepted); signals come back as `Float64Array`
//! and WAV files as `Uint8Array`. Sample rates and seeds are ordinary JS numbers
//! that must be whole, so JavaScript never sees a `BigInt`; trailing settings
//! left `undefined` (or `null`) take the Rust defaults.

use serde::Serialize;
use serde::de::DeserializeOwned;
use sonification_core_rs as imp;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;

/// The largest whole number a JS number holds exactly, 2⁵³ − 1.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = JSON, js_name = parse)]
    fn json_parse(text: &str) -> JsValue;
    #[wasm_bindgen(catch, js_namespace = JSON, js_name = stringify)]
    fn json_stringify(value: &JsValue, replacer: &JsValue) -> Result<JsValue, JsValue>;
    #[wasm_bindgen(js_namespace = ArrayBuffer, js_name = isView)]
    fn is_view(value: &JsValue) -> bool;
    #[wasm_bindgen(js_namespace = Array, js_name = from)]
    fn array_from(value: &JsValue) -> JsValue;
}

/// `x` as a whole number of type `T`, or an error naming the parameter.
fn whole<T: TryFrom<u64>>(name: &str, x: f64) -> Result<T, JsError> {
    if (0.0..=MAX_SAFE_INTEGER).contains(&x)
        && x.fract() == 0.0
        && let Ok(v) = T::try_from(x as u64)
    {
        return Ok(v);
    }
    Err(JsError::new(&format!(
        "{name} must be a whole number in range, got {x}"
    )))
}

/// A JSON-shaped JS value read as the Rust value `T`, through `JSON.stringify`
/// (with typed arrays turned into plain arrays).
fn from_js<T: DeserializeOwned>(name: &str, value: &JsValue) -> Result<T, JsError> {
    let plain = Closure::<dyn Fn(JsValue, JsValue) -> JsValue>::new(|_key: JsValue, v: JsValue| {
        if is_view(&v) { array_from(&v) } else { v }
    });
    let text = json_stringify(value, plain.as_ref())
        .ok()
        .and_then(|s| s.as_string())
        .ok_or_else(|| JsError::new(&format!("{name} must be a JSON value")))?;
    serde_json::from_str(&text).map_err(|e| JsError::new(&format!("{name}: {e}")))
}

/// A Rust value as plain JS objects and arrays, through `JSON.parse`.
fn to_js(value: &impl Serialize) -> Result<JsValue, JsError> {
    Ok(json_parse(&serde_json::to_string(value)?))
}

fn settings(sample_rate: Option<f64>, seed: Option<f64>) -> Result<imp::RenderSettings, JsError> {
    let sample_rate = sample_rate.map_or(Ok(imp::SR), |x| whole("sample_rate", x))?;
    let seed = seed.map_or(Ok(0), |x| whole("seed", x))?;
    Ok(imp::RenderSettings { sample_rate, seed })
}

fn params(
    hop_s: Option<f64>,
    window_s: Option<f64>,
    threshold_db: Option<f64>,
    min_gap_s: Option<f64>,
    floor_db: Option<f64>,
) -> imp::OnsetParams {
    let d = imp::OnsetParams::default();
    imp::OnsetParams {
        hop_s: hop_s.unwrap_or(d.hop_s),
        window_s: window_s.unwrap_or(d.window_s),
        threshold_db: threshold_db.unwrap_or(d.threshold_db),
        min_gap_s: min_gap_s.unwrap_or(d.min_gap_s),
        floor_db: floor_db.unwrap_or(d.floor_db),
    }
}

/// The crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// The audio time of real time `real_s`: offset_s + real_s / speedup.
#[wasm_bindgen]
pub fn to_audio(time_map: &JsValue, real_s: f64) -> Result<f64, JsError> {
    Ok(from_js::<imp::TimeMap>("time_map", time_map)?.to_audio(real_s)?)
}

/// The real time of audio time `audio_s`: (audio_s − offset_s) · speedup.
#[wasm_bindgen]
pub fn to_real(time_map: &JsValue, audio_s: f64) -> Result<f64, JsError> {
    Ok(from_js::<imp::TimeMap>("time_map", time_map)?.to_real(audio_s)?)
}

/// A parameter map (constant, linear or exponential) applied to the data value `x`.
#[wasm_bindgen]
pub fn map_value(param_map: &JsValue, x: f64) -> Result<f64, JsError> {
    Ok(from_js::<imp::ParamMap>("param_map", param_map)?.map_value(x)?)
}

/// A pitch map (hz, midi, rhythm or fold) applied to the data value `x`: a frequency in Hz.
#[wasm_bindgen]
pub fn map_pitch(pitch_map: &JsValue, x: f64) -> Result<f64, JsError> {
    Ok(from_js::<imp::PitchMap>("pitch_map", pitch_map)?.map_pitch(x)?)
}

/// The scale note nearest to the MIDI note number `midi`; exact halves go to the higher note.
#[wasm_bindgen]
pub fn quantise(scale: &JsValue, midi: f64) -> Result<f64, JsError> {
    Ok(from_js::<imp::Scale>("scale", scale)?.quantise(midi)? as f64)
}

/// The frequency at which a rhythm of period `period_s` is heard at speed-up `speedup`.
#[wasm_bindgen]
pub fn heard_hz(period_s: f64, speedup: f64) -> Result<f64, JsError> {
    Ok(imp::heard_hz(period_s, speedup)?)
}

/// `lifesound.audify`: `values` (every `src_dt` seconds) played `speedup` times faster; audio that would not vary throws.
#[wasm_bindgen]
pub fn audify(
    values: &[f64],
    src_dt: f64,
    speedup: f64,
    sample_rate: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let sr = sample_rate.map_or(Ok(imp::SR), |x| whole("sample_rate", x))?;
    Ok(imp::audify(values, src_dt, speedup, sr)?)
}

/// The mapping applied to a data series: one event object per point `(times[i], values[i])`.
#[wasm_bindgen]
pub fn sonify(
    mapping: &JsValue,
    times: &[f64],
    values: &[f64],
    claim: Option<String>,
) -> Result<JsValue, JsError> {
    let mapping: imp::Mapping = from_js("mapping", mapping)?;
    to_js(&imp::sonify(&mapping, times, values, claim.as_deref())?)
}

/// The events rendered into an interleaved stereo `Float64Array` (l0, r0, l1, r1, …).
#[wasm_bindgen]
pub fn render(
    events: &JsValue,
    sample_rate: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let events: Vec<imp::Event> = from_js("events", events)?;
    Ok(imp::render(&events, &settings(sample_rate, seed)?)?)
}

/// The events rendered as a 16-bit PCM stereo WAV file (`Uint8Array`).
#[wasm_bindgen]
pub fn render_wav(
    events: &JsValue,
    sample_rate: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<u8>, JsError> {
    let events: Vec<imp::Event> = from_js("events", events)?;
    Ok(imp::render_wav(&events, &settings(sample_rate, seed)?)?)
}

/// The SHA-256 of `data` as 64 lowercase hexadecimal digits.
#[wasm_bindgen]
pub fn sha256_hex(data: &[u8]) -> String {
    imp::sha256_hex(data)
}

/// Sonifies and renders the series; returns `[manifest, wav]`, the manifest object and the WAV it hashes.
#[wasm_bindgen]
pub fn write_manifest(
    info: &JsValue,
    series: &JsValue,
    sample_rate: Option<f64>,
    seed: Option<f64>,
) -> Result<Vec<JsValue>, JsError> {
    let info: imp::ManifestInfo = from_js("info", info)?;
    let series: Vec<imp::Series> = from_js("series", series)?;
    let (manifest, wav) = imp::write_manifest(&info, &series, &settings(sample_rate, seed)?)?;
    Ok(vec![to_js(&manifest)?, wav.into()])
}

/// Onset times in seconds of an interleaved stereo signal: sharp rises of its energy.
#[wasm_bindgen]
pub fn detect_onsets(
    stereo: &[f64],
    sample_rate: Option<f64>,
    hop_s: Option<f64>,
    window_s: Option<f64>,
    threshold_db: Option<f64>,
    min_gap_s: Option<f64>,
    floor_db: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let sr = sample_rate.map_or(Ok(imp::SR), |x| whole("sample_rate", x))?;
    let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
    Ok(imp::detect_onsets(stereo, sr, &p)?)
}

/// Checks rendered stereo audio against its manifest: the timing (onsets matched to events, real times recovered) and the length.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn verify(
    manifest: &JsValue,
    stereo: &[f64],
    tolerance_s: Option<f64>,
    hop_s: Option<f64>,
    window_s: Option<f64>,
    threshold_db: Option<f64>,
    min_gap_s: Option<f64>,
    floor_db: Option<f64>,
) -> Result<JsValue, JsError> {
    let manifest: imp::Manifest = from_js("manifest", manifest)?;
    let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
    let tolerance = tolerance_s.unwrap_or(imp::DEFAULT_TOLERANCE_S);
    to_js(&imp::verify(&manifest, stereo, tolerance, &p)?)
}

/// Checks a WAV file (`Uint8Array`) against its manifest: `verify` on its samples, and that it is the file the manifest's events render to.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn verify_wav(
    manifest: &JsValue,
    wav: &[u8],
    tolerance_s: Option<f64>,
    hop_s: Option<f64>,
    window_s: Option<f64>,
    threshold_db: Option<f64>,
    min_gap_s: Option<f64>,
    floor_db: Option<f64>,
) -> Result<JsValue, JsError> {
    let manifest: imp::Manifest = from_js("manifest", manifest)?;
    let p = params(hop_s, window_s, threshold_db, min_gap_s, floor_db);
    let tolerance = tolerance_s.unwrap_or(imp::DEFAULT_TOLERANCE_S);
    to_js(&imp::verify_wav(&manifest, wav, tolerance, &p)?)
}
