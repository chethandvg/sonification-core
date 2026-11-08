// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! Golden-vector tests. The Python and WebAssembly bindings check the same
//! `vectors/*.json` cases, so all three implementations give the same answers.

use golden_vectors::{Outcome, f64_arg, i64_arg};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sonification_core as sc;
use std::error::Error;

type Res = Result<Value, Box<dyn Error>>;

/// Argument `i` as a whole number of type `T`; out-of-range values (a negative
/// sample rate, say) are rejected, as the Python and JavaScript bindings reject them.
fn int<T: TryFrom<i64>>(args: &[Value], i: usize) -> Result<T, Box<dyn Error>>
where
    T::Error: Error + 'static,
{
    Ok(T::try_from(i64_arg(args, i))?)
}

/// Argument `i` read as `T` from its JSON, as the bindings read dicts and objects.
fn arg<T: DeserializeOwned>(args: &[Value], i: usize) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_value(args[i].clone())?)
}

/// The render settings in arguments `i` (sample rate) and `i + 1` (seed).
fn settings(args: &[Value], i: usize) -> Result<sc::RenderSettings, Box<dyn Error>> {
    Ok(sc::RenderSettings {
        sample_rate: int(args, i)?,
        seed: int(args, i + 1)?,
    })
}

/// The onset settings in arguments `i` to `i + 4`.
fn params(args: &[Value], i: usize) -> sc::OnsetParams {
    sc::OnsetParams {
        hop_s: f64_arg(args, i),
        window_s: f64_arg(args, i + 1),
        threshold_db: f64_arg(args, i + 2),
        min_gap_s: f64_arg(args, i + 3),
        floor_db: f64_arg(args, i + 4),
    }
}

/// Runs the crate's function `name`; `None` if there is no such function.
fn call(name: &str, a: &[Value]) -> Option<Res> {
    let x = |i| f64_arg(a, i);
    let floats = |i| arg::<Vec<f64>>(a, i);
    let run = |f: &dyn Fn() -> Res| Some(f());
    match name {
        "to_audio" => run(&|| Ok(json!(arg::<sc::TimeMap>(a, 0)?.to_audio(x(1))?))),
        "to_real" => run(&|| Ok(json!(arg::<sc::TimeMap>(a, 0)?.to_real(x(1))?))),
        "map_value" => run(&|| Ok(json!(arg::<sc::ParamMap>(a, 0)?.map_value(x(1))?))),
        "map_pitch" => run(&|| Ok(json!(arg::<sc::PitchMap>(a, 0)?.map_pitch(x(1))?))),
        "quantise" => run(&|| Ok(json!(arg::<sc::Scale>(a, 0)?.quantise(x(1))?))),
        "heard_hz" => run(&|| Ok(json!(sc::heard_hz(x(0), x(1))?))),
        "audify" => run(&|| Ok(json!(sc::audify(&floats(0)?, x(1), x(2), int(a, 3)?)?))),
        "sonify" => run(&|| {
            let claim: Option<String> = arg(a, 3)?;
            let events = sc::sonify(&arg(a, 0)?, &floats(1)?, &floats(2)?, claim.as_deref())?;
            Ok(serde_json::to_value(events)?)
        }),
        "render" => run(&|| {
            let events: Vec<sc::Event> = arg(a, 0)?;
            Ok(json!(sc::render(&events, &settings(a, 1)?)?))
        }),
        "write_manifest" => run(&|| {
            let series: Vec<sc::Series> = arg(a, 1)?;
            let (manifest, _wav) = sc::write_manifest(&arg(a, 0)?, &series, &settings(a, 2)?)?;
            Ok(serde_json::to_value(manifest)?)
        }),
        "detect_onsets" => run(&|| {
            Ok(json!(sc::detect_onsets(
                &floats(0)?,
                int(a, 1)?,
                &params(a, 2)
            )?))
        }),
        "verify" => run(&|| {
            let report = sc::verify(&arg(a, 0)?, &floats(1)?, x(2), &params(a, 3))?;
            Ok(serde_json::to_value(report)?)
        }),
        "verify_wav" => run(&|| {
            let wav: Vec<u8> = arg(a, 1)?;
            let report = sc::verify_wav(&arg(a, 0)?, &wav, x(2), &params(a, 3))?;
            Ok(serde_json::to_value(report)?)
        }),
        _ => None,
    }
}

/// Maps each vector's `fn` name to this crate's implementation.
fn dispatch(name: &str, args: &[Value]) -> Outcome {
    call(name, args).map_or_else(|| Outcome::unknown(name, args), Outcome::from)
}

#[test]
fn golden_vectors() {
    golden_vectors::check_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/vectors"), dispatch);
}
