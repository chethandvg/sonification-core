# Sonification core: Python bindings

Python bindings for [sonification-core](https://github.com/chethandvg/sonification-core), built from the Rust core with PyO3 and maturin. Function names and results are identical to the Rust crate; the shared golden vectors check it in CI.

```bash
pip install "git+https://github.com/chethandvg/sonification-core@v0.1.0#subdirectory=bindings/python"
```

Mappings, events, manifests and reports are plain dicts and lists in the JSON shape of the Rust types (numpy arrays inside them are accepted); signals come back as float64 numpy arrays and WAV files as `bytes`. Inside mappings and the other JSON-shaped arguments, a whole float such as `4.0` is accepted where a whole number is expected; the `sample_rate` and `seed` arguments must be Python ints. `write_manifest(info, series, sample_rate=SR, seed=0)` returns `(manifest, wav)`, and `verify_wav(manifest, wav)` checks that file against its manifest; every sample-rate argument is called `sample_rate`. Seeds are at most `MAX_SEED` (2⁵³ − 1). Invalid input raises `ValueError` (`OverflowError` for a negative or too-large integer argument, `TypeError` for a non-integer one). The schema test needs `jsonschema` (`pip install ".[test]"`; CI gets it as a dependency of evidence-kernel) and is skipped without it.
