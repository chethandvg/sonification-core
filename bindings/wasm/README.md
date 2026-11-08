# Sonification core: WebAssembly bindings

WebAssembly build of [sonification-core](https://github.com/chethandvg/sonification-core) for browsers and Node. Function names match the Rust crate. WebAssembly has its own maths library, so results that use exp, log, pow, sin or cos (rendered samples among them) can differ from the native build in their last bits; CI checks every result against the shared golden vectors within their tolerances (1e-12 for renders), and the SHA-256s of rendered WAV files exactly.

```bash
npm install
npm run build   # pkg/web for browsers, pkg/node for Node
npm test
```

Mappings, events, manifests and reports are plain objects (typed arrays inside them are accepted); signals come back as `Float64Array` and WAV files as `Uint8Array`. `write_manifest(info, series, sample_rate, seed)` returns `[manifest, wav]`, and `verify_wav(manifest, wav)` checks that file against its manifest. Sample rates must be whole numbers up to 2³² − 1 and seeds whole numbers up to 2⁵³ − 1; trailing settings left `undefined` take the defaults (44 100 Hz, seed 0, and the detector's 1 ms hop, 20 ms windows, 6 dB threshold, 50 ms gap and −90 dB floor).

Building needs Rust with the `wasm32-unknown-unknown` target and the `wasm-bindgen` CLI at the version pinned in `Cargo.toml`.
