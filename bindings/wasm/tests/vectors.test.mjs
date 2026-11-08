// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
// The shared golden vectors, and the round trip, checked against the WebAssembly build.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import { checkDir } from "golden-vectors";

const impl = createRequire(import.meta.url)("../pkg/node/sonification_core_wasm.js");

test("golden vectors", () => {
  // write_manifest returns [manifest, wav]; the vectors check the manifest, whose SHA-256 covers the wav.
  const vectors = { ...impl, write_manifest: (...args) => impl.write_manifest(...args)[0] };
  assert.ok(checkDir(new URL("../../../vectors/", import.meta.url), vectors) > 0);
});

const chimes = {
  time: { speedup: 600, offset_s: 0.05 },
  pitch: {
    kind: "midi",
    map: { kind: "linear", x_lo: 0, x_hi: 1, out_lo: 88, out_hi: 104, clamp: true },
    scale: { root: 4, pitch_classes: [0, 2, 4, 7, 9] },
  },
  gain_db: { kind: "constant", value: -12 },
  duration_s: { kind: "constant", value: 0.4 },
  instrument: { kind: "fm_bell" },
};
const info = { title: "Chimes", creator: "Chethan Dandebala Venkatarama", licence: "CC-BY-4.0" };

function piece(offset_s = 0.05) {
  const times = Float64Array.from({ length: 8 }, (_, i) => 360 * i + 10 * (i % 3)); // typed arrays inside objects work
  const values = Float64Array.from({ length: 8 }, (_, i) => i / 7);
  const mapping = { ...chimes, time: { speedup: 600, offset_s } };
  return impl.write_manifest(info, [{ name: "chimes", mapping, times, values }], 22050, 1);
}

test("the round trip recovers the data's timing, and fails for shifted times", () => {
  const [manifest, wav] = piece();
  assert.ok(wav instanceof Uint8Array);
  assert.equal(manifest.audio.sha256, createHash("sha256").update(wav).digest("hex"));
  assert.equal(impl.sha256_hex(wav), manifest.audio.sha256);
  const events = manifest.layers.flatMap((layer) => layer.events);
  const stereo = impl.render(events, 22050, 1);
  assert.ok(stereo instanceof Float64Array && stereo.length === 2 * manifest.audio.frames);
  assert.deepEqual(impl.render_wav(events, 22050, 1), wav);
  for (const report of [impl.verify(manifest, stereo), impl.verify_wav(manifest, wav)]) {
    assert.ok(report.pass && report.frames_match && report.matched.length === 8, JSON.stringify(report.extra));
    assert.ok(report.max_audio_error_s <= 0.0005);
  }
  assert.equal(impl.verify_wav(manifest, wav).wav_reproduces, true);
  const [late] = piece(0.06);
  const shifted = impl.verify(late, stereo);
  assert.ok(shifted.events_reproduce && !shifted.pass && shifted.missing.length === 8);
  const [quiet, quietWav] = piece(0.3); // 0.3 s of dithered silence before the first note
  assert.ok(impl.verify_wav(quiet, quietWav).pass);
  assert.ok(!impl.verify_wav(late, wav).pass);
});

test("defaults, whole numbers and errors", () => {
  const events = impl.sonify(chimes, [0, 600], [0.25, 0.5], null);
  assert.deepEqual(
    events.map((e) => e.audio_s),
    [0.05, 1.05],
  );
  assert.deepEqual(events[0].instrument, { kind: "fm_bell", ratio: 1.4, index: 10 });
  assert.equal(impl.render(events).length, 2 * (46305 + 17640)); // 44.1 kHz by default
  assert.equal(impl.quantise({ root: 4, pitch_classes: [0, 2, 4, 7, 9] }, 89), 90);
  assert.equal(impl.heard_hz(600, 60000), 100);
  assert.throws(() => impl.audify([0.5, 0.5], 1, 2, 100), /nothing to audify/);
  assert.throws(() => impl.audify([0, 1], 1, 2, 100.5), /sample_rate must be a whole number/);
  assert.throws(() => impl.audify([0.1, 0.1, 0.1, 0.1, 0], 1, 300, 100), /nothing to audify/);
  assert.throws(() => impl.verify_wav(piece()[0], new Uint8Array(4)), /bytes must be a RIFF WAVE file/);
  assert.throws(() => impl.render(events, 44100, -1), /seed must be a whole number/);
  assert.throws(() => impl.render(events, 2 ** 53), /sample_rate must be a whole number/);
  assert.throws(() => impl.to_audio({ speedup: 2, start: 1 }, 0), /time_map: unknown field/);
  assert.throws(() => impl.map_value({ kind: "constant", value: NaN }, 0), /param_map/);
  assert.throws(() => impl.to_audio(undefined, 0), /time_map must be a JSON value/);
  // Errors name their own argument.
  assert.throws(() => impl.render([], 0), /^Error: sample_rate must be at least 1/);
  assert.throws(() => impl.detect_onsets(new Float64Array(2), 0), /^Error: sample_rate must be at least 1/);
  assert.throws(() => impl.detect_onsets(new Float64Array(2), 8000, 0.001, 0.02, 6, -0.05), /^Error: min_gap_s must be/);
  assert.throws(() => impl.write_manifest({ ...info, title: "\u001c" }, [{ name: "n", mapping: chimes, times: [0], values: [0.5] }]), /^Error: title must be/);
});

test("the schema's patterns, as ECMA-262 reads them, end at the end of the text", () => {
  const schema = JSON.parse(readFileSync(new URL("../../../schema/manifest.schema.json", import.meta.url), "utf8"));
  const cases = [
    [schema.$defs.text.pattern, "Chimes"],
    [schema.$defs.claim.pattern, "a:B"],
    [schema.$defs.source.properties.doi.pattern, "10.1/x"],
    [schema.properties.audio.properties.sha256.pattern, "0".repeat(64)],
  ];
  for (const [pattern, text] of cases) {
    const re = new RegExp(pattern, "u");
    assert.ok(re.test(text) && !re.test(`${text}\n`) && !re.test(`${text}\u0085`), pattern);
  }
});

test("ten seconds of 44.1 kHz notes render and verify quickly", () => {
  const start = performance.now();
  const times = Array.from({ length: 40 }, (_, i) => 150 * i);
  const values = times.map((_, i) => (i % 9) / 8);
  const mapping = { ...chimes, time: { speedup: 600 }, duration_s: { kind: "constant", value: 0.2 } };
  const [manifest] = impl.write_manifest(info, [{ name: "n", mapping, times, values }]);
  const stereo = impl.render(manifest.layers[0].events);
  const report = impl.verify(manifest, stereo);
  const seconds = (performance.now() - start) / 1000;
  assert.ok(report.pass);
  console.log(`# rendered, hashed and verified 40 notes (${(stereo.length / 88200).toFixed(1)} s) in ${seconds.toFixed(2)} s`);
  assert.ok(seconds < 10, `took ${seconds} s`);
});
