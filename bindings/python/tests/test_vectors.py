# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama
"""The shared golden vectors, and the round trip, checked against the Python bindings."""

import hashlib
import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import sonification_core as sc
from evidence_kernel.vectors import check_dir

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "vectors"


def plain(value):
    """numpy arrays and tuples as lists, for the vector comparison."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


def by_name(module):
    """Every public function of `module`, under its own name, returning plain values.

    write_manifest returns (manifest, wav); the vectors check the manifest, whose SHA-256 covers the wav.
    """

    def wrap(fn):
        return lambda *args: plain(fn(*args))

    impl = {name: wrap(fn) for name, fn in vars(module).items() if callable(fn) and not name.startswith("_")}
    impl["write_manifest"] = lambda *args: module.write_manifest(*args)[0]
    return impl


def test_golden_vectors():
    assert check_dir(VECTORS, by_name(sc)) > 0


CHIMES = {
    "time": {"speedup": 600.0, "offset_s": 0.05},
    "pitch": {
        "kind": "midi",
        "map": {"kind": "linear", "x_lo": 0, "x_hi": 1, "out_lo": 88, "out_hi": 104, "clamp": True},
        "scale": {"root": 4, "pitch_classes": [0, 2, 4, 7, 9]},
    },
    "gain_db": {"kind": "constant", "value": -12.0},
    "pan": {"kind": "linear", "x_lo": 0, "x_hi": 1, "out_lo": -0.7, "out_hi": 0.7},
    "duration_s": {"kind": "constant", "value": 0.4},
    "instrument": {"kind": "modal_chime"},
}
INFO = {"title": "Chimes", "creator": "Chethan Dandebala Venkatarama", "licence": "CC-BY-4.0"}


def piece(offset_s=0.05, sr=22050, seed=1):
    times = np.arange(8) * 360.0 + np.array([0, 50, 10, 90, 30, 70, 20, 0])
    values = np.linspace(0.0, 1.0, 8)
    mapping = CHIMES | {"time": {"speedup": 600.0, "offset_s": offset_s}}
    return sc.write_manifest(INFO, [{"name": "chimes", "mapping": mapping, "times": times, "values": values}], sr, seed)


def stereo_of(manifest):
    events = [e for layer in manifest["layers"] for e in layer["events"]]
    return sc.render(events, manifest["render"]["sample_rate"], manifest["render"]["seed"])


def decode(wav):
    """The WAV file's samples, as signal-core's read_wav scales them."""
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32767.0


@pytest.mark.parametrize("offset_s", [0.05, 0.3])
def test_round_trip_recovers_the_data_timing(offset_s):
    manifest, wav = piece(offset_s)
    for report in (
        sc.verify(manifest, stereo_of(manifest)),
        sc.verify(manifest, decode(wav)),
        sc.verify_wav(manifest, wav),
    ):
        assert report["pass"] and report["events_reproduce"] and report["frames_match"]
        assert len(report["matched"]) == 8 and report["max_audio_error_s"] <= 0.0005
        for m in report["matched"]:
            assert abs(m["real_error_s"]) <= 600.0 * report["tolerance_s"]
    assert sc.verify_wav(manifest, wav)["wav_reproduces"] is True
    assert sc.verify(manifest, decode(wav))["wav_reproduces"] is None
    late, _ = piece(offset_s + 0.01)  # the same data, 10 ms later: self-consistent, but not this audio
    report = sc.verify(late, stereo_of(manifest))
    assert report["events_reproduce"] and not report["pass"] and len(report["missing"]) == 8
    assert not sc.verify_wav(late, wav)["pass"]


def test_manifests_validate_and_round_trip():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schema" / "manifest.schema.json").read_text(encoding="utf-8"))
    manifest, wav = piece()
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema"] == sc.MANIFEST_SCHEMA == "sonification-manifest/0.1-draft"
    assert manifest["software"] == {"name": "sonification-core", "version": sc.__version__}
    for case in json.loads((VECTORS / "manifests.json").read_text(encoding="utf-8"))["cases"]:
        jsonschema.Draft202012Validator(schema).validate(case["expected"])


# Every control character, Unicode's 25 White_Space characters, and a few other unusual ones.
OTHERS = (0x1680, 0x180E, 0x200B, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF, 0x1F600)
UNUSUAL = [chr(i) for i in [*range(0xA1), *range(0x2000, 0x200B), *OTHERS]]


def test_text_ending_in_a_line_feed_is_invalid():
    """The schema's patterns end at the end of the text under Python's re too, where $ would not."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schema" / "manifest.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    manifest, _ = piece()
    manifest["sources"] = [{"claim": "a:B", "doi": "10.1/x", "values": []}]
    assert validator.is_valid(manifest)
    for change in (
        lambda m: m.update(title=m["title"] + "\n"),
        lambda m: m["sources"][0].update(claim="a:B\n"),
        lambda m: m["sources"][0].update(doi="10.1/x\n"),
        lambda m: m["audio"].update(sha256=m["audio"]["sha256"] + "\n"),
    ):
        broken = json.loads(json.dumps(manifest))
        change(broken)
        assert not validator.is_valid(broken)


def test_accepted_text_validates_under_python_re():
    """Whatever text write_manifest accepts, jsonschema (Python's re) accepts too."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "schema" / "manifest.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    mapping = CHIMES | {"duration_s": {"kind": "constant", "value": 0.01}}
    accepted = 0
    for c in UNUSUAL:
        for title, claim, doi in [(c, "a:B", "10.1/x"), ("t", f"a:{c}B", "10.1/x"), ("t", "a:B", f"10.1/{c}x")]:
            info = INFO | {"title": title, "sources": [{"claim": claim, "doi": doi, "values": []}]}
            series = [{"name": "n", "mapping": mapping, "times": [0.0], "values": [0.5], "claim": claim}]
            try:
                manifest, _ = sc.write_manifest(info, series, 1000)
            except ValueError:
                continue
            accepted += 1
            assert validator.is_valid(manifest), f"U+{ord(c):04X}"
    assert accepted > 0


def test_errors_name_their_argument():
    for call, name in [
        (lambda: sc.render([], 0), "sample_rate"),
        (lambda: sc.detect_onsets([0.0, 0.0], 0), "sample_rate"),
        (lambda: sc.detect_onsets([0.0, 0.0], 8000, hop_s=1e9), "hop_s"),
        (lambda: sc.detect_onsets([0.0, 0.0], 8000, window_s=1e9), "window_s"),
        (lambda: sc.detect_onsets([0.0, 0.0], 8000, min_gap_s=-0.05), "min_gap_s"),
        (
            lambda: sc.write_manifest(
                INFO | {"title": "\x1c"}, [{"name": "n", "mapping": CHIMES, "times": [0.0], "values": [0.5]}]
            ),
            "title",
        ),
        (lambda: sc.sonify(CHIMES | {"pitch": {"kind": "fold", "low_hz": 1e308}}, [], []), "low_hz"),
        (lambda: sc.sonify(CHIMES | {"pitch": {"kind": "rhythm", "octaves": 5000}}, [], []), "octaves"),
        (
            lambda: sc.sonify(CHIMES | {"pitch": {"kind": "rhythm", "speedup": 1e-320, "octaves": -1074}}, [], []),
            "octaves",
        ),
        (lambda: sc.sonify(CHIMES | {"gain_db": {"kind": "constant", "value": 1e6}}, [], []), "gain_db"),
    ]:
        with pytest.raises(ValueError, match=f"^{name} must be"):
            call()


def test_rendering_is_deterministic():
    (a, wav_a), (b, wav_b), (c, _) = piece(seed=5), piece(seed=5), piece(seed=6)
    assert wav_a == wav_b and a == b
    assert a["audio"]["sha256"] == sc.sha256_hex(wav_a) == hashlib.sha256(wav_a).hexdigest()
    assert a["audio"]["sha256"] != c["audio"]["sha256"]
    events = [e for layer in a["layers"] for e in layer["events"]]
    assert sc.render_wav(events, 22050, 5) == wav_a
    assert a["audio"]["frames"] == (len(wav_a) - 44) // 4


def test_results_defaults_and_errors():
    events = sc.sonify(CHIMES, [0.0, 600.0], np.array([0.25, 0.5]))
    assert [e["audio_s"] for e in events] == [0.05, 1.05] and "claim" not in events[0]
    assert events[0]["instrument"] == {"kind": "modal_chime", "brightness": 1.0}
    stereo = sc.render(events)
    assert isinstance(stereo, np.ndarray) and stereo.dtype == np.float64
    assert stereo.shape == (2 * sc.SR * 145 // 100,)  # the second note starts at 1.05 s and lasts 0.4 s
    assert np.array_equal(stereo, sc.render(events, sc.SR, 0))
    onsets = sc.detect_onsets(stereo)
    assert isinstance(onsets, np.ndarray) and len(onsets) == 2
    assert sc.heard_hz(600.0, 60000.0) == 100.0
    assert sc.quantise({"root": 4, "pitch_classes": [0, 2, 4, 7, 9]}, 89.0) == 90
    with pytest.raises(ValueError, match="nothing to audify"):
        sc.audify([0.0, 0.0, 0.0], 1.0, 2.0)
    with pytest.raises(ValueError, match="nothing to audify"):
        sc.audify([0.1] * 10 + [0.0], 1.0, 300.0, sample_rate=100)  # the change comes after the last sample
    assert sc.map_pitch({"kind": "rhythm", "octaves": np.float64(4.0)}, 10.0) == 160.0  # whole floats are whole
    with pytest.raises(ValueError, match="seed"):
        sc.render(events, 8000, sc.MAX_SEED + 1)
    mono = io.BytesIO()
    with wave.open(mono, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(bytes(20))
    with pytest.raises(ValueError, match="channels"):
        sc.verify_wav(piece()[0], mono.getvalue())
    with pytest.raises(ValueError, match="time_map: unknown field"):
        sc.to_audio({"speedup": 2.0, "start": 1.0}, 0.0)
    with pytest.raises(ValueError):
        sc.map_value({"kind": "constant", "value": float("nan")}, 0.0)  # NaN is not JSON
    with pytest.raises(OverflowError):
        sc.render(events, 8000, -1)
    with pytest.raises(TypeError):
        sc.audify([0.0, 1.0], 1.0, 2.0, 100.5)
