// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Chethan Dandebala Venkatarama

//! The sonification manifest: a piece's sources, mappings, events and audio hash.

use crate::event::{Event, Mapping, sonify};
use crate::render::{Instrument, RenderSettings, encode, peak, render, sha256_hex};
use crate::{Result, SonifyError, all_finite, require};
use serde::{Deserialize, Serialize};

/// The format of the manifests this crate writes. The open standard is
/// planned for the `sonification-manifest` repository; this is its draft.
pub const MANIFEST_SCHEMA: &str = "sonification-manifest/0.1-draft";

/// The audio encoding the manifest's hash covers.
const ENCODING: &str = "wav-pcm16";

/// The audio's channels: stereo.
const CHANNELS: u32 = 2;

/// The length of signal-core's WAV header; 4 bytes per stereo frame follow it.
const WAV_HEADER: usize = 44;

/// The software that wrote a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Software {
    /// The package name, `sonification-core`.
    pub name: String,
    /// Its version.
    pub version: String,
}

/// A data source: the claim it is recorded under, its DOI, and the values the piece uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// The claim id (for example `pitch:TEN-MINUTE-RHYTHM-NOTE`).
    pub claim: String,
    /// The DOI of the publication the values come from, if there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    /// The values taken from the source.
    pub values: Vec<f64>,
}

/// What a manifest says about a piece besides its data: its title, creator,
/// the audio's licence and the data's sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestInfo {
    /// The piece's title.
    pub title: String,
    /// Who made it.
    pub creator: String,
    /// The audio's licence, as an SPDX identifier (for example `CC-BY-4.0`).
    pub licence: String,
    /// The data sources (default none).
    #[serde(default)]
    pub sources: Vec<Source>,
}

/// A data series to sonify: a name, a mapping, the points and their source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Series {
    /// The layer's name (for example `calcium spikes`).
    pub name: String,
    /// How the points become notes.
    pub mapping: Mapping,
    /// The points' real times, in seconds.
    pub times: Vec<f64>,
    /// The points' values.
    pub values: Vec<f64>,
    /// The claim id of the series' source, which must be one of the manifest's sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<String>,
}

/// One layer of a manifest: a series' name, claim, mapping and events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// The layer's name.
    pub name: String,
    /// The claim id of the layer's source, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<String>,
    /// The mapping, with every constant.
    pub mapping: Mapping,
    /// Every event, in the order of the series.
    pub events: Vec<Event>,
}

/// How the audio was rendered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderInfo {
    /// Samples per second.
    pub sample_rate: u32,
    /// The seed of every random element (see [`RenderSettings::seed`]), at most 2⁵³ − 1.
    pub seed: u64,
    /// Every distinct instrument the events use, in order of first use.
    pub instruments: Vec<Instrument>,
}

/// The rendered audio file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioInfo {
    /// `wav-pcm16`: a 16-bit PCM WAV file as signal-core's `write_wav` writes it.
    pub encoding: String,
    /// Channels per frame, 2.
    pub channels: u32,
    /// Frames (samples per channel).
    pub frames: u64,
    /// The largest `|sample|` of the render before it was written. Above 1,
    /// the WAV file is clipped there: lower the gains.
    pub peak: f64,
    /// The SHA-256 of the whole WAV file, header included, in lowercase hexadecimal.
    pub sha256: String,
}

/// A sonification manifest (`sonification-manifest/0.1-draft`): everything
/// needed to reproduce a rendered piece and check it note by note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// The piece's title.
    pub title: String,
    /// Who made it.
    pub creator: String,
    /// The audio's licence (an SPDX identifier).
    pub licence: String,
    /// The software that wrote the manifest and rendered the audio.
    pub software: Software,
    /// The data sources.
    pub sources: Vec<Source>,
    /// The layers: each series' mapping and events.
    pub layers: Vec<Layer>,
    /// The render settings.
    pub render: RenderInfo,
    /// The audio file and its hash.
    pub audio: AudioInfo,
}

/// The largest difference, `1e-12 · max(1, |b|)`, by which a stored number may
/// differ from its recomputed value: the last-bit differences of `exp`, `pow`
/// and `sin` between platforms.
const SAME: f64 = 1e-12;

/// Whether `a` is `b` within `1e-12 · max(1, |b|)`.
pub(crate) fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= SAME * b.abs().max(1.0)
}

fn same_event(a: &Event, b: &Event) -> bool {
    a.claim == b.claim
        && a.instrument == b.instrument
        && [
            (a.real_s, b.real_s),
            (a.value, b.value),
            (a.audio_s, b.audio_s),
            (a.hz, b.hz),
            (a.gain_db, b.gain_db),
            (a.pan, b.pan),
            (a.dur_s, b.dur_s),
        ]
        .iter()
        .all(|&(x, y)| close(x, y))
}

/// Whether `c` is blank: a control character or white space (Unicode's `Cc`
/// and `White_Space`). The schema's patterns list the same characters by
/// code point, so that no dialect's own idea of white space enters, and end
/// in `(?![\s\S])`, the end of the text in ECMA-262 and in Python's `re`
/// alike (where `$` would also match before a final line feed).
fn blank(c: char) -> bool {
    c.is_control() || c.is_whitespace()
}

/// A title, creator or licence: no control characters, and not all blank.
fn text(name: &'static str, value: &str) -> Result<()> {
    if value.contains(char::is_control) || value.chars().all(blank) {
        Err(SonifyError::text(
            name,
            "text that is not blank and holds no control characters",
            value,
        ))
    } else {
        Ok(())
    }
}

/// The rules for a manifest's descriptive fields: the title, creator and
/// licence are [`text`], and every source has a well-formed claim id, DOI (if
/// any) and finite values.
fn check_info(title: &str, creator: &str, licence: &str, sources: &[Source]) -> Result<()> {
    text("title", title)?;
    text("creator", creator)?;
    text("licence", licence)?;
    for s in sources {
        if !s.claim.contains(':') || s.claim.contains(blank) {
            return Err(SonifyError::text(
                "claim",
                "a claim id of the form prefix:NAME, with no blank characters",
                &s.claim,
            ));
        }
        match &s.doi {
            Some(doi) if !(doi.starts_with("10.") && doi.contains('/') && !doi.contains(blank)) => {
                return Err(SonifyError::text("doi", "a DOI such as 10.1000/xyz", doi));
            }
            _ => {}
        }
        all_finite("values", &s.values)?;
    }
    Ok(())
}

/// Whether `claim`, if there is one, is the claim of one of `sources`.
fn cited(sources: &[Source], claim: Option<&str>) -> bool {
    claim.is_none_or(|c| sources.iter().any(|s| s.claim == c))
}

/// Every distinct instrument of `events`, in order of first use.
fn instruments_of<'a>(events: impl IntoIterator<Item = &'a Event>) -> Vec<Instrument> {
    let mut instruments = Vec::new();
    for e in events {
        if !instruments.contains(&e.instrument) {
            instruments.push(e.instrument);
        }
    }
    instruments
}

/// Sonifies `series`, renders every event with `settings` and writes the
/// manifest that describes the result. Returns the manifest and the WAV file
/// (16-bit PCM, stereo) whose SHA-256 it records.
///
/// Each series becomes a [`Layer`] through [`sonify`], which checks its
/// mapping even when it has no points; the events of all layers, in order,
/// are rendered to a WAV file by [`render_wav`](crate::render_wav), and the
/// render's peak is recorded (above 1, the file is clipped). The title,
/// creator and licence must not be blank or hold control characters (tabs,
/// line breaks), every source needs a claim id `prefix:NAME` with no blank
/// characters (white space or control characters; a DOI, if given, starts
/// `10.`, holds a `/` and no blank characters), every series' claim must be
/// one of the sources' claims, there must be at least one event, the sample
/// rate must be at least 1 and the seed at most [`MAX_SEED`](crate::MAX_SEED).
pub fn write_manifest(
    info: &ManifestInfo,
    series: &[Series],
    settings: &RenderSettings,
) -> Result<(Manifest, Vec<u8>)> {
    check_info(&info.title, &info.creator, &info.licence, &info.sources)?;
    let mut layers = Vec::with_capacity(series.len());
    for s in series {
        if !cited(&info.sources, s.claim.as_deref()) {
            return Err(SonifyError::text(
                "claim",
                "the claim of one of the manifest's sources",
                s.claim.as_deref().unwrap_or_default(),
            ));
        }
        layers.push(Layer {
            name: s.name.clone(),
            claim: s.claim.clone(),
            mapping: s.mapping.clone(),
            events: sonify(&s.mapping, &s.times, &s.values, s.claim.as_deref())?,
        });
    }
    let events: Vec<Event> = layers.iter().flat_map(|l| l.events.clone()).collect();
    require(
        !events.is_empty(),
        "series",
        0.0,
        "at least one data point to sonify",
    )?;
    let stereo = render(&events, settings)?;
    let wav = encode(&stereo, settings)?;
    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        title: info.title.clone(),
        creator: info.creator.clone(),
        licence: info.licence.clone(),
        software: Software {
            name: env!("CARGO_PKG_NAME").to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
        },
        sources: info.sources.clone(),
        layers,
        render: RenderInfo {
            sample_rate: settings.sample_rate,
            seed: settings.seed,
            instruments: instruments_of(&events),
        },
        audio: AudioInfo {
            encoding: ENCODING.to_owned(),
            channels: CHANNELS,
            frames: ((wav.len() - WAV_HEADER) / 4) as u64,
            peak: peak(&stereo),
            sha256: sha256_hex(&wav),
        },
    };
    Ok((manifest, wav))
}

impl Manifest {
    /// Every event of every layer, in order: the order in which they are rendered.
    pub fn events(&self) -> impl Iterator<Item = &Event> {
        self.layers.iter().flat_map(|l| &l.events)
    }

    /// The render settings the manifest records.
    pub fn settings(&self) -> RenderSettings {
        RenderSettings {
            sample_rate: self.render.sample_rate,
            seed: self.render.seed,
        }
    }

    /// Whether the manifest describes its own events, as [`write_manifest`]
    /// writes them: every layer's mapping passes [`Mapping::check`] and its
    /// events are that mapping applied to the events' own real times and
    /// values (within `1e-12 · max(1, |x|)` for each number, the last-bit
    /// differences between platforms); every layer's claim, if it has one, is
    /// one of the sources' claims; `render.instruments` is the events' distinct
    /// instruments in order of first use; the audio is `wav-pcm16` in 2
    /// channels; and the title, creator, licence and sources have the form
    /// `write_manifest` requires (not blank, no control characters, claim ids
    /// and DOIs well formed). The audio's frames, peak and hash are compared
    /// with the audio itself by [`verify`](crate::verify) and
    /// [`verify_wav`](crate::verify_wav).
    pub fn reproduces_events(&self) -> bool {
        check_info(&self.title, &self.creator, &self.licence, &self.sources).is_ok()
            && self.audio.encoding == ENCODING
            && self.audio.channels == CHANNELS
            && self.render.instruments == instruments_of(self.events())
            && self.layers.iter().all(|layer| {
                let (times, values): (Vec<f64>, Vec<f64>) =
                    layer.events.iter().map(|e| (e.real_s, e.value)).unzip();
                cited(&self.sources, layer.claim.as_deref())
                    && sonify(&layer.mapping, &times, &values, layer.claim.as_deref()).is_ok_and(
                        |again| {
                            again.len() == layer.events.len()
                                && again
                                    .iter()
                                    .zip(&layer.events)
                                    .all(|(a, b)| same_event(a, b))
                        },
                    )
            })
    }

    /// The manifest as pretty-printed JSON (two-space indent). Numbers are
    /// written in their shortest exact form, so [`from_json`](Self::from_json)
    /// gives back the same manifest.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Reads a manifest from JSON; its `schema` must be [`MANIFEST_SCHEMA`].
    pub fn from_json(text: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(text)?;
        if manifest.schema != MANIFEST_SCHEMA {
            return Err(SonifyError::text(
                "schema",
                MANIFEST_SCHEMA,
                &manifest.schema,
            ));
        }
        Ok(manifest)
    }
}
