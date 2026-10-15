//! MP4 recording of the mirrored H.264 stream.
//!
//! The daemon already sends H.264 in AVCC framing (4-byte length-prefixed NALs)
//! and prepends SPS/PPS to every keyframe, so recording is a mux and not an
//! encode: samples are copied into the file untouched and the parameter sets go
//! into the track's avcC. Nothing is re-encoded, so the cost per frame is one
//! buffered write.
//!
//! The wire format carries no timestamps, so sample durations come from arrival
//! time on this side, which is the clock the viewer sees. The device encodes
//! with frame reordering disabled, so samples arrive in presentation order and
//! can be written as they come.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;
use mp4::{AvcConfig, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType};

use crate::protocol::{VIDEO_FLAG_CONFIG, VIDEO_FLAG_H264, VIDEO_FLAG_KEYFRAME};

/// Sample timescale. 90 kHz is the MPEG convention, and it lands the device's
/// capture cadences on whole ticks (45 fps = 2000, 30 fps = 3000, 60 fps = 1500).
const TIMESCALE: u32 = 90_000;
/// Movie timescale, used for the mvhd/tkhd durations.
const MOVIE_TIMESCALE: u32 = 1_000;
/// Duration for the last sample when no successor tells us how long it was.
const LAST_SAMPLE_FALLBACK: u32 = 2_000; // 1/45 s
const NANOS_PER_SEC: u128 = 1_000_000_000;

/// Writes the mirrored stream to an MP4 file. Frames that arrive before the
/// first keyframe are dropped, since an MP4 video track has to start on one.
pub struct Recorder {
    path: PathBuf,
    writer: Option<TrackWriter>,
    /// Reset when the track opens, so the first written sample lands at t=0.
    origin: Instant,
    /// The newest sample, held back until the next one reveals its duration.
    pending: Option<Pending>,
    last_duration: u32,
    samples: u64,
    bytes: u64,
    skipped: u64,
}

struct TrackWriter {
    mp4: Mp4Writer<BufWriter<File>>,
    track_id: u32,
}

struct Pending {
    start: u64,
    bytes: Vec<u8>,
    sync: bool,
}

impl Recorder {
    /// Prepare a recording to `path`. The file is created (truncating an older
    /// one) right away so a bad path fails before the session starts. The MP4
    /// header itself waits for the first keyframe.
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| {
                format!("could not create the recording directory {}", dir.display())
            })?;
        }
        File::create(path)
            .with_context(|| format!("could not create the recording file {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: None,
            origin: Instant::now(),
            pending: None,
            last_duration: 0,
            samples: 0,
            bytes: 0,
            skipped: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// True once the track is open and frames are going into the file.
    pub fn is_recording(&self) -> bool {
        self.writer.is_some()
    }

    /// Feed one video frame exactly as it came off the wire. Non-H.264 frames
    /// (MJPEG) are ignored, plus anything that arrives before a keyframe.
    pub fn push(&mut self, width: u32, height: u32, flags: u32, avcc: &[u8]) -> Result<()> {
        if flags & VIDEO_FLAG_H264 == 0 {
            return Ok(());
        }
        if self.writer.is_none() {
            // A track can only start on a keyframe, and the muxer needs SPS/PPS
            // for the avcC: the daemon sends both together on every IDR.
            if flags & VIDEO_FLAG_KEYFRAME == 0 || flags & VIDEO_FLAG_CONFIG == 0 {
                self.skipped += 1;
                return Ok(());
            }
            let Some((sps, pps)) = parameter_sets(avcc) else {
                self.skipped += 1;
                return Ok(());
            };
            self.open(width, height, sps, pps)?;
        }

        // The previous sample's duration is the gap up to this frame, so a stall
        // shows up as a longer-held frame rather than a jump cut.
        let start = self.ticks();
        if let Some(previous) = self.pending.take() {
            let duration = start.saturating_sub(previous.start).max(1) as u32;
            self.last_duration = duration;
            self.write_sample(previous, duration)?;
        }
        self.pending = Some(Pending {
            start,
            bytes: avcc.to_vec(),
            sync: flags & VIDEO_FLAG_KEYFRAME != 0,
        });
        Ok(())
    }

    /// Flush the last sample and write the index (`moov`), which is what makes
    /// the file playable.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(previous) = self.pending.take() {
            let duration = if self.last_duration > 0 {
                self.last_duration
            } else {
                LAST_SAMPLE_FALLBACK
            };
            self.write_sample(previous, duration)?;
        }

        let Some(mut writer) = self.writer.take() else {
            // No usable frame ever arrived, so don't leave an empty file behind.
            let _ = std::fs::remove_file(&self.path);
            crate::warn!(
                "recording: no keyframe arrived, nothing was written to {}",
                self.path.display()
            );
            return Ok(());
        };

        writer
            .mp4
            .write_end()
            .context("could not finalize the MP4 index")?;
        // Mp4Writer owns a BufWriter, and dropping it would swallow a failed
        // flush, so take it back and flush it by hand.
        writer
            .mp4
            .into_writer()
            .flush()
            .context("could not flush the recording to disk")?;

        crate::info!(
            "recording saved: {} ({} frames, {:.1} MB)",
            self.path.display(),
            self.samples,
            self.bytes as f64 / 1_048_576.0
        );
        Ok(())
    }

    fn ticks(&self) -> u64 {
        (self.origin.elapsed().as_nanos() * TIMESCALE as u128 / NANOS_PER_SEC) as u64
    }

    fn open(&mut self, width: u32, height: u32, sps: Vec<u8>, pps: Vec<u8>) -> Result<()> {
        let file = File::create(&self.path)
            .with_context(|| format!("could not open {} for recording", self.path.display()))?;
        let config = Mp4Config {
            major_brand: "isom".parse().unwrap(),
            minor_version: 512,
            compatible_brands: ["isom", "iso2", "avc1", "mp41"]
                .iter()
                .map(|b| b.parse().unwrap())
                .collect(),
            timescale: MOVIE_TIMESCALE,
        };
        let mut mp4 = Mp4Writer::write_start(BufWriter::new(file), &config)
            .context("could not start the MP4")?;
        mp4.add_track(&TrackConfig {
            track_type: TrackType::Video,
            timescale: TIMESCALE,
            language: "und".to_string(),
            media_conf: MediaConfig::AvcConfig(AvcConfig {
                width: width as u16,
                height: height as u16,
                seq_param_set: sps,
                pic_param_set: pps,
            }),
        })
        .context("could not add the video track")?;

        self.writer = Some(TrackWriter { mp4, track_id: 1 });
        self.origin = Instant::now();
        crate::info!(
            "recording to {} ({width}x{height} H.264 passthrough, no re-encode)",
            self.path.display()
        );
        Ok(())
    }

    fn write_sample(&mut self, sample: Pending, duration: u32) -> Result<()> {
        let payload = Bytes::from(sample.bytes);
        let len = payload.len() as u64;
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        writer
            .mp4
            .write_sample(
                writer.track_id,
                &Mp4Sample {
                    start_time: sample.start,
                    duration,
                    rendering_offset: 0,
                    is_sync: sample.sync,
                    bytes: payload,
                },
            )
            .context("could not write a video sample")?;
        self.samples += 1;
        self.bytes += len;
        Ok(())
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // A file without its moov does not play, so close the track even when the
        // session ends without a clean stop.
        if self.writer.is_some() {
            let _ = self.finish();
        }
    }
}

/// Pull SPS (type 7) and PPS (type 8) out of an AVCC sample, which is what the
/// muxer needs for the avcC box. `None` if either is missing or the framing is
/// malformed.
fn parameter_sets(avcc: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let (mut sps, mut pps) = (None, None);
    let mut i = 0;
    while i + 4 <= avcc.len() {
        let len = u32::from_be_bytes([avcc[i], avcc[i + 1], avcc[i + 2], avcc[i + 3]]) as usize;
        i += 4;
        if len == 0 || i + len > avcc.len() {
            return None;
        }
        let nal = &avcc[i..i + len];
        match nal[0] & 0x1f {
            7 => sps = Some(nal.to_vec()),
            8 => pps = Some(nal.to_vec()),
            _ => {}
        }
        i += len;
    }
    if i != avcc.len() {
        return None;
    }
    match (sps, pps) {
        (Some(sps), Some(pps)) => Some((sps, pps)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp4::Mp4Reader;
    use std::io::BufReader;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Frame an AVCC sample out of raw NAL bodies.
    fn avcc(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            out.extend_from_slice(nal);
        }
        out
    }

    fn temp_path() -> PathBuf {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ioscpy-record-test-{}-{n}.mp4", std::process::id()))
    }

    fn no_params() -> Vec<u8> {
        // A non-IDR slice NAL with a one byte body.
        avcc(&[&[0x41, 0x9a]])
    }

    fn keyframe() -> Vec<u8> {
        // Leads with SPS/PPS, exactly like the daemon does on every IDR.
        avcc(&[
            &[0x67, 0x42, 0xc0, 0x1e],
            &[0x68, 0xce, 0x3c, 0x80],
            &[0x65, 0x88],
        ])
    }

    #[test]
    fn parameter_sets_only_come_from_frames_that_carry_them() {
        assert!(parameter_sets(&keyframe()).is_some());
        assert!(parameter_sets(&no_params()).is_none());
        // Truncated framing is rejected rather than guessed at.
        assert!(parameter_sets(&[0, 0, 0, 9, 0x67]).is_none());
    }

    #[test]
    fn recording_waits_for_a_keyframe_and_produces_a_playable_file() {
        let path = temp_path();
        {
            let mut rec = Recorder::new(&path).unwrap();
            // A P frame before the first keyframe is dropped, not written.
            rec.push(64, 48, VIDEO_FLAG_H264, &no_params()).unwrap();
            assert!(!rec.is_recording());

            rec.push(
                64,
                48,
                VIDEO_FLAG_H264 | VIDEO_FLAG_KEYFRAME | VIDEO_FLAG_CONFIG,
                &keyframe(),
            )
            .unwrap();
            assert!(rec.is_recording());
            rec.push(64, 48, VIDEO_FLAG_H264, &no_params()).unwrap();
            rec.finish().unwrap();
        }

        let file = File::open(&path).unwrap();
        let size = file.metadata().unwrap().len();
        let mut mp4 = Mp4Reader::read_header(BufReader::new(file), size).unwrap();
        assert_eq!(mp4.tracks().len(), 1);
        let track = mp4.tracks().values().next().unwrap();
        let track_id = track.track_id();
        assert_eq!(track.width(), 64);
        assert_eq!(track.height(), 48);
        assert_eq!(track.sample_count(), 2);
        // The keyframe is flagged as a sync sample; the P frame is not.
        assert!(mp4.read_sample(track_id, 1).unwrap().unwrap().is_sync);
        assert!(!mp4.read_sample(track_id, 2).unwrap().unwrap().is_sync);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_empty_recording_does_not_leave_a_file_behind() {
        let path = temp_path();
        {
            let mut rec = Recorder::new(&path).unwrap();
            rec.push(64, 48, VIDEO_FLAG_H264, &no_params()).unwrap();
            rec.finish().unwrap();
        }
        assert!(!path.exists());
    }
}
