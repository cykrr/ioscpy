//! Video decode. Turns a JPEG or H.264 frame into a DecodedFrame ready for
//! rendering. Planar YUV 4:2:0 is preserved for direct GPU shader conversion,
//! bypassing slow CPU color packing and rotation.

use zune_jpeg::JpegDecoder;

/// Underlying pixel buffer payload of a decoded video frame.
#[derive(Clone)]
pub enum FrameData {
    /// 32-bit packed BGRA/RGBA pixels (0x00RRGGBB). Used for JPEG and macOS VideoToolbox.
    Rgba(Vec<u32>),
    /// Planar YUV 4:2:0. Used for Linux/Windows OpenH264 decoder with zero CPU conversion.
    Yuv420 {
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
        y_stride: usize,
        u_stride: usize,
        v_stride: usize,
    },
}

/// A decoded frame carrying image data, native capture dimensions, and orientation.
#[derive(Clone)]
pub struct DecodedFrame {
    pub data: FrameData,
    pub width: usize,
    pub height: usize,
    pub orientation: u8,
}

impl DecodedFrame {
    pub fn new_rgba(buf: Vec<u32>, width: usize, height: usize, orientation: u8) -> Self {
        Self {
            data: FrameData::Rgba(buf),
            width,
            height,
            orientation,
        }
    }

    pub fn new_yuv420(
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
        y_stride: usize,
        u_stride: usize,
        v_stride: usize,
        width: usize,
        height: usize,
        orientation: u8,
    ) -> Self {
        Self {
            data: FrameData::Yuv420 {
                y,
                u,
                v,
                y_stride,
                u_stride,
                v_stride,
            },
            width,
            height,
            orientation,
        }
    }

    /// Display width after accounting for upright orientation turns.
    #[inline]
    pub fn display_width(&self) -> usize {
        let turns = upright_turns(self.orientation);
        if turns & 1 == 1 {
            self.height
        } else {
            self.width
        }
    }

    /// Display height after accounting for upright orientation turns.
    #[inline]
    pub fn display_height(&self) -> usize {
        let turns = upright_turns(self.orientation);
        if turns & 1 == 1 {
            self.width
        } else {
            self.height
        }
    }
}

/// Clockwise quarter-turns needed to show a captured orientation upright
/// (1=portrait, 2=upsideDown, 3=landscapeLeft, 4=landscapeRight). These match the
/// device's touch-rotation labels, so display and touch stay in step.
pub fn upright_turns(orientation: u8) -> u8 {
    match orientation {
        2 => 2, // upside down
        3 => 1, // landscape left
        4 => 3, // landscape right
        _ => 0, // portrait or unknown, already upright
    }
}

/// Rotate a decoded frame `turns` clockwise quarter-turns (0..=3). 0 is a no-op.
/// In the modern GPU pipeline rotation is performed in the vertex shader, so this
/// simply adjusts the orientation metadata in O(1) time without copying memory.
#[allow(dead_code)]
pub fn rotate_cw(mut frame: DecodedFrame, turns: u8) -> DecodedFrame {
    if (turns & 3) == 0 {
        return frame;
    }
    let cur_turns = upright_turns(frame.orientation);
    let new_turns = (cur_turns + (turns & 3)) & 3;
    frame.orientation = match new_turns {
        1 => 3, // landscape left (turns 1)
        2 => 2, // upside down (turns 2)
        3 => 4, // landscape right (turns 3)
        _ => 1, // portrait (turns 0)
    };
    frame
}

/// Pack a tightly-packed RGB888 buffer into the window's `0x00RRGGBB` layout.
/// `rgb.len()` must be at least `width * height * 3`; extra bytes are ignored.
pub fn pack_rgb888(rgb: &[u8], width: usize, height: usize) -> Vec<u32> {
    let mut buf = vec![0u32; width * height];
    for (i, px) in buf.iter_mut().enumerate() {
        let r = rgb[i * 3] as u32;
        let g = rgb[i * 3 + 1] as u32;
        let b = rgb[i * 3 + 2] as u32;
        *px = (r << 16) | (g << 8) | b;
    }
    buf
}

/// Decode a JPEG into a packed 0RGB buffer. Returns `None` on a bad frame.
/// Color JPEGs decode to RGB by default, which is what the packing expects.
pub fn decode_jpeg(jpeg: &[u8], orientation: u8) -> Option<DecodedFrame> {
    let mut decoder = JpegDecoder::new(jpeg);
    let pixels = decoder.decode().ok()?;
    let (width, height) = decoder.dimensions()?;

    if pixels.len() < width * height * 3 {
        return None;
    }

    let buf = pack_rgb888(&pixels, width, height);
    Some(DecodedFrame::new_rgba(buf, width, height, orientation))
}
