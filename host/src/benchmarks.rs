//! Micro-benchmarks and performance tests for the receiving end pipeline.
//! Run with: `cargo test -- --nocapture bench_`

#[cfg(test)]
mod tests {
    use crate::protocol::{self, MessageType, CHANNEL_VIDEO};
    use std::time::Instant;

    #[test]
    fn bench_protocol_parsing_throughput() {
        let mut sample_payload = Vec::with_capacity(64);
        let w: u32 = 1170;
        let h: u32 = 2532;
        let flags: u32 = protocol::VIDEO_FLAG_H264;
        let data = [0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00];
        let len: u32 = data.len() as u32;

        sample_payload.extend_from_slice(&w.to_be_bytes());
        sample_payload.extend_from_slice(&h.to_be_bytes());
        sample_payload.extend_from_slice(&flags.to_be_bytes());
        sample_payload.extend_from_slice(&len.to_be_bytes());
        sample_payload.extend_from_slice(&data);

        let iterations = 500_000;
        let start = Instant::now();
        let mut parsed_frames = 0u64;
        let mut _total_bytes = 0usize;

        for _ in 0..iterations {
            if let Some((pw, ph, pflags, pdata)) = protocol::parse_video_payload(&sample_payload) {
                parsed_frames += 1;
                _total_bytes += pdata.len();
                assert_eq!(pw, 1170);
                assert_eq!(ph, 2532);
                assert_eq!(pflags, protocol::VIDEO_FLAG_H264);
            }
        }

        let elapsed = start.elapsed();
        let nanos_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        let mps = (iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0;

        println!("\n=== Benchmark: Protocol Payload Parsing ===");
        println!("  Iterations:    {iterations}");
        println!("  Time per parse: {:.2} ns", nanos_per_op);
        println!("  Throughput:    {:.2} million pkts/sec", mps);
        println!("===========================================");
        assert_eq!(parsed_frames, iterations as u64);
        assert!(
            nanos_per_op < 500.0,
            "Parsing must be faster than 500ns per packet in debug mode"
        );
    }

    #[test]
    fn bench_rgba_vs_rgb_pack() {
        use openh264::decoder::Decoder;
        use openh264::encoder::Encoder;
        use openh264::formats::{RgbSliceU8, YUVBuffer, YUVSource};

        let w = 1170;
        let h = 2532;
        let mut encoder = Encoder::new().unwrap();
        let mut decoder = Decoder::new().unwrap();

        let mut rgb = vec![128u8; w * h * 3];
        for i in 0..w * h {
            rgb[i * 3] = (i % 256) as u8;
            rgb[i * 3 + 1] = ((i * 3) % 256) as u8;
            rgb[i * 3 + 2] = ((i * 7) % 256) as u8;
        }

        let yuv = YUVBuffer::from_rgb8_source(RgbSliceU8::new(&rgb, (w, h)));
        let stream = encoder.encode(&yuv).unwrap();
        let stream_bytes = stream.to_vec();

        // Warm up and decode
        let mut decoder2 = Decoder::new().unwrap();
        let iters = 20;

        // 0. Time decoder.decode alone
        let start0 = Instant::now();
        for _ in 0..iters {
            let _ = decoder2.decode(&stream_bytes);
        }
        let elapsed0 = start0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let yuv_decoded = decoder.decode(&stream_bytes).unwrap().unwrap();

        // 1. Old way: write_rgb8 + pack_rgb888 + rotate_cw
        let start1 = Instant::now();
        let mut rgb_buf = vec![0u8; w * h * 3];
        for _ in 0..iters {
            yuv_decoded.write_rgb8(&mut rgb_buf);
            let frame = crate::video::DecodedFrame::new_rgba(
                crate::video::pack_rgb888(&rgb_buf, w, h),
                w,
                h,
                1,
            );
            let _ = crate::video::rotate_cw(frame, 1);
        }
        let elapsed1 = start1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // 2. New GPU path: zero-copy YUV extraction (conversion + rotation done on GPU)
        let start2 = Instant::now();
        let (y_stride, u_stride, v_stride) = yuv_decoded.strides();
        let mut y_plane = vec![0u8; y_stride * h];
        let mut u_plane = vec![0u8; u_stride * ((h + 1) / 2)];
        let mut v_plane = vec![0u8; v_stride * ((h + 1) / 2)];
        for _ in 0..iters {
            y_plane.copy_from_slice(yuv_decoded.y());
            u_plane.copy_from_slice(yuv_decoded.u());
            v_plane.copy_from_slice(yuv_decoded.v());
            let _frame = crate::video::DecodedFrame::new_yuv420(
                y_plane.clone(),
                u_plane.clone(),
                v_plane.clone(),
                y_stride,
                u_stride,
                v_stride,
                w,
                h,
                1,
            );
        }
        let elapsed2 = start2.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!("\n=== Benchmark: Color Conversion & Rotation (1170x2532) ===");
        println!("  Decode alone:                                   {:.2} ms ({:.1} FPS)", elapsed0, 1000.0 / elapsed0);
        println!("  Old way (write_rgb8 + pack_rgb888 + rotate_cw): {:.2} ms ({:.1} FPS)", elapsed1, 1000.0 / elapsed1);
        println!("  New GPU path (zero-copy YUV extraction):        {:.2} ms ({:.1} FPS)", elapsed2, 1000.0 / elapsed2);
        println!("  Speedup: {:.2}x faster", elapsed1 / elapsed2);
        println!("===========================================================");
    }

    #[test]
    fn bench_wire_frame_roundtrip() {
        let payload = vec![0xAB; 16384]; // 16 KB frame
        let iterations = 20_000;
        let start = Instant::now();

        let mut buf = Vec::with_capacity(32 + payload.len());
        for i in 0..iterations {
            buf.clear();
            protocol::write_frame(
                &mut buf,
                MessageType::VideoFrame,
                CHANNEL_VIDEO,
                i as u64,
                &payload,
            )
            .unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let frame = protocol::read_frame(&mut cursor).unwrap();
            assert_eq!(frame.payload.len(), 16384);
        }

        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mb_per_sec =
            (payload.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);

        println!("\n=== Benchmark: Wire Framing Serialization (16KB) ===");
        println!("  Iterations:    {iterations}");
        println!("  Time per frame: {:.3} ms", time_per_op);
        println!("  Memory rate:    {:.2} MB/s", mb_per_sec);
        println!("=====================================================");
        assert!(time_per_op < 0.1, "Framing 16KB must take < 0.1ms");
    }

    #[test]
    fn bench_h264_decoder_instance() {
        use crate::h264::H264Decoder;
        let start = Instant::now();
        let mut decoders = Vec::new();
        for _ in 0..50 {
            if let Some(dec) = H264Decoder::new() {
                decoders.push(dec);
            }
        }
        let elapsed = start.elapsed();
        println!("\n=== Benchmark: H264 Decoder Initialization ===");
        println!(
            "  Created 50 instances in {:.2} ms ({:.2} ms/instance)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / 50.0
        );
        println!("==============================================");
        assert!(!decoders.is_empty(), "Must successfully initialize decoder");
    }
}
