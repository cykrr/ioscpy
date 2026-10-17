//! Micro-benchmarks and performance tests for the receiving end pipeline.
//! Run with: `cargo test -- --nocapture bench_`

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use crate::protocol::{self, MessageType, CHANNEL_VIDEO};

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
        assert!(nanos_per_op < 500.0, "Parsing must be faster than 500ns per packet in debug mode");
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
            ).unwrap();

            let mut cursor = std::io::Cursor::new(&buf);
            let frame = protocol::read_frame(&mut cursor).unwrap();
            assert_eq!(frame.payload.len(), 16384);
        }

        let elapsed = start.elapsed();
        let time_per_op = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let mb_per_sec = (payload.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);

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
        println!("  Created 50 instances in {:.2} ms ({:.2} ms/instance)",
                 elapsed.as_secs_f64() * 1000.0,
                 elapsed.as_secs_f64() * 1000.0 / 50.0);
        println!("==============================================");
        assert!(!decoders.is_empty(), "Must successfully initialize decoder");
    }
}
