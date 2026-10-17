// Hardware H.264 encoder built on VideoToolbox. Lives in the tweak next to the
// capture path so the screen pixels never leave SpringBoard uncompressed.
//
// Everything here is public VideoToolbox/CoreMedia/CoreVideo, so it works on any
// iPhone with a hardware H.264 encoder (every model we target). If a session
// can't be created on a given device/OS, encoding reports failure and the caller
// drops back to the JPEG path. Nothing is assumed about availability.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Whether the VideoToolbox encode path is plausibly present. The real test is
// creating a session, which the encoder reports per-frame.
BOOL IOSPYH264Available(void);

#ifdef __cplusplus
}
#endif

typedef void (^IOSPYEncodeHandler)(NSData *avcc, BOOL isKeyframe, BOOL failed);

@interface IOSPYH264Encoder : NSObject

// Encode a BGRA IOSurface to H.264 asynchronously. The hardware compression
// session produces frames in presentation order without pipeline stalls or
// synchronous thread blocking. The handler is invoked when hardware encoding
// completes.
- (BOOL)encodeSurface:(IOSurfaceRef)surface
                width:(int)width
               height:(int)height
                  fps:(int)fps
        forceKeyframe:(BOOL)forceKeyframe
              handler:(IOSPYEncodeHandler)handler;

// Tear down the underlying session (e.g. when the stream stops).
- (void)invalidate;

@end
