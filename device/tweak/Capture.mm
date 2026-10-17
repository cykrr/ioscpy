#import "Capture.h"
#import <UIKit/UIKit.h>
#import <IOSurface/IOSurfaceRef.h>
#import <ImageIO/ImageIO.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <CoreServices/UTCoreTypes.h>
#import <dlfcn.h>

// The render server can blit the live display straight into an IOSurface. It's
// a long-standing private QuartzCore entry point; we resolve it at runtime so a
// build of iOS that lacks it just reports "unavailable" instead of failing to
// load.
typedef void (*CARenderServerRenderDisplayFn)(uint32_t client, CFStringRef display,
                                               IOSurfaceRef surface, int x, int y);

static CARenderServerRenderDisplayFn renderDisplayFn(void) {
    static CARenderServerRenderDisplayFn fn = NULL;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        fn = (CARenderServerRenderDisplayFn)dlsym(RTLD_DEFAULT, "CARenderServerRenderDisplay");
    });
    return fn;
}

BOOL IOSPYCaptureAvailable(void) {
    return renderDisplayFn() != NULL;
}

static double nowMs(void) {
    return CFAbsoluteTimeGetCurrent() * 1000.0;
}

// Native screen size in pixels (cached; doesn't change at runtime).
static CGSize nativeScreenSize(void) {
    static CGSize size = {0, 0};
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        if ([NSThread isMainThread]) {
            size = [UIScreen mainScreen].nativeBounds.size;
        } else {
            __block CGSize s = CGSizeZero;
            dispatch_sync(dispatch_get_main_queue(), ^{
                s = [UIScreen mainScreen].nativeBounds.size;
            });
            size = s;
        }
    });
    return size;
}

// The render server addresses displays by name; the main display's name varies
// by device, so read it rather than hardcoding "LCD".
static CFStringRef mainDisplayName(void) {
    static CFStringRef name = NULL;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSString *resolved = nil;
        Class displayClass = NSClassFromString(@"CADisplay");
        if (displayClass && [displayClass respondsToSelector:@selector(mainDisplay)]) {
            id display = [displayClass performSelector:@selector(mainDisplay)];
            if ([display respondsToSelector:@selector(name)]) {
                resolved = [display performSelector:@selector(name)];
            }
        }
        if (resolved.length == 0) {
            resolved = @"LCD";
        }
        name = (CFStringRef)CFBridgingRetain(resolved);
    });
    return name;
}

// 3-surface ring buffer so the CoreAnimation GPU render and VideoToolbox hardware
// encoder run concurrently in a pipelined fashion without waiting on each other.
#define IOSPY_SURFACE_POOL_SIZE 3
static IOSurfaceRef gSurfaces[IOSPY_SURFACE_POOL_SIZE] = {NULL, NULL, NULL};
static int gSurfaceIndex = 0;
static int cachedW = 0, cachedH = 0;

static IOSurfaceRef nextSurfaceForSize(int width, int height) {
    if (cachedW != width || cachedH != height) {
        for (int i = 0; i < IOSPY_SURFACE_POOL_SIZE; i++) {
            if (gSurfaces[i]) {
                CFRelease(gSurfaces[i]);
                gSurfaces[i] = NULL;
            }
        }
        cachedW = width;
        cachedH = height;
    }
    int idx = gSurfaceIndex;
    gSurfaceIndex = (gSurfaceIndex + 1) % IOSPY_SURFACE_POOL_SIZE;
    if (!gSurfaces[idx]) {
        NSDictionary *props = @{
            (id)kIOSurfaceWidth: @(width),
            (id)kIOSurfaceHeight: @(height),
            (id)kIOSurfaceBytesPerElement: @4,
            (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA'),
        };
        gSurfaces[idx] = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    }
    return gSurfaces[idx];
}

// Fast path: CARenderServer blits the native screen straight into a GPU IOSurface,
// which is then wrapped directly by VideoToolbox for hardware H.264 encoding with
// zero-copy and zero CPU usage.
IOSurfaceRef IOSPYCaptureScreenSurface(CGFloat maxDimension, int *outWidth, int *outHeight) {
    @autoreleasepool {
        CARenderServerRenderDisplayFn render = renderDisplayFn();
        if (!render) {
            return NULL;
        }
        CGSize native = nativeScreenSize();
        if (native.width < 1 || native.height < 1) {
            return NULL;
        }
        // Round down to even dimensions for 4:2:0 H.264.
        int nw = ((int)native.width) & ~1;
        int nh = ((int)native.height) & ~1;
        if (nw < 2 || nh < 2) {
            return NULL;
        }

        IOSurfaceRef src = nextSurfaceForSize(nw, nh);
        if (!src) {
            return NULL;
        }

        render(0, mainDisplayName(), src, 0, 0);

        if (outWidth) {
            *outWidth = nw;
        }
        if (outHeight) {
            *outHeight = nh;
        }
        return src;
    }
}

NSData *IOSPYCaptureScreenJPEG(CGFloat maxDimension, CGFloat quality,
                               int *outWidth, int *outHeight,
                               double *outRenderMs, double *outEncodeMs) {
    @autoreleasepool {
        CARenderServerRenderDisplayFn render = renderDisplayFn();
        if (!render) {
            return nil;
        }

        CGSize native = nativeScreenSize();
        if (native.width < 1 || native.height < 1) {
            return nil;
        }
        int nw = (int)native.width;
        int nh = (int)native.height;

        // The render server blits 1:1 and clips to the surface, so capture the
        // whole screen at native size, then downscale the concrete pixels.
        CGSize target = native;
        if (maxDimension > 0) {
            CGFloat longest = MAX(native.width, native.height);
            if (longest > maxDimension) {
                CGFloat factor = maxDimension / longest;
                target = CGSizeMake(round(native.width * factor), round(native.height * factor));
            }
        }
        int tw = (int)target.width;
        int th = (int)target.height;

        IOSurfaceRef surface = nextSurfaceForSize(nw, nh);
        if (!surface) {
            return nil;
        }

        const uint32_t bgra = kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little;
        CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();

        double renderStart = nowMs();
        render(0, mainDisplayName(), surface, 0, 0);

        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
        void *base = IOSurfaceGetBaseAddress(surface);
        size_t bytesPerRow = IOSurfaceGetBytesPerRow(surface);

        // Wrap the surface memory without copying, then draw it (downscaled, cheap
        // interpolation) into the target bitmap.
        CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, base, bytesPerRow * nh, NULL);
        CGImageRef nativeImage = CGImageCreate(nw, nh, 8, 32, bytesPerRow, space, bgra, provider,
                                               NULL, false, kCGRenderingIntentDefault);
        CGContextRef ctx = CGBitmapContextCreate(NULL, tw, th, 8, 0, space, bgra);
        CGImageRef image = NULL;
        if (ctx && nativeImage) {
            CGContextSetInterpolationQuality(ctx, kCGInterpolationLow);
            CGContextDrawImage(ctx, CGRectMake(0, 0, tw, th), nativeImage);
            image = CGBitmapContextCreateImage(ctx);
        }
        if (ctx) {
            CGContextRelease(ctx);
        }
        CGImageRelease(nativeImage);
        CGDataProviderRelease(provider);
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        CGColorSpaceRelease(space);
        if (outRenderMs) {
            *outRenderMs = nowMs() - renderStart;
        }
        if (!image) {
            return nil;
        }

        double encodeStart = nowMs();
        NSMutableData *data = [NSMutableData data];
        CGImageDestinationRef dest =
            CGImageDestinationCreateWithData((__bridge CFMutableDataRef)data, kUTTypeJPEG, 1, NULL);
        BOOL ok = NO;
        if (dest) {
            NSDictionary *options = @{(__bridge id)kCGImageDestinationLossyCompressionQuality: @(quality)};
            CGImageDestinationAddImage(dest, image, (__bridge CFDictionaryRef)options);
            ok = CGImageDestinationFinalize(dest);
            CFRelease(dest);
        }
        CGImageRelease(image);
        if (outEncodeMs) {
            *outEncodeMs = nowMs() - encodeStart;
        }

        if (!ok) {
            return nil;
        }
        if (outWidth) {
            *outWidth = tw;
        }
        if (outHeight) {
            *outHeight = th;
        }
        return data;
    }
}
