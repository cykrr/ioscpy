// ioscpyctl: small on-device helper for status, diagnostics, and the privileged
// maintenance actions the Settings pane and the host driver rely on.

#import <Foundation/Foundation.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <arpa/inet.h>
#import <unistd.h>
#import <spawn.h>
#import <sys/wait.h>
#import <errno.h>

#import "Protocol.h"
#import "Detect.h"
#import "Paths.h"

extern char **environ;

static int connectLoopback(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// Handshake with the daemon and return the parsed HELLO_ACK, or nil.
static NSDictionary *fetchHandshake(uint16_t port) {
    int fd = connectLoopback(port);
    if (fd < 0) {
        return nil;
    }
    NSDictionary *hello = @{
        @"role": @"ctl",
        @"host_version": @"0.1.5",
        @"protocol_version": @(IOSPY_PROTOCOL_VERSION),
        @"nonce": @"00",
    };
    NSData *body = [NSJSONSerialization dataWithJSONObject:hello options:0 error:nil];
    NSDictionary *result = nil;
    if (IOSPYWriteFrame(fd, IOSPYMsgHello, IOSPY_CHANNEL_CONTROL, 0, body)) {
        IOSPYFrameHeader hdr;
        NSData *payload = nil;
        if (IOSPYReadFrame(fd, &hdr, &payload) && hdr.type == IOSPYMsgHelloAck) {
            result = [NSJSONSerialization JSONObjectWithData:payload options:0 error:nil];
        }
    }
    close(fd);
    return result;
}

static void printDeviceInfo(void) {
    printf("model       %s\n", IOSPYDeviceModel().UTF8String);
    printf("ios         %s\n", IOSPYSystemVersion().UTF8String);
    printf("jailbreak   %s\n", IOSPYLayoutName(IOSPYDetectLayout()).UTF8String);
    NSString *prefix = IOSPYJBPrefix();
    printf("prefix      %s\n", prefix.length ? prefix.UTF8String : "/");
    printf("injection   %s\n", IOSPYInjectionFramework().UTF8String);
}

static int cmdStatus(void) {
    printDeviceInfo();
    NSDictionary *ack = fetchHandshake(IOSPY_DEFAULT_PORT);
    if (ack) {
        printf("daemon      running (version %s, protocol %s)\n",
               [ack[@"daemon_version"] description].UTF8String,
               [ack[@"protocol_version"] description].UTF8String);
    } else {
        printf("daemon      not reachable on 127.0.0.1:%u\n", IOSPY_DEFAULT_PORT);
    }
    BOOL hook = [[NSFileManager defaultManager]
        fileExistsAtPath:IOSPYPath(@"/Library/MobileSubstrate/DynamicLibraries/ioscpyhook.dylib")];
    printf("tweak       %s\n", hook ? "installed" : "missing");
    return ack ? 0 : 1;
}

static int cmdCapabilities(void) {
    NSDictionary *ack = fetchHandshake(IOSPY_DEFAULT_PORT);
    if (!ack) {
        fprintf(stderr, "daemon not reachable\n");
        return 1;
    }
    NSData *pretty = [NSJSONSerialization dataWithJSONObject:ack[@"capabilities"]
                                                     options:NSJSONWritingPrettyPrinted
                                                       error:nil];
    fwrite(pretty.bytes, 1, pretty.length, stdout);
    printf("\n");
    return 0;
}

// Run a shell command (system() isn't available on iOS). PATH points at the
// prefix bins so tools like launchctl, tar, and chmod resolve under any layout.
static int runShell(NSString *command) {
    NSString *prefix = IOSPYJBPrefix();
    NSString *wrapped = [NSString stringWithFormat:
        @"export PATH=%@/usr/bin:%@/bin:%@/usr/sbin:%@/sbin:/usr/bin:/bin:/usr/sbin:/sbin; %@",
        prefix, prefix, prefix, prefix, command];

    NSString *sh = IOSPYPath(@"/bin/sh");
    const char *argv[] = {sh.UTF8String, "-c", wrapped.UTF8String, NULL};
    pid_t pid = 0;
    int rc = posix_spawn(&pid, sh.UTF8String, NULL, NULL, (char *const *)argv, environ);
    if (rc != 0) {
        return 1;
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != pid) {
        return 1; // never reaped the child, don't claim success
    }
    return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : 1;
}

static int cmdRestartDaemon(void) {
    // Reload through launchd; fall back to a fresh bootstrap if it isn't loaded.
    NSString *plist = IOSPYPath(@"/Library/LaunchDaemons/com.ioscpy.daemon.plist");
    NSString *cmd = [NSString stringWithFormat:
        @"launchctl kickstart -k system/com.ioscpy.daemon 2>/dev/null || launchctl bootstrap system %@",
        plist];
    return runShell(cmd);
}

static int cmdReloadHooks(void) {
    return runShell(@"sbreload 2>/dev/null || killall -9 SpringBoard");
}

static int cmdRepairPermissions(void) {
    NSString *cmd = [NSString stringWithFormat:
        @"chmod 755 %@ %@ 2>/dev/null; chown root:wheel %@ %@ 2>/dev/null",
        IOSPYPath(@"/usr/bin/ioscpyd"), IOSPYPath(@"/usr/bin/ioscpyctl"),
        IOSPYPath(@"/usr/bin/ioscpyd"), IOSPYPath(@"/usr/bin/ioscpyctl")];
    return runShell(cmd);
}

static int cmdExportDiagnostics(void) {
    NSString *log = IOSPYPath(@"/var/log/ioscpy");
    NSString *out = @"/tmp/ioscpy-diagnostics.tar.gz";
    NSString *cmd = [NSString stringWithFormat:@"tar czf %@ %@ 2>/dev/null", out, log];
    int rc = runShell(cmd);
    if (rc == 0) {
        printf("%s\n", out.UTF8String);
    }
    return rc;
}

static int cmdBench(int seconds) {
    if (seconds <= 0) seconds = 5;
    printf("Starting on-device stream benchmark for %d seconds...\n", seconds);
    int fd = connectLoopback(IOSPY_DEFAULT_PORT);
    if (fd < 0) {
        printf("Error: Could not connect to ioscpyd on 127.0.0.1:%d\n", IOSPY_DEFAULT_PORT);
        return 1;
    }
    // Handshake
    NSDictionary *hello = @{
        @"role": @"ctl-bench",
        @"host_version": @"0.1.5",
        @"protocol_version": @(IOSPY_PROTOCOL_VERSION),
        @"nonce": @"00",
    };
    NSData *body = [NSJSONSerialization dataWithJSONObject:hello options:0 error:nil];
    if (!IOSPYWriteFrame(fd, IOSPYMsgHello, IOSPY_CHANNEL_CONTROL, 0, body)) {
        printf("Error: Failed to send HELLO\n");
        close(fd);
        return 1;
    }
    IOSPYFrameHeader hdr;
    NSData *payload = nil;
    if (!IOSPYReadFrame(fd, &hdr, &payload) || hdr.type != IOSPYMsgHelloAck) {
        printf("Error: Failed to receive HELLO_ACK\n");
        close(fd);
        return 1;
    }
    printf("Connected to daemon. Starting H.264 stream benchmark...\n");
    uint8_t codec = IOSPY_VIDEO_CODEC_H264;
    NSData *codecData = [NSData dataWithBytes:&codec length:1];
    IOSPYWriteFrame(fd, IOSPYMsgStartStream, IOSPY_CHANNEL_CONTROL, 0, codecData);

    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    CFAbsoluteTime lastFrameTime = start;
    uint64_t frameCount = 0;
    uint64_t totalBytes = 0;
    double maxInterval = 0;
    double minInterval = 999.0;

    while (CFAbsoluteTimeGetCurrent() - start < seconds) {
        if (!IOSPYReadFrame(fd, &hdr, &payload)) {
            break;
        }
        CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
        if (hdr.type == IOSPYMsgVideoFrame && payload.length >= 16) {
            frameCount++;
            totalBytes += payload.length;
            if (frameCount > 1) {
                double interval = (now - lastFrameTime) * 1000.0;
                if (interval > maxInterval) maxInterval = interval;
                if (interval < minInterval) minInterval = interval;
            }
            lastFrameTime = now;
        } else if (hdr.type == IOSPYMsgLog && payload.length > 0) {
            NSDictionary *log = [NSJSONSerialization JSONObjectWithData:payload options:0 error:nil];
            if (log) {
                printf("  [device perf] %s\n", [log[@"message"] UTF8String]);
            }
        }
    }
    IOSPYWriteFrame(fd, IOSPYMsgStopStream, IOSPY_CHANNEL_CONTROL, 0, nil);
    close(fd);

    double totalElapsed = CFAbsoluteTimeGetCurrent() - start;
    double fps = (double)frameCount / totalElapsed;
    double mbps = (double)totalBytes * 8.0 / (totalElapsed * 1000000.0);
    printf("\n=== Device Stream Benchmark Results ===\n");
    printf("Duration:        %.2f s\n", totalElapsed);
    printf("Frames captured: %llu\n", (unsigned long long)frameCount);
    printf("Framerate:       %.2f FPS\n", fps);
    printf("Bitrate:         %.2f Mbps (%.1f KB/frame)\n", mbps, frameCount ? (double)totalBytes / frameCount / 1024.0 : 0.0);
    if (frameCount > 1) {
        printf("Frame interval:  min=%.2f ms, avg=%.2f ms, max=%.2f ms\n", minInterval, 1000.0 / fps, maxInterval);
    }
    printf("=======================================\n");
    return 0;
}

static void usage(void) {
    printf("usage: ioscpyctl <command>\n");
    printf("  status              device + daemon + tweak summary\n");
    printf("  capabilities        capability map reported by the daemon\n");
    printf("  bench [seconds]     benchmark live capture & stream speed on device\n");
    printf("  print-device-info   jailbreak layout and device facts\n");
    printf("  restart-daemon      reload ioscpyd through launchd\n");
    printf("  reload-hooks        respring to reload the tweak\n");
    printf("  repair-permissions  fix exec bits / ownership\n");
    printf("  export-diagnostics  bundle logs into /tmp\n");
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            usage();
            return 2;
        }
        NSString *cmd = [NSString stringWithUTF8String:argv[1]];
        if ([cmd isEqualToString:@"status"]) {
            return cmdStatus();
        } else if ([cmd isEqualToString:@"capabilities"]) {
            return cmdCapabilities();
        } else if ([cmd isEqualToString:@"bench"]) {
            int secs = (argc >= 3) ? atoi(argv[2]) : 5;
            return cmdBench(secs);
        } else if ([cmd isEqualToString:@"print-device-info"]) {
            printDeviceInfo();
            return 0;
        } else if ([cmd isEqualToString:@"restart-daemon"]) {
            return cmdRestartDaemon();
        } else if ([cmd isEqualToString:@"reload-hooks"]) {
            return cmdReloadHooks();
        } else if ([cmd isEqualToString:@"repair-permissions"]) {
            return cmdRepairPermissions();
        } else if ([cmd isEqualToString:@"export-diagnostics"]) {
            return cmdExportDiagnostics();
        }
        usage();
        return 2;
    }
}
