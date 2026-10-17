//! Render window and event-driven pipeline using winit and wgpu.
//!
//! Opens a native window sized from the first frame and streams video via a 2D GPU
//! texture and fullscreen quad shader with PresentMode::Immediate.
//! The network thread decodes frames and drops the latest one into a shared slot,
//! notifying the event loop to request redraws. Keyboard and mouse inputs are captured
//! and forwarded as device-space messages.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState, NamedKey, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::clipboard;
use crate::input::{map_to_norm, InputFrame};
use crate::protocol::{self, KeyCode as ProtoKeyCode, MessageType, SystemAction, TouchPhase};
use crate::sidebar;
use crate::video::DecodedFrame;

/// Where the network thread drops the most recent decoded frame, alongside frame tracking stats.
#[derive(Default)]
pub struct FrameSlotInner {
    pub frame: Option<DecodedFrame>,
    pub incoming: u64,
    pub dropped: u64,
    pub waker: Option<Arc<dyn Fn() + Send + Sync>>,
}

pub type FrameSlot = Arc<Mutex<FrameSlotInner>>;

pub fn new_frame_slot() -> FrameSlot {
    Arc::new(Mutex::new(FrameSlotInner::default()))
}

#[derive(Debug)]
pub enum UserEvent {
    NewFrame,
    RemoteClipboard(String),
}

fn send_action(tx: &Sender<InputFrame>, action: SystemAction) {
    let _ = tx.send(InputFrame::new(
        MessageType::SystemAction,
        protocol::encode_system_action(action),
    ));
}

fn send_key(tx: &Sender<InputFrame>, code: ProtoKeyCode) {
    let _ = tx.send(InputFrame::new(
        MessageType::InputKey,
        protocol::encode_key(code),
    ));
}

fn send_touch(tx: &Sender<InputFrame>, phase: TouchPhase, x: f32, y: f32) {
    let _ = tx.send(InputFrame::new(
        MessageType::InputTouch,
        protocol::encode_touch(phase, 0, x, y),
    ));
}

/// Translate a sidebar button press into the same messages its matching
/// keyboard shortcut sends.
fn dispatch_sidebar_action(
    tx: &Sender<InputFrame>,
    clip: &Arc<Mutex<ClipState>>,
    action: sidebar::Action,
) {
    use sidebar::Action::*;
    match action {
        Home => send_action(tx, SystemAction::Home),
        Lock => send_action(tx, SystemAction::Lock),
        AppSwitcher => send_action(tx, SystemAction::AppSwitcher),
        Rotate => send_action(tx, SystemAction::RotateLeft),
        Back => send_action(tx, SystemAction::Back),
        SelectAll => send_key(tx, ProtoKeyCode::SelectAll),
        Copy => send_key(tx, ProtoKeyCode::Copy),
        Paste => paste_now(tx, clip),
        Cut => send_key(tx, ProtoKeyCode::Cut),
        Undo => send_key(tx, ProtoKeyCode::Undo),
    }
}

/// Per-session clipboard bookkeeping, shared between the change poll and the Cmd/Ctrl+V
/// handler. A change whose hash we already synced is just the echo of what the
/// device gave us (or our own paste), so we don't re-send it.
#[derive(Default)]
struct ClipState {
    last_change_count: i64,
    last_synced_hash: Option<u64>,
}

/// Send a clipboard set: `[flags:u8][utf8]`, flags bit0 means paste after setting.
fn send_clipboard_set(tx: &Sender<InputFrame>, text: &str, paste: bool) {
    let mut p = Vec::with_capacity(1 + text.len());
    p.push(if paste { 0x01 } else { 0x00 });
    p.extend_from_slice(text.as_bytes());
    let _ = tx.send(InputFrame::new(MessageType::ClipboardSet, p));
}

/// Push the host clipboard to the device then paste it, so cross-device paste
/// carries the right text. Falls back to a plain iOS paste if the host
/// clipboard isn't usable text (empty, too large, or no display). Shared by
/// the Cmd/Ctrl+V shortcut.
fn paste_now(tx: &Sender<InputFrame>, clip: &Arc<Mutex<ClipState>>) {
    match clipboard::read_text() {
        Some(text) if !text.is_empty() && text.len() <= clipboard::MAX_CLIPBOARD_BYTES => {
            if let Ok(mut st) = clip.lock() {
                st.last_synced_hash = Some(clipboard::hash_text(&text));
            }
            send_clipboard_set(tx, &text, true);
        }
        _ => send_key(tx, ProtoKeyCode::Paste),
    }
}

/// Put a clipboard value the device sent into the host pasteboard, recording the
/// hash and change count so our own poll doesn't echo it back.
fn apply_remote_clipboard(text: &str, clip: &Arc<Mutex<ClipState>>) {
    if text.is_empty() {
        return;
    }
    let h = clipboard::hash_text(text);
    let mut st = clip.lock().unwrap();
    if st.last_synced_hash == Some(h) {
        return; // already have it
    }
    st.last_synced_hash = Some(h);
    st.last_change_count = clipboard::write_text(text);
}

/// If the host clipboard changed to new text, push it to the device.
fn poll_clipboard(tx: &Sender<InputFrame>, clip: &Arc<Mutex<ClipState>>) {
    let cc = clipboard::change_count();
    {
        let mut st = clip.lock().unwrap();
        if cc == st.last_change_count {
            return;
        }
        st.last_change_count = cc;
    }
    let text = match clipboard::read_text() {
        Some(t) if !t.is_empty() && t.len() <= clipboard::MAX_CLIPBOARD_BYTES => t,
        _ => return,
    };
    let h = clipboard::hash_text(&text);
    {
        let mut st = clip.lock().unwrap();
        if st.last_synced_hash == Some(h) {
            return; // already in sync (our own paste / device echo)
        }
        st.last_synced_hash = Some(h);
    }
    send_clipboard_set(tx, &text, false);
}

#[cfg(target_os = "macos")]
unsafe fn nsstring_to_string(ns: *mut objc::runtime::Object) -> String {
    use objc::{msg_send, sel, sel_impl};
    let utf8: *const std::os::raw::c_char = msg_send![ns, UTF8String];
    if utf8.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(utf8)
        .to_string_lossy()
        .into_owned()
}

/// Capture the Mac keyboard and forward it to the device via a local Cocoa monitor.
/// Handled events return nil so they are not forwarded to Cocoa's default handler,
/// preventing the system error alert beep on unhandled key equivalents.
#[cfg(target_os = "macos")]
fn install_key_monitor(
    tx: Sender<InputFrame>,
    clip: Arc<Mutex<ClipState>>,
    fps_toggle: Arc<AtomicBool>,
) {
    use block::ConcreteBlock;
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};

    const NS_EVENT_MASK_KEY_DOWN: u64 = 1 << 10;
    const NS_CMD: u64 = 1 << 20; // NSEventModifierFlagCommand

    let handler = ConcreteBlock::new(move |event: *mut Object| -> *mut Object {
        let nil: *mut Object = std::ptr::null_mut();
        unsafe {
            let flags: u64 = msg_send![event, modifierFlags];
            let repeat: bool = msg_send![event, isARepeat];

            if (flags & NS_CMD) != 0 {
                if repeat {
                    return nil; // Cmd combos never auto-repeat into the device
                }
                let chars: *mut Object = msg_send![event, charactersIgnoringModifiers];
                let len: u64 = if chars.is_null() {
                    0
                } else {
                    msg_send![chars, length]
                };
                if len >= 1 {
                    let ch: u16 = msg_send![chars, characterAtIndex: 0u64];
                    let mut handled = true;
                    match ch as u8 {
                        b'j' | b'J' => send_action(&tx, SystemAction::Home),
                        b'l' | b'L' => send_action(&tx, SystemAction::Lock),
                        b't' | b'T' => send_action(&tx, SystemAction::AppSwitcher),
                        b'r' | b'R' => send_action(&tx, SystemAction::RotateLeft),
                        b'i' | b'I' => {
                            let prev = fps_toggle.fetch_xor(true, Ordering::Relaxed);
                            if !prev {
                                println!("[ioscpy] FPS counter started");
                            } else {
                                println!("[ioscpy] FPS counter stopped");
                            }
                        }
                        b'a' | b'A' => send_key(&tx, ProtoKeyCode::SelectAll),
                        b'c' | b'C' => send_key(&tx, ProtoKeyCode::Copy),
                        b'v' | b'V' => paste_now(&tx, &clip),
                        b'x' | b'X' => send_key(&tx, ProtoKeyCode::Cut),
                        b'z' | b'Z' => send_key(&tx, ProtoKeyCode::Undo),
                        _ => handled = false,
                    }
                    if handled {
                        return nil;
                    }
                }
                return event; // other Cmd combos: let macOS have them
            }

            // Editing keys (no Cmd), by macOS virtual keycode.
            let keycode: u16 = msg_send![event, keyCode];
            // Esc maps to the iOS back gesture, not a literal Escape key.
            if keycode == 53 {
                send_action(&tx, SystemAction::Back);
                return nil;
            }
            let special = match keycode {
                36 | 76 => Some(ProtoKeyCode::Enter),
                51 => Some(ProtoKeyCode::Backspace),
                48 => Some(ProtoKeyCode::Tab),
                123 => Some(ProtoKeyCode::Left),
                124 => Some(ProtoKeyCode::Right),
                125 => Some(ProtoKeyCode::Down),
                126 => Some(ProtoKeyCode::Up),
                _ => None,
            };
            if let Some(k) = special {
                send_key(&tx, k);
                return nil;
            }

            // Everything else is text: forward the resolved characters.
            let chars: *mut Object = msg_send![event, characters];
            if !chars.is_null() {
                let s = nsstring_to_string(chars);
                if !s.is_empty() && !s.chars().all(char::is_control) {
                    let _ = tx.send(InputFrame::new(
                        MessageType::InputText,
                        protocol::encode_text(&s),
                    ));
                    return nil;
                }
            }
        }
        event
    });
    let handler = handler.copy();
    unsafe {
        let _: *mut Object = msg_send![class!(NSEvent),
            addLocalMonitorForEventsMatchingMask: NS_EVENT_MASK_KEY_DOWN
            handler: &*handler];
    }
    std::mem::forget(handler);
}

#[cfg(not(target_os = "macos"))]
fn install_key_monitor(
    _tx: Sender<InputFrame>,
    _clip: Arc<Mutex<ClipState>>,
    _fps_toggle: Arc<AtomicBool>,
) {
}

/// Become a regular Dock app and set its icon (the icon is embedded in the
/// binary). Call this before the window opens so the Dock shows our icon from the
/// start instead of briefly flashing the default executable one. Main thread only.
#[cfg(target_os = "macos")]
pub fn set_app_icon() {
    use objc::runtime::{Object, BOOL};
    use objc::{class, msg_send, sel, sel_impl};
    const ICON: &[u8] = include_bytes!("../assets/AppIcon.png");
    unsafe {
        let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
        if app.is_null() {
            return;
        }
        // NSApplicationActivationPolicyRegular = 0.
        let _: BOOL = msg_send![app, setActivationPolicy: 0isize];
        let data: *mut Object =
            msg_send![class!(NSData), dataWithBytes: ICON.as_ptr() length: ICON.len()];
        if data.is_null() {
            return;
        }
        let image: *mut Object = msg_send![class!(NSImage), alloc];
        let image: *mut Object = msg_send![image, initWithData: data];
        if image.is_null() {
            return;
        }
        let _: () = msg_send![app, setApplicationIconImage: image];
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_app_icon() {}

/// The main screen's usable size in points (menu bar and dock excluded). Used to
/// fit the window on screen. Otherwise macOS clamps an over-tall window's height
/// and the device ends up letterboxed with side bars.
#[cfg(target_os = "macos")]
fn screen_visible_size() -> (f64, f64) {
    use cocoa::foundation::NSRect;
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};
    unsafe {
        let screen: *mut Object = msg_send![class!(NSScreen), mainScreen];
        if screen.is_null() {
            return (1440.0, 900.0);
        }
        let frame: NSRect = msg_send![screen, visibleFrame];
        (frame.size.width, frame.size.height)
    }
}

#[cfg(not(target_os = "macos"))]
fn screen_visible_size() -> (f64, f64) {
    (1440.0, 900.0)
}

/// Window size for a `w`x`h` device frame at the given scale, clamped to at most
/// about 92% of the screen while keeping the aspect ratio. `sidebar_w` is a
/// fixed-width budget added to the window width.
fn scaled_fit(w: usize, h: usize, scale: f32, sidebar_w: usize) -> (usize, usize) {
    let (sw, sh) = screen_visible_size();
    let (max_w, max_h) = (
        ((sw * 0.92) as f32 - sidebar_w as f32).max(160.0),
        (sh * 0.92) as f32,
    );
    let tw = w as f32 * scale;
    let th = h as f32 * scale;
    let clamp = (max_w / tw).min(max_h / th).min(1.0);
    (
        ((tw * clamp) as usize).max(160) + sidebar_w,
        ((th * clamp) as usize).max(160),
    )
}

const SHADER_SOURCE: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle covering [-1, 1] NDC
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let p = pos[in_vertex_index];
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    // NDC (-1, 1) -> (0, 0) top-left; NDC (1, -1) -> (1, 1) bottom-right
    out.tex_coords = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    return vec4<f32>(color.rgb, 1.0);
}
"#;

struct WgpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    texture_format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    // Mirror frame texture
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    bind_group: Option<wgpu::BindGroup>,
    texture_size: (u32, u32),

    // Sidebar button strip texture
    sidebar_texture: Option<wgpu::Texture>,
    sidebar_texture_view: Option<wgpu::TextureView>,
    sidebar_bind_group: Option<wgpu::BindGroup>,
    sidebar_size: (u32, u32),
    sidebar_buf: Vec<u32>,
}

impl WgpuRenderer {
    fn new(window: Arc<Window>, initial_frame: &DecodedFrame) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("failed to find an appropriate GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ioscpy Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .context("failed to create GPU device")?;

        let capabilities = surface.get_capabilities(&adapter);

        // Prefer a non-sRGB surface format if available so the decoded video
        // frame values pass straight through 1:1 without double-gamma conversion.
        // If the surface only supports sRGB (e.g. some Wayland/Metal compositors),
        // we use an sRGB texture format so the GPU hardware sRGB -> linear -> sRGB
        // roundtrip cancels out and preserves the original video contrast.
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(capabilities.formats[0]);

        let texture_format = if surface_format.is_srgb() {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            wgpu::TextureFormat::Bgra8Unorm
        };

        // Use PresentMode::Immediate for minimal latency as requested, falling back to
        // Mailbox or Fifo if unsupported by the backend/compositor.
        let present_mode = if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            wgpu::PresentMode::Immediate
        } else if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            capabilities.present_modes[0]
        };

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fullscreen Quad Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let mut renderer = Self {
            surface,
            device,
            queue,
            config,
            texture_format,
            render_pipeline,
            bind_group_layout,
            sampler,
            texture: None,
            texture_view: None,
            bind_group: None,
            texture_size: (0, 0),
            sidebar_texture: None,
            sidebar_texture_view: None,
            sidebar_bind_group: None,
            sidebar_size: (0, 0),
            sidebar_buf: Vec::new(),
        };

        renderer.update_texture(initial_frame);
        Ok(renderer)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update_texture(&mut self, frame: &DecodedFrame) {
        let width = frame.width as u32;
        let height = frame.height as u32;

        if self.texture_size != (width, height) || self.texture.is_none() {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Mirror Frame Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.texture_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mirror Frame Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.texture = Some(texture);
            self.texture_view = Some(view);
            self.bind_group = Some(bind_group);
            self.texture_size = (width, height);
        }

        if let Some(texture) = &self.texture {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&frame.buf),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn update_sidebar(&mut self, width: u32, height: u32, pressed: Option<usize>) {
        if width == 0 || height == 0 {
            return;
        }

        let w = width as usize;
        let h = height as usize;
        let needed = w * h;
        if self.sidebar_buf.len() != needed {
            self.sidebar_buf.resize(needed, 0);
        }

        sidebar::draw_into(&mut self.sidebar_buf, w, 0, w, h, pressed);

        if self.sidebar_size != (width, height) || self.sidebar_texture.is_none() {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Sidebar Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.texture_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sidebar Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.sidebar_texture = Some(texture);
            self.sidebar_texture_view = Some(view);
            self.sidebar_bind_group = Some(bind_group);
            self.sidebar_size = (width, height);
        }

        if let Some(texture) = &self.sidebar_texture {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&self.sidebar_buf),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn render(&mut self, scale_factor: f32) -> Result<(), wgpu::SurfaceError> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let sidebar_w_px = (sidebar::WIDTH as f32 * scale_factor)
            .round()
            .min(self.config.width as f32);
        let content_w_px = (self.config.width as f32 - sidebar_w_px).max(1.0);
        let gh = self.config.height as f32;

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 1. Draw the mirror frame into its letterboxed area on the left
            if let Some(bind_group) = &self.bind_group {
                let fw = self.texture_size.0 as f32;
                let fh = self.texture_size.1 as f32;

                if fw > 0.0 && fh > 0.0 {
                    let img_aspect = fw / fh;
                    let area_aspect = content_w_px / gh;

                    let (img_w, img_h, x_off, y_off) = if area_aspect > img_aspect {
                        let w = gh * img_aspect;
                        (w, gh, (content_w_px - w) / 2.0, 0.0)
                    } else {
                        let h = content_w_px / img_aspect;
                        (content_w_px, h, 0.0, (gh - h) / 2.0)
                    };

                    render_pass.set_pipeline(&self.render_pipeline);
                    render_pass.set_viewport(x_off, y_off, img_w, img_h, 0.0, 1.0);
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }

            // 2. Draw the sidebar into its strip on the right
            if sidebar_w_px > 0.0 {
                if let Some(sb_bind_group) = &self.sidebar_bind_group {
                    render_pass.set_pipeline(&self.render_pipeline);
                    render_pass.set_viewport(content_w_px, 0.0, sidebar_w_px, gh, 0.0, 1.0);
                    render_pass.set_bind_group(0, sb_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

#[derive(Default)]
struct InputState {
    touching: bool,
    last: (f32, f32),
    sidebar_down: Option<usize>,
}

struct App {
    title: String,
    frames: FrameSlot,
    stop: Arc<AtomicBool>,
    input_tx: Sender<InputFrame>,
    clip: Arc<Mutex<ClipState>>,
    fps_enabled: Arc<AtomicBool>,
    display_scale: f32,

    current_frame: DecodedFrame,
    last_dims: (usize, usize),
    input_state: InputState,
    cursor_pos: (f32, f32),
    modifiers: ModifiersState,

    last_clip_poll: Instant,
    rendered_frames: u64,
    last_fps_instant: Instant,
    last_rendered_count: u64,
    last_dropped_count: u64,
    last_fps_was_enabled: bool,

    window: Option<Arc<Window>>,
    renderer: Option<WgpuRenderer>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        title: String,
        frames: FrameSlot,
        stop: Arc<AtomicBool>,
        input_tx: Sender<InputFrame>,
        clip: Arc<Mutex<ClipState>>,
        fps_enabled: Arc<AtomicBool>,
        display_scale: f32,
        first_frame: DecodedFrame,
    ) -> Self {
        let print_fps = fps_enabled.load(Ordering::Relaxed);
        let last_dims = (first_frame.width, first_frame.height);
        Self {
            title,
            frames,
            stop,
            input_tx,
            clip,
            fps_enabled,
            display_scale,
            current_frame: first_frame,
            last_dims,
            input_state: InputState::default(),
            cursor_pos: (0.0, 0.0),
            modifiers: ModifiersState::default(),
            last_clip_poll: Instant::now(),
            rendered_frames: 0,
            last_fps_instant: Instant::now(),
            last_rendered_count: 0,
            last_dropped_count: 0,
            last_fps_was_enabled: print_fps,
            window: None,
            renderer: None,
        }
    }

    fn scale_factor(&self) -> f32 {
        self.window
            .as_ref()
            .map(|w| w.scale_factor() as f32)
            .unwrap_or(1.0)
    }

    fn update_sidebar_texture(&mut self, pressed: Option<usize>) {
        let scale_factor = self.scale_factor();
        if let Some(renderer) = &mut self.renderer {
            let sidebar_w_px = (sidebar::WIDTH as f32 * scale_factor).round() as u32;
            let h = renderer.config.height;
            renderer.update_sidebar(sidebar_w_px, h, pressed);
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn handle_key_input(&mut self, event: KeyEvent) {
        if !event.state.is_pressed() {
            return;
        }

        let ctrl = self.modifiers.control_key();

        if ctrl {
            let mut handled = true;
            match &event.logical_key {
                Key::Character(s) => match s.as_str() {
                    "j" | "J" => send_action(&self.input_tx, SystemAction::Home),
                    "l" | "L" => send_action(&self.input_tx, SystemAction::Lock),
                    "t" | "T" => send_action(&self.input_tx, SystemAction::AppSwitcher),
                    "r" | "R" => send_action(&self.input_tx, SystemAction::RotateLeft),
                    "i" | "I" => {
                        let prev = self.fps_enabled.fetch_xor(true, Ordering::Relaxed);
                        if !prev {
                            println!("[ioscpy] FPS counter started");
                        } else {
                            println!("[ioscpy] FPS counter stopped");
                        }
                    }
                    "a" | "A" => send_key(&self.input_tx, ProtoKeyCode::SelectAll),
                    "c" | "C" => send_key(&self.input_tx, ProtoKeyCode::Copy),
                    "v" | "V" => paste_now(&self.input_tx, &self.clip),
                    "x" | "X" => send_key(&self.input_tx, ProtoKeyCode::Cut),
                    "z" | "Z" => send_key(&self.input_tx, ProtoKeyCode::Undo),
                    _ => handled = false,
                },
                _ => handled = false,
            }
            if !handled {
                // Fallback check physical key code
                if let PhysicalKey::Code(code) = event.physical_key {
                    use winit::keyboard::KeyCode as WinitKeyCode;
                    match code {
                        WinitKeyCode::KeyJ => send_action(&self.input_tx, SystemAction::Home),
                        WinitKeyCode::KeyL => send_action(&self.input_tx, SystemAction::Lock),
                        WinitKeyCode::KeyT => {
                            send_action(&self.input_tx, SystemAction::AppSwitcher)
                        }
                        WinitKeyCode::KeyR => send_action(&self.input_tx, SystemAction::RotateLeft),
                        WinitKeyCode::KeyI => {
                            let prev = self.fps_enabled.fetch_xor(true, Ordering::Relaxed);
                            if !prev {
                                println!("[ioscpy] FPS counter started");
                            } else {
                                println!("[ioscpy] FPS counter stopped");
                            }
                        }
                        WinitKeyCode::KeyA => send_key(&self.input_tx, ProtoKeyCode::SelectAll),
                        WinitKeyCode::KeyC => send_key(&self.input_tx, ProtoKeyCode::Copy),
                        WinitKeyCode::KeyV => paste_now(&self.input_tx, &self.clip),
                        WinitKeyCode::KeyX => send_key(&self.input_tx, ProtoKeyCode::Cut),
                        WinitKeyCode::KeyZ => send_key(&self.input_tx, ProtoKeyCode::Undo),
                        _ => {}
                    }
                }
            }
            return;
        }

        // Non-Ctrl keys: editing keys and text
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                send_action(&self.input_tx, SystemAction::Back);
                return;
            }
            Key::Named(NamedKey::Enter) => {
                send_key(&self.input_tx, ProtoKeyCode::Enter);
                return;
            }
            Key::Named(NamedKey::Tab) => {
                send_key(&self.input_tx, ProtoKeyCode::Tab);
                return;
            }
            Key::Named(NamedKey::Backspace) => {
                send_key(&self.input_tx, ProtoKeyCode::Backspace);
                return;
            }
            Key::Named(NamedKey::ArrowLeft) => {
                send_key(&self.input_tx, ProtoKeyCode::Left);
                return;
            }
            Key::Named(NamedKey::ArrowRight) => {
                send_key(&self.input_tx, ProtoKeyCode::Right);
                return;
            }
            Key::Named(NamedKey::ArrowUp) => {
                send_key(&self.input_tx, ProtoKeyCode::Up);
                return;
            }
            Key::Named(NamedKey::ArrowDown) => {
                send_key(&self.input_tx, ProtoKeyCode::Down);
                return;
            }
            _ => {}
        }

        if let Some(text) = event.text {
            if !text.is_empty() && !text.chars().all(char::is_control) {
                let _ = self.input_tx.send(InputFrame::new(
                    MessageType::InputText,
                    protocol::encode_text(text.as_str()),
                ));
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn handle_key_input(&mut self, _event: KeyEvent) {
        // macOS routes keys through install_key_monitor (Cocoa local event monitor)
        // to properly swallow handled shortcuts and prevent the system alert beep.
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let (win_w, win_h) = scaled_fit(
            self.current_frame.width,
            self.current_frame.height,
            self.display_scale,
            sidebar::WIDTH,
        );

        let window_attributes = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(win_w as f64, win_h as f64));

        let window = match event_loop.create_window(window_attributes) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("ioscpy: error creating window: {e:#}");
                event_loop.exit();
                return;
            }
        };

        match WgpuRenderer::new(window.clone(), &self.current_frame) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
                self.update_sidebar_texture(None);
            }
            Err(e) => {
                eprintln!("ioscpy: error initializing wgpu: {e:#}");
                event_loop.exit();
                return;
            }
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::NewFrame => {
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            UserEvent::RemoteClipboard(text) => {
                apply_remote_clipboard(&text, &self.clip);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.stop.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(new_size);
                }
                self.update_sidebar_texture(self.input_state.sidebar_down);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.as_ref().map(|w| w.inner_size());
                if let Some(size) = size {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(size);
                    }
                }
                self.update_sidebar_texture(self.input_state.sidebar_down);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.stop.load(Ordering::Relaxed) {
                    event_loop.exit();
                    return;
                }

                // Drain any fresh decoded frame from the slot
                let (new_frame, _incoming_total, dropped_total) = {
                    let mut lock = self.frames.lock().unwrap();
                    (lock.frame.take(), lock.incoming, lock.dropped)
                };

                if let Some(decoded) = new_frame {
                    self.current_frame = decoded;
                    self.rendered_frames += 1;
                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_texture(&self.current_frame);
                    }
                }

                // Check for device orientation change and request window resize
                let cur_dims = (self.current_frame.width, self.current_frame.height);
                if cur_dims != self.last_dims {
                    self.last_dims = cur_dims;
                    let (nw, nh) =
                        scaled_fit(cur_dims.0, cur_dims.1, self.display_scale, sidebar::WIDTH);
                    if let Some(w) = &self.window {
                        let _ = w.request_inner_size(LogicalSize::new(nw as f64, nh as f64));
                    }
                }

                // FPS tracking
                let is_fps_on = self.fps_enabled.load(Ordering::Relaxed);
                if self.last_fps_was_enabled && !is_fps_on {
                    if let Some(w) = &self.window {
                        w.set_title(&self.title);
                    }
                }
                self.last_fps_was_enabled = is_fps_on;

                if self.last_fps_instant.elapsed() >= Duration::from_secs(1) {
                    let elapsed = self.last_fps_instant.elapsed().as_secs_f64();
                    let rendered_delta = self.rendered_frames - self.last_rendered_count;
                    let dropped_delta = dropped_total - self.last_dropped_count;
                    self.last_rendered_count = self.rendered_frames;
                    self.last_dropped_count = dropped_total;
                    self.last_fps_instant = Instant::now();

                    let fps = rendered_delta as f64 / elapsed;

                    if is_fps_on {
                        if dropped_delta > 0 {
                            println!("{:.1} fps (+{} frames dropped)", fps, dropped_delta);
                            if let Some(w) = &self.window {
                                w.set_title(&format!(
                                    "{} ({:.1} fps, {} dropped)",
                                    self.title, fps, dropped_delta
                                ));
                            }
                        } else {
                            println!("{:.1} fps", fps);
                            if let Some(w) = &self.window {
                                w.set_title(&format!("{} ({:.1} fps)", self.title, fps));
                            }
                        }
                    }
                }

                let scale_factor = self.scale_factor();
                if let Some(renderer) = &mut self.renderer {
                    match renderer.render(scale_factor) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            if let Some(w) = &self.window {
                                renderer.resize(w.inner_size());
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("ioscpy: wgpu out of memory");
                            event_loop.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout | wgpu::SurfaceError::Other) => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x as f32, position.y as f32);
                if self.input_state.touching {
                    if let Some(renderer) = &self.renderer {
                        let scale_factor = self.scale_factor();
                        let sidebar_w_px = (sidebar::WIDTH as f32 * scale_factor)
                            .round()
                            .min(renderer.config.width as f32);
                        let content_w_px = (renderer.config.width as f32 - sidebar_w_px).max(1.0);
                        let gh = renderer.config.height as f32;

                        let fw = self.current_frame.width;
                        let fh = self.current_frame.height;
                        if fw > 0 && fh > 0 {
                            let (nx, ny) = map_to_norm(
                                self.cursor_pos.0,
                                self.cursor_pos.1,
                                content_w_px as usize,
                                gh as usize,
                                fw,
                                fh,
                            );

                            if nx != self.input_state.last.0 || ny != self.input_state.last.1 {
                                send_touch(&self.input_tx, TouchPhase::Move, nx, ny);
                                self.input_state.last = (nx, ny);
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    if let Some(renderer) = &self.renderer {
                        let scale_factor = self.scale_factor();
                        let sidebar_w_px = (sidebar::WIDTH as f32 * scale_factor)
                            .round()
                            .min(renderer.config.width as f32);
                        let content_w_px = (renderer.config.width as f32 - sidebar_w_px).max(1.0);
                        let gh = renderer.config.height as f32;

                        let mx = self.cursor_pos.0;
                        let my = self.cursor_pos.1;

                        if mx < content_w_px {
                            // Touch event on mirror screen
                            let fw = self.current_frame.width as f32;
                            let fh = self.current_frame.height as f32;
                            if fw > 0.0 && fh > 0.0 {
                                let img_aspect = fw / fh;
                                let area_aspect = content_w_px / gh;
                                let (img_w, img_h, x_off, y_off) = if area_aspect > img_aspect {
                                    let w = gh * img_aspect;
                                    (w, gh, (content_w_px - w) / 2.0, 0.0)
                                } else {
                                    let h = content_w_px / img_aspect;
                                    (content_w_px, h, 0.0, (gh - h) / 2.0)
                                };

                                if mx >= x_off
                                    && mx <= x_off + img_w
                                    && my >= y_off
                                    && my <= y_off + img_h
                                {
                                    let (nx, ny) = map_to_norm(
                                        mx,
                                        my,
                                        content_w_px as usize,
                                        gh as usize,
                                        self.current_frame.width,
                                        self.current_frame.height,
                                    );
                                    send_touch(&self.input_tx, TouchPhase::Down, nx, ny);
                                    self.input_state.touching = true;
                                    self.input_state.last = (nx, ny);
                                }
                            }
                        } else if sidebar_w_px > 0.0 {
                            // Click on sidebar button
                            let sb_x = (mx - content_w_px) / scale_factor;
                            let sb_y = my / scale_factor;
                            let sb_h = gh / scale_factor;
                            if let Some(idx) = sidebar::hit_test(sb_x, sb_y, sb_h) {
                                self.input_state.sidebar_down = Some(idx);
                                dispatch_sidebar_action(
                                    &self.input_tx,
                                    &self.clip,
                                    sidebar::BUTTONS[idx],
                                );
                                self.update_sidebar_texture(Some(idx));
                                if let Some(w) = &self.window {
                                    w.request_redraw();
                                }
                            }
                        }
                    }
                }
                ElementState::Released => {
                    if self.input_state.touching {
                        send_touch(
                            &self.input_tx,
                            TouchPhase::Up,
                            self.input_state.last.0,
                            self.input_state.last.1,
                        );
                        self.input_state.touching = false;
                    }
                    if self.input_state.sidebar_down.take().is_some() {
                        self.update_sidebar_texture(None);
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
            },
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key_input(event);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.stop.load(Ordering::Relaxed) {
            event_loop.exit();
            return;
        }

        // Push host clipboard changes to the device (rate-limited)
        if self.last_clip_poll.elapsed() >= Duration::from_millis(300) {
            self.last_clip_poll = Instant::now();
            poll_clipboard(&self.input_tx, &self.clip);
        }

        // Check if a new frame is waiting in the slot
        let has_pending_frame = {
            let lock = self.frames.lock().unwrap();
            lock.frame.is_some()
        };
        if has_pending_frame {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }

        // Event-driven wait: wake up on events or after 50ms for clipboard poll
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(50),
        ));
    }
}

/// Run the window loop on the calling (main) thread until the window closes or
/// `stop` is set. Renders the latest frame via wgpu and forwards input.
pub fn run_window(
    title: &str,
    frames: FrameSlot,
    stop: Arc<AtomicBool>,
    input_tx: Sender<InputFrame>,
    clip_in: Receiver<String>,
    print_fps: bool,
) -> Result<()> {
    let first = match wait_for_first_frame(&frames, &stop) {
        Some(f) => f,
        None => return Ok(()), // stopped before the first frame arrived
    };

    let (sw, sh) = screen_visible_size();
    let (sw, sh) = (sw as f32, sh as f32);
    let long = first.width.max(first.height).max(1) as f32;
    let short = first.width.min(first.height).max(1) as f32;
    let display_scale = ((sh * 0.92) / long).min((sw * 0.92) / short);

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    // Attach waker to frame slot so incoming frames wake up the winit event loop
    {
        let p = proxy.clone();
        let mut lock = frames.lock().unwrap();
        lock.waker = Some(Arc::new(move || {
            let _ = p.send_event(UserEvent::NewFrame);
        }));
    }

    // Forward remote clipboard changes to the event loop
    {
        let p = proxy.clone();
        std::thread::spawn(move || {
            while let Ok(text) = clip_in.recv() {
                if p.send_event(UserEvent::RemoteClipboard(text)).is_err() {
                    break;
                }
            }
        });
    }

    let clip = Arc::new(Mutex::new(ClipState::default()));
    let fps_enabled = Arc::new(AtomicBool::new(print_fps));
    if print_fps {
        println!("[ioscpy] FPS counter started");
    }

    install_key_monitor(input_tx.clone(), clip.clone(), fps_enabled.clone());

    let mut app = App::new(
        title.to_string(),
        frames,
        stop.clone(),
        input_tx,
        clip,
        fps_enabled,
        display_scale,
        first,
    );

    event_loop.run_app(&mut app)?;
    stop.store(true, Ordering::Relaxed);
    Ok(())
}

fn wait_for_first_frame(frames: &FrameSlot, stop: &Arc<AtomicBool>) -> Option<DecodedFrame> {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
        if let Some(decoded) = frames.lock().unwrap().frame.take() {
            return Some(decoded);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}
