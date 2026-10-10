//! Windowed (winit + wgpu surface) and headless (`--shot`) terrain rendering.
//!
//! Soft camera model:
//! - wheel = smooth zoom (exponential, eased toward a target)
//! - middle-button drag = smooth rotation (yaw around the grid centre; the
//!   shader applies the angle, so the mesh is never re-uploaded)
//! - Shift+middle / left / right drag = pan; the mouse near a window edge
//!   pans smoothly (edge scrolling, speed grows toward the edge)
//! - WASD/arrows = pan; Q/E = rotate 90°; R = reset; Esc = quit
//!
//! Every frame the eased camera state feeds the shared shader uniforms
//! (`center`/`scale`/`angle`/`time`); water shimmer animates via `time`.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use kolomna_render::{
    Params, TerrainRenderer, VIEW_H, VIEW_W, Vertex, ZSCALE, build_mesh, grid_side, load_atlas,
    load_terrain, render_offscreen_with, water_level, write_png,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

/// keyboard pan speed (screen px per second)
const PAN_PX_PER_S: f32 = 480.0;
/// edge-scrolling speed at the window border (screen px per second)
const EDGE_MAX_PX_PER_S: f32 = 700.0;
/// edge-scrolling activation margin (px)
const EDGE_PX: f64 = 36.0;
const ZOOM_MIN: f32 = 0.12;
const ZOOM_MAX: f32 = 12.0;

// ---------------------------------------------------------------- scene ----

pub struct Scene {
    base: Vec<Vertex>,
    atlas: Vec<u8>,
    grid: f32,
    zspan: f32,
}

pub fn load_scene(city: &str, dat: &str) -> Result<Scene, Box<dyn Error>> {
    let terrain = load_terrain(city)?;
    let n = grid_side(&terrain);
    if n * n != terrain.len() {
        return Err(format!("terrain grid is not square ({} values)", terrain.len()).into());
    }
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    println!("terrain: {n}x{n} vertices, altitude {mn:.1}..{mx:.1} m, water surface {wl:.2} m");

    let atlas = load_atlas(dat)?;
    println!("atlas: 6 textures -> 256x256 px layers");

    let base = build_mesh(&terrain, mn, mx, wl);
    println!("mesh: {} triangles", base.len() / 3);

    Ok(Scene {
        base,
        atlas,
        grid: (n - 1) as f32,
        zspan: (mx - mn) * ZSCALE,
    })
}

/// Headless mode: render exactly one frame offscreen with the default camera
/// (`zoom` scales the framing; 1.0 = whole grid) and write it as a PNG.
pub fn shot(out: &str, city: &str, dat: &str, zoom: f32) -> Result<(), Box<dyn Error>> {
    let scene = load_scene(city, dat)?;
    let rgba = render_offscreen_with(
        VIEW_W,
        VIEW_H,
        &scene.atlas,
        &scene.base,
        scene.grid,
        scene.zspan,
        zoom,
    )?;
    write_png(out, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {out}");
    Ok(())
}

// --------------------------------------------------------------- camera ----

struct Camera {
    /// camera target in iso screen space (the shader's `center` units)
    target: [f32; 2],
    zoom: f32,
    zoom_target: f32,
    rot: f32,
    rot_target: f32,
    /// smoothed pan velocity (screen px/s)
    vel: [f32; 2],
    /// seconds since start (water shimmer)
    time: f32,
}

impl Camera {
    fn new(grid: f32, zspan: f32) -> Self {
        Self {
            target: [0.0, grid * 0.25 - zspan * 0.5],
            zoom: 1.0,
            zoom_target: 1.0,
            rot: 0.0,
            rot_target: 0.0,
            vel: [0.0; 2],
            time: 0.0,
        }
    }

    fn reset(&mut self, grid: f32, zspan: f32) {
        *self = Self::new(grid, zspan);
    }

    fn zoom_by(&mut self, factor: f32) {
        self.zoom_target = (self.zoom_target * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    }

    fn rotate_by(&mut self, delta: f32) {
        self.rot_target += delta;
    }

    /// Pan by a screen-space mouse delta (px).
    fn pan_by_px(&mut self, dx: f32, dy: f32, scale: f32) {
        let s = scale.max(1e-3);
        self.target[0] -= dx / s;
        self.target[1] -= dy / s;
    }

    /// Advance smoothing; `base_scale` is the zoom=1 world->screen scale.
    fn step(
        &mut self,
        dt: f32,
        keys: &HashSet<KeyCode>,
        cursor: Option<(f64, f64)>,
        win: (f64, f64),
        base_scale: f32,
    ) {
        // desired pan velocity: keyboard + window edges
        let mut desired = [0.0f32; 2];
        if keys.contains(&KeyCode::ArrowLeft) || keys.contains(&KeyCode::KeyA) {
            desired[0] -= PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowRight) || keys.contains(&KeyCode::KeyD) {
            desired[0] += PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowUp) || keys.contains(&KeyCode::KeyW) {
            desired[1] -= PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowDown) || keys.contains(&KeyCode::KeyS) {
            desired[1] += PAN_PX_PER_S;
        }
        if let Some((x, y)) = cursor {
            let (w, h) = win;
            let fx = if x < EDGE_PX {
                -((1.0 - x / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else if x > w - EDGE_PX {
                ((1.0 - (w - x) / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else {
                0.0
            };
            let fy = if y < EDGE_PX {
                -((1.0 - y / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else if y > h - EDGE_PX {
                ((1.0 - (h - y) / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else {
                0.0
            };
            desired[0] += fx as f32;
            desired[1] += fy as f32;
        }
        let k = 1.0 - (-8.0 * dt).exp();
        self.vel[0] += (desired[0] - self.vel[0]) * k;
        self.vel[1] += (desired[1] - self.vel[1]) * k;
        let s = (base_scale * self.zoom).max(1e-3);
        self.target[0] += self.vel[0] * dt / s;
        self.target[1] += self.vel[1] * dt / s;

        // eased zoom and rotation
        let kz = 1.0 - (-13.0 * dt).exp();
        self.zoom += (self.zoom_target - self.zoom) * kz;
        let kr = 1.0 - (-14.0 * dt).exp();
        self.rot += (self.rot_target - self.rot) * kr;
        self.time += dt;
    }

    fn params(&self, w: f32, h: f32, grid: f32, base_scale: f32) -> Params {
        Params {
            center: self.target,
            scale: base_scale * self.zoom,
            hz: 1.0,
            half: [w / 2.0, h / 2.0],
            angle: self.rot,
            time: self.time,
            pivot: grid * 0.5,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

// ------------------------------------------------------------- renderer ----

fn make_targets(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let msaa = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("app-msaa"),
        size,
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("app-depth"),
        size,
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    (
        msaa.create_view(&wgpu::TextureViewDescriptor::default()),
        depth.create_view(&wgpu::TextureViewDescriptor::default()),
    )
}

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    terrain: TerrainRenderer,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    fn new(window: Arc<Window>, scene: &Scene) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no graphics adapter found: {e}"))?;
        let info = adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("kolomna-app"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;

        let caps = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("window surface is not supported by the adapter")?;
        // prefer an sRGB surface format so on-screen colors match the offscreen reference
        if let Some(f) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = f;
        }
        println!(
            "surface: {}x{} {:?} ({:?})",
            config.width, config.height, config.format, config.present_mode
        );
        surface.configure(&device, &config);

        let terrain = TerrainRenderer::new(&device, &queue, config.format, 4, &scene.atlas, &scene.base);
        let (msaa_view, depth_view) = make_targets(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            terrain,
            msaa_view,
            depth_view,
        })
    }

    /// zoom=1 world->screen scale for the current window size.
    fn base_scale(&self, scene: &Scene) -> f32 {
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let vspan = scene.grid * 0.5 + scene.zspan;
        ((w - 60.0) / scene.grid).min((h - 60.0) / vspan)
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        let (msaa_view, depth_view) = make_targets(&self.device, &self.config);
        self.msaa_view = msaa_view;
        self.depth_view = depth_view;
    }

    fn render(&mut self, scene: &Scene, cam: &Camera) -> Result<(), Box<dyn Error>> {
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let params = cam.params(w, h, scene.grid, self.base_scale(scene));
        self.terrain.set_params(&self.queue, &params);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("surface validation error".into());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("app-enc"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("app-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.60,
                            g: 0.72,
                            b: 0.86,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.terrain.draw(&mut pass);
        }
        self.queue.submit(Some(enc.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

// ------------------------------------------------------------ winit app ----

struct App {
    city: String,
    dat: String,
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    camera: Option<Camera>,
    keys: HashSet<KeyCode>,
    shift: bool,
    cursor: Option<(f64, f64)>,
    win_size: (f64, f64),
    drag_rot: bool,
    drag_pan: bool,
    last_frame: Instant,
    title_acc: f32,
    fps_acc: f32,
    frames: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let title = format!(
            "Kolomna — {}",
            std::path::Path::new(&self.city)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.city.clone())
        );
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(VIEW_W, VIEW_H));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("error: failed to create window: {e}");
                std::process::exit(1);
            }
        };
        let scene = match load_scene(&self.city, &self.dat) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let renderer = match Renderer::new(window.clone(), &scene) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let size = window.inner_size();
        self.win_size = (size.width as f64, size.height as f64);
        println!("Управление:");
        println!("  колесо               — мягкий зум");
        println!("  средняя кнопка       — мягкий поворот");
        println!("  Shift+средняя / ЛКМ / ПКМ — перенос карты");
        println!("  края окна            — навигация (карта едет, если мышь у края)");
        println!("  WASD/стрелки — перенос · Q/E — поворот 90° · R — сброс · Esc — выход");
        self.camera = Some(Camera::new(scene.grid, scene.zspan));
        self.scene = Some(scene);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.last_frame = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(mods) => {
                self.shift = mods.state().contains(ModifiersState::SHIFT);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Escape {
                                event_loop.exit();
                                return;
                            }
                            if !event.repeat {
                                if let Some(scene) = self.scene.as_ref() {
                                    if code == KeyCode::KeyR {
                                        if let Some(cam) = self.camera.as_mut() {
                                            cam.reset(scene.grid, scene.zspan);
                                        }
                                    }
                                }
                                if let Some(cam) = self.camera.as_mut() {
                                    match code {
                                        KeyCode::KeyQ => cam.rotate_by(-std::f32::consts::FRAC_PI_2),
                                        KeyCode::KeyE => cam.rotate_by(std::f32::consts::FRAC_PI_2),
                                        _ => {}
                                    }
                                }
                            }
                            self.keys.insert(code);
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cam) = self.camera.as_mut() {
                    let factor = match delta {
                        MouseScrollDelta::LineDelta(_, y) => 1.12f32.powf(y),
                        MouseScrollDelta::PixelDelta(p) => (p.y as f32 * 0.002).exp(),
                    };
                    cam.zoom_by(factor);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Middle => {
                        if pressed && self.shift {
                            self.drag_pan = true;
                        } else {
                            self.drag_rot = pressed;
                        }
                    }
                    MouseButton::Left | MouseButton::Right => self.drag_pan = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some(prev) = self.cursor {
                    let dx = (pos.0 - prev.0) as f32;
                    let dy = (pos.1 - prev.1) as f32;
                    if self.drag_rot && !self.shift {
                        if let Some(cam) = self.camera.as_mut() {
                            cam.rotate_by(dx * 0.007);
                        }
                    }
                    if self.drag_pan || (self.drag_rot && self.shift) {
                        let scale = self
                            .renderer
                            .as_ref()
                            .zip(self.scene.as_ref())
                            .zip(self.camera.as_ref())
                            .map(|((r, s), c)| r.base_scale(s) * c.zoom)
                            .unwrap_or(1.0)
                            .max(1e-3);
                        if let Some(cam) = self.camera.as_mut() {
                            cam.pan_by_px(dx, dy, scale);
                        }
                    }
                }
                self.cursor = Some(pos);
            }
            WindowEvent::Resized(size) => {
                self.win_size = (size.width as f64, size.height as f64);
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                self.fps_acc += dt;
                self.frames += 1;
                let mut frame_ok = true;
                if let Some(scene) = self.scene.as_ref() {
                    if let (Some(cam), Some(renderer)) =
                        (self.camera.as_mut(), self.renderer.as_mut())
                    {
                        let base = renderer.base_scale(scene);
                        cam.step(dt, &self.keys, self.cursor, self.win_size, base);
                        if let Err(e) = renderer.render(scene, cam) {
                            eprintln!("error: {e}");
                            frame_ok = false;
                        }
                        // once per 0.5 s: window title with zoom/angle/fps
                        self.title_acc += dt;
                        if self.title_acc >= 0.5 {
                            self.title_acc = 0.0;
                            let fps = self.frames as f32 / self.fps_acc.max(1e-3);
                            self.fps_acc = 0.0;
                            self.frames = 0;
                            if let Some(w) = self.window.as_ref() {
                                let stem = std::path::Path::new(&self.city)
                                    .file_stem()
                                    .map(|s| s.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                let deg = cam.rot.to_degrees().round() as i32;
                                w.set_title(&format!(
                                    "Kolomna — {stem} · {fps:.0} fps · зум ×{:.2} · {deg}°",
                                    cam.zoom
                                ));
                            }
                        }
                    }
                }
                if !frame_ok {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

/// Interactive mode: open a window and run the event loop until the user quits.
pub fn run(city: &str, dat: &str) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        city: city.to_string(),
        dat: dat.to_string(),
        window: None,
        scene: None,
        renderer: None,
        camera: None,
        keys: HashSet::new(),
        shift: false,
        cursor: None,
        win_size: (VIEW_W as f64, VIEW_H as f64),
        drag_rot: false,
        drag_pan: false,
        last_frame: Instant::now(),
        title_acc: 0.0,
        fps_acc: 0.0,
        frames: 0,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
