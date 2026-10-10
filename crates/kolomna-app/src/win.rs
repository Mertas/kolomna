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

use kolomna_render::assets::fsh_rgba_search;
use kolomna_render::model::{ModelMesh, ModelRenderer, PlacedModel};
use kolomna_render::road::{build_road_mesh, load_road_tiles, RoadMesh, RoadRenderer};
use kolomna_render::{
    Params, TerrainRenderer, VIEW_H, VIEW_W, Vertex, ZSCALE, build_mesh, grid_side, load_atlas,
    load_terrain, smooth_field, water_level, write_png,
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
const ZOOM_MAX: f32 = 100.0;

// ---------------------------------------------------------------- scene ----

pub struct Scene {
    base: Vec<Vertex>,
    atlas: Vec<u8>,
    grid: f32,
    zspan: f32,
    /// terrain min altitude (metres) — feeds the shader's sand blend
    mn: f32,
    /// water surface level (metres)
    wl: f32,
    /// smoothed height grid (metres, water-clamped) — the exact field the
    /// mesh is built from; the camera raycasts it for the rotation pivot
    sm: Vec<f32>,
    /// grid side length of `sm`
    n: usize,
    /// road network from the save (16 m tiles)
    road_mesh: RoadMesh,
    road_textures: Vec<(u32, u32, Vec<u8>)>,
    /// generated Petersburg tenements (mesh, textures, ground position)
    village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3])>,
}

pub fn load_scene(city: &str, dat: &str) -> Result<Scene, Box<dyn Error>> {
    let mut terrain = load_terrain(city)?;
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

    // (the terrain mesh is built after the tenement pads are levelled below)

    // ---- road network from the save (one 16 m tile per record)
    let tiles = load_road_tiles(city)?;
    println!("roads: {} tiles", tiles.len());
    let road_mesh = build_road_mesh(&tiles, mn, wl);
    let mut road_textures = Vec::new();
    for id in &road_mesh.texture_ids {
        road_textures.push(fsh_rgba_search(dat, *id)?);
    }
    println!("road textures: {}", road_textures.len());

    // (The 3DMD house family is retired from the scene — the focus is now the
    // generated Petersburg tenements; the 3DMD parser stays for the examples.)
    let t = &tiles[tiles.len() / 2];
    let alt = (t.verts[0][1] + t.verts[2][1]) * 0.5;
    let anchor = [
        (t.verts[0][0] + t.verts[2][0]) * 0.5 / 16.0,
        (t.verts[0][2] + t.verts[2][2]) * 0.5 / 16.0 - 1.0,
    ];
    println!(
        "street anchor at cell ({:.1}, {:.1}), altitude {alt:.1} m",
        anchor[0], anchor[1]
    );
    let mut dx = t.verts[1][0] - t.verts[0][0];
    let mut dz = t.verts[1][2] - t.verts[0][2];
    let dl = (dx * dx + dz * dz).sqrt().max(1e-6);
    dx /= dl;
    dz /= dl;

    // ---- a street of Petersburg tenements (доходные дома, E:/My/Clicker) ----
    // A tenement on a slope: cut a level pad under the footprint into the
    // height grid BEFORE the mesh is built (terraced lots do the same). The
    // pad height is the smoothed height at the house centre, so the hill is
    // cut from behind and filled from below — the house sits on a terrace
    // that follows the hill, and no corner gets sliced by the slope.
    let sm0 = smooth_field(&terrain, wl);
    let h0 = |x: f32, z: f32| -> f32 {
        let i = x.clamp(0.0, (n - 1) as f32);
        let j = z.clamp(0.0, (n - 1) as f32);
        sample_h(&sm0, n, i, j)
    };
    let mut village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3])> = Vec::new();
    for k in 0..4u32 {
        let along = (k as f32 - 1.5) * 2.6;
        let side: f32 = -4.8;
        let gx = anchor[0] + dx * along - dz * side;
        let gz = anchor[1] + dz * along + dx * side;
        let (mesh, texs) = kolomna_render::piter::generate_manor(7100 + k as u64 * 31);
        let h_pad = h0(gx, gz).max(wl + 2.0);
        // half extents incl. a 1–2 m apron (metres; the grid cell is 16 m)
        kolomna_render::flatten_pad(&mut terrain, n, gx * 16.0, gz * 16.0, 13.0, 8.0, h_pad, 22.0);
        village.push((mesh, texs, [gx, gz, h_pad]));
    }
    println!("village: {} Petersburg tenements", village.len());

    // ---- terrain mesh (after the pads were levelled) -----------------------
    let base = build_mesh(&terrain, mn, mx, wl);
    let sm = smooth_field(&terrain, wl);
    println!("mesh: {} triangles", base.len() / 3);
    // final placement heights: sample the levelled field at each centre
    for (_, _, p) in village.iter_mut() {
        p[2] = (sample_h(&sm, n, p[0], p[1]) - mn) * ZSCALE + 0.02;
    }

    Ok(Scene {
        base,
        atlas,
        grid: (n - 1) as f32,
        zspan: (mx - mn) * ZSCALE,
        mn,
        wl,
        sm,
        n,
        road_mesh,
        road_textures,
        village,
    })
}

/// Headless mode: render exactly one frame offscreen with the default camera
/// (`zoom` scales the framing; 1.0 = whole grid) and write it as a PNG.
pub fn shot(out: &str, city: &str, dat: &str, zoom: f32) -> Result<(), Box<dyn Error>> {
    let scene = load_scene(city, dat)?;
    let mut placed = Vec::new();
    for (mesh, texs, pos) in &scene.village {
        placed.push(PlacedModel {
            mesh,
            textures: texs,
            pos: *pos,
            rot: 0.0,
            spin: 0.0,
        });
    }
    let tilt = std::env::var("KOLOMNA_TILT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let angle = std::env::var("KOLOMNA_ANGLE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    let view = scene
        .village
        .get(scene.village.len().saturating_sub(2))
        .map(|(_, _, p)| {
            // the shader rotates world points about the grid centre before the
            // iso projection — rotate the anchor the same way, or a frame with
            // a non-zero angle aims at the wrong place (empty ground).
            let c = scene.grid * 0.5;
            let (sa, ca) = angle.sin_cos();
            let d = [p[0] - c, p[1] - c];
            let rx = c + d[0] * ca - d[1] * sa;
            let rz = c + d[0] * sa + d[1] * ca;
            (
                [
                    (rx - rz) * 0.5,
                    // + ~half the tenement height: keeps a ~17 m house centred
                    (rx + rz) * (0.5 * tilt) - p[2] - 0.55,
                ],
                angle,
            )
        });
    let rgba = kolomna_render::model::render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &scene.atlas,
        &scene.base,
        scene.grid,
        scene.zspan,
        zoom,
        scene.mn,
        scene.wl,
        view,
        &placed,
        None,
        &[],
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
    /// last `rot` the pivot compensation was applied at
    rot_applied: f32,
    /// rotation pivot, cell coords (set under the cursor on middle-press)
    pivot: [f32; 2],
    /// grid side (cells) — pivot compensation needs it
    grid: f32,
    /// camera tilt = sin(elevation): 0.5 = 2:1 iso, 1.0 = top-down,
    /// 0.0 = straight ahead; vertical middle-drag drives it
    tilt: f32,
    tilt_target: f32,
    /// last `tilt` the pivot compensation was applied at
    tilt_applied: f32,
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
            rot_applied: 0.0,
            tilt: 0.5,
            tilt_target: 0.5,
            tilt_applied: 0.5,
            pivot: [grid * 0.5, grid * 0.5],
            grid,
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

    /// Middle-drag up/down: 0.5 = the classic 2:1 iso view, up to 1.0
    /// (straight down) and down to 0.0 (level, houses seen from the side).
    fn tilt_by(&mut self, delta: f32) {
        self.tilt_target = (self.tilt_target + delta).clamp(0.0, 1.0);
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
            desired[0] += fx;
            desired[1] += fy;
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
        // keep the rotation pivot pinned under the cursor: rotating the world
        // about the map centre moves every point on screen; shifting `target`
        // by the pivot's own screen displacement cancels that exactly at the
        // pivot, so the ground under the mouse stays put while rotating
        if (self.rot - self.rot_applied).abs() > 1e-6 {
            let c0 = self.grid * 0.5;
            let d = [self.pivot[0] - c0, self.pivot[1] - c0];
            let (s0, c0c) = self.rot_applied.sin_cos();
            let (s1, c1) = self.rot.sin_cos();
            let v0 = [d[0] * c0c - d[1] * s0, d[0] * s0 + d[1] * c0c];
            let v1 = [d[0] * c1 - d[1] * s1, d[0] * s1 + d[1] * c1];
            let dd = [v1[0] - v0[0], v1[1] - v0[1]];
            self.target[0] += (dd[0] - dd[1]) * 0.5;
            // sy of a rotated delta is (dx + dy) * (0.5 * tilt): the hardcoded
            // 0.25 only matched tilt = 0.5, so at any other tilt the pivot
            // drifted away from the cursor while rotating.
            let t = self.tilt.clamp(0.05, 1.0);
            self.target[1] += (dd[0] + dd[1]) * (0.5 * t);
            self.rot_applied = self.rot;
        }
        // eased tilt; the view reclines about the ground point under the
        // SCREEN CENTRE (the OrbitControls convention: the focus point keeps
        // its screen spot). Anchoring on the click pivot instead made close
        // views swing hard — any offset from the cursor got stretched by
        // (x + y) * dtilt and the whole street flew off at high zoom.
        let kt = 1.0 - (-14.0 * dt).exp();
        self.tilt += (self.tilt_target - self.tilt) * kt;
        if (self.tilt - self.tilt_applied).abs() > 1e-6 {
            // sy = (x + y) * 0.5 * tilt - z  =>  the ground point held in the
            // screen centre has (x + y) * 0.5 = target[1] / tilt
            let t0 = self.tilt_applied.max(0.07);
            let p = (self.target[1] / t0).clamp(-2.0 * self.grid, 2.0 * self.grid);
            self.target[1] += p * (self.tilt - self.tilt_applied);
            self.tilt_applied = self.tilt;
        }
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
            // the water plane height is filled by TerrainRenderer::set_params
            _pad0: 0.0,
            tilt: self.tilt,
            _pad2: 0.0,
        }
    }

    /// Point the rotation pivot at the terrain under a screen position:
    /// invert the iso projection, then walk the view ray onto the smoothed
    /// height field (the mesh is drawn from the same field, so the pivot
    /// lands where the user clicked).
    fn set_pivot_from_screen(
        &mut self,
        sm: &[f32],
        n: usize,
        mn: f32,
        px: f64,
        py: f64,
        win: (f64, f64),
        base_scale: f32,
    ) {
        let scale = (base_scale * self.zoom).max(1e-3);
        let a = self.target[0] + (px as f32 - win.0 as f32 * 0.5) / scale;
        let b = self.target[1] + (py as f32 - win.1 as f32 * 0.5) / scale;
        // Iso ray: x = s + a, y = s - a and sy = (x + y) * (0.5 * tilt) - z,
        // so s = (b + z(s)) / tilt.  The old code hardcoded tilt = 0.5 (s =
        // 2*(b+z)) which put the pivot at the wrong map point for every other
        // tilt — the view then orbited around a phantom axis.
        let t = self.tilt.clamp(0.07, 1.0);
        let smax = 2.0 * self.grid;
        let mut s = (b / t).clamp(0.0, smax);
        for _ in 0..20 {
            let cx = (s + a).clamp(0.0, self.grid);
            let cy = (s - a).clamp(0.0, self.grid);
            let zmesh = (sample_h(sm, n, cx, cy) - mn) * ZSCALE;
            s = ((b + zmesh) / t).clamp(0.0, smax);
        }
        // The ray inversion yields camera-rotated ("rp") coordinates, but the
        // pivot compensation in `step` needs a world point: undo the yaw here,
        // otherwise every rotation after the first orbits a wrong spot.
        let rp = [s + a, s - a];
        let c = self.grid * 0.5;
        let (sr, cr) = self.rot.sin_cos();
        let d = [rp[0] - c, rp[1] - c];
        self.pivot = [
            (c + d[0] * cr + d[1] * sr).clamp(0.0, self.grid),
            (c - d[0] * sr + d[1] * cr).clamp(0.0, self.grid),
        ];
    }
}

/// Bilinear sample of the smoothed height grid (metres) at cell coordinates.
fn sample_h(sm: &[f32], n: usize, cx: f32, cy: f32) -> f32 {
    let x = cx.clamp(0.0, (n - 1) as f32);
    let y = cy.clamp(0.0, (n - 1) as f32);
    let x0 = (x as usize).min(n.saturating_sub(2));
    let y0 = (y as usize).min(n.saturating_sub(2));
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let h00 = sm[y0 * n + x0];
    let h10 = sm[y0 * n + x0 + 1];
    let h01 = sm[(y0 + 1) * n + x0];
    let h11 = sm[(y0 + 1) * n + x0 + 1];
    (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fy) + (h01 * (1.0 - fx) + h11 * fx) * fy
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
    road: RoadRenderer,
    village: Vec<ModelRenderer>,
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

        let terrain = TerrainRenderer::new(
            &device,
            &queue,
            config.format,
            4,
            &scene.atlas,
            &scene.base,
            scene.mn,
            scene.wl,
        );
        let road = RoadRenderer::new(
            &device,
            &queue,
            config.format,
            4,
            &scene.road_mesh,
            &scene.road_textures,
        );
        let village: Vec<ModelRenderer> = scene
            .village
            .iter()
            .map(|(mesh, texs, pos)| {
                let r = ModelRenderer::new(&device, &queue, config.format, 4, mesh, texs);
                r.set_placement(&queue, *pos, 0.0, 0.0);
                r
            })
            .collect();
        let (msaa_view, depth_view) = make_targets(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            terrain,
            road,
            village,
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
        for r in &self.village {
            r.set_params(&self.queue, &params);
        }

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
            for r in &self.village {
                r.draw(&mut pass);
            }
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
                            if pressed {
                                // rotation pivots around the terrain point
                                // under the cursor
                                let base_scale = self
                                    .renderer
                                    .as_ref()
                                    .zip(self.scene.as_ref())
                                    .map(|(r, s)| r.base_scale(s))
                                    .unwrap_or(1.0);
                                if let (Some(scene), Some(cam), Some((px, py))) = (
                                    self.scene.as_ref(),
                                    self.camera.as_mut(),
                                    self.cursor,
                                ) {
                                    cam.set_pivot_from_screen(
                                        &scene.sm,
                                        scene.n,
                                        scene.mn,
                                        px,
                                        py,
                                        self.win_size,
                                        base_scale,
                                    );
                                }
                            }
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
                            cam.rotate_by(dx * 0.005);
                            // vertical drag = camera tilt (top-down <-> level);
                            // the closer the camera, the softer the tilt — the
                            // on-screen motion of any point scales with (x + y),
                            // which explodes at high zoom (OrbitControls-style
                            // zoom-dependent damping).
                            let soft = (12.0 / cam.zoom).clamp(0.25, 1.0);
                            cam.tilt_by(dy as f32 * 0.0020 * soft);
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
