//! Kolomna render library: terrain extraction, meshing and isometric wgpu rendering.
//!
//! Reads the terrain height grid from a city save (chunk a9dd6ff4/e98f9525/1:
//! u16 version, then an n*n grid of f32 altitudes in meters, n = 129),
//! meshes it as iso diamonds and textures it with its own terrain atlas.
//!
//! Surface rules (v2):
//! - water is a flat plane at the detected surface level, shaded procedurally
//!   (noise waves, sun glints, bed-depth tint, shore foam);
//! - a thin sandy rim marks the banks; steep cliffs (> STEEP_M between cell
//!   corners) get the dirt texture; grass comes in three height-bands
//!   (light meadow / mid / dark forest) dithered at the edges;
//! - land textures use a continuous world UV with REPEAT sampling (each
//!   texture instance spans REPEAT_CELLS x REPEAT_CELLS cells) so nothing
//!   looks like per-cell squares.
//!
//! The rendering internals are shared by the `city_terrain` example (offscreen
//! CLI) and `kolomna-app` (windowed + `--shot`).

use std::borrow::Cow;
use std::collections::HashMap;
use std::io;

use bytemuck::{Pod, Zeroable};

pub const VIEW_W: u32 = 1600;
pub const VIEW_H: u32 = 1200;
pub const TILE: u32 = 256;
pub const FSH_TYPE: u32 = 0x7AB50E44;
pub const TERRAIN_TYPE: u32 = 0xa9dd6ff4;
pub const TERRAIN_GROUP: u32 = 0xe98f9525;
pub const TERRAIN_INSTANCE: u32 = 1;
/// map units per meter of altitude (visual relief exaggeration)
pub const ZSCALE: f32 = 0.14;
/// world cells per texture repeat (texture instance spans this many cells)
pub const REPEAT_CELLS: f32 = 5.0;
/// water surface sits this many metres above the flat-bed mode (see water_level)
pub const WATER_MARGIN: f32 = 2.2;
/// sandy rim band above the water surface (m)
pub const BANK_BAND: f32 = 2.0;
/// corner-height spread (m) within a cell that paints it as a cliff (dirt)
pub const STEEP_M: f32 = 12.0;

/// data2.dat terrain textures (group, instance), all 256x256 DXT1.
/// Layers: 0 = water (unused by shaders), 1 = dark forest green, 2 = sand,
/// 3 = dirt, 4 = mid green, 5 = light meadow green.
pub const SOURCES: [(u32, u32); 6] = [
    (0x891b0e1a, 0x00000013),
    (0x891b0e1a, 0x00000031),
    (0x891b0e1a, 0x00000032),
    (0x891b0e1a, 0x00000033),
    (0x891b0e1a, 0x00000034),
    (0x891b0e1a, 0x00000035),
];
const LAYER_SAND: f32 = 2.0;
const LAYER_DIRT: f32 = 3.0;
const LAYER_GRASS_MID: f32 = 4.0;
const LAYER_GRASS_LIGHT: f32 = 5.0;
const LAYER_GRASS_DARK: f32 = 1.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    /// continuous world uv (cell coordinate / REPEAT_CELLS), REPEAT sampling
    pub uv: [f32; 2],
    /// water bed depth in meters (>= 0.10) for water vertices; -1 for dry land
    pub wdep: f32,
    /// texture layer index for land vertices (ignored for water)
    pub layer: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub center: [f32; 2],
    pub scale: f32,
    pub hz: f32,
    pub half: [f32; 2],
    /// world rotation around `pivot` (radians)
    pub angle: f32,
    /// seconds since start (water shimmer phase)
    pub time: f32,
    /// rotation pivot (grid / 2)
    pub pivot: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

const SHADER: &str = r#"
struct Params {
    center: vec2<f32>,
    scale: f32,
    hz: f32,
    half: vec2<f32>,
    angle: f32,
    time: f32,
    pivot: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var atlas: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) wpos: vec3<f32>,
    @location(2) wdep: f32,
    @location(3) layer: f32,
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) wdep: f32, @location(3) layer: f32) -> VsOut {
    var o: VsOut;
    let d = p.xy - vec2<f32>(params.pivot);
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rp = vec2<f32>(d.x * ca - d.y * sa, d.x * sa + d.y * ca) + vec2<f32>(params.pivot);
    let iso = vec2<f32>((rp.x - rp.y) * 0.5, (rp.x + rp.y) * 0.25 - p.z * params.hz);
    let scr = (iso - params.center) * params.scale;
    let ndc = vec2<f32>(scr.x / params.half.x, -scr.y / params.half.y);
    o.pos = vec4<f32>(ndc.x, ndc.y, (ndc.y + 1.0) * 0.499 + 0.001, 1.0);
    o.uv = uv;
    o.wpos = p;
    o.wdep = wdep;
    o.layer = layer;
    return o;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 345.45));
    q = q + dot(q, q + 34.345);
    return fract(q.x * q.y);
}

// two-octave value noise: non-repeating wave field for water normals
fn wwater(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n1 = mix(mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), u.x),
                 mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
    let p2 = p * 2.7 + vec2<f32>(13.7, 7.3);
    let i2 = floor(p2);
    let f2 = fract(p2);
    let u2 = f2 * f2 * (3.0 - 2.0 * f2);
    let n2 = mix(mix(hash21(i2), hash21(i2 + vec2<f32>(1.0, 0.0)), u2.x),
                 mix(hash21(i2 + vec2<f32>(0.0, 1.0)), hash21(i2 + vec2<f32>(1.0, 1.0)), u2.x), u2.y);
    return n1 + 0.5 * n2;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // water: procedural wave normals + fresnel sky + sun glint + depth tint + shore foam
    if (in.wdep > -0.5) {
        let drift = vec2<f32>(params.time * 0.11, params.time * 0.07);
        let wp = in.wpos.xy * 1.3 + drift;
        let e = 0.30;
        let h0 = wwater(wp);
        let hx = wwater(wp + vec2<f32>(e, 0.0));
        let hy = wwater(wp + vec2<f32>(0.0, e));
        let g = (vec2<f32>(h0 - hx, h0 - hy) / e) * 0.12;
        let n = normalize(vec3<f32>(g.x, g.y, 1.0));
        let sun = normalize(vec3<f32>(-0.30, -0.50, 0.55));
        let eye = normalize(vec3<f32>(0.15, 0.25, 0.95));
        let spec = pow(max(dot(reflect(-sun, n), eye), 0.0), 90.0) * 1.1;
        let fres = pow(1.0 - max(dot(n, eye), 0.0), 2.0);
        let dep = clamp(in.wdep / 6.0, 0.0, 1.0);
        let shallow = vec3<f32>(0.16, 0.28, 0.38);
        let deep = vec3<f32>(0.02, 0.06, 0.11);
        let sky = vec3<f32>(0.55, 0.67, 0.82);
        var col = mix(shallow, deep, dep);
        col = col + sky * (0.10 + fres * 0.55);
        col = col + vec3<f32>(1.0, 0.98, 0.92) * spec;
        let shore = 1.0 - smoothstep(0.1, 0.85, in.wdep);
        col = mix(col, vec3<f32>(0.30, 0.34, 0.33), shore * 0.35);
        return vec4<f32>(col, 1.0);
    }
    let n = normalize(cross(dpdx(in.wpos), dpdy(in.wpos)));
    let l = normalize(vec3<f32>(-0.40, -0.55, 0.85));
    let lit = 0.72 + 0.38 * max(dot(n, l), 0.0);
    let c = textureSample(atlas, samp, in.uv, i32(in.layer + 0.5));
    return vec4<f32>(c.rgb * lit, 1.0);
}
"#;

/// Grid side n of a square terrain grid (`(len as f64).sqrt().round()`).
pub fn grid_side(vals: &[f32]) -> usize {
    (vals.len() as f64).sqrt().round() as usize
}

/// Read the terrain chunk (u16 version + n*n f32 altitudes).
pub fn load_terrain(city: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let archive = kolomna_dbpf::Archive::open(city)?;
    let e = archive
        .find(TERRAIN_TYPE, TERRAIN_GROUP, TERRAIN_INSTANCE)
        .or_else(|| {
            archive
                .entries_of_type(TERRAIN_TYPE)
                .find(|e| e.tgi.instance_id == TERRAIN_INSTANCE)
        })
        .ok_or_else(|| {
            format!("terrain chunk {TERRAIN_TYPE:08x}/.../{TERRAIN_INSTANCE} not found in {city}")
        })?;
    let raw = archive.read_subfile(e)?;
    if raw.len() < 6 || (raw.len() - 2) % 4 != 0 {
        return Err(format!("terrain chunk: unexpected size {}", raw.len()).into());
    }
    let n = ((raw.len() - 2) / 4) as f64;
    let n = n.sqrt().round() as usize;
    if n * n * 4 + 2 != raw.len() {
        return Err(format!(
            "terrain chunk: {} bytes is not a square f32 grid",
            raw.len()
        )
        .into());
    }
    let vals = (0..n * n)
        .map(|k| {
            let o = 2 + 4 * k;
            f32::from_le_bytes(raw[o..o + 4].try_into().unwrap())
        })
        .collect();
    Ok(vals)
}

/// Water SURFACE level. The save stores the water bed; flat beds cluster on a
/// tight family whose mode is the deep-flat value (e.g. 198.00 m), and the
/// shallow shelves ripple up to ~2 m above it. We take the mode + WATER_MARGIN
/// as the surface so shallow cells are water (not grass poking through).
/// Fallback when there is no flat family: min + 6 m.
pub fn water_level(vals: &[f32], mn: f32) -> f32 {
    let mut hist: HashMap<u32, usize> = HashMap::new();
    for &v in vals {
        if v < mn + 30.0 {
            *hist.entry((v * 4096.0).round() as u32).or_insert(0) += 1;
        }
    }
    match hist.iter().max_by_key(|&(_, &c)| c) {
        Some((&q, &cnt)) if cnt >= 24 => q as f32 / 4096.0 + WATER_MARGIN,
        _ => mn + 6.0,
    }
}

/// Mesh the height grid as iso diamonds with continuous world UVs.
/// Water cells form a flat plane at `wl` and carry a per-vertex bed-depth
/// attribute for the procedural water shading; land is painted by rules
/// (sand rim / cliffs / three grass bands) — no per-cell texture squares.
pub fn build_mesh(vals: &[f32], mn: f32, _mx: f32, wl: f32) -> Vec<Vertex> {
    let n = grid_side(vals);
    let zof = |h: f32| (h.max(wl) - mn) * ZSCALE;
    let mut verts = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for cy in 0..n - 1 {
        for cx in 0..n - 1 {
            let h00 = vals[cy * n + cx];
            let h10 = vals[cy * n + cx + 1];
            let h01 = vals[(cy + 1) * n + cx];
            let h11 = vals[(cy + 1) * n + cx + 1];
            let havg = (h00 + h10 + h01 + h11) * 0.25;

            let is_water = havg <= wl + 0.02;
            let layer: f32 = if is_water {
                0.0
            } else if havg < wl + BANK_BAND {
                LAYER_SAND
            } else {
                let spread = h00.max(h10).max(h01).max(h11) - h00.min(h10).min(h01).min(h11);
                if spread > STEEP_M {
                    LAYER_DIRT
                } else {
                    // three grass tints by altitude, dithered at the band edges
                    let hsh = (cx as u32).wrapping_mul(73_856_093)
                        ^ (cy as u32).wrapping_mul(19_349_663);
                    let jit = ((hsh >> 8) & 0xff) as f32 / 255.0 * 16.0 - 8.0;
                    let h = havg + jit;
                    if h > wl + 60.0 {
                        LAYER_GRASS_DARK
                    } else if h > wl + 25.0 {
                        LAYER_GRASS_MID
                    } else {
                        LAYER_GRASS_LIGHT
                    }
                }
            };

            // per-vertex bed depth in meters for the water shader; -1 = dry land
            let wd = |h: f32| {
                if is_water {
                    (wl - h).clamp(0.10, 8.0)
                } else {
                    -1.0
                }
            };
            let (z00, z10, z01, z11) = (zof(h00), zof(h10), zof(h01), zof(h11));
            let (d00, d10, d01, d11) = (wd(h00), wd(h10), wd(h01), wd(h11));
            let x0 = cx as f32;
            let x1 = (cx + 1) as f32;
            let y0 = cy as f32;
            let y1 = (cy + 1) as f32;
            // continuous UV: shared corners share UVs -> texture flows across cells
            let (u0, u1) = (x0 / REPEAT_CELLS, x1 / REPEAT_CELLS);
            let (v0, v1) = (y0 / REPEAT_CELLS, y1 / REPEAT_CELLS);
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
                wdep: d00,
                layer,
            });
            verts.push(Vertex {
                pos: [x1, y0, z10],
                uv: [u1, v0],
                wdep: d10,
                layer,
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
                wdep: d11,
                layer,
            });
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
                wdep: d00,
                layer,
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
                wdep: d11,
                layer,
            });
            verts.push(Vertex {
                pos: [x0, y1, z01],
                uv: [u0, v1],
                wdep: d01,
                layer,
            });
        }
    }
    verts
}

/// Load the terrain textures from the archive as a flat RGBA8 buffer:
/// SOURCES.len() layers of TILE x TILE x 4 bytes, layer order = SOURCES order.
pub fn load_atlas(dat_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let archive = kolomna_dbpf::Archive::open(dat_path)?;
    let layer_bytes = (TILE * TILE * 4) as usize;
    let mut layers = vec![0u8; layer_bytes * SOURCES.len()];
    for (i, (group, instance)) in SOURCES.iter().enumerate() {
        let e = archive
            .find(FSH_TYPE, *group, *instance)
            .ok_or_else(|| format!("texture {group:08x}/{instance:08x} not found in {dat_path}"))?;
        let raw = archive.read_subfile(e)?;
        let fsh = kolomna_assets::fsh::Fsh::parse(&raw)?;
        let img = fsh
            .decode_entry(&raw, 0)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if img.width != TILE || img.height != TILE {
            return Err(format!(
                "texture {group:08x}/{instance:08x}: unexpected size {}x{}",
                img.width, img.height
            )
            .into());
        }
        layers[i * layer_bytes..(i + 1) * layer_bytes].copy_from_slice(&img.rgba);
    }
    Ok(layers)
}

/// Render-ready terrain: pipeline + bind group + buffers, usable with any
/// color target format (offscreen Rgba8UnormSrgb or a window surface).
pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    verts_buf: wgpu::Buffer,
    nverts: u32,
}

impl TerrainRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        sample_count: u32,
        layers: &[u8],
        verts: &[Vertex],
    ) -> Self {
        let layer_bytes = (TILE * TILE * 4) as usize;
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas-array"),
            size: wgpu::Extent3d {
                width: TILE,
                height: TILE,
                depth_or_array_layers: SOURCES.len() as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (i, chunk) in layers.chunks(layer_bytes).enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                chunk,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TILE * 4),
                    rows_per_image: Some(TILE),
                },
                wgpu::Extent3d {
                    width: TILE,
                    height: TILE,
                    depth_or_array_layers: 1,
                },
            );
        }
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let verts_buf = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("verts"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, 1 => Float32x2, 2 => Float32, 3 => Float32
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            params_buf,
            verts_buf,
            nverts: verts.len() as u32,
        }
    }

    pub fn set_params(&self, queue: &wgpu::Queue, params: &Params) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.verts_buf.slice(..));
        pass.draw(0..self.nverts, 0..1);
    }
}

/// Default camera params that frame the whole grid (same framing as before).
pub fn default_params(w: f32, h: f32, grid: f32, zspan: f32) -> Params {
    let vspan = grid * 0.5 + zspan;
    let scale = ((w - 60.0) / grid).min((h - 60.0) / vspan);
    Params {
        center: [0.0, grid * 0.25 - zspan * 0.5],
        scale,
        hz: 1.0,
        half: [w / 2.0, h / 2.0],
        angle: 0.0,
        time: 0.0,
        pivot: grid * 0.5,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    }
}

/// Render the terrain mesh offscreen (4x MSAA, resolve, readback) and return
/// the RGBA8 pixels (w * h * 4 bytes).
pub fn render_offscreen(
    w: u32,
    h: u32,
    layers: &[u8],
    verts: &[Vertex],
    grid: f32,
    zspan: f32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no graphics adapter found");
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("kolomna-render"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("failed to create device");

    let sample_count = 4u32;
    let renderer = TerrainRenderer::new(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        sample_count,
        layers,
        verts,
    );
    renderer.set_params(&queue, &default_params(w as f32, h as f32, grid, zspan));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target-msaa"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let resolved = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target-resolved"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let resolved_view = resolved.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("terrain-enc"),
    });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: Some(&resolved_view),
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
                view: &depth_view,
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
        renderer.draw(&mut pass);
    }

    let bpr = w * 4; // multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256)
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolved,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map channel closed").expect("map failed");
    let data = slice.get_mapped_range()?.to_vec();
    readback.unmap();
    Ok(data)
}

/// Encode RGBA8 pixels (w * h * 4 bytes) as a PNG file.
pub fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}
