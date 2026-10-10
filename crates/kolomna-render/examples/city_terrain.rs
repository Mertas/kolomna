//! Offscreen isometric render of a saved city's terrain.
//!
//! Reads the terrain height grid from a city save (chunk a9dd6ff4/e98f9525/1:
//! u16 version, then an n*n grid of f32 altitudes in meters, n = 129),
//! meshes it as iso diamonds and textures it with its own terrain atlas.
//! Water cells (flat plateaus near the minimum) get the water tile; a thin
//! sandy rim marks the banks; land is a mosaic of grass variants.
//!
//! usage: cargo run -p kolomna-render --example city_terrain -- <city.save> <data2.dat> <out.png>

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::io;

use bytemuck::{Pod, Zeroable};

const VIEW_W: u32 = 1600;
const VIEW_H: u32 = 1200;
const TILE: u32 = 256;
const ATLAS_COLS: u32 = 3;
const ATLAS_ROWS: u32 = 2;
const FSH_TYPE: u32 = 0x7AB50E44;
const TERRAIN_TYPE: u32 = 0xa9dd6ff4;
const TERRAIN_GROUP: u32 = 0xe98f9525;
const TERRAIN_INSTANCE: u32 = 1;
/// map units per meter of altitude (visual relief exaggeration)
const ZSCALE: f32 = 0.14;

/// data2.dat textures for the atlas (group, instance), all 256x256 DXT1.
const SOURCES: [(u32, u32); 6] = [
    (0x891b0e1a, 0x00000013),
    (0x891b0e1a, 0x00000031),
    (0x891b0e1a, 0x00000032),
    (0x891b0e1a, 0x00000033),
    (0x891b0e1a, 0x00000034),
    (0x891b0e1a, 0x00000035),
];
/// atlas cell used for water cells
const WATER_TILE: usize = 0;
/// atlas cells mixed stably over the land (grass with occasional dry patches)
const LAND_TILES: [usize; 6] = [5, 4, 5, 4, 2, 1];
/// atlas cell used for the sandy rim just above the water
const BANK_TILE: usize = 2;
/// texture block size in cells (texture instance spans BLOCK x BLOCK cells)
const BLOCK: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    center: [f32; 2],
    scale: f32,
    hz: f32,
    half: [f32; 2],
    _pad: [f32; 2],
}

const SHADER: &str = r#"
struct Params {
    center: vec2<f32>,
    scale: f32,
    hz: f32,
    half: vec2<f32>,
    pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) wpos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    let iso = vec2<f32>((p.x - p.y) * 0.5, (p.x + p.y) * 0.25 - p.z * params.hz);
    let scr = (iso - params.center) * params.scale;
    let ndc = vec2<f32>(scr.x / params.half.x, -scr.y / params.half.y);
    o.pos = vec4<f32>(ndc.x, ndc.y, (ndc.y + 1.0) * 0.499 + 0.001, 1.0);
    o.uv = uv;
    o.wpos = p;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(cross(dpdx(in.wpos), dpdy(in.wpos)));
    let l = normalize(vec3<f32>(-0.40, -0.55, 0.85));
    let lit = 0.66 + 0.44 * max(dot(n, l), 0.0);
    let c = textureSample(atlas, samp, in.uv);
    return vec4<f32>(c.rgb * lit, 1.0);
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: city_terrain <city.save> <data2.dat> <out.png>");
        std::process::exit(1);
    }

    let terrain = load_terrain(&args[0])?;
    let n = (terrain.len() as f64).sqrt().round() as usize;
    if n * n != terrain.len() {
        return Err(format!("terrain grid is not square ({} values)", terrain.len()).into());
    }
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    println!("terrain: {n}x{n} vertices, altitude {mn:.1}..{mx:.1} m, water level {wl:.2} m");

    let atlas = load_atlas(&args[1])?;
    println!(
        "atlas: {} textures -> {}x{} px",
        SOURCES.len(),
        ATLAS_COLS * TILE,
        ATLAS_ROWS * TILE
    );

    let verts = build_mesh(&terrain, mn, mx, wl);
    println!("mesh: {} triangles", verts.len() / 3);

    let zspan = (mx - mn) * ZSCALE;
    let rgba = render(VIEW_W, VIEW_H, &atlas, &verts, (n - 1) as f32, zspan)?;
    write_png(&args[2], VIEW_W, VIEW_H, &rgba)?;
    println!("saved {}", args[2]);
    Ok(())
}

/// Read the terrain chunk (u16 version + n*n f32 altitudes).
fn load_terrain(city: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

/// Water level = the most frequent exact altitude near the minimum (flat water
/// surfaces are stored as a tight family around one value, e.g. 198.00..198.40);
/// take the flat mode + half a metre. Fallback: min + 6 m.
fn water_level(vals: &[f32], mn: f32) -> f32 {
    let mut hist: HashMap<u32, usize> = HashMap::new();
    for &v in vals {
        if v < mn + 30.0 {
            *hist.entry((v * 4096.0).round() as u32).or_insert(0) += 1;
        }
    }
    match hist.iter().max_by_key(|&(_, &c)| c) {
        Some((&q, &cnt)) if cnt >= 24 => q as f32 / 4096.0 + 0.45,
        _ => mn + 6.0,
    }
}

/// Mesh the height grid as iso diamonds; per-cell texture from the atlas.
fn build_mesh(vals: &[f32], mn: f32, _mx: f32, wl: f32) -> Vec<Vertex> {
    let n = (vals.len() as f64).sqrt().round() as usize;
    let aw = (ATLAS_COLS * TILE) as f32;
    let ah = (ATLAS_ROWS * TILE) as f32;
    let zof = |h: f32| (h.max(wl) - mn) * ZSCALE;
    let mut verts = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for cy in 0..n - 1 {
        for cx in 0..n - 1 {
            let h00 = vals[cy * n + cx];
            let h10 = vals[cy * n + cx + 1];
            let h01 = vals[(cy + 1) * n + cx];
            let h11 = vals[(cy + 1) * n + cx + 1];
            let havg = (h00 + h10 + h01 + h11) * 0.25;

            let cell = if havg <= wl + 0.02 {
                WATER_TILE
            } else if havg < wl + 3.0 {
                BANK_TILE
            } else {
                let bx = (cx / BLOCK) as u32;
                let by = (cy / BLOCK) as u32;
                let hsh = bx.wrapping_mul(73_856_093) ^ by.wrapping_mul(19_349_663);
                LAND_TILES[(hsh as usize) % LAND_TILES.len()]
            };

            // water: whole tile; land: quadrant of a BLOCK x BLOCK block
            let (u0, v0, u1, v1) = if cell == WATER_TILE {
                let col = (cell as u32) % ATLAS_COLS;
                let row = (cell as u32) / ATLAS_COLS;
                (
                    ((col * TILE) as f32 + 0.5) / aw,
                    ((row * TILE) as f32 + 0.5) / ah,
                    (((col + 1) * TILE) as f32 - 0.5) / aw,
                    (((row + 1) * TILE) as f32 - 0.5) / ah,
                )
            } else {
                let col = (cell as u32) % ATLAS_COLS;
                let row = (cell as u32) / ATLAS_COLS;
                let qx = (cx % BLOCK) as f32;
                let qy = (cy % BLOCK) as f32;
                let qw = (TILE / BLOCK as u32) as f32;
                (
                    ((col * TILE) as f32 + qx * qw + 0.5) / aw,
                    ((row * TILE) as f32 + qy * qw + 0.5) / ah,
                    ((col * TILE) as f32 + (qx + 1.0) * qw - 0.5) / aw,
                    ((row * TILE) as f32 + (qy + 1.0) * qw - 0.5) / ah,
                )
            };

            let (z00, z10, z01, z11) = (zof(h00), zof(h10), zof(h01), zof(h11));
            let x0 = cx as f32;
            let x1 = (cx + 1) as f32;
            let y0 = cy as f32;
            let y1 = (cy + 1) as f32;
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
            });
            verts.push(Vertex {
                pos: [x1, y0, z10],
                uv: [u1, v0],
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
            });
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
            });
            verts.push(Vertex {
                pos: [x0, y1, z01],
                uv: [u0, v1],
            });
        }
    }
    verts
}

/// Build the atlas from archive textures inside the given .dat archive.
fn load_atlas(dat_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let archive = kolomna_dbpf::Archive::open(dat_path)?;
    let aw = ATLAS_COLS * TILE;
    let ah = ATLAS_ROWS * TILE;
    let mut atlas = vec![128u8; (aw * ah * 4) as usize];
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
        let col = (i as u32) % ATLAS_COLS;
        let row = (i as u32) / ATLAS_COLS;
        for y in 0..TILE {
            let src = (y * TILE * 4) as usize;
            let dst = (((row * TILE + y) * aw + col * TILE) * 4) as usize;
            atlas[dst..dst + (TILE * 4) as usize]
                .copy_from_slice(&img.rgba[src..src + (TILE * 4) as usize]);
        }
    }
    Ok(atlas)
}

fn render(
    w: u32,
    h: u32,
    atlas: &[u8],
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
        label: Some("city-terrain"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("failed to create device");

    let atlas_w = ATLAS_COLS * TILE;
    let atlas_h = ATLAS_ROWS * TILE;
    let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("atlas"),
        size: wgpu::Extent3d {
            width: atlas_w,
            height: atlas_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        atlas,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(atlas_w * 4),
            rows_per_image: Some(atlas_h),
        },
        wgpu::Extent3d {
            width: atlas_w,
            height: atlas_h,
            depth_or_array_layers: 1,
        },
    );
    let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("atlas-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let center = [0.0f32, grid * 0.25 - zspan * 0.5];
    let vspan = grid * 0.5 + zspan;
    let scale = ((w as f32 - 60.0) / grid).min((h as f32 - 60.0) / vspan);
    let params = Params {
        center,
        scale,
        hz: 1.0,
        half: [w as f32 / 2.0, h as f32 / 2.0],
        _pad: [0.0; 2],
    };
    let params_buf = {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
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
                    view_dimension: wgpu::TextureViewDimension::D2,
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
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
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
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
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
                resolve_target: None,
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, verts_buf.slice(..));
        pass.draw(0..verts.len() as u32, 0..1);
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
            texture: &target,
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

fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}
