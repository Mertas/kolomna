//! Offscreen isometric terrain preview.
//!
//! Renders a textured diamond terrain grid to a PNG: archive textures
//! (loaded from a local archive) on a procedural demo heightmap.
//!
//! usage: cargo run -p kolomna-render --example terrain_preview -- <file.dat> <out.png> [grid_side]

use std::borrow::Cow;
use std::env;
use std::io;

use bytemuck::{Pod, Zeroable};

const VIEW_W: u32 = 1600;
const VIEW_H: u32 = 900;
const DEF_GRID: usize = 48;
const TILE: u32 = 256;
const ATLAS_COLS: u32 = 3;
const ATLAS_ROWS: u32 = 2;
const FSH_TYPE: u32 = 0x7AB50E44;
const HEIGHT_MAX: f32 = 8.0;

/// data2.dat textures used for the demo atlas (group, instance), all 256x256 DXT1.
const SOURCES: [(u32, u32); 6] = [
    (0x891b0e1a, 0x00000013),
    (0x891b0e1a, 0x00000031),
    (0x891b0e1a, 0x00000032),
    (0x891b0e1a, 0x00000033),
    (0x891b0e1a, 0x00000034),
    (0x891b0e1a, 0x00000035),
];

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
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var o: VsOut;
    let iso = vec2<f32>((p.x - p.y) * 0.5, (p.x + p.y) * 0.25 - p.z * params.hz);
    let scr = (iso - params.center) * params.scale;
    o.pos = vec4<f32>(scr.x / params.half.x, -scr.y / params.half.y, 0.5 - p.z * 0.03, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(atlas, samp, in.uv);
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: terrain_preview <file.dat> <out.png> [grid_side]");
        std::process::exit(1);
    }
    let grid: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(DEF_GRID);

    let atlas = load_atlas(&args[0])?;
    println!(
        "atlas: {} textures -> {}x{} px",
        SOURCES.len(),
        ATLAS_COLS * TILE,
        ATLAS_ROWS * TILE
    );

    let (verts, hmax) = build_mesh(grid);
    println!(
        "mesh: {}x{} cells, {} vertices, height max {:.1}",
        grid,
        grid,
        verts.len(),
        hmax
    );

    let rgba = render(VIEW_W, VIEW_H, &atlas, &verts, grid as f32)?;
    write_png(&args[1], VIEW_W, VIEW_H, &rgba)?;
    println!("saved {}", args[1]);
    Ok(())
}

/// Build the demo atlas from archive textures inside the given .dat archive.
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

/// Deterministic value noise in 0..1.
fn hash2(x: i32, y: i32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(374_761_393)
        .wrapping_add((y as u32).wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (h ^ (h >> 16)) as f32 / u32::MAX as f32
}

fn value_noise(x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (xf, yf) = (x - xi as f32, y - yi as f32);
    let (sx, sy) = (xf * xf * (3.0 - 2.0 * xf), yf * yf * (3.0 - 2.0 * yf));
    let v00 = hash2(xi, yi);
    let v10 = hash2(xi + 1, yi);
    let v01 = hash2(xi, yi + 1);
    let v11 = hash2(xi + 1, yi + 1);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

fn fbm(x: f32, y: f32) -> f32 {
    let (mut amp, mut fr, mut sum, mut norm) = (1.0f32, 1.0f32, 0.0f32, 0.0f32);
    for _ in 0..4 {
        sum += value_noise(x * fr, y * fr) * amp;
        norm += amp;
        amp *= 0.5;
        fr *= 2.0;
    }
    sum / norm
}

/// Procedural demo heightmap meshed as iso diamonds; texture band picked by height.
fn build_mesh(grid: usize) -> (Vec<Vertex>, f32) {
    let g = grid + 1;
    let mut heights = vec![0f32; g * g];
    let mut hmax = 0f32;
    for y in 0..g {
        for x in 0..g {
            let h = fbm(x as f32 * 0.075, y as f32 * 0.075).powf(1.6) * HEIGHT_MAX;
            heights[y * g + x] = h;
            hmax = hmax.max(h);
        }
    }

    let aw = (ATLAS_COLS * TILE) as f32;
    let ah = (ATLAS_ROWS * TILE) as f32;
    let corner = |x: usize, y: usize| -> [f32; 3] { [x as f32, y as f32, heights[y * g + x]] };
    let mut verts = Vec::with_capacity(grid * grid * 6);
    for cy in 0..grid {
        for cx in 0..grid {
            let havg = (heights[cy * g + cx]
                + heights[cy * g + cx + 1]
                + heights[(cy + 1) * g + cx]
                + heights[(cy + 1) * g + cx + 1])
                * 0.25;
            let band = ((havg / HEIGHT_MAX) * (SOURCES.len() as f32 - 0.001)) as usize;
            let col = (band as u32) % ATLAS_COLS;
            let row = (band as u32) / ATLAS_COLS;
            // half-texel inset to avoid bleeding between atlas tiles
            let u0 = ((col * TILE) as f32 + 0.5) / aw;
            let v0 = ((row * TILE) as f32 + 0.5) / ah;
            let u1 = (((col + 1) * TILE) as f32 - 0.5) / aw;
            let v1 = (((row + 1) * TILE) as f32 - 0.5) / ah;

            let p00 = corner(cx, cy);
            let p10 = corner(cx + 1, cy);
            let p01 = corner(cx, cy + 1);
            let p11 = corner(cx + 1, cy + 1);
            verts.push(Vertex {
                pos: p00,
                uv: [u0, v0],
            });
            verts.push(Vertex {
                pos: p10,
                uv: [u1, v0],
            });
            verts.push(Vertex {
                pos: p11,
                uv: [u1, v1],
            });
            verts.push(Vertex {
                pos: p00,
                uv: [u0, v0],
            });
            verts.push(Vertex {
                pos: p11,
                uv: [u1, v1],
            });
            verts.push(Vertex {
                pos: p01,
                uv: [u0, v1],
            });
        }
    }
    (verts, hmax)
}

fn render(
    w: u32,
    h: u32,
    atlas: &[u8],
    verts: &[Vertex],
    grid: f32,
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
        label: Some("kolomna-terrain-preview"),
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

    let center = [0.0f32, grid * 0.25 - HEIGHT_MAX * 0.5];
    let vspan = grid * 0.5 + HEIGHT_MAX;
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
            depth_compare: Some(wgpu::CompareFunction::Less),
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

    let bpr = w * 4; // 6400 px rows: multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256)
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
