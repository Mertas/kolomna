//! Road / network tiles from a city save: parse the `c9c05c6e`
//! (network occupant) records, build one textured mesh and render it on top
//! of the terrain with the shared isometric camera.
//!
//! Each record is one 16 x 16 m tile: 4 corner vertices (world metres, y =
//! absolute altitude) with UVs, a `texture_id` pointing at an FSH in group
//! 1abe787d, orientation, crossings and walls. Roads are drawn as flat
//! textured quads, so the mesh is triangles with an along-ray depth bias that
//! keeps the road above the ground it lies on.

use std::collections::BTreeMap;
use std::error::Error;

use bytemuck::{Pod, Zeroable};

use crate::{Params, ZSCALE};

pub const NETWORK_TYPE: u32 = 0xc9c0_5c6e;
pub const CITY_GROUP: u32 = 0x299b_2d1b;
/// 1 cell in metres (roads and terrain grids use 16 m per cell).
pub const CELL_M: f32 = 16.0;

#[derive(Clone, Debug)]
pub struct RoadTile {
    /// 4 corners in world metres: x horizontal, y altitude, z horizontal.
    pub verts: [[f32; 3]; 4],
    pub uv: [[f32; 2]; 4],
    /// FSH instance in group 1abe787d.
    pub tex: u32,
    pub orient: u8,
    pub crossings: Vec<[u8; 5]>,
}

// ------------------------------------------------------------- parsing ----

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, o: 0 }
    }
    fn remain(&self) -> usize {
        self.b.len().saturating_sub(self.o)
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.remain() < n {
            return None;
        }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn skip(&mut self, n: usize) -> Option<()> {
        self.take(n).map(|_| ())
    }
}

/// SGProp array: u32 count, then records (name x2, unknown u32, type u8,
/// key u8, unknown u16, value). Values we don't care about are skipped.
fn skip_sgprops(r: &mut Rd<'_>) -> Option<()> {
    let n = r.u32()?;
    for _ in 0..n {
        r.skip(12)?;
        let t = r.u8()?;
        let key = r.u8()?;
        r.skip(2)?;
        let sz: usize = match t {
            0x01 | 0x0b => 1,
            0x02 => 2,
            0x03 | 0x07 | 0x09 => 4,
            0x08 => 8,
            0x0c => {
                let len = r.u32()? as usize;
                r.skip(len)?;
                continue;
            }
            _ => 0,
        };
        if key == 0x80 {
            let count = r.u32()? as usize;
            r.skip(count.checked_mul(sz)?)?;
        } else {
            r.skip(sz)?;
        }
    }
    Some(())
}

fn parse_record(rec: &[u8]) -> Option<RoadTile> {
    let mut r = Rd::new(rec);
    let _size = r.u32()?;
    let _crc = r.u32()?;
    let _mem = r.u32()?;
    let _vmaj = r.u16()?;
    let _vmin = r.u16()?;
    let _appearance = r.u32()?;
    let _unknown = r.u32()?;
    r.skip(8)?; // tract info
    skip_sgprops(&mut r)?;
    r.skip(12)?; // gti (group / type / instance)
    r.skip(1)?; // unknown byte
    let _pos = (r.f32()?, r.f32()?, r.f32()?);
    let mut verts = [[0f32; 3]; 4];
    let mut uv = [[0f32; 2]; 4];
    for k in 0..4 {
        verts[k] = [r.f32()?, r.f32()?, r.f32()?];
        uv[k] = [r.f32()?, r.f32()?];
        r.skip(4)?; // vertex colour
    }
    let tex = r.u32()?;
    let _wealth = r.u8()?;
    let _base = r.u32()?;
    let orient = r.u8()?;
    r.skip(2)?;
    let nc = r.u8()? as usize + 1;
    let mut crossings = Vec::with_capacity(nc);
    for _ in 0..nc {
        let c = r.take(5)?;
        crossings.push([c[0], c[1], c[2], c[3], c[4]]);
    }
    let nwalls = r.u32()? as usize;
    r.skip(nwalls.checked_mul(28)?)?; // texture dword + vertex (20 + 4 colour)
    r.skip(6 * 4)?; // bbox (range mode)
    r.skip(4 * 2 + 12 + 8)?; // construction states, alt path id, 3 unknown, costs
    if r.remain() != 0 {
        return None; // record didn't parse cleanly
    }
    Some(RoadTile {
        verts,
        uv,
        tex,
        orient,
        crossings,
    })
}

/// Read every network tile from the save (chunk `c9c05c6e`).
pub fn load_road_tiles(city: &str) -> Result<Vec<RoadTile>, Box<dyn Error>> {
    let archive = kolomna_dbpf::Archive::open(city)?;
    let e = archive
        .find(NETWORK_TYPE, CITY_GROUP, 0)
        .or_else(|| archive.entries_of_type(NETWORK_TYPE).next())
        .ok_or_else(|| format!("network chunk {NETWORK_TYPE:08x} not found in {city}"))?;
    let raw = archive.read_subfile(e)?;
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 4 <= raw.len() {
        let size = u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        if size < 8 || o + size > raw.len() {
            break;
        }
        if let Some(t) = parse_record(&raw[o..o + size]) {
            out.push(t);
        }
        o += size;
    }
    Ok(out)
}

// ---------------------------------------------------------------- mesh ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RoadVertex {
    /// x / y in cells, z in cell units (`(altitude - mn) * ZSCALE`).
    pub pos: [f32; 3],
    pub uv: [f32; 2],
}

pub struct RoadGroup {
    pub first: u32,
    pub count: u32,
    /// index into `RoadMesh::texture_ids`
    pub slot: u32,
}

pub struct RoadMesh {
    pub verts: Vec<RoadVertex>,
    pub indices: Vec<u32>,
    pub groups: Vec<RoadGroup>,
    pub texture_ids: Vec<u32>,
}

/// Convert tiles to a render mesh: world metres -> cells, altitude -> mesh z.
/// Tiles are grouped per texture so each unique texture is one draw call.
pub fn build_road_mesh(tiles: &[RoadTile], mn: f32, wl: f32) -> RoadMesh {
    let mut by_tex: BTreeMap<u32, Vec<&RoadTile>> = BTreeMap::new();
    for t in tiles {
        by_tex.entry(t.tex).or_default().push(t);
    }
    let mut mesh = RoadMesh {
        verts: Vec::new(),
        indices: Vec::new(),
        groups: Vec::new(),
        texture_ids: Vec::new(),
    };
    for (tex, list) in by_tex {
        let slot = match mesh.texture_ids.iter().position(|&x| x == tex) {
            Some(k) => k as u32,
            None => {
                mesh.texture_ids.push(tex);
                (mesh.texture_ids.len() - 1) as u32
            }
        };
        let first = mesh.indices.len() as u32;
        for t in list {
            let base = mesh.verts.len() as u32;
            for k in 0..4 {
                let p = t.verts[k];
                let x = p[0] / CELL_M;
                let y = p[2] / CELL_M;
                let z = (p[1].max(wl) - mn) * ZSCALE;
                mesh.verts.push(RoadVertex {
                    pos: [x, y, z],
                    uv: t.uv[k],
                });
            }
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let count = mesh.indices.len() as u32 - first;
        mesh.groups.push(RoadGroup { first, count, slot });
    }
    mesh
}

// ------------------------------------------------------------ renderer ----

const ROAD_SHADER: &str = r#"
struct Params {
    center: vec2<f32>,
    scale: f32,
    hz: f32,
    half: vec2<f32>,
    angle: f32,
    time: f32,
    pivot: f32,
    water_z: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_road(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    let d = p.xy - vec2<f32>(params.pivot);
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rp = vec2<f32>(d.x * ca - d.y * sa, d.x * sa + d.y * ca) + vec2<f32>(params.pivot);
    let iso = vec2<f32>((rp.x - rp.y) * 0.5, (rp.x + rp.y) * 0.25 - p.z * params.hz);
    let scr = (iso - params.center) * params.scale;
    let ndc = vec2<f32>(scr.x / params.half.x, -scr.y / params.half.y);
    // same along-ray depth as the terrain, biased slightly toward the camera
    // so the road always wins against the ground it lies on
    let dep = (rp.x + rp.y + p.z * 0.5) * (0.9 / (4.0 * params.pivot + 40.0)) + 0.0008;
    var o: VsOut;
    o.pos = vec4<f32>(ndc.x, ndc.y, 0.95 - dep, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs_road(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

pub struct RoadRenderer {
    pipeline: wgpu::RenderPipeline,
    group0: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    parts: Vec<(RoadGroup, wgpu::BindGroup)>,
}

impl RoadRenderer {
    /// `textures` aligns with `mesh.texture_ids`: (width, height, rgba8).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        sample_count: u32,
        mesh: &RoadMesh,
        textures: &[(u32, u32, Vec<u8>)],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("road-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("road-bgl0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("road-bg0"),
            layout: &bgl0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("road-bgl1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("road-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut tex_bgs: Vec<wgpu::BindGroup> = Vec::new();
        for (k, _id) in mesh.texture_ids.iter().enumerate() {
            let (tw, th, rgba) = textures.get(k).cloned().unwrap_or_else(|| {
                // missing texture: magenta/black checker, 8x8
                let mut px = vec![0u8; 8 * 8 * 4];
                for y in 0..8usize {
                    for x in 0..8usize {
                        let o = (y * 8 + x) * 4;
                        let on = (x / 2 + y / 2) % 2 == 0;
                        px[o] = if on { 255 } else { 32 };
                        px[o + 1] = 32;
                        px[o + 2] = if on { 255 } else { 32 };
                        px[o + 3] = 255;
                    }
                }
                (8u32, 8u32, px)
            });
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("road-texture"),
                size: wgpu::Extent3d {
                    width: tw.max(1),
                    height: th.max(1),
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
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(tw.max(1) * 4),
                    rows_per_image: Some(th.max(1)),
                },
                wgpu::Extent3d {
                    width: tw.max(1),
                    height: th.max(1),
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            tex_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("road-tex-bg"),
                layout: &bgl1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }));
        }
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("road-verts"),
            contents: bytemuck::cast_slice(&mesh.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("road-indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("road-shader"),
            source: wgpu::ShaderSource::Wgsl(ROAD_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("road-layout"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("road-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_road"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RoadVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_road"),
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
        let parts = mesh
            .groups
            .iter()
            .map(|g| {
                let bg = tex_bgs
                    .get(g.slot as usize)
                    .expect("texture slot out of range")
                    .clone();
                (
                    RoadGroup {
                        first: g.first,
                        count: g.count,
                        slot: g.slot,
                    },
                    bg,
                )
            })
            .collect();
        Self {
            pipeline,
            group0,
            params_buf,
            vbuf,
            ibuf,
            parts,
        }
    }

    pub fn set_params(&self, queue: &wgpu::Queue, params: &Params) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group0, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        for (g, bg) in &self.parts {
            pass.set_bind_group(1, bg, &[]);
            pass.draw_indexed(g.first..g.first + g.count, 0, 0..1);
        }
    }
}
