//! Model rendering: meshes from parsed models + a textured wgpu
//! pipeline that shares the terrain camera (same `Params` uniform, same iso
//! projection and along-ray depth, so models occlude correctly against terrain).
//!
//! Model space conventions (from archived models):
//! - x / z are horizontal in METRES, y is up in METRES (bbox matches the save's
//!   occupied size in metres, e.g. 15.9 x 7 x 8 for a small house);
//! - one cell = 16 m, mesh units = cells horizontally and
//!   `metres * ZSCALE` vertically;
//! - material texture ids are FSH instances inside group 1abe787d.
//!
//! Each ANIM group's frame-0 block (vertexGroup, indexGroup, primGroup,
//! materialGroup) yields one textured submesh; vanilla non-animated models
//! have exactly one block. `render_scene_with_models` renders the terrain
//! plus any number of placed models offscreen to RGBA8 (4x MSAA).

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use kolomna_assets::mesh3d;

use crate::{Params, TerrainRenderer, Vertex};

const MODEL_SHADER: &str = r#"
struct Params {
    center: vec2<f32>,
    scale: f32,
    hz: f32,
    half: vec2<f32>,
    angle: f32,
    time: f32,
    pivot: f32,
    water_z: f32,
    /// camera tilt = sin(elevation)
    tilt: f32,
    pad2: f32,
};
struct ModelParams {
    pos: vec3<f32>,
    rot: f32,
    spin: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> mp: ModelParams;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) anim: f32,
    @location(3) wpos: vec3<f32>,
};

/// Travelling micro-ripples used for the water normal (analytic sines).
fn wv(p: vec2<f32>, t: f32) -> f32 {
    return sin(p.x * 1.9 + p.y * 1.1 + t * 1.3) * 0.45
         + sin(p.x * 3.7 - p.y * 2.9 - t * 1.9) * 0.35
         + sin(p.x * 8.1 + p.y * 6.7 + t * 2.7) * 0.20;
}

fn h21(v: vec2<f32>) -> f32 {
    let f = fract(v * vec2<f32>(0.1031, 0.1030));
    let d = dot(f, f.yx + vec2<f32>(33.33, 33.33));
    let g = f + vec2<f32>(d, d);
    return fract((g.x + g.y) * g.x);
}

@vertex
fn vs_model(@location(0) pp: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) nrm: vec3<f32>, @location(3) anim: f32) -> VsOut {
    // ---- water animation (fountain jets, sheets, basin ripples) ---------
    // fast and free: pure vertex-shader displacement, no CPU work per frame
    let tw = params.time;
    var wp = pp;
    if (anim >= 3.0) {
        // falling water sheet: the bottom sways, the rim edge stays put
        let w = clamp(anim - 3.0, 0.0, 1.0);
        let ph = h21(floor(vec2<f32>(pp.x, pp.z) * 2.13)) * 6.28318;
        wp.x = wp.x + sin(tw * 4.1 + ph + w * 3.0) * 0.012 * (1.0 - w);
        wp.z = wp.z + cos(tw * 3.3 + ph) * 0.010 * (1.0 - w);
    } else if (anim >= 2.0) {
        // a jet: pulsing, bobbing, swaying in the wind; the top moves most
        let w = clamp(anim - 2.0, 0.0, 1.0);
        let ph = h21(floor(vec2<f32>(pp.x, pp.z) * 1.31) + vec2<f32>(5.0, 9.0)) * 6.28318;
        let bob = 0.24 * sin(tw * 3.4 + ph);
        let sway = 0.13 * sin(tw * 2.1 + ph * 1.7) + 0.065 * sin(tw * 4.3 + ph * 2.3);
        let pulse = 0.52 * pow(0.5 + 0.5 * sin(tw * 5.6 + ph), 3.0);
        wp.y = wp.y + bob * (0.20 + 0.80 * w) + pulse * w;
        wp.x = wp.x + cos(ph * 1.3) * sway * w;
        wp.z = wp.z + sin(ph * 1.3) * sway * w;
    } else if (anim >= 1.0) {
        // basin / tier water: crossed ripples
        wp.y = wp.y + 0.035 * sin(pp.x * 6.0 + pp.z * 6.0 + tw * 2.6)
                     + 0.024 * sin(pp.x * 9.0 - pp.z * 9.0 - tw * 2.1);
    }
    let p = wp;
    // model y-up metres -> scene z-up; horizontal metres -> cells (16 m)
    // 3DMD buildings spin with the camera (spin = 1): the picked orientation
    // variant always faces the viewer, its rot compensating the camera angle.
    // Full 3-D models (procedural houses, spin = 0) are world-fixed.
    let rot = mp.rot - params.angle * mp.spin;
    let c = cos(rot);
    let s = sin(rot);
    let wx = (p.x * c - p.z * s) / 16.0 + mp.pos.x;
    let wy = (p.x * s + p.z * c) / 16.0 + mp.pos.y;
    let wz = p.y * 0.035 + mp.pos.z;
    let d = vec2<f32>(wx, wy) - vec2<f32>(params.pivot);
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rp = vec2<f32>(d.x * ca - d.y * sa, d.x * sa + d.y * ca) + vec2<f32>(params.pivot);
    let iso = vec2<f32>((rp.x - rp.y) * 0.5, (rp.x + rp.y) * (0.5 * params.tilt) - wz * params.hz);
    let scr = (iso - params.center) * params.scale;
    let ndc = vec2<f32>(scr.x / params.half.x, -scr.y / params.half.y);
    let ce = sqrt(max(1.0 - params.tilt * params.tilt, 0.01));
    let tk = params.tilt / ce;
    let dep = (rp.x + rp.y + wz * tk) * (0.9 / (4.0 * params.pivot + 40.0));
    var o: VsOut;
    o.pos = vec4<f32>(ndc.x, ndc.y, 0.95 - dep, 1.0);
    o.uv = uv;
    o.anim = anim;
    o.wpos = vec3<f32>(wx, wy, wz);
    // normals: model y-up -> scene z-up, then the scene rotation
    let nx0 = nrm.x * c - nrm.z * s;
    let nz0 = nrm.x * s + nrm.z * c;
    let ns = vec3<f32>(nx0, nz0, nrm.y);
    o.nrm = vec3<f32>(ns.x * ca - ns.y * sa, ns.x * sa + ns.y * ca, ns.z);
    return o;
}

@fragment
fn fs_model(in: VsOut) -> @location(0) vec4<f32> {
    var uv2 = in.uv;
    var shim = 0.0;
    if (in.anim >= 3.0) {
        // falling sheet: the streams run down fast with a lateral waver
        uv2.y = uv2.y + params.time * 1.35;
        uv2.x = uv2.x + sin(params.time * 2.3 + uv2.y * 9.0) * 0.010;
        shim = 0.08 * sin(uv2.y * 15.0 + params.time * 3.4);
    } else if (in.anim >= 2.0) {
        // a jet: the water streams up fast, with rising shimmer bands
        uv2.y = uv2.y - params.time * 1.70;
        shim = 0.12 * sin(uv2.y * 12.57 + params.time * 3.4);
    } else if (in.anim >= 1.0) {
        // basin / tier water: crossed drifts + a travelling sparkle
        uv2 = uv2 + vec2<f32>(params.time * 0.035, params.time * 0.022);
        shim = 0.09 * sin(uv2.x * 21.0 + uv2.y * 17.0 + params.time * 2.4);
    }
    var t = textureSample(tex, samp, uv2);
    if (t.a < 0.5) {
        discard;
    }
    var caust = 0.0;
    if (in.anim > 0.4 && in.anim < 0.6) {
        // underwater floor: a moving net of light (cheap caustics)
        caust = abs(wv(in.wpos.xy * 9.0, params.time * 0.9));
    }
    var floor_tint = 1.0;
    if (in.anim > 0.4 && in.anim < 0.6) {
        floor_tint = 0.84;
    }
    let n = normalize(in.nrm);
    let l = normalize(vec3<f32>(-0.40, -0.55, 0.85));
    var lit = 0.60 + 0.50 * max(abs(dot(n, l)), 0.0);
    var rgb = t.rgb;
    var alpha = 1.0;
    if (in.anim >= 1.0) {
        // ---- realistic water: rippled normals, sun glints, fresnel sky ----
        let ce2 = sqrt(max(1.0 - params.tilt * params.tilt, 0.01));
        let tk2 = params.tilt / ce2;
        let v = normalize(vec3<f32>(1.0, 1.0, tk2)); // towards the camera
        let q = in.wpos.xy * 6.0;
        let e = 0.14;
        let gx = wv(q + vec2<f32>(e, 0.0), params.time) - wv(q - vec2<f32>(e, 0.0), params.time);
        let gy = wv(q + vec2<f32>(0.0, e), params.time) - wv(q - vec2<f32>(0.0, e), params.time);
        let n2 = normalize(vec3<f32>(-gx * 2.4, -gy * 2.4, 1.0));
        let h = normalize(l + v);
        let dnh = max(dot(n2, h), 0.0);
        let spec = pow(dnh, 60.0) * 1.10 + pow(dnh, 22.0) * 0.30 + pow(dnh, 8.0) * 0.15
                 + pow(max(dot(n2, v), 0.0), 24.0) * 0.12;
        let fres = pow(1.0 - clamp(dot(n2, v), 0.0, 1.0), 3.5);
        let sky = vec3<f32>(0.60, 0.71, 0.83);
        lit = 0.62 + 0.48 * max(abs(dot(n2, l)), 0.0);
        if (in.anim >= 3.0) {
            // falling sheets
            alpha = 0.70;
            rgb = mix(t.rgb, sky, fres * 0.35);
        } else if (in.anim >= 2.0) {
            // jets
            alpha = 0.78;
            rgb = mix(t.rgb, sky, fres * 0.40);
        } else if (in.anim >= 1.5) {
            // churn / foam
            alpha = 0.90;
            rgb = mix(t.rgb, vec3<f32>(0.95, 0.975, 0.99), 0.50);
        } else {
            let shallow = clamp((in.anim - 1.1) / 0.3, 0.0, 1.0);
            rgb = mix(t.rgb * 0.70, t.rgb, shallow); // deeper reads darker
            rgb = mix(rgb, sky, fres * 0.70);
            alpha = mix(0.48, 0.66, shallow) + fres * 0.30;
        }
        rgb = rgb + vec3<f32>(spec, spec, spec) * 0.9;
    }
    rgb = rgb * floor_tint * (1.0 + 0.55 * caust);
    let c = rgb * lit * (1.0 + shim);
    return vec4<f32>(c, alpha);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub nrm: [f32; 3],
    /// water animation channel: 0 static; 1..1.9 basin/tier water ripples;
    /// 2..2.9 fountain jet (fraction = height along the column); 3..3.9 a
    /// falling water sheet (fraction = height along the sheet)
    pub anim: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelParams {
    pub pos: [f32; 3],
    pub rot: f32,
    /// 1.0 = spin with the camera (rotated house meshes), 0.0 = world-fixed
    pub spin: f32,
    pub _p0: f32,
    pub _p1: f32,
    pub _p2: f32,
}

pub struct ModelPart {
    pub first_index: u32,
    pub index_count: u32,
    /// index into `ModelMesh::texture_ids`
    pub texture: u32,
}

pub struct ModelMesh {
    pub verts: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<ModelPart>,
    /// unique FSH instance ids (group 1abe787d) used by the submeshes
    pub texture_ids: Vec<u32>,
    /// material slots (indices into `texture_ids`) that are WATER: they are
    /// drawn in a second blended pass after all opaque geometry
    pub water_slots: Vec<u32>,
}

fn slot_for(mesh: &mut ModelMesh, id: u32) -> u32 {
    match mesh.texture_ids.iter().position(|&x| x == id) {
        Some(k) => k as u32,
        None => {
            mesh.texture_ids.push(id);
            (mesh.texture_ids.len() - 1) as u32
        }
    }
}

fn push_submesh(m: &mesh3d::Mesh3d, mesh: &mut ModelMesh, vg: usize, ig: usize, mg: usize) {
    let (Some(vgrp), Some(igrp)) = (m.vertex_groups.get(vg), m.index_groups.get(ig)) else {
        return;
    };
    if vgrp.verts.is_empty() || igrp.indices.is_empty() {
        return;
    }
    let base = mesh.verts.len() as u32;
    // per-vertex normals from face normals (model space, y-up)
    let mut nrms = vec![[0.0f32; 3]; vgrp.verts.len()];
    for t in igrp.indices.chunks_exact(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if a >= vgrp.verts.len() || b >= vgrp.verts.len() || c >= vgrp.verts.len() {
            continue;
        }
        let pa = vgrp.verts[a].pos;
        let pb = vgrp.verts[b].pos;
        let pc = vgrp.verts[c].pos;
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &i in &[a, b, c] {
            nrms[i][0] += n[0];
            nrms[i][1] += n[1];
            nrms[i][2] += n[2];
        }
    }
    for (v, n) in vgrp.verts.iter().zip(nrms.iter()) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        mesh.verts.push(ModelVertex {
            pos: v.pos,
            uv: v.uv,
            nrm: [n[0] / len, n[2] / len, n[1] / len],
            anim: 0.0,
        });
    }
    let tex_id = m
        .material_groups
        .get(mg)
        .and_then(|g| g.textures.first())
        .map(|t| t.id)
        .unwrap_or(0);
    let slot = slot_for(mesh, tex_id);
    let first_index = mesh.indices.len() as u32;
    for &i in &igrp.indices {
        mesh.indices.push(base + i as u32);
    }
    mesh.parts.push(ModelPart {
        first_index,
        index_count: igrp.indices.len() as u32,
        texture: slot,
    });
}

/// Build a render mesh from a parsed 3DMD model (one textured submesh per
/// ANIM group frame-0 block).
pub fn mesh_from_mesh3d(m: &mesh3d::Mesh3d) -> ModelMesh {
    let mut mesh = ModelMesh {
        water_slots: Vec::new(),
        verts: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
        texture_ids: Vec::new(),
    };
    let mut any = false;
    for g in &m.animation.groups {
        if let Some(b) = g.blocks.first() {
            push_submesh(m, &mut mesh, b[0] as usize, b[1] as usize, b[3] as usize);
            any = true;
        }
    }
    if !any {
        // fallback: one submesh per group index
        let n = m
            .vertex_groups
            .len()
            .min(m.index_groups.len())
            .min(m.material_groups.len().max(1));
        for k in 0..n {
            push_submesh(m, &mut mesh, k, k, k.min(m.material_groups.len().saturating_sub(1)));
        }
    }
    mesh
}

impl ModelMesh {
    /// An empty mesh (no geometry): renders as nothing; used to clear the
    /// live-preview layers without rebuilding their renderer.
    pub fn empty() -> Self {
        Self {
            water_slots: Vec::new(),
            verts: Vec::new(),
            indices: Vec::new(),
            parts: Vec::new(),
            texture_ids: Vec::new(),
        }
    }

    /// [`empty`] + declared texture slots: a renderer built from it still
    /// creates one texture bind group per slot, so later `set_mesh` calls
    /// may reference any of the first `n_tex` texture ids.  Used to
    /// pre-warm the live road-stroke layer.
    pub fn empty_with_slots(n_tex: usize) -> Self {
        let mut m = Self::empty();
        m.texture_ids = (0..n_tex as u32).collect();
        m
    }
}

pub struct ModelRenderer {
    pipeline: wgpu::RenderPipeline,
    pipeline_water: wgpu::RenderPipeline,
    group0: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    model_buf: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    tex_bgs: Vec<wgpu::BindGroup>,
    parts: Vec<(ModelPart, wgpu::BindGroup, bool)>,
}

impl ModelRenderer {
    /// `textures` must align with `mesh.texture_ids()`: (width, height, rgba8).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        sample_count: u32,
        mesh: &ModelMesh,
        textures: &[(u32, u32, Vec<u8>)],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model-params"),
            size: (std::mem::size_of::<Params>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model-placement"),
            size: (std::mem::size_of::<ModelParams>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model-bgl0"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model-bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: model_buf.as_entire_binding(),
                },
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model-bgl1"),
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
            label: Some("model-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // one texture bind group per unique texture id
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
                label: Some("model-texture"),
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
                label: Some("model-tex-bg"),
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
            label: Some("model-verts"),
            contents: bytemuck::cast_slice(&mesh.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model-indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model-shader"),
            source: wgpu::ShaderSource::Wgsl(MODEL_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model-layout"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_model"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_model"),
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
        // water: blended, no depth write; drawn after all opaque parts
        let pipeline_water = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model-pipeline-water"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_model"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3, 3 => Float32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_model"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                depth_write_enabled: Some(false),
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
            .parts
            .iter()
            .map(|p| {
                let bg = tex_bgs
                    .get(p.texture as usize)
                    .expect("texture slot out of range")
                    .clone();
                (
                    ModelPart {
                        first_index: p.first_index,
                        index_count: p.index_count,
                        texture: p.texture,
                    },
                    bg,
                    mesh.water_slots.contains(&p.texture),
                )
            })
            .collect();
        Self {
            pipeline,
            pipeline_water,
            group0,
            params_buf,
            model_buf,
            vbuf,
            ibuf,
            tex_bgs,
            parts,
        }
    }

    /// Swap the geometry in place — fresh vertex/index buffers and part
    /// ranges, while the pipeline and the texture bind groups stay.  The
    /// mesh must use the same texture slots this renderer was built with
    /// (the road textures are process-cached, so the live preview always
    /// does).  Meant for per-interaction rebuilds: shader compilation and
    /// texture uploads are not repeated.
    pub fn set_mesh(&mut self, device: &wgpu::Device, mesh: &ModelMesh) {
        use wgpu::util::DeviceExt;
        static DV: ModelVertex = ModelVertex {
            pos: [0.0; 3],
            uv: [0.0; 2],
            nrm: [0.0; 3],
            anim: 0.0,
        };
        static DI: [u32; 1] = [0];
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model-verts"),
            contents: if mesh.verts.is_empty() {
                bytemuck::bytes_of(&DV)
            } else {
                bytemuck::cast_slice(&mesh.verts)
            },
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model-indices"),
            contents: if mesh.indices.is_empty() {
                bytemuck::bytes_of(&DI)
            } else {
                bytemuck::cast_slice(&mesh.indices)
            },
            usage: wgpu::BufferUsages::INDEX,
        });
        self.parts = mesh
            .parts
            .iter()
            .map(|p| {
                let bg = self
                    .tex_bgs
                    .get(p.texture as usize)
                    .expect("texture slot out of range for set_mesh")
                    .clone();
                (
                    ModelPart {
                        first_index: p.first_index,
                        index_count: p.index_count,
                        texture: p.texture,
                    },
                    bg,
                    mesh.water_slots.contains(&p.texture),
                )
            })
            .collect();
        self.vbuf = vbuf;
        self.ibuf = ibuf;
    }

    pub fn set_params(&self, queue: &wgpu::Queue, params: &Params) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn set_placement(&self, queue: &wgpu::Queue, pos: [f32; 3], rot: f32, spin: f32) {
        let mp = ModelParams {
            pos,
            rot,
            spin,
            _p0: 0.0,
            _p1: 0.0,
            _p2: 0.0,
        };
        queue.write_buffer(&self.model_buf, 0, bytemuck::bytes_of(&mp));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        // An empty mesh (e.g. the editor's roads layer before the first chain
        // lands) has no parts; binding its zero-sized buffers panics in wgpu
        // ("buffer slice can not be empty"), so bail out first.
        if self.parts.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.group0, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        // pass 1: opaque parts only (they write depth)
        pass.set_pipeline(&self.pipeline);
        for (part, bg, water) in &self.parts {
            if *water {
                continue;
            }
            pass.set_bind_group(1, bg, &[]);
            pass.draw_indexed(
                part.first_index..part.first_index + part.index_count,
                0,
                0..1,
            );
        }
        // pass 2: water (blended, tested against the opaque depth)
        pass.set_pipeline(&self.pipeline_water);
        for (part, bg, water) in &self.parts {
            if !*water {
                continue;
            }
            pass.set_bind_group(1, bg, &[]);
            pass.draw_indexed(
                part.first_index..part.first_index + part.index_count,
                0,
                0..1,
            );
        }
    }
}

/// A model placed into the scene: mesh + textures (aligned with
/// `mesh.texture_ids()`) + scene position (cell x, cell y, mesh z) and yaw.
pub struct PlacedModel<'a> {
    pub mesh: &'a ModelMesh,
    pub textures: &'a [(u32, u32, Vec<u8>)],
    pub pos: [f32; 3],
    pub rot: f32,
    /// 1.0 = spin with the camera (3DMD buildings), 0.0 = world-fixed
    pub spin: f32,
}

/// Offscreen render of terrain + placed models -> RGBA8 (w*h*4).
/// Mirrors `crate::render_offscreen_with`; keep the two in sync.
#[allow(clippy::too_many_arguments)]
pub fn render_scene_with_models(
    w: u32,
    h: u32,
    layers: &[u8],
    verts: &[Vertex],
    grid: f32,
    zspan: f32,
    zoom: f32,
    mn: f32,
    wl: f32,
    // optional camera override (iso centre in cells, angle); None frames the grid
    view: Option<([f32; 2], f32)>,
    models: &[PlacedModel<'_>],
    roads: Option<&crate::road::RoadMesh>,
    road_textures: &[(u32, u32, Vec<u8>)],
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
        label: Some("kolomna-scene"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("failed to create device");

    let sample_count = 4u32;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let terrain = TerrainRenderer::new(&device, &queue, format, sample_count, layers, verts, mn, wl);
    let mut params = crate::default_params(w as f32, h as f32, grid, zspan);
    // shots can freeze the water animation at a given phase (KOLOMNA_TIME)
    if let Ok(s) = std::env::var("KOLOMNA_TIME") {
        if let Ok(v) = s.parse::<f32>() {
            params.time = v;
        }
    }
    params.scale *= zoom;
    if std::env::var("KOLOMNA_DEBUG").is_ok() {
        eprintln!(
            "params: center={:?} scale={:.3} tilt={:.3} grid={grid} zspan={zspan:.2} zoom={zoom}",
            params.center, params.scale, params.tilt
        );
    }
    if let Some((center, angle)) = view {
        params.center = center;
        params.angle = angle;
    }
    terrain.set_params(&queue, &params);

    let model_renderers: Vec<ModelRenderer> = models
        .iter()
        .map(|m| {
            let r = ModelRenderer::new(&device, &queue, format, sample_count, m.mesh, m.textures);
            r.set_params(&queue, &params);
            r.set_placement(&queue, m.pos, m.rot, m.spin);
            r
        })
        .collect();

    let road_renderer = roads.map(|rm| {
        let r = crate::road::RoadRenderer::new(&device, &queue, format, sample_count, rm, road_textures);
        r.set_params(&queue, &params);
        r
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-msaa"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let resolved = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-resolved"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let resolved_view = resolved.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
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
        label: Some("scene-enc"),
    });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene-pass"),
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
        terrain.draw(&mut pass);
        for r in &model_renderers {
            r.draw(&mut pass);
        }
        if let Some(r) = &road_renderer {
            r.draw(&mut pass);
        }
    }

    let bpr = w * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene-readback"),
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

/// Debug helper: count how many vertices each texture id is used by.
pub fn texture_usage(mesh: &ModelMesh) -> BTreeMap<u32, usize> {
    let mut m = BTreeMap::new();
    for p in &mesh.parts {
        if let Some(id) = mesh.texture_ids.get(p.texture as usize) {
            *m.entry(*id).or_insert(0) += p.index_count as usize / 3;
        }
    }
    m
}
