//! Kolomna render library: terrain extraction, meshing and isometric wgpu rendering.
//!
//! Reads the terrain height grid from a city save (chunk a9dd6ff4/e98f9525/1:
//! u16 version, then an n*n grid of f32 altitudes in meters, n = 129),
//! meshes it as iso diamonds and textures it with its own terrain atlas.
//!
//! Surface rules (v4):
//! - water is a flat plane at the detected surface level, shaded procedurally
//!   (noise waves, sun glints, bed-depth tint, shore foam);
//! - land textures are blended CONTINUOUSLY in the fragment shader — sand by
//!   metres above the water plane, dirt/rock by slope from smooth vertex
//!   normals, grass by three height-bands (dithered). No per-cell binary
//!   classification for sand/cliffs, so boundaries can never look like
//!   squares cut into a slope.
//! - land UVs are continuous world coordinates with REPEAT sampling (each
//!   texture instance spans REPEAT_CELLS x REPEAT_CELLS cells); steep faces
//!   blend in a vertical projection so texels do not streak.
//!
//! Depth (v3): the fragment depth is the dot of the rotated world position
//! with the iso view ray (1,1,0.5) — strictly monotone along the ray, so
//! overlapping fragments never tie. (Screen-derived depth did tie: the
//! LessEqual test then resolved by float rounding, which flickered while
//! zooming and let water show through slopes.)
//!
//! Textures (v3): full mip chain generated at startup (blit until 1x1) with
//! Linear mipmap filtering — no minification shimmer on zoom.
//!
//! Lighting (v4): smooth per-vertex normals (central differences of the
//! height field) instead of flat per-triangle derivatives.
//!
//! The rendering internals are shared by the `city_terrain` example (offscreen
//! CLI) and `kolomna-app` (windowed + `--shot`).

use std::borrow::Cow;
use std::collections::HashMap;
use std::io;

use bytemuck::{Pod, Zeroable};

pub mod model;
pub mod piter;
pub mod road;
pub mod assets;
pub mod procgen;
pub mod streets;
pub mod summer_garden;
pub mod terrain_gen;

pub const VIEW_W: u32 = 1600;
pub const VIEW_H: u32 = 1200;
pub const TILE: u32 = 256;
pub const FSH_TYPE: u32 = 0x7AB50E44;
pub const TERRAIN_TYPE: u32 = 0xa9dd6ff4;
pub const TERRAIN_GROUP: u32 = 0xe98f9525;
pub const TERRAIN_INSTANCE: u32 = 1;
/// map units (mesh units) per meter of altitude (visual relief exaggeration)
pub const ZSCALE: f32 = 0.035;
/// world cells per texture repeat (texture instance spans this many cells)
pub const REPEAT_CELLS: f32 = 5.0;
/// Land-vertex `layer` sentinel marking the wooden diorama base: the shader
/// skips the grass blend for it and samples the wood atlas layer instead.
pub const WOOD_LAYER: f32 = -1000.0;
/// water surface sits this many metres above the flat-bed mode (see water_level)
pub const WATER_MARGIN: f32 = 2.2;

/// data2.dat terrain textures (group, instance), all 256x256 DXT1.
/// Layers: 0 = water (unused by shaders), 1 = dark forest green, 2 = sand,
/// 3 = dirt, 4 = mid green, 5 = light meadow green.
/// Sand (2) and dirt (3) are sampled with CONSTANT indices by the shader;
/// the per-vertex `layer` picks among the three green tints only.
pub const SOURCES: [(u32, u32); 6] = [
    (0x891b0e1a, 0x00000013),
    (0x891b0e1a, 0x00000031),
    (0x891b0e1a, 0x00000032),
    (0x891b0e1a, 0x00000033),
    (0x891b0e1a, 0x00000034),
    (0x891b0e1a, 0x00000035),
];


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    /// continuous world uv (cell coordinate / REPEAT_CELLS), REPEAT sampling
    pub uv: [f32; 2],
    /// water bed depth in meters (>= 0.10) for water vertices; -1 for dry land
    pub wdep: f32,
    /// continuous grass-blend coordinate: metres above water + forest noise
    /// (ignored for water; the shader blends the three greens over it)
    pub layer: f32,
    /// smooth mesh-space normal (for land lighting and slope blending)
    pub nrm: [f32; 3],
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
    /// water plane height in mesh units — filled by TerrainRenderer::set_params
    pub _pad0: f32,
    /// camera tilt = sin(elevation); see Params docs
    pub tilt: f32,
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
    // water plane height in mesh units (set by the renderer)
    water_z: f32,
    /// camera tilt = sin(elevation): 0.5 = 2:1 iso, 1.0 = top-down, 0.0 = level
    tilt: f32,
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
    @location(4) nrm: vec3<f32>,
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) wdep: f32, @location(3) layer: f32, @location(4) nrm: vec3<f32>) -> VsOut {
    var o: VsOut;
    let d = p.xy - vec2<f32>(params.pivot);
    let ca = cos(params.angle);
    let sa = sin(params.angle);
    let rp = vec2<f32>(d.x * ca - d.y * sa, d.x * sa + d.y * ca) + vec2<f32>(params.pivot);
    let iso = vec2<f32>((rp.x - rp.y) * 0.5, (rp.x + rp.y) * (0.5 * params.tilt) - p.z * params.hz);
    let scr = (iso - params.center) * params.scale;
    let ndc = vec2<f32>(scr.x / params.half.x, -scr.y / params.half.y);
    // True view depth: the iso camera looks along (1, 1, 0.5) in rotated mesh
    // space, so a surface is closer to the eye when dot(rp.xy + z*0.5) is
    // LARGER. Monotone along the whole view ray -> overlapping fragments get
    // distinct depths (no tie lottery -> no flicker, no water through slopes).
    let ce = sqrt(max(1.0 - params.tilt * params.tilt, 0.01));
    let tk = params.tilt / ce;
    let dep = (rp.x + rp.y + p.z * tk) * (0.9 / (4.0 * params.pivot + 40.0));
    o.pos = vec4<f32>(ndc.x, ndc.y, 0.95 - dep, 1.0);
    o.uv = uv;
    o.wpos = p;
    o.wdep = wdep;
    o.layer = layer;
    o.nrm = nrm;
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
    // land: smooth vertex normal -> no flat-shaded facets
    let n = normalize(in.nrm);
    let l = normalize(vec3<f32>(-0.40, -0.55, 0.85));
    let lit = 0.72 + 0.38 * max(dot(n, l), 0.0);
    // the wooden diorama base: marked verts sample the wood layer; vertical
    // faces darken so the base reads as one solid wooden volume
    if (in.layer < -500.0) {
        let w = textureSample(atlas, samp, in.uv, 6).rgb;
        let side = clamp(1.0 - n.z, 0.0, 1.0);
        return vec4<f32>(w * lit * (1.0 - 0.30 * side), 1.0);
    }
    // grass: CONTINUOUS three-way blend over `layer` (metres above water with
    // forest noise) — soft organic meadow/wood edges, never per-cell squares
    let t1 = smoothstep(20.0, 26.0, in.layer);
    let t2 = smoothstep(48.0, 56.0, in.layer);
    let gm = textureSample(atlas, samp, in.uv, 4).rgb;
    let g = mix(mix(textureSample(atlas, samp, in.uv, 5).rgb, gm, t1),
                textureSample(atlas, samp, in.uv, 1).rgb, t2);
    // sand: CONTINUOUS fade by metres above the water plane — a hard per-cell
    // rule here was what cut green squares into sandy slopes.
    let zs = 0.035; // ZSCALE (mesh units per metre)
    let above = (in.wpos.z - params.water_z) / zs;
    let ws = 1.0 - smoothstep(0.5, 3.5, above);
    let sand = textureSample(atlas, samp, in.uv, 2);
    var col = mix(g, sand.rgb, ws);
    // cliffs: blend dirt/rock by slope from the smooth normals; the sample is
    // TRIPLANAR (top + both vertical faces, weighted by n^2) so steep slopes
    // show no stretched streaks and no cell-shaped patches.
    let slope = 1.0 - n.z;
    let wd = smoothstep(0.12, 0.50, slope);
    let sx = textureSample(atlas, samp, vec2<f32>(in.uv.y, in.wpos.z / 5.0), 3);
    let sy = textureSample(atlas, samp, vec2<f32>(in.uv.x, in.wpos.z / 5.0), 3);
    let sz = textureSample(atlas, samp, in.uv, 3);
    let wts = n * n + vec3<f32>(1e-4);
    let dirt = (sx.rgb * wts.x + sy.rgb * wts.y + sz.rgb * wts.z) / (wts.x + wts.y + wts.z);
    col = mix(col, dirt, wd);
    return vec4<f32>(col * lit, 1.0);
}
"#;

/// Fullscreen-triangle blit used to build the texture mip chain.
const MIP_SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct MipOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_mip(@builtin(vertex_index) i: u32) -> MipOut {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    var o: MipOut;
    o.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, 1.0 - y);
    return o;
}

@fragment
fn fs_mip(in: MipOut) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
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

/// Clamped height field + one 3x3 [1 2 1]^2 blur pass. The save's height field
/// is ragged at cell scale (31% of cells deviate >1 m from their 4-neighbours),
/// and cell-shaped spikes / patchy cliff-grass speckle read as "protruding
/// squares" on screen; a light pass turns them into smooth slopes while leaving
/// real cliffs and plateaus intact. Exposed separately so the camera can
/// raycast the *visual* surface (rotation pivot under the cursor).
pub fn smooth_field(vals: &[f32], wl: f32) -> Vec<f32> {
    let n = grid_side(vals);
    let mut hc: Vec<f32> = vals.iter().map(|&h| h.max(wl)).collect();
    let src = hc.clone();
    for gy in 1..n - 1 {
        for gx in 1..n - 1 {
            let s = src[(gy - 1) * n + (gx - 1)]
                + 2.0 * src[(gy - 1) * n + gx]
                + src[(gy - 1) * n + (gx + 1)]
                + 2.0 * src[gy * n + (gx - 1)]
                + 4.0 * src[gy * n + gx]
                + 2.0 * src[gy * n + (gx + 1)]
                + src[(gy + 1) * n + (gx - 1)]
                + 2.0 * src[(gy + 1) * n + gx]
                + src[(gy + 1) * n + (gx + 1)];
            hc[gy * n + gx] = (s / 16.0).max(wl);
        }
    }
    hc
}

/// Mesh the height grid as iso diamonds with continuous world UVs and smooth
/// per-vertex normals. Water cells form a flat plane at `wl` and carry a
/// per-vertex bed-depth attribute for the procedural water shading; land
/// vertices carry only a green-tint layer — sand and dirt are blended
/// continuously in the shader, so no per-cell classification squares exist.
pub fn build_mesh(vals: &[f32], mn: f32, _mx: f32, wl: f32) -> Vec<Vertex> {
    let n = grid_side(vals);
    // grid step in cells: 129 nodes span 128 cells (16 m); a 513-node grid
    // spans the same 128 cells at 4 m — the render mesh samples finer so
    // slopes show no visible triangle facets
    let step = 128.0 / (n as f32 - 1.0);
    // clamped + blurred field (see smooth_field): geometry and normals treat
    // the water bed as the surface
    let hc = smooth_field(vals, wl);
    let zof = |h: f32| (h.max(wl) - mn) * ZSCALE;
    // metres above the water level with the soft forest noise folded in — the
    // continuous coordinate the fragment shader blends the three greens over;
    // no per-cell classification => no square dither at any mesh resolution
    let lay = |gx: usize, gy: usize| -> f32 {
        hc[gy * n + gx] - wl + crate::terrain_gen::forest_noise(gx as f32 * step, gy as f32 * step)
    };

    // per-grid-vertex normals in mesh units (central differences, one-sided at borders)
    let mut normals = vec![[0.0f32, 0.0, 1.0]; n * n];
    for gy in 0..n {
        for gx in 0..n {
            let xl = gx.saturating_sub(1);
            let xr = (gx + 1).min(n - 1);
            let yl = gy.saturating_sub(1);
            let yr = (gy + 1).min(n - 1);
            let wx = if gx == 0 || gx + 1 == n { step } else { 2.0 * step };
            let wy = if gy == 0 || gy + 1 == n { step } else { 2.0 * step };
            let dzdx = (hc[gy * n + xr] - hc[gy * n + xl]) / wx * ZSCALE;
            let dzdy = (hc[yr * n + gx] - hc[yl * n + gx]) / wy * ZSCALE;
            let inv = 1.0 / (dzdx * dzdx + dzdy * dzdy + 1.0).sqrt();
            normals[gy * n + gx] = [-dzdx * inv, -dzdy * inv, inv];
        }
    }

    let mut verts = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for cy in 0..n - 1 {
        for cx in 0..n - 1 {
            let h00 = vals[cy * n + cx];
            let h10 = vals[cy * n + cx + 1];
            let h01 = vals[(cy + 1) * n + cx];
            let h11 = vals[(cy + 1) * n + cx + 1];
            let havg = (h00 + h10 + h01 + h11) * 0.25;

            let is_water = havg <= wl + 0.02;
            // per-vertex grass blend coordinate — water vertices keep 0.0
            let (l00, l10, l01, l11) = if is_water {
                (0.0, 0.0, 0.0, 0.0)
            } else {
                (
                    lay(cx, cy),
                    lay(cx + 1, cy),
                    lay(cx, cy + 1),
                    lay(cx + 1, cy + 1),
                )
            };

            // per-vertex bed depth in meters for the water shader; -1 = dry land
            let wd = |h: f32| {
                if is_water {
                    (wl - h).clamp(0.10, 8.0)
                } else {
                    -1.0
                }
            };
            let (z00, z10, z01, z11) = (
                zof(hc[cy * n + cx]),
                zof(hc[cy * n + cx + 1]),
                zof(hc[(cy + 1) * n + cx]),
                zof(hc[(cy + 1) * n + cx + 1]),
            );
            let (d00, d10, d01, d11) = (wd(h00), wd(h10), wd(h01), wd(h11));
            let (n00, n10, n01, n11) = (
                normals[cy * n + cx],
                normals[cy * n + cx + 1],
                normals[(cy + 1) * n + cx],
                normals[(cy + 1) * n + cx + 1],
            );
            let x0 = cx as f32 * step;
            let x1 = (cx + 1) as f32 * step;
            let y0 = cy as f32 * step;
            let y1 = (cy + 1) as f32 * step;
            // continuous UV: shared corners share UVs -> texture flows across cells
            let (u0, u1) = (x0 / REPEAT_CELLS, x1 / REPEAT_CELLS);
            let (v0, v1) = (y0 / REPEAT_CELLS, y1 / REPEAT_CELLS);
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
                wdep: d00,
                layer: l00,
                nrm: n00,
            });
            verts.push(Vertex {
                pos: [x1, y0, z10],
                uv: [u1, v0],
                wdep: d10,
                layer: l10,
                nrm: n10,
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
                wdep: d11,
                layer: l11,
                nrm: n11,
            });
            verts.push(Vertex {
                pos: [x0, y0, z00],
                uv: [u0, v0],
                wdep: d00,
                layer: l00,
                nrm: n00,
            });
            verts.push(Vertex {
                pos: [x1, y1, z11],
                uv: [u1, v1],
                wdep: d11,
                layer: l11,
                nrm: n11,
            });
            verts.push(Vertex {
                pos: [x0, y1, z01],
                uv: [u0, v1],
                wdep: d01,
                layer: l01,
                nrm: n01,
            });
        }
    }

    // ------------------------------------------------ diorama wooden base:
    // The map sits on a flat wooden base like an exhibition model: the
    // terrain is cut at its rim (a soil wall falls from the rim line down to
    // the board), the board top is a level wooden ring all around, and the
    // board side is a wooden fascia dropping from the ring's outer edge.
    // Wood vertices carry layer = WOOD_LAYER: the shader skips the grass
    // blend for them and samples the wood atlas layer instead.
    const BOARD_MARGIN_M: f32 = 34.0; // wooden ring width, metres
    const BOARD_DROP_M: f32 = 2.0; // board top below the water plane, metres
    const FASCIA_M: f32 = 22.0; // board side height, metres
    let board_w = BOARD_MARGIN_M / 16.0; // cells
    let board_z = (wl - BOARD_DROP_M - mn) * ZSCALE;
    let fascia_z = FASCIA_M * ZSCALE;
    // mesh z -> uv units at the same texel density as the ground (16 m <-> 1 cell)
    let zcell = |z: f32| z / (16.0 * ZSCALE * REPEAT_CELLS);
    let edge = |verts: &mut Vec<Vertex>, pts: &[(usize, usize)], out_xy: [f32; 2], u_along: bool| {
        let uvw = |p: [f32; 2], z: f32| -> [f32; 2] {
            if u_along {
                [p[0] / REPEAT_CELLS, zcell(z)]
            } else {
                [zcell(z), p[1] / REPEAT_CELLS]
            }
        };
        for k in 0..pts.len() - 1 {
            let (g0x, g0y) = pts[k];
            let (g1x, g1y) = pts[k + 1];
            let z0 = zof(hc[g0y * n + g0x]);
            let z1 = zof(hc[g1y * n + g1x]);
            let (l0, l1) = (lay(g0x, g0y), lay(g1x, g1y));
            let (p0, p1) = (
                [g0x as f32 * step, g0y as f32 * step],
                [g1x as f32 * step, g1y as f32 * step],
            );
            let (q0, q1) = (
                [p0[0] + out_xy[0] * board_w, p0[1] + out_xy[1] * board_w],
                [p1[0] + out_xy[0] * board_w, p1[1] + out_xy[1] * board_w],
            );
            let wn = [out_xy[0], out_xy[1], 0.0];
            // cut-soil wall: the rim line falls vertically to the board
            // (six verts per quad: TriangleList consumes triples, a 4-vert
            // quad would feed its partner triangle from the next quad)
            for (pos, layer, uv) in [
                ([p0[0], p0[1], z0], l0, uvw(p0, z0)),
                ([p1[0], p1[1], z1], l1, uvw(p1, z1)),
                ([p1[0], p1[1], board_z], l1, uvw(p1, board_z)),
                ([p0[0], p0[1], z0], l0, uvw(p0, z0)),
                ([p1[0], p1[1], board_z], l1, uvw(p1, board_z)),
                ([p0[0], p0[1], board_z], l0, uvw(p0, board_z)),
            ] {
                verts.push(Vertex {
                    pos,
                    uv,
                    wdep: -1.0,
                    layer,
                    nrm: wn,
                });
            }
            // the level wooden ring around the map (six verts per quad)
            for (pos, uv) in [
                (
                    [p0[0], p0[1], board_z],
                    [p0[0] / REPEAT_CELLS, p0[1] / REPEAT_CELLS],
                ),
                (
                    [p1[0], p1[1], board_z],
                    [p1[0] / REPEAT_CELLS, p1[1] / REPEAT_CELLS],
                ),
                (
                    [q1[0], q1[1], board_z],
                    [q1[0] / REPEAT_CELLS, q1[1] / REPEAT_CELLS],
                ),
                (
                    [p0[0], p0[1], board_z],
                    [p0[0] / REPEAT_CELLS, p0[1] / REPEAT_CELLS],
                ),
                (
                    [q1[0], q1[1], board_z],
                    [q1[0] / REPEAT_CELLS, q1[1] / REPEAT_CELLS],
                ),
                (
                    [q0[0], q0[1], board_z],
                    [q0[0] / REPEAT_CELLS, q0[1] / REPEAT_CELLS],
                ),
            ] {
                verts.push(Vertex {
                    pos,
                    uv,
                    wdep: -1.0,
                    layer: crate::WOOD_LAYER,
                    nrm: [0.0, 0.0, 1.0],
                });
            }
            // wooden fascia: the ring's outer edge drops as one solid side
            // (six verts per quad)
            for (pos, uv) in [
                ([q0[0], q0[1], board_z], uvw(q0, board_z)),
                ([q1[0], q1[1], board_z], uvw(q1, board_z)),
                ([q1[0], q1[1], board_z - fascia_z], uvw(q1, board_z - fascia_z)),
                ([q0[0], q0[1], board_z], uvw(q0, board_z)),
                ([q1[0], q1[1], board_z - fascia_z], uvw(q1, board_z - fascia_z)),
                ([q0[0], q0[1], board_z - fascia_z], uvw(q0, board_z - fascia_z)),
            ] {
                verts.push(Vertex {
                    pos,
                    uv,
                    wdep: -1.0,
                    layer: crate::WOOD_LAYER,
                    nrm: wn,
                });
            }
        }
    };
    let front: Vec<(usize, usize)> = (0..n).map(|i| (i, 0)).collect();
    let right: Vec<(usize, usize)> = (0..n).map(|i| (n - 1, i)).collect();
    let back: Vec<(usize, usize)> = (0..n).map(|i| (n - 1 - i, n - 1)).collect();
    let left: Vec<(usize, usize)> = (0..n).map(|i| (0, n - 1 - i)).collect();
    edge(&mut verts, &front, [0.0, -1.0], true);
    edge(&mut verts, &right, [1.0, 0.0], false);
    edge(&mut verts, &back, [0.0, 1.0], true);
    edge(&mut verts, &left, [-1.0, 0.0], false);
    // the four 90-degree corners: the strips of two neighbouring sides meet in
    // a wedge — fill the outer corner square with the same bands (the soil
    // walls need no filler: the two vertical wall planes share the corner
    // line). o1 is the side whose outward direction is (0, sy).
    for &(cx, cy, sx, sy) in &[
        (0usize, 0usize, -1.0f32, -1.0f32),
        (n - 1, 0, 1.0, -1.0),
        (n - 1, n - 1, 1.0, 1.0),
        (0, n - 1, -1.0, 1.0),
    ] {
        let c = [cx as f32 * step, cy as f32 * step];
        let o1 = [0.0, sy];
        let o2 = [sx, 0.0];
        let a = c;
        let b = [c[0] + o1[0] * board_w, c[1] + o1[1] * board_w];
        let m = [
            c[0] + (o1[0] + o2[0]) * board_w,
            c[1] + (o1[1] + o2[1]) * board_w,
        ];
        let d = [c[0] + o2[0] * board_w, c[1] + o2[1] * board_w];
        // level wooden ring: the corner square
        for p in [a, b, m, a, m, d] {
            verts.push(Vertex {
                pos: [p[0], p[1], board_z],
                uv: [p[0] / REPEAT_CELLS, p[1] / REPEAT_CELLS],
                wdep: -1.0,
                layer: crate::WOOD_LAYER,
                nrm: [0.0, 0.0, 1.0],
            });
        }
        // wooden fascia: the corner square's two outward faces
        for (e0, e1, wn) in [(b, m, [o1[0], o1[1], 0.0]), (m, d, [o2[0], o2[1], 0.0])] {
            let uvz = |p: [f32; 2], z: f32| -> [f32; 2] {
                if wn[0] == 0.0 {
                    [p[0] / REPEAT_CELLS, zcell(z)]
                } else {
                    [zcell(z), p[1] / REPEAT_CELLS]
                }
            };
            for (pos, uv) in [
                ([e0[0], e0[1], board_z], uvz(e0, board_z)),
                ([e1[0], e1[1], board_z], uvz(e1, board_z)),
                (
                    [e1[0], e1[1], board_z - fascia_z],
                    uvz(e1, board_z - fascia_z),
                ),
                ([e0[0], e0[1], board_z], uvz(e0, board_z)),
                (
                    [e1[0], e1[1], board_z - fascia_z],
                    uvz(e1, board_z - fascia_z),
                ),
                (
                    [e0[0], e0[1], board_z - fascia_z],
                    uvz(e0, board_z - fascia_z),
                ),
            ] {
                verts.push(Vertex {
                    pos,
                    uv,
                    wdep: -1.0,
                    layer: crate::WOOD_LAYER,
                    nrm: wn,
                });
            }
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

/// Build the full mip chain of `tex` (array layers, level 0 uploaded already)
/// by successive fullscreen blits: level i = box-ish average of level i-1.
fn generate_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    levels: u32,
    layers: u32,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mip-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MIP_SHADER)),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mip-bgl"),
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
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mip-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mip-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_mip"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_mip"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mip-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mip-enc"),
    });
    for level in 1..levels {
        for layer in 0..layers {
            let src = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let dst = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mip-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mip-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
    queue.submit(Some(enc.finish()));
}

/// Render-ready terrain: pipeline + bind group + buffers, usable with any
/// color target format (offscreen Rgba8UnormSrgb or a window surface).
pub struct TerrainRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    verts_buf: wgpu::Buffer,
    nverts: u32,
    /// water plane height in mesh units (feeds the shader's sand blend)
    water_z: f32,
}

impl TerrainRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        sample_count: u32,
        layers: &[u8],
        verts: &[Vertex],
        mn: f32,
        wl: f32,
    ) -> Self {
        let water_z = (wl - mn) * ZSCALE;
        let layer_bytes = (TILE * TILE * 4) as usize;
        let nlayers = (layers.len() / layer_bytes) as u32;
        let mip_levels = (TILE as f32).log2() as u32 + 1;
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas-array"),
            size: wgpu::Extent3d {
                width: TILE,
                height: TILE,
                depth_or_array_layers: nlayers,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
        generate_mips(device, queue, &atlas_tex, mip_levels, nlayers);
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
            // sampler default here is Nearest for mips -> shimmer while zooming
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
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
                        0 => Float32x3, 1 => Float32x2, 2 => Float32, 3 => Float32, 4 => Float32x3
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
            water_z,
        }
    }

    pub fn set_params(&self, queue: &wgpu::Queue, params: &Params) {
        // the water plane height is renderer state, not caller state
        let mut p = *params;
        p._pad0 = self.water_z;
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
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
    // debug hook so `--shot` can render tilt variants from the shell
    let tilt = std::env::var("KOLOMNA_TILT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let vspan = grid * 0.5 * tilt + zspan;
    let scale = ((w - 60.0) / grid).min((h - 60.0) / vspan);
    Params {
        center: [0.0, grid * 0.5 * tilt - zspan * 0.5],
        scale,
        hz: 1.0,
        half: [w / 2.0, h / 2.0],
        angle: 0.0,
        time: 0.0,
        pivot: grid * 0.5,
        _pad0: 0.0,
        tilt: tilt,
        _pad2: 0.0,
    }
}

/// Render the terrain mesh offscreen (4x MSAA, resolve, readback) and return
/// the RGBA8 pixels (w * h * 4 bytes). `zoom` = 1.0 frames the whole grid;
/// `mn` / `wl` are the terrain min altitude and water surface level (metres).
pub fn render_offscreen_with(
    w: u32,
    h: u32,
    layers: &[u8],
    verts: &[Vertex],
    grid: f32,
    zspan: f32,
    zoom: f32,
    mn: f32,
    wl: f32,
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
        mn,
        wl,
    );
    let mut params = default_params(w as f32, h as f32, grid, zspan);
    params.scale *= zoom;
    renderer.set_params(&queue, &params);

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

/// Whole-grid framing (zoom = 1.0) — thin wrapper over `render_offscreen_with`.
pub fn render_offscreen(
    w: u32,
    h: u32,
    layers: &[u8],
    verts: &[Vertex],
    grid: f32,
    zspan: f32,
    mn: f32,
    wl: f32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    render_offscreen_with(w, h, layers, verts, grid, zspan, 1.0, mn, wl)
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
