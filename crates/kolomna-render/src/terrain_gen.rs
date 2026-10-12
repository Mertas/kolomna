//! Procedural terrain — nothing external involved.
//!
//! The scene is a 2048 m x 2048 m map (128 cells of 16 m). One analytic
//! height field is sampled at two resolutions:
//! - **coarse** 129x129 (16 m cells): the scene grid — house placement,
//!   street anchors, camera raycast;
//! - **fine** 513x513 (4 m cells): the render mesh, so slopes show no
//!   visible triangle facets.
//!
//! Composition: the city stands on a flat bank plateau of a wide river that
//! crosses the whole map west to east just south of it; both banks get gentle
//! sandy waterlines; beyond the river there are low meadows and groves; north
//! of the city soft wooded hills rise. The outer 48 m of the map rim blends
//! toward a smoothed perimeter silhouette so the map-edge skirt reads clean.
//!
//! `terrain_textures()` paints six 256x256 RGBA layers in the order the
//! terrain shader expects.

use std::f32::consts::PI;

use crate::TILE;

/// water surface height in metres (the scene datum)
pub const WATER_LEVEL: f32 = 200.0;
/// coarse scene grid side (matches the old save grid: 129 nodes, 16 m cells)
pub const LAND_N: usize = 129;
/// render mesh subdivision: 4 m cells over the same 128-cell span
pub const FINE_SUB: usize = 4;
/// map size in metres (128 cells x 16 m)
pub const MAP_M: f32 = 2048.0;

pub struct Terrain {
    pub coarse: Vec<f32>,
    pub fine: Vec<f32>,
    pub wl: f32,
    pub mn: f32,
    pub mx: f32,
}

// ---------------------------------------------------------------- noise ----

fn h2(ix: i32, iy: i32, s: u32) -> f32 {
    let mut n = (ix as u32)
        .wrapping_mul(0x8da6_b343)
        ^ (iy as u32).wrapping_mul(0xd816_3841)
        ^ s.wrapping_mul(0x9e37_79b9);
    n ^= n >> 13;
    n = n.wrapping_mul(0x85eb_ca6b);
    n ^= n >> 16;
    (n & 0x00ff_ffff) as f32 / 0x0100_0000 as f32
}

fn vnoise(x: f32, y: f32, s: u32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let (ix, iy) = (ix as i32, iy as i32);
    let a = h2(ix, iy, s);
    let b = h2(ix + 1, iy, s);
    let c = h2(ix, iy + 1, s);
    let d = h2(ix + 1, iy + 1, s);
    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

/// 4-octave value fBm, roughly [-0.95, 0.95].
fn fbm(x: f32, y: f32, s: u32) -> f32 {
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut f = 1.0;
    for k in 0..4u32 {
        v += amp * (vnoise(x * f, y * f, s.wrapping_add(k * 101)) * 2.0 - 1.0);
        amp *= 0.5;
        f *= 2.03;
    }
    v
}

/// Value noise on a wrapped lattice: seamless when `x`/`y` span exactly
/// `period` over the texture and the texture side is the period multiple.
fn tnoise(x: f32, y: f32, period: i32, s: u32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let (ix, iy) = (ix as i32, iy as i32);
    let f = |a: i32, b: i32| h2(a.rem_euclid(period), b.rem_euclid(period), s);
    let ab = f(ix, iy) + (f(ix + 1, iy) - f(ix, iy)) * ux;
    let cd = f(ix, iy + 1) + (f(ix + 1, iy + 1) - f(ix, iy + 1)) * ux;
    ab + (cd - ab) * uy
}

/// tileable fBm: `x`,`y` are pixel coords in [0, 256); `period` is the base
/// lattice period across the tile; returns roughly [-0.95, 0.95].
fn tfbm(x: f32, y: f32, period: i32, octaves: u32, s: u32) -> f32 {
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut p = period;
    for k in 0..octaves {
        let pf = p as f32;
        v += amp * (tnoise(x / 256.0 * pf, y / 256.0 * pf, p, s.wrapping_add(k * 17)) * 2.0 - 1.0);
        amp *= 0.5;
        p *= 2;
    }
    v
}

fn sstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ----------------------------------------------------------- height field --

// the city band: a flat bank plateau around the plan anchor at cell (64, 64)
const CITY_X0: f32 = 768.0;
const CITY_X1: f32 = 1472.0;
const CITY_Z0: f32 = 832.0;
const CITY_Z1: f32 = 1264.0;
const CITY_BLEND: f32 = 40.0;

fn rect_dist(x: f32, z: f32) -> f32 {
    let dx = (CITY_X0 - x).max(x - CITY_X1).max(0.0);
    let dz = (CITY_Z0 - z).max(z - CITY_Z1).max(0.0);
    (dx * dx + dz * dz).sqrt()
}

/// near river bank (metres, z of the waterline); meanders along x
fn shore_near(x: f32, s: u32) -> f32 {
    1344.0 + 26.0 * (x / 260.0 + 2.0).sin() + 22.0 * fbm(x / 700.0, 3.7, s + 501)
}

/// far river bank across the water
fn shore_far(x: f32, s: u32) -> f32 {
    1648.0 + 34.0 * (x / 340.0 + 0.5).sin() + 26.0 * fbm(x / 470.0, 9.1, s + 977)
}

fn height(x: f32, z: f32, s: u32) -> f32 {
    let d_out = rect_dist(x, z);
    let sn = shore_near(x, s);
    let sf = shore_far(x, s);
    let micro = 0.55 * fbm(x / 95.0, z / 95.0, s + 311)
        + 0.35 * fbm(x / 37.0 + 5.2, z / 37.0 + 11.8, s + 433);
    let h_raw = 30.0 * fbm(x / 460.0, z / 460.0, s + 11)
        + 15.0 * fbm(x / 190.0 + 7.3, z / 190.0 + 3.1, s + 77)
        + 6.0 * fbm(x / 80.0 + 13.7, z / 80.0 + 9.4, s + 131);

    let mut h;
    if z >= sf {
        // far bank: low meadows with groves, rising gently to the south edge
        let far = WATER_LEVEL + 2.4
            + 0.32 * h_raw
            + 0.5 * micro
            + 16.0 * sstep(0.0, 380.0, z - sf);
        h = far.max(WATER_LEVEL + 1.0);
    } else {
        // near side: calm near the city and near the river, rolling further out
        let m_h = sstep(0.0, 300.0, d_out);
        let d_riv = if z < sn { sn - z } else if z > sf { z - sf } else { 0.0 };
        let m_rivw = sstep(60.0, 260.0, d_riv);
        let north_rise = 34.0 * sstep(860.0, 520.0, z) * sstep(0.0, 200.0, d_out);
        h = 209.0 + micro + m_h * m_rivw * (h_raw + 14.0) + north_rise;
        h = h.max(WATER_LEVEL + 6.0 + 0.6 * micro).min(281.0);
    }

    // city plateau (a whisker of tilt toward the river)
    let wz = sstep(CITY_BLEND, 0.0, d_out);
    let city = 206.3 - 0.0032 * (z - 1024.0);
    h = mix(h, city, wz);

    // river carve: bed dips to ~9.5 m below the surface at mid-channel
    let d_near_r = z - sn;
    let d_far_r = sf - z;
    let m_riv = sstep(-72.0, -10.0, d_near_r) * sstep(-72.0, -10.0, d_far_r);
    let t = d_near_r / (sf - sn).max(1.0);
    let depth = if t > 0.0 && t < 1.0 {
        9.5 * (PI * t).sin().powf(1.15)
    } else {
        0.0
    };
    h = mix(h, WATER_LEVEL - depth, m_riv);
    h
}

/// Organic forest / dark-grass mottling in metres of altitude, keyed by mesh
/// cell coordinates. Fixed seed: the mottling pattern is stable across runs.
pub fn forest_noise(xc: f32, yc: f32) -> f32 {
    const FS: u32 = 0x5EED_0A11;
    let x = xc * 16.0;
    let y = yc * 16.0;
    24.0 * fbm(x / 240.0, y / 240.0, FS) + 9.0 * fbm(x / 90.0 + 3.3, y / 90.0 + 8.8, FS + 333)
}

/// Blend the outer 48 m of the map toward a smoothed perimeter silhouette so
/// the map-edge skirt line is clean (no jagged rim).
fn smooth_rim(fine: &mut [f32], n: usize) {
    let inset = 12usize; // fine cells: 48 m
    let seg = n - 1;
    let mut ring = vec![0.0f32; seg * 4];
    for i in 0..seg {
        ring[i] = fine[inset * n + i];
        ring[seg + i] = fine[i * n + (n - 1 - inset)];
        ring[2 * seg + i] = fine[(n - 1 - inset) * n + (n - 1 - i)];
        ring[3 * seg + i] = fine[(n - 1 - i) * n + inset];
    }
    for _ in 0..3 {
        let src = ring.clone();
        for i in 0..ring.len() {
            let a = src[(i + ring.len() - 1) % ring.len()];
            let b = src[i];
            let c = src[(i + 1) % ring.len()];
            ring[i] = (a + 2.0 * b + c) * 0.25;
        }
    }
    for gy in 0..n {
        for gx in 0..n {
            let d = gx.min(gy).min(n - 1 - gx).min(n - 1 - gy);
            if d >= inset {
                continue;
            }
            let w = 1.0 - sstep(0.0, inset as f32, d as f32);
            let dl = gx;
            let dr = n - 1 - gx;
            let du = gy;
            let dd = n - 1 - gy;
            let m = dl.min(dr).min(du).min(dd);
            let idx = if m == dl {
                3 * seg + (n - 1 - gy)
            } else if m == dr {
                seg + gy
            } else if m == du {
                gx
            } else {
                2 * seg + (n - 1 - gx)
            };
            let target = ring[idx % ring.len()];
            fine[gy * n + gx] = mix(fine[gy * n + gx], target, w);
        }
    }
}

/// Generate the terrain for `seed`: both grids plus level stats.
pub fn generate(seed: u64) -> Terrain {
    let s = (seed ^ (seed >> 32)) as u32;
    let nf = (LAND_N - 1) * FINE_SUB + 1;
    let dm = MAP_M / (nf - 1) as f32;
    let mut fine = vec![0.0f32; nf * nf];
    for gy in 0..nf {
        let z = gy as f32 * dm;
        for gx in 0..nf {
            let x = gx as f32 * dm;
            fine[gy * nf + gx] = height(x, z, s);
        }
    }
    smooth_rim(&mut fine, nf);
    let mut coarse = vec![0.0f32; LAND_N * LAND_N];
    for gy in 0..LAND_N {
        for gx in 0..LAND_N {
            coarse[gy * LAND_N + gx] = fine[gy * FINE_SUB * nf + gx * FINE_SUB];
        }
    }
    let mn = fine.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = fine.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Terrain {
        coarse,
        fine,
        wl: WATER_LEVEL,
        mn,
        mx,
    }
}

// -------------------------------------------------------------- textures ---

fn add3(b: [f32; 3], v: f32) -> [f32; 3] {
    [b[0] + v, b[1] + v, b[2] + v]
}

fn paint(layer: usize, x: usize, y: usize) -> [f32; 3] {
    let xf = x as f32;
    let yf = y as f32;
    let grain = (h2(x as i32, y as i32, 0x7777 ^ (layer as u32) * 31) - 0.5) * 2.0;
    let grain2 = (h2(x as i32 + 4111, y as i32 + 907, 0x1234 ^ (layer as u32) * 17) - 0.5) * 2.0;
    match layer {
        // 0: water bed (the shader shades water procedurally; this is silt)
        0 => {
            let v = tfbm(xf, yf, 5, 3, 0xB01) * 9.0 + grain * 5.0;
            add3([62.0, 71.0, 82.0], v)
        }
        // 1: dark forest green — clumps + fine grain, a few shadow specks
        1 => {
            let m = tfbm(xf, yf, 4, 4, 0xF0E1);
            let d = tfbm(xf + 77.0, yf + 19.0, 12, 3, 0xF02);
            let mut v = m * 15.0 + d * 6.0 + grain * 7.0;
            if grain2 > 0.86 {
                v -= 13.0;
            }
            add3([56.0, 70.0, 33.0], v)
        }
        // 2: sand — fine grain + faint ripples
        2 => {
            let rip = (yf / 256.0 * 6.0 * 2.0 * PI + tfbm(xf, yf, 3, 2, 0x5A2) * PI).sin() * 3.5;
            let v = tfbm(xf, yf, 6, 3, 0x5A2 ^ 0x99) * 8.0 + grain * 7.0 + rip;
            add3([183.0, 158.0, 112.0], v)
        }
        // 3: dirt / soil — clumps, pebbles, dark specks (used triplanar)
        3 => {
            let m = tfbm(xf, yf, 5, 4, 0xD17);
            let mut v = m * 17.0 + grain * 9.0;
            if grain2 > 0.93 {
                v += 30.0;
            } else if grain2 < -0.93 {
                v -= 22.0;
            }
            add3([138.0, 114.0, 80.0], v)
        }
        // 4: mid green grass
        4 => {
            let m = tfbm(xf, yf, 4, 4, 0x64E);
            let v = m * 12.0 + grain * 6.0 + (if grain2 > 0.96 { 12.0 } else { 0.0 });
            add3([86.0, 100.0, 45.0], v)
        }
        // 5: light meadow green
        _ => {
            let m = tfbm(xf, yf, 4, 4, 0x11B);
            let v = m * 12.0 + grain * 6.0 + (if grain2 > 0.96 { 12.0 } else { 0.0 });
            add3([109.0, 121.0, 56.0], v)
        }
    }
}

/// Six 256x256 RGBA layers in flat order — the procedural replacement for the
/// terrain atlas (water bed, dark green, sand, dirt, mid green, light).
pub fn terrain_textures() -> Vec<u8> {
    let t = TILE as usize;
    let mut out = vec![0u8; t * t * 4 * 6];
    for layer in 0..6usize {
        for y in 0..t {
            for x in 0..t {
                let c = paint(layer, x, y);
                let o = (layer * t * t + y * t + x) * 4;
                out[o] = c[0].clamp(0.0, 255.0) as u8;
                out[o + 1] = c[1].clamp(0.0, 255.0) as u8;
                out[o + 2] = c[2].clamp(0.0, 255.0) as u8;
                out[o + 3] = 255;
            }
        }
    }
    out
}
