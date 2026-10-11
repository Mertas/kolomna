//! Procedurally generated modern streets for the Petersburg village: an
//! asphalt roadway with a dashed centre line, granite kerbs, tiled
//! sidewalks, ground skirts and zebra crossings — all conforming to the
//! smoothed terrain. The whole network is a single world-space mesh
//! (absolute metres, placed at [0, 0, 0] like the plan overlay).
//!
//! Plan-frame coordinates: `u` runs along the main street (the line of
//! block fronts), `v` across it. They map to the world exactly like the
//! house placements do: `world_rel = eu * u + nvec * v`.

use crate::model::ModelMesh;
use crate::procgen::Builder;

pub const M_ASPHALT: u32 = 0;
pub const M_KERB: u32 = 1;
pub const M_WALK: u32 = 2;
pub const M_PAINT: u32 = 3;

const ROAD_H: f32 = 0.14; // roadway sits this high above the highest ground
const KERB_H: f32 = 0.15; // kerb / sidewalk step above the roadway
const KERB_W: f32 = 0.15;
const STEP: f32 = 1.5; // cross-section spacing, m

pub struct Params {
    pub main_t0: f32,
    pub main_t1: f32,
    pub cross_half: f32,
    /// cross lanes: (centre u, roadway half-width, corridor half-width)
    pub cross: Vec<(f32, f32, f32)>,
}

struct Ctx<'a> {
    h_at: &'a dyn Fn(f32, f32) -> f32, // terrain altitude at cell coords
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
}

impl Ctx<'_> {
    /// world-relative metres of a plan point
    fn wm(&self, u: f32, v: f32) -> [f32; 2] {
        [
            self.eu[0] * u + self.nvec[0] * v,
            self.eu[1] * u + self.nvec[1] * v,
        ]
    }

    /// terrain altitude at a plan point
    fn hh(&self, u: f32, v: f32) -> f32 {
        let w = self.wm(u, v);
        (self.h_at)(self.anchor[0] + w[0] / 16.0, self.anchor[1] + w[1] / 16.0)
    }

    /// absolute-mesh vertex at a plan point, altitude in metres
    fn pt(&self, u: f32, v: f32, z: f32) -> [f32; 3] {
        let w = self.wm(u, v);
        [
            self.anchor[0] * 16.0 + w[0],
            z - self.mn,
            self.anchor[1] * 16.0 + w[1],
        ]
    }
}

/// A cross-section: station `t` along the street, `c` the across-coordinate
/// of the centreline, `hr`/`hw` the roadway / sidewalk half-widths.
struct Sec {
    t: f32,
    h0: f32, // roadway top (altitude)
    hn: f32, // lowest ground across the section
}

fn pair(u_along: bool, t: f32, c: f32, x: f32) -> (f32, f32) {
    if u_along {
        (t, c + x)
    } else {
        (c + x, t)
    }
}

fn section(ctx: &Ctx, u_along: bool, t: f32, c: f32, hr: f32, hw: f32) -> Sec {
    let mut hm = f32::MIN;
    let mut hn = f32::MAX;
    for a in [0.0f32, hr, -hr, hw, -hw] {
        let (u, v) = pair(u_along, t, c, a);
        let h = ctx.hh(u, v);
        hm = hm.max(h);
        hn = hn.min(h);
    }
    Sec { t, h0: hm + ROAD_H, hn }
}

/// Top-facing quad (normal up): the winding flips with the axis pair.
#[allow(clippy::too_many_arguments)]
fn quad_top(
    b: &mut Builder,
    ctx: &Ctx,
    u_along: bool,
    c: f32,
    a: &Sec,
    e: &Sec,
    x0: f32,
    x1: f32,
    dz: f32,
    mat: u32,
    uk: f32,
    xk: f32,
) {
    let vx = |s: &Sec, x: f32| {
        let (u, v) = pair(u_along, s.t, c, x);
        ctx.pt(u, v, s.h0 + dz)
    };
    let (p, uv) = if u_along {
        (
            [vx(a, x0), vx(a, x1), vx(e, x1), vx(e, x0)],
            [
                [a.t / uk, (c + x0) / xk],
                [a.t / uk, (c + x1) / xk],
                [e.t / uk, (c + x1) / xk],
                [e.t / uk, (c + x0) / xk],
            ],
        )
    } else {
        (
            [vx(a, x0), vx(e, x0), vx(e, x1), vx(a, x1)],
            [
                [(c + x0) / xk, a.t / uk],
                [(c + x0) / xk, e.t / uk],
                [(c + x1) / xk, e.t / uk],
                [(c + x1) / xk, a.t / uk],
            ],
        )
    };
    b.quad(mat, p, uv);
}

/// One street segment: roadway, kerbs, sidewalks — the pieces the caller
/// asks for. Kerb faces and sidewalk skirts run down far enough to stay
/// buried where the terrain drops away across the section.
#[allow(clippy::too_many_arguments)]
fn seg(
    b: &mut Builder,
    ctx: &Ctx,
    t0: f32,
    t1: f32,
    u_along: bool,
    c: f32,
    hr: f32,
    hw: f32,
    road: bool,
    kerb: bool,
    walk: bool,
) {
    let n = ((t1 - t0) / STEP).ceil().max(1.0) as usize;
    let secs: Vec<Sec> = (0..=n)
        .map(|i| section(ctx, u_along, t0 + (t1 - t0) * i as f32 / n as f32, c, hr, hw))
        .collect();
    for k in 0..n {
        let (a, e) = (&secs[k], &secs[k + 1]);
        if road {
            quad_top(b, ctx, u_along, c, a, e, -hr, hr, 0.0, M_ASPHALT, 4.0, 4.0);
        }
        if kerb {
            // granite kerb tops either side
            quad_top(b, ctx, u_along, c, a, e, hr, hr + KERB_W, KERB_H, M_KERB, 2.0, 2.0);
            quad_top(b, ctx, u_along, c, a, e, -hr - KERB_W, -hr, KERB_H, M_KERB, 2.0, 2.0);
            // kerb faces toward the roadway, dropped well below grade
            for sx in [hr, -hr] {
                let va = |s: &Sec, z: f32| {
                    let (u, v) = pair(u_along, s.t, c, sx);
                    ctx.pt(u, v, z)
                };
                let lo = |s: &Sec| s.hn - 0.5;
                let p = [
                    va(a, a.h0 + KERB_H),
                    va(e, e.h0 + KERB_H),
                    va(e, lo(e)),
                    va(a, lo(a)),
                ];
                if sx < 0.0 {
                    b.quad(
                        M_KERB,
                        [p[1], p[0], p[3], p[2]],
                        [
                            [e.t / 2.0, 1.0],
                            [a.t / 2.0, 1.0],
                            [a.t / 2.0, 0.0],
                            [e.t / 2.0, 0.0],
                        ],
                    );
                } else {
                    b.quad(
                        M_KERB,
                        p,
                        [
                            [a.t / 2.0, 1.0],
                            [e.t / 2.0, 1.0],
                            [e.t / 2.0, 0.0],
                            [a.t / 2.0, 0.0],
                        ],
                    );
                }
            }
        }
        if walk {
            // tiled sidewalks, flush with the kerb tops
            quad_top(
                b, ctx, u_along, c, a, e,
                hr + KERB_W, hw, KERB_H, M_WALK, 2.0, 2.0,
            );
            quad_top(
                b, ctx, u_along, c, a, e,
                -hw, -hr - KERB_W, KERB_H, M_WALK, 2.0, 2.0,
            );
            // outer edges drop to their own local ground
            for sx in [hw, -hw] {
                let va = |s: &Sec, z: f32| {
                    let (u, v) = pair(u_along, s.t, c, sx);
                    ctx.pt(u, v, z)
                };
                let gb = |s: &Sec| {
                    let (u, v) = pair(u_along, s.t, c, sx);
                    ctx.hh(u, v) - 0.5
                };
                let p = [
                    va(a, a.h0 + KERB_H),
                    va(e, e.h0 + KERB_H),
                    va(e, gb(e)),
                    va(a, gb(a)),
                ];
                let uv = [
                    [a.t / 2.0, 0.9],
                    [e.t / 2.0, 0.9],
                    [e.t / 2.0, 0.0],
                    [a.t / 2.0, 0.0],
                ];
                if sx < 0.0 {
                    b.quad(
                        M_WALK,
                        [p[1], p[0], p[3], p[2]],
                        [uv[1], uv[0], uv[3], uv[2]],
                    );
                } else {
                    b.quad(M_WALK, p, uv);
                }
            }
        }
    }
}

/// Paint stripe on the roadway: a quad at +0.05 over the slab, spanning
/// [x0, x1] across and one stripe width along the street.
#[allow(clippy::too_many_arguments)]
fn stripe(
    b: &mut Builder,
    ctx: &Ctx,
    u_along: bool,
    c: f32,
    hr: f32,
    hw: f32,
    t0: f32,
    t1: f32,
    x0: f32,
    x1: f32,
) {
    let sa = section(ctx, u_along, t0, c, hr, hw);
    let se = section(ctx, u_along, t1, c, hr, hw);
    quad_top(b, ctx, u_along, c, &sa, &se, x0, x1, 0.05, M_PAINT, 2.0, 2.0);
}

pub fn build_streets(
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
    p: &Params,
) -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>) {
    let ctx = Ctx {
        h_at,
        anchor,
        mn,
        eu,
        nvec,
    };
    let mut b = Builder::new();
    let (hr, hw) = (3.5f32, 6.5f32); // main street: 7 m roadway + kerbs + walks

    // ---- main street: the roadway runs through, kerbs and sidewalks
    // break at every cross lane
    seg(&mut b, &ctx, p.main_t0, p.main_t1, true, 0.0, hr, hw, true, false, false);
    let mut cuts = p.cross.clone();
    cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ranges: Vec<(f32, f32)> = Vec::new();
    let mut prev = p.main_t0;
    for &(uc, _, cw) in &cuts {
        ranges.push((prev, uc - cw - 0.05));
        prev = uc + cw + 0.05;
    }
    ranges.push((prev, p.main_t1));
    for &(a, e) in &ranges {
        if e - a > 0.5 {
            seg(&mut b, &ctx, a, e, true, 0.0, hr, hw, false, true, true);
        }
    }

    // ---- cross lanes through the gaps between block columns (each sized to
    // the gap it runs through)
    for &(uc, cr, cw) in &p.cross {
        for (t0, t1) in [(3.5, p.cross_half), (-p.cross_half, -3.5)] {
            if t1 > t0 + 0.5 {
                seg(&mut b, &ctx, t0, t1, false, uc, cr, cw, true, true, false);
            }
        }
        for (t0, t1) in [(6.5, p.cross_half), (-p.cross_half, -6.5)] {
            if t1 > t0 + 0.5 {
                seg(&mut b, &ctx, t0, t1, false, uc, cr, cw, false, false, true);
            }
        }
    }

    // ---- centre line: 3 m dashes, 3 m gaps, kept out of the junctions
    let mut s = p.main_t0 + 1.0;
    while s + 3.0 < p.main_t1 {
        let clear = p
            .cross
            .iter()
            .all(|&(uc, _, _)| s + 3.0 < uc - 10.0 || s > uc + 10.0);
        if clear {
            stripe(
                &mut b, &ctx, true, 0.0, hr, hw, s, s + 3.0, -0.125, 0.125,
            );
        }
        s += 6.0;
    }

    // ---- zebra crossings at every junction approach
    for &(uc, cr, cw) in &p.cross {
        // across the main roadway, both sides of the junction
        for k in 0..5 {
            let off = cw + 0.25 + k as f32 * 1.1;
            stripe(
                &mut b, &ctx, true, 0.0, hr, hw, uc + off, uc + off + 0.55, -hr, hr,
            );
            stripe(
                &mut b, &ctx, true, 0.0, hr, hw, uc - off - 0.55, uc - off, -hr, hr,
            );
        }
        // across the cross lane, both sides of the main street
        for k in 0..5 {
            let off = 3.9 + k as f32 * 1.1;
            stripe(&mut b, &ctx, false, uc, cr, cw, off, off + 0.55, -cr, cr);
            stripe(&mut b, &ctx, false, uc, cr, cw, -off - 0.55, -off, -cr, cr);
        }
    }

    (b.finish(4), textures())
}

// ------------------------------------------------------------- textures ----

fn hash2(x: u32, y: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

fn push(v: &mut Vec<u8>, r: i32, g: i32, b: i32) {
    v.extend_from_slice(&[
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        255,
    ]);
}

/// 4 m of asphalt: dark grey, fine grain, a few worn patches, sparse grit.
fn tex_asphalt() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            // fine grain only — big features would tile visibly along the street
            let n = (hash2(x, y) % 25) as i32 - 12;
            let grit = if hash2(x ^ 99, y ^ 7) % 89 == 0 { 26 } else { 0 };
            let base = 62 + n + grit;
            push(&mut v, base, base, base + 4);
        }
    }
    (256, 256, v)
}

/// Granite kerb: warm grey-pink stone, block joints every 1 m, a chamfer
/// highlight along the top.
fn tex_kerb() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 500, y + 71) % 17) as i32 - 8;
            let joint = x % 128 < 3;
            let (mut r, mut g, mut b) = (172 + n, 158 + n, 150 + n);
            if joint {
                r -= 52;
                g -= 50;
                b -= 48;
            }
            if y < 18 {
                r += 14;
                g += 14;
                b += 14;
            }
            push(&mut v, r, g, b);
        }
    }
    (256, 256, v)
}

/// 2 m of concrete sidewalk: 0.5 m tile grid, per-tile tone jitter.
fn tex_walk() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let j = (hash2(x / 64, y / 64) % 13) as i32 - 6;
            let n = (hash2(x + 17, y + 300) % 9) as i32 - 4;
            let line = x % 64 < 2 || y % 64 < 2;
            let base = 152 + j + n;
            if line {
                push(&mut v, base - 26, base - 26, base - 24);
            } else {
                push(&mut v, base, base, base - 3);
            }
        }
    }
    (256, 256, v)
}

/// Road paint: off-white, a little wear.
fn tex_paint() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 900, y + 900) % 13) as i32 - 6;
            let worn = if hash2(x ^ 3, y ^ 11) % 47 == 0 { -22 } else { 0 };
            let base = 238 + n + worn;
            push(&mut v, base, base, base - 6);
        }
    }
    (256, 256, v)
}

/// User-drawn roads (the editor): free chains of plan points. Every
/// segment gets the same cross-section as the main street — a 7 m
/// roadway, granite kerbs and tiled sidewalks — sampled on the same
/// smoothed height field, so a drawn road conforms to the terrain exactly
/// like the procedural one. Consecutive segments overlap ~3.4 m at
/// internal nodes (corners close up); free ends keep only a small stub.
pub fn build_roads(
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
    chains: &[Vec<[f32; 2]>],
) -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>) {
    let ctx = Ctx { h_at, anchor, mn, eu, nvec };
    let mut b = Builder::new();
    let (hr, hw) = (3.5f32, 6.5f32);

    // top-facing quad with the winding forced so its normal points up
    let quad_up = |b: &mut Builder, mat: u32, mut p: [[f32; 3]; 4], uv: [[f32; 2]; 4]| {
        let cz = (p[1][0] - p[0][0]) * (p[2][1] - p[1][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[1][0]);
        if cz < 0.0 {
            p = [p[3], p[2], p[1], p[0]];
            let uv = [uv[3], uv[2], uv[1], uv[0]];
            b.quad(mat, p, uv);
        } else {
            b.quad(mat, p, uv);
        }
    };

    for ch in chains {
        if ch.len() < 2 {
            continue;
        }
        for k in 0..ch.len() - 1 {
            let p0 = ch[k];
            let p1 = ch[k + 1];
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let l0 = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if l0 < 0.6 {
                continue;
            }
            let dir = [d[0] / l0, d[1] / l0];
            let per = [-dir[1], dir[0]];
            let e_a = if k == 0 { 0.4 } else { 3.4 };
            let e_b = if k + 2 == ch.len() { 0.4 } else { 3.4 };
            let (t0, t1) = (-e_a, l0 + e_b);
            let g_at = |t: f32, x: f32| -> [f32; 2] {
                [p0[0] + dir[0] * t + per[0] * x, p0[1] + dir[1] * t + per[1] * x]
            };
            let ns = ((t1 - t0) / STEP).ceil().max(1.0) as usize;
            let secs: Vec<Sec> = (0..=ns)
                .map(|i| {
                    let t = t0 + (t1 - t0) * i as f32 / ns as f32;
                    let mut hm = f32::MIN;
                    let mut hn = f32::MAX;
                    for a in [0.0f32, hr, -hr, hw, -hw] {
                        let q = g_at(t, a);
                        let h = ctx.hh(q[0], q[1]);
                        hm = hm.max(h);
                        hn = hn.min(h);
                    }
                    Sec { t, h0: hm + ROAD_H, hn }
                })
                .collect();
            let g3 = |s: &Sec, x: f32, z: f32| -> [f32; 3] {
                let q = g_at(s.t, x);
                ctx.pt(q[0], q[1], z)
            };
            for j in 0..ns {
                let (a, e) = (&secs[j], &secs[j + 1]);
                // roadway slab
                quad_up(
                    &mut b,
                    M_ASPHALT,
                    [g3(a, -hr, a.h0), g3(e, -hr, e.h0), g3(e, hr, e.h0), g3(a, hr, a.h0)],
                    [[a.t / 4.0, 0.0], [e.t / 4.0, 0.0], [e.t / 4.0, 1.0], [a.t / 4.0, 1.0]],
                );
                // kerb tops
                quad_up(
                    &mut b,
                    M_KERB,
                    [
                        g3(a, hr, a.h0 + KERB_H),
                        g3(e, hr, e.h0 + KERB_H),
                        g3(e, hr + KERB_W, e.h0 + KERB_H),
                        g3(a, hr + KERB_W, a.h0 + KERB_H),
                    ],
                    [[a.t / 2.0, 0.0], [e.t / 2.0, 0.0], [e.t / 2.0, 1.0], [a.t / 2.0, 1.0]],
                );
                quad_up(
                    &mut b,
                    M_KERB,
                    [
                        g3(a, -hr - KERB_W, a.h0 + KERB_H),
                        g3(e, -hr - KERB_W, e.h0 + KERB_H),
                        g3(e, -hr, e.h0 + KERB_H),
                        g3(a, -hr, a.h0 + KERB_H),
                    ],
                    [[a.t / 2.0, 0.0], [e.t / 2.0, 0.0], [e.t / 2.0, 1.0], [a.t / 2.0, 1.0]],
                );
                // kerb faces toward the roadway
                for sx in [hr, -hr] {
                    let p = [
                        g3(a, sx, a.h0 + KERB_H),
                        g3(e, sx, e.h0 + KERB_H),
                        g3(e, sx, e.hn - 0.5),
                        g3(a, sx, a.hn - 0.5),
                    ];
                    let uv = [
                        [a.t / 2.0, 1.0],
                        [e.t / 2.0, 1.0],
                        [e.t / 2.0, 0.0],
                        [a.t / 2.0, 0.0],
                    ];
                    if sx < 0.0 {
                        b.quad(M_KERB, [p[1], p[0], p[3], p[2]], [uv[1], uv[0], uv[3], uv[2]]);
                    } else {
                        b.quad(M_KERB, p, uv);
                    }
                }
                // sidewalks, flush with the kerb tops
                quad_up(
                    &mut b,
                    M_WALK,
                    [
                        g3(a, hr + KERB_W, a.h0 + KERB_H),
                        g3(e, hr + KERB_W, e.h0 + KERB_H),
                        g3(e, hw, e.h0 + KERB_H),
                        g3(a, hw, a.h0 + KERB_H),
                    ],
                    [[a.t / 2.0, 0.0], [e.t / 2.0, 0.0], [e.t / 2.0, 1.0], [a.t / 2.0, 1.0]],
                );
                quad_up(
                    &mut b,
                    M_WALK,
                    [
                        g3(a, -hw, a.h0 + KERB_H),
                        g3(e, -hw, e.h0 + KERB_H),
                        g3(e, -hr - KERB_W, e.h0 + KERB_H),
                        g3(a, -hr - KERB_W, a.h0 + KERB_H),
                    ],
                    [[a.t / 2.0, 0.0], [e.t / 2.0, 0.0], [e.t / 2.0, 1.0], [a.t / 2.0, 1.0]],
                );
                // outer edges drop to their own local ground
                for sx in [hw, -hw] {
                    let gb = |s: &Sec| {
                        let q = g_at(s.t, sx);
                        ctx.hh(q[0], q[1]) - 0.5
                    };
                    let p = [
                        g3(a, sx, a.h0 + KERB_H),
                        g3(e, sx, e.h0 + KERB_H),
                        g3(e, sx, gb(e)),
                        g3(a, sx, gb(a)),
                    ];
                    let uv = [
                        [a.t / 2.0, 0.9],
                        [e.t / 2.0, 0.9],
                        [e.t / 2.0, 0.0],
                        [a.t / 2.0, 0.0],
                    ];
                    if sx < 0.0 {
                        b.quad(M_WALK, [p[1], p[0], p[3], p[2]], [uv[1], uv[0], uv[3], uv[2]]);
                    } else {
                        b.quad(M_WALK, p, uv);
                    }
                }
            }
        }
    }
    (b.finish(4), textures())
}

fn textures() -> Vec<(u32, u32, Vec<u8>)> {
    vec![tex_asphalt(), tex_kerb(), tex_walk(), tex_paint()]
}
