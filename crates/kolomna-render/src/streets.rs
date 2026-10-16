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
use crate::procgen::{Builder, Rng};

pub const M_ASPHALT: u32 = 0;
pub const M_KERB: u32 = 1;
pub const M_WALK: u32 = 2;
pub const M_PAINT: u32 = 3;
pub const M_IRON: u32 = 4;
pub const M_GLASS: u32 = 5;
pub const M_BARK: u32 = 6;
pub const M_LEAF: u32 = 7;
pub const M_LEAF2: u32 = 8;
pub const M_SIGN: u32 = 9;
pub const M_WOOD: u32 = 10;
pub const M_WHITE: u32 = 11;
pub const M_LIGHTS: u32 = 12;

const ROAD_H: f32 = 0.14; // roadway sits this high above the highest ground
const KERB_H: f32 = 0.15; // kerb / sidewalk step above the roadway
const KERB_W: f32 = 0.15;
const STEP: f32 = 1.5; // cross-section spacing, m

#[derive(Clone, Default)]
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
    cuts: &NetCuts,
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
    // junctions: the cross lanes AND every drawn-road junction interrupt the
    // kerbs, sidewalks and furniture along the main street; interruptions
    // that sit close together merge into one junction zone
    let lane_cuts = merge_intervals(
        p.cross
            .iter()
            .map(|&(uc, _, cw)| (uc - cw - 0.05, uc + cw + 0.05))
            .collect(),
        0.2,
    );
    let mut all_cuts = lane_cuts.clone();
    all_cuts.extend(cuts.main.iter().copied());
    all_cuts = merge_intervals(all_cuts, 0.6);
    let mut ranges: Vec<(f32, f32)> = Vec::new();
    let mut prev = p.main_t0;
    for &(a, e) in &all_cuts {
        if a > prev + 0.3 {
            ranges.push((prev, a));
        }
        prev = prev.max(e);
    }
    if p.main_t1 > prev + 0.3 {
        ranges.push((prev, p.main_t1));
    }
    for (ri, &(a, e)) in ranges.iter().enumerate() {
        if e - a > 0.5 {
            seg(&mut b, &ctx, a, e, true, 0.0, hr, hw, false, true, true);
        }
        furniture_axis(&mut b, &ctx, 0x5100 + ri as u32, true, a, e, 0.0, hr, hw);
    }

    // ---- cross lanes through the gaps between block columns (each sized to
    // the gap it runs through)
    for (uci, &(uc, cr, cw)) in p.cross.iter().enumerate() {
        // drawn-road junctions interrupt this lane's kerbs / walks / furniture
        let lc = merge_intervals(
            cuts.lanes[uci]
                .iter()
                .map(|&(a, e)| (a - 0.35, e + 0.35))
                .collect(),
            0.6,
        );
        let sub = |lo: f32, hi: f32| -> Vec<(f32, f32)> {
            let mut out = Vec::new();
            let mut prev = lo;
            for &(a, e) in &lc {
                if a > prev + 0.2 && prev < hi {
                    out.push((prev, a.min(hi)));
                }
                prev = prev.max(e);
            }
            if hi > prev + 0.2 {
                out.push((prev, hi));
            }
            out
        };
        for (k, (t0, t1)) in [(3.5f32, p.cross_half), (-p.cross_half, -3.5)]
            .into_iter()
            .enumerate()
        {
            for (a, e) in sub(t0, t1) {
                if e - a > 0.5 {
                    seg(&mut b, &ctx, a, e, false, uc, cr, cw, true, true, false);
                }
                furniture_axis(&mut b, &ctx, 0x6200 + uci as u32 + k as u32 * 64, false, a, e, uc, cr, cw);
            }
        }
        for (t0, t1) in [(6.5, p.cross_half), (-p.cross_half, -6.5)] {
            for (a, e) in sub(t0, t1) {
                if e - a > 0.5 {
                    seg(&mut b, &ctx, a, e, false, uc, cr, cw, false, false, true);
                }
            }
        }
    }

    // ---- centre line: 3 m dashes, 3 m gaps, kept out of the junctions
    let mut s = p.main_t0 + 1.0;
    while s + 3.0 < p.main_t1 {
        let clear = p
            .cross
            .iter()
            .all(|&(uc, _, _)| s + 3.0 < uc - 10.0 || s > uc + 10.0)
            && cuts
                .main
                .iter()
                .all(|&(z0, z1)| s + 3.0 < z0 - 2.0 || s > z1 + 2.0);
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

    (b.finish(13), textures())
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

// ------------------------------------------------------- road furniture ----
// Street-side dressing along any road path: cast-iron lamps alternating
// sides, small trees, bush clusters, enamel signs and the odd bench — all
// deterministic (hash-driven) so a reload rebuilds the very same street.

fn vsub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn vadd(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn vmul(a: [f32; 2], k: f32) -> [f32; 2] {
    [a[0] * k, a[1] * k]
}

fn vlen(a: [f32; 2]) -> f32 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

fn vunit(a: [f32; 2]) -> [f32; 2] {
    let l = vlen(a).max(1e-6);
    [a[0] / l, a[1] / l]
}

fn vperp(a: [f32; 2]) -> [f32; 2] {
    [-a[1], a[0]]
}

fn vdot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// Oriented box between two plan points (as in the garden): sides, caps, top.
fn obb(b: &mut Builder, ctx: &Ctx, a: [f32; 2], e: [f32; 2], wid: f32, z0: f32, z1: f32, mat: u32, uvm: f32) {
    let d = vsub(e, a);
    let l = vlen(d).max(0.02);
    let dir = vunit(d);
    let n = vperp(dir);
    let w = wid * 0.5;
    let p = |t: f32, s: f32, z: f32| -> [f32; 3] {
        let q = vadd(a, vadd(vmul(dir, t), vmul(n, s)));
        ctx.pt(q[0], q[1], z)
    };
    let ul = l / uvm;
    let uw = wid / uvm;
    let vz1 = (z1 - z0) / uvm;
    b.quad(mat, [p(0.0, w, z0), p(l, w, z0), p(l, w, z1), p(0.0, w, z1)], [[0.0, 0.0], [ul, 0.0], [ul, vz1], [0.0, vz1]]);
    b.quad(mat, [p(l, -w, z0), p(0.0, -w, z0), p(0.0, -w, z1), p(l, -w, z1)], [[0.0, 0.0], [ul, 0.0], [ul, vz1], [0.0, vz1]]);
    b.quad(mat, [p(0.0, -w, z0), p(0.0, w, z0), p(0.0, w, z1), p(0.0, -w, z1)], [[0.0, 0.0], [uw, 0.0], [uw, vz1], [0.0, vz1]]);
    b.quad(mat, [p(l, w, z0), p(l, -w, z0), p(l, -w, z1), p(l, w, z1)], [[0.0, 0.0], [uw, 0.0], [uw, vz1], [0.0, vz1]]);
    b.quad(mat, [p(0.0, -w, z1), p(0.0, w, z1), p(l, w, z1), p(l, -w, z1)], [[0.0, 0.0], [uw, 0.0], [uw, ul], [0.0, ul]]);
}

/// Tapered cylinder (vertical) at plan point p, top cap when r1 > 0.03.
fn cyl(b: &mut Builder, ctx: &Ctx, p: [f32; 2], r0: f32, r1: f32, z0: f32, z1: f32, sides: usize, mat: u32, uvm: f32) {
    let ring = |r: f32, z: f32, j: usize| -> [f32; 3] {
        let a = j as f32 / sides as f32 * std::f32::consts::TAU;
        let q = [p[0] + a.cos() * r, p[1] + a.sin() * r];
        ctx.pt(q[0], q[1], z)
    };
    for j in 0..sides {
        let j1 = (j + 1) % sides;
        let u0 = j as f32 / sides as f32 * (std::f32::consts::TAU * r0.max(r1) / uvm);
        let u1 = (j + 1) as f32 / sides as f32 * (std::f32::consts::TAU * r0.max(r1) / uvm);
        let hz = (z1 - z0) / uvm;
        b.quad(mat, [ring(r0, z0, j), ring(r1, z1, j), ring(r1, z1, j1), ring(r0, z0, j1)], [[u0, 0.0], [u0, hz], [u1, hz], [u1, 0.0]]);
    }
    if r1 > 0.03 {
        let c = ctx.pt(p[0], p[1], z1);
        for j in 0..sides {
            let j1 = (j + 1) % sides;
            b.tri(mat, [c, ring(r1, z1, j), ring(r1, z1, j1)], [[r1 / uvm, r1 / uvm], [0.0, 0.0], [std::f32::consts::TAU * r1 / uvm, 0.0]]);
        }
    }
}

/// Jittered blob (crown / bush) around a world point.
fn blob(b: &mut Builder, ctx: &Ctx, c: [f32; 3], r: f32, mat: u32, uvm: f32, rng: &mut Rng) {
    let lats = [-1.0f32, -0.33, 0.33, 1.0];
    let lons = 7;
    let ring = |lat: f32, j: usize, rng: &mut Rng| -> [f32; 3] {
        let a = j as f32 / lons as f32 * std::f32::consts::TAU;
        let rr = r * lat.cos() * rng.range(0.9, 1.1);
        [c[0] + a.cos() * rr, c[1] + lat.sin() * r * rng.range(0.92, 1.05), c[2] + a.sin() * rr]
    };
    let mut rings: Vec<Vec<[f32; 3]>> = Vec::new();
    for &lat in &lats {
        let mut v = Vec::with_capacity(lons);
        for j in 0..lons {
            v.push(ring(lat, j, rng));
        }
        rings.push(v);
    }
    for k in 0..lats.len() - 1 {
        for j in 0..lons {
            let j1 = (j + 1) % lons;
            let (a, e) = (&rings[k], &rings[k + 1]);
            let uv = |j: usize, k: usize| [j as f32 / lons as f32 * 2.0 * r / uvm, k as f32 * 0.8 * r / uvm];
            b.quad(mat, [a[j], e[j], e[j1], a[j1]], [uv(j, k), uv(j, k + 1), uv(j1, k + 2), uv(j, k + 2)]);
        }
    }
    let bot = [c[0], c[1] - r * 0.95, c[2]];
    let top = [c[0], c[1] + r * 1.02, c[2]];
    let r0 = &rings[0];
    let rt = &rings[lats.len() - 1];
    for j in 0..lons {
        let j1 = (j + 1) % lons;
        b.tri(mat, [bot, r0[j1], r0[j]], [[0.5, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        b.tri(mat, [top, rt[j], rt[j1]], [[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]]);
    }
}

/// Street lamp: stone base, cast-iron column, glazed lantern, finial.
fn st_lamp(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32) {
    obb(b, ctx, [p[0] - 0.21, p[1]], [p[0] + 0.21, p[1]], 0.42, z - 0.35, z + 0.34, M_KERB, 0.9);
    cyl(b, ctx, p, 0.095, 0.06, z + 0.34, z + 3.35, 8, M_IRON, 1.0);
    cyl(b, ctx, p, 0.15, 0.13, z + 1.62, z + 1.74, 8, M_IRON, 1.0);
    cyl(b, ctx, p, 0.13, 0.105, z + 3.35, z + 3.46, 8, M_IRON, 1.0);
    for j in 0..4 {
        let a0 = j as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let a1 = (j + 1) as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let q0 = [p[0] + a0.cos() * 0.185, p[1] + a0.sin() * 0.185];
        let q1 = [p[0] + a1.cos() * 0.185, p[1] + a1.sin() * 0.185];
        b.quad(
            M_GLASS,
            [ctx.pt(q0[0], q0[1], z + 3.46), ctx.pt(q1[0], q1[1], z + 3.46), ctx.pt(q1[0], q1[1], z + 4.06), ctx.pt(q0[0], q0[1], z + 4.06)],
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.6], [0.0, 0.6]],
        );
    }
    cyl(b, ctx, p, 0.30, 0.05, z + 4.06, z + 4.38, 4, M_IRON, 1.0);
    cyl(b, ctx, p, 0.045, 0.02, z + 4.38, z + 4.56, 5, M_IRON, 0.6);
}

/// Street tree: trunk plus two or three jittered crowns (three habits);
/// the sidewalk tree stands in a low granite kerb surround (tree pit).
fn st_tree(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32, s: f32, v: u32, along: [f32; 2]) {
    // tree pit: granite kerb rim + soil bed, aligned with the street
    let lat = vperp(along);
    let a = |to: f32, fo: f32| -> [f32; 2] { vadd(p, vadd(vmul(along, to), vmul(lat, fo))) };
    for fo in [0.715f32, -0.715] {
        obb(b, ctx, a(-0.775, fo), a(0.775, fo), 0.12, z, z + 0.14, M_KERB, 0.9);
    }
    for to in [0.715f32, -0.715] {
        obb(b, ctx, a(to, -0.655), a(to, 0.655), 0.12, z, z + 0.14, M_KERB, 0.9);
    }
    obb(b, ctx, a(-0.655, 0.0), a(0.655, 0.0), 1.31, z, z + 0.115, M_BARK, 1.2);
    let mut rng = Rng::new((v as u64).wrapping_mul(0x9E37_79B9) | 1);
    let variant = (v >> 5) % 3;
    let trunk_top = match variant {
        0 => 1.85,
        1 => 2.45,
        _ => 2.10,
    } * s;
    cyl(b, ctx, p, 0.20 * s, 0.12 * s, z - 0.35, z + trunk_top, 7, M_BARK, 1.6);
    let base = z + trunk_top;
    match variant {
        0 => {
            blob(b, ctx, ctx.pt(p[0], p[1], base + 0.62 * s), 1.25 * s, M_LEAF, 3.0, &mut rng);
            for _ in 0..2 {
                let dx = rng.range(-0.55, 0.55) * s;
                let dz = rng.range(-0.55, 0.55) * s;
                let dzz = rng.range(-0.1, 0.2) * s;
                blob(b, ctx, ctx.pt(p[0] + dx, p[1] + dz, base + 1.15 * s + dzz), 0.80 * s, M_LEAF, 3.0, &mut rng);
            }
        }
        1 => {
            blob(b, ctx, ctx.pt(p[0], p[1], base + 0.90 * s), 1.00 * s, M_LEAF, 3.0, &mut rng);
            blob(b, ctx, ctx.pt(p[0], p[1], base + 1.80 * s), 0.82 * s, M_LEAF, 3.0, &mut rng);
            let dx = rng.range(-0.25, 0.25) * s;
            blob(b, ctx, ctx.pt(p[0] + dx, p[1], base + 2.50 * s), 0.55 * s, M_LEAF2, 3.0, &mut rng);
        }
        _ => {
            blob(b, ctx, ctx.pt(p[0], p[1], base + 0.75 * s), 1.15 * s, M_LEAF, 3.0, &mut rng);
            let dx = rng.range(-0.4, 0.4) * s;
            let dz = rng.range(-0.4, 0.4) * s;
            blob(b, ctx, ctx.pt(p[0] + dx, p[1] + dz, base + 1.50 * s), 0.70 * s, M_LEAF2, 3.0, &mut rng);
        }
    }
}

/// Bush cluster: two or three low blobs.
fn st_bush(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32, v: u32) {
    let mut rng = Rng::new((v as u64).wrapping_mul(0xC2B2_AE35) | 1);
    let n = 2 + rng.below(2) as usize;
    for k in 0..n {
        let dx = rng.range(-0.38, 0.38);
        let dz = rng.range(-0.38, 0.38);
        let r = rng.range(0.40, 0.62);
        let mat = if k % 2 == 0 { M_LEAF2 } else { M_LEAF };
        blob(b, ctx, ctx.pt(p[0] + dx, p[1] + dz, z + r * 0.5), r, mat, 2.2, &mut rng);
    }
}

/// Enamel street sign: iron pole, white plate, blue faces, white bar.
fn st_sign(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32, along: [f32; 2]) {
    let lat = vperp(along);
    let (z0, z1) = (z + 2.02, z + 2.68);
    cyl(b, ctx, p, 0.055, 0.045, z - 0.30, z + 2.20, 6, M_IRON, 0.8);
    let a = [p[0] - lat[0] * 0.50, p[1] - lat[1] * 0.50];
    let e = [p[0] + lat[0] * 0.50, p[1] + lat[1] * 0.50];
    obb(b, ctx, a, e, 0.055, z0, z1, M_WHITE, 1.0);
    for sdir in [1.0f32, -1.0] {
        let off = [p[0] + along[0] * 0.05 * sdir, p[1] + along[1] * 0.05 * sdir];
        let ba = [off[0] - lat[0] * 0.42, off[1] - lat[1] * 0.42];
        let be = [off[0] + lat[0] * 0.42, off[1] + lat[1] * 0.42];
        obb(b, ctx, ba, be, 0.012, z0 + 0.05, z1 - 0.05, M_SIGN, 0.8);
        let off2 = [p[0] + along[0] * 0.06 * sdir, p[1] + along[1] * 0.06 * sdir];
        let ca = [off2[0] - lat[0] * 0.22, off2[1] - lat[1] * 0.22];
        let ce = [off2[0] + lat[0] * 0.22, off2[1] + lat[1] * 0.22];
        obb(b, ctx, ca, ce, 0.012, z + 2.28, z + 2.42, M_WHITE, 0.6);
    }
}

/// Bench facing `face`, standing on the given surface height.
fn st_bench(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32, face: [f32; 2]) {
    let t = vperp(face);
    let q = |to: f32, fo: f32| -> [f32; 2] { vadd(p, vadd(vmul(t, to), vmul(face, fo))) };
    for s in [-0.82f32, 0.0, 0.82] {
        let w2 = if s == 0.0 { 0.34 } else { 0.5 };
        obb(b, ctx, q(s - 0.03, -0.24), q(s + 0.03, 0.24), w2, z - 0.25, z + 0.44, M_IRON, 0.8);
    }
    for s in [-0.82f32, 0.82] {
        obb(b, ctx, q(s - 0.03, 0.20), q(s + 0.03, 0.26), 0.1, z + 0.44, z + 1.0, M_IRON, 0.8);
    }
    for f in [-0.195f32, -0.065, 0.065, 0.195] {
        obb(b, ctx, q(-0.9, f - 0.052), q(0.9, f + 0.052), 0.104, z + 0.44, z + 0.49, M_WOOD, 1.2);
    }
    for (z0, z1) in [(0.62f32, 0.70f32), (0.79, 0.87), (0.93, 1.0)] {
        obb(b, ctx, q(-0.9, 0.235), q(0.9, 0.295), 0.09, z + z0, z + z1, M_WOOD, 1.2);
    }
}

// ------------------------------------------------------- smoothed paths ----

/// Station along a road path: arc length, plan point, left normal (perp of
/// the tangent), roadway height and the lowest ground across the section.
struct St {
    s: f32,
    p: [f32; 2],
    n: [f32; 2],
    h0: f32,
    hn: f32,
}

/// Stations every STEP m along a smoothed chain, heights sampled across the
/// section exactly like the straight streets do.
fn path_stations(ctx: &Ctx, path: &[[f32; 2]], hr: f32, hw: f32) -> Vec<St> {
    let mut lens = vec![0.0f32];
    let mut total = 0.0f32;
    for i in 1..path.len() {
        total += vlen(vsub(path[i], path[i - 1]));
        lens.push(total);
    }
    if total < 0.5 {
        return Vec::new();
    }
    let n = (total / STEP).ceil().max(2.0) as usize;
    let mut out = Vec::with_capacity(n + 1);
    let mut k = 1usize;
    for i in 0..=n {
        let s = total * i as f32 / n as f32;
        while k < lens.len() - 1 && lens[k] < s {
            k += 1;
        }
        let (l0, l1) = (lens[k - 1], lens[k]);
        let f = if l1 > l0 { (s - l0) / (l1 - l0) } else { 0.0 };
        let p = [
            path[k - 1][0] + (path[k][0] - path[k - 1][0]) * f,
            path[k - 1][1] + (path[k][1] - path[k - 1][1]) * f,
        ];
        let d = vunit(vsub(path[k], path[k - 1]));
        let nv = vperp(d);
        let mut hm = f32::MIN;
        let mut hn = f32::MAX;
        for a in [0.0f32, hr, -hr, hw, -hw] {
            let q = [p[0] + nv[0] * a, p[1] + nv[1] * a];
            let h = ctx.hh(q[0], q[1]);
            hm = hm.max(h);
            hn = hn.min(h);
        }
        out.push(St { s, p, n: nv, h0: hm + ROAD_H, hn });
    }
    out
}

/// Interpolated station at arc length s.
fn st_at(stns: &[St], s: f32) -> St {
    if stns.is_empty() {
        return St { s: 0.0, p: [0.0, 0.0], n: [0.0, 1.0], h0: 0.0, hn: 0.0 };
    }
    let s = s.clamp(0.0, stns[stns.len() - 1].s);
    let mut k = 1usize;
    while k < stns.len() - 1 && stns[k].s < s {
        k += 1;
    }
    let (a, e) = (&stns[k - 1], &stns[k]);
    let f = if e.s > a.s { (s - a.s) / (e.s - a.s) } else { 0.0 };
    St {
        s,
        p: [a.p[0] + (e.p[0] - a.p[0]) * f, a.p[1] + (e.p[1] - a.p[1]) * f],
        n: vunit([a.n[0] + (e.n[0] - a.n[0]) * f, a.n[1] + (e.n[1] - a.n[1]) * f]),
        h0: a.h0 + (e.h0 - a.h0) * f,
        hn: a.hn + (e.hn - a.hn) * f,
    }
}

/// Corner-filleted dense polyline: every interior vertex is rounded with an
/// arc of radius r (capped by the neighbouring segment lengths), so the
/// roadway sweeps through turns instead of meeting them at a sharp angle.
/// Returns the points and the arc-length positions of the corner apexes.
fn smooth_chain(pts: &[[f32; 2]], r: f32) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut clean: Vec<[f32; 2]> = Vec::new();
    for &p in pts {
        if clean.last().map(|q| vlen(vsub(p, *q)) > 0.3).unwrap_or(true) {
            clean.push(p);
        }
    }
    if clean.len() < 2 {
        return (clean, Vec::new());
    }
    // small stubs at the free ends so the slab covers the very last click
    let d0 = vunit(vsub(clean[1], clean[0]));
    let dn = vunit(vsub(clean[clean.len() - 1], clean[clean.len() - 2]));
    let start = vsub(clean[0], vmul(d0, 0.6));
    let end = vadd(clean[clean.len() - 1], vmul(dn, 0.6));
    fn push(out: &mut Vec<[f32; 2]>, s_run: &mut f32, p: [f32; 2]) {
        *s_run += vlen(vsub(p, *out.last().unwrap()));
        out.push(p);
    }
    let mut out = vec![start];
    let mut s_run = vlen(vsub(clean[0], start));
    let mut corners = Vec::new();
    for i in 1..clean.len() - 1 {
        let p = clean[i - 1];
        let vpt = clean[i];
        let q = clean[i + 1];
        let d1 = vunit(vsub(vpt, p));
        let d2 = vunit(vsub(q, vpt));
        let dot = (d1[0] * d2[0] + d1[1] * d2[1]).clamp(-1.0, 1.0);
        let turn = dot.acos();
        if turn < 0.12 {
            push(&mut out, &mut s_run, vpt);
            continue;
        }
        let half = turn * 0.5;
        let t = (r * half.tan())
            .min(vlen(vsub(vpt, p)) * 0.48)
            .min(vlen(vsub(q, vpt)) * 0.48)
            .max(0.05);
        let rr = t / half.tan();
        let ta = vsub(vpt, vmul(d1, t));
        let tb = vadd(vpt, vmul(d2, t));
        let cross = d1[0] * d2[1] - d1[1] * d2[0];
        let sg = if cross >= 0.0 { 1.0f32 } else { -1.0f32 };
        let nc = [-d1[1] * sg, d1[0] * sg];
        let c = vadd(ta, vmul(nc, rr));
        let a0 = (ta[1] - c[1]).atan2(ta[0] - c[0]);
        let steps = ((rr * turn) / 1.1).ceil().max(2.0) as usize;
        push(&mut out, &mut s_run, ta);
        let s_arc0 = s_run;
        for k in 1..=steps {
            let a = a0 + sg * turn * (k as f32 / steps as f32);
            push(&mut out, &mut s_run, [c[0] + a.cos() * rr, c[1] + a.sin() * rr]);
        }
        let _ = tb;
        corners.push(s_arc0 + rr * half);
    }
    push(&mut out, &mut s_run, end);
    (out, corners)
}

/// Deterministic street furniture along [0, len]: lamps alternate sides,
/// trees and bushes fill between, signs occasionally, benches near a few
/// lamps.  `get(s, x)` maps arc length + lateral offset to (plan point,
/// sidewalk top, along direction, outward normal).
fn street_furniture(
    b: &mut Builder,
    ctx: &Ctx,
    seed: u32,
    hr: f32,
    hw: f32,
    len: f32,
    get: &dyn Fn(f32, f32) -> ([f32; 2], f32, [f32; 2], [f32; 2]),
    corners: &[f32],
    blocked: &dyn Fn([f32; 2]) -> bool,
) {
    if len < 14.0 {
        return;
    }
    let x_lamp = hr + 0.85;
    let x_tree = (hr + 1.90).min(hw - 0.30);
    let x_bush = (hr + 2.35).min(hw - 0.20);
    let x_sign = hr + 0.80;
    let step = 4.3f32;
    let mut s = 6.0f32;
    let mut i = 0u32;
    while s < len - 7.0 {
        let j = hash2(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i), 0x51ED);
        let ja = (j & 0xFF) as f32 / 255.0; // 0..1 along jitter
        let jb = ((j >> 8) & 0xFF) as f32 / 255.0; // 0..1 lateral jitter
        let period = i / 8;
        let slot = i % 8;
        let side_a = if period % 2 == 0 { 1.0f32 } else { -1.0f32 };
        let is_sign = slot == 3 && period % 3 == 2;
        let (x, item): (f32, u8) = match slot {
            0 => (x_lamp, 0),
            1 => (x_bush, 2),
            2 => (x_tree, 1),
            3 => {
                if is_sign {
                    (x_sign, 3)
                } else {
                    (x_bush, 2)
                }
            }
            4 => (x_lamp, 0),
            5 => (x_bush, 2),
            6 => (x_tree, 1),
            _ => (x_bush, 2),
        };
        let side = match slot {
            0 | 3 | 5 | 6 => side_a,
            _ => -side_a,
        };
        let ss = s + (jb - 0.5) * 1.4;
        let clear = ss > 4.0
            && ss < len - 4.0
            && corners.iter().all(|&c| (ss - c).abs() > 3.0);
        if clear {
            let (p, z, along, outw) = get(ss, side * (x + (ja - 0.5) * 0.5));
            if blocked(p) {
                s += step + ja * 1.1;
                i += 1;
                continue;
            }
            let variant = j ^ hash2(i ^ 0x77, seed);
            match item {
                0 => {
                    st_lamp(b, ctx, p, z);
                    if variant % 5 == 1 {
                        let (bp, bz, _, _) = get(ss + 2.6, side * (x_lamp + 1.35));
                        // back to the sidewalk, seat facing the roadway
                        let face = if side > 0.0 { outw } else { vmul(outw, -1.0) };
                        st_bench(b, ctx, bp, bz, face);
                    }
                }
                1 => st_tree(b, ctx, p, z, 0.80 + ((j >> 20) & 7) as f32 * 0.035, variant, along),
                2 => st_bush(b, ctx, p, z, variant),
                _ => st_sign(b, ctx, p, z, along),
            }
        }
        s += step + ja * 1.1;
        i += 1;
    }
}

/// Furniture along one straight axis piece of the grid streets.
#[allow(clippy::too_many_arguments)]
fn furniture_axis(
    b: &mut Builder,
    ctx: &Ctx,
    seed: u32,
    u_along: bool,
    t0: f32,
    t1: f32,
    c: f32,
    hr: f32,
    hw: f32,
) {
    let len = t1 - t0;
    if len < 14.0 {
        return;
    }
    let along = if u_along { ctx.eu } else { ctx.nvec };
    let outw = if u_along { ctx.nvec } else { ctx.eu };
    let get = |s: f32, x: f32| -> ([f32; 2], f32, [f32; 2], [f32; 2]) {
        let t = t0 + s;
        let (u, v) = pair(u_along, t, c, x);
        let sec = section(ctx, u_along, t, c, hr, hw);
        ([u, v], sec.h0 + KERB_H, along, outw)
    };
    street_furniture(b, ctx, seed, hr, hw, len, &get, &[], &|_| false);
}

// ------------------------------------------------------------ junctions --

/// The cuts the junctions impose on the procedural grid + stats.
#[derive(Clone, Default)]
pub struct NetCuts {
    /// main-street axis intervals (u, metres)
    pub main: Vec<(f32, f32)>,
    /// per cross lane: intervals along v (metres)
    pub lanes: Vec<Vec<(f32, f32)>>,
    /// crossings + attachments found (self-test counters)
    pub events: usize,
    /// chain free ends that sit on another way
    pub attaches: usize,
}

enum WKind {
    Chain(usize),
    Main,
    Lane(usize),
}

/// A way of the junction network: a plan polyline plus its half-widths.
struct JWay<'a> {
    pts: &'a [[f32; 2]],
    hr: f32,
    hw: f32,
    kind: WKind,
}

/// Proper segment intersection: returns the point and the parameters.
fn seg_int(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> Option<([f32; 2], f32, f32)> {
    let da = vsub(a1, a0);
    let db = vsub(b1, b0);
    let den = da[0] * db[1] - da[1] * db[0];
    if den.abs() < 1e-6 {
        return None;
    }
    let d = vsub(b0, a0);
    let t = (d[0] * db[1] - d[1] * db[0]) / den;
    let u = (d[0] * da[1] - d[1] * da[0]) / den;
    if !(-0.001..=1.001).contains(&t) || !(-0.001..=1.001).contains(&u) {
        return None;
    }
    Some((vadd(a0, vmul(da, t)), t, u))
}

/// Intersection of two infinite lines (point + direction).
fn line_int2(p: [f32; 2], d: [f32; 2], q: [f32; 2], e: [f32; 2]) -> Option<[f32; 2]> {
    let den = d[0] * e[1] - d[1] * e[0];
    if den.abs() < 1e-6 {
        return None;
    }
    let r = vsub(q, p);
    let t = (r[0] * e[1] - r[1] * e[0]) / den;
    Some(vadd(p, vmul(d, t)))
}

/// Nearest point on a polyline and its distance.
pub fn near_poly(pts: &[[f32; 2]], p: [f32; 2]) -> ([f32; 2], f32) {
    let mut best = (p, f32::MAX);
    for k in 0..pts.len().saturating_sub(1) {
        let a = pts[k];
        let ab = vsub(pts[k + 1], a);
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        if l2 < 1e-9 {
            continue;
        }
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0);
        let q = vadd(a, vmul(ab, t));
        let d = vlen(vsub(p, q));
        if d < best.1 {
            best = (q, d);
        }
    }
    best
}

/// The smoothed centre line of a chain (the same fillet path the mesh uses)
/// — house clearing and park splitting measure against the road as drawn.
pub fn smooth_path(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    smooth_chain(pts, 9.0).0
}

/// Nearest point on a polyline plus the local unit normal (its direction
/// rotated 90°); used to snap onto a road's cross-section edge.
pub fn near_poly_n(pts: &[[f32; 2]], p: [f32; 2]) -> ([f32; 2], [f32; 2], f32) {
    let mut best = (p, [0.0f32, 1.0], f32::MAX);
    for k in 0..pts.len().saturating_sub(1) {
        let a = pts[k];
        let ab = vsub(pts[k + 1], a);
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        if l2 < 1e-9 {
            continue;
        }
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0);
        let q = vadd(a, vmul(ab, t));
        let d = vlen(vsub(p, q));
        if d < best.2 {
            let dir = vunit(ab);
            best = (q, [-dir[1], dir[0]], d);
        }
    }
    best
}

/// Shortest distance between two segments.
pub fn seg_seg_dist(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> f32 {
    let seg_pt = |p: [f32; 2], q0: [f32; 2], q1: [f32; 2]| -> f32 {
        let d = vsub(q1, q0);
        let l2 = d[0] * d[0] + d[1] * d[1];
        if l2 < 1e-9 {
            return vlen(vsub(p, q0));
        }
        let t = (((p[0] - q0[0]) * d[0] + (p[1] - q0[1]) * d[1]) / l2).clamp(0.0, 1.0);
        vlen(vsub(p, vadd(q0, vmul(d, t))))
    };
    let d1 = vsub(a1, a0);
    let d2 = vsub(b1, b0);
    let den = d1[0] * d2[1] - d1[1] * d2[0];
    if den.abs() > 1e-6 {
        let r = vsub(b0, a0);
        let t = (r[0] * d2[1] - r[1] * d2[0]) / den;
        let s = (r[0] * d1[1] - r[1] * d1[0]) / den;
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&s) {
            return 0.0;
        }
    }
    seg_pt(a0, b0, b1)
        .min(seg_pt(a1, b0, b1))
        .min(seg_pt(b0, a0, a1))
        .min(seg_pt(b1, a0, a1))
}

/// Signed area of a closed polygon.
fn poly_area(p: &[[f32; 2]]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..p.len() {
        let a = p[k];
        let e = p[(k + 1) % p.len()];
        s += a[0] * e[1] - e[0] * a[1];
    }
    (s * 0.5).abs()
}

/// Sutherland–Hodgman clip of a closed polygon against a half-plane given
/// by its signed distance function `keep`.
fn clip_half(part: &[[f32; 2]], a: [f32; 2], nrm: [f32; 2], keep: impl Fn(f32) -> bool) -> Vec<[f32; 2]> {
    let sd = |p: [f32; 2]| (p[0] - a[0]) * nrm[0] + (p[1] - a[1]) * nrm[1];
    let mut out: Vec<[f32; 2]> = Vec::new();
    for k in 0..part.len() {
        let c = part[k];
        let e = part[(k + 1) % part.len()];
        let (sc, se) = (sd(c), sd(e));
        if keep(sc) {
            out.push(c);
        }
        if keep(sc) != keep(se) {
            let t = sc / (sc - se);
            out.push([c[0] + (e[0] - c[0]) * t, c[1] + (e[1] - c[1]) * t]);
        }
    }
    out
}

/// Cut a park outline by the road corridors that cross it: each smoothed
/// road segment splits every part it truly crosses into the two halves
/// pulled `margin` metres off the centre line, so the fence then runs
/// parallel to the roadway.  Slivers below the size floor are dropped;
/// a road-free outline comes back whole.
pub fn split_outline_by_chains(
    outline: &[[f32; 2]],
    chains: &[Vec<[f32; 2]>],
    margin: f32,
) -> Vec<Vec<[f32; 2]>> {
    let mut parts: Vec<Vec<[f32; 2]>> = vec![outline.to_vec()];
    for ch in chains {
        if ch.len() < 2 {
            continue;
        }
        let (path, _) = smooth_chain(ch, 9.0);
        for w in path.windows(2) {
            let (a, e) = (w[0], w[1]);
            let l = vlen(vsub(e, a));
            if l < 0.15 {
                continue;
            }
            let dir = vmul(vsub(e, a), 1.0 / l);
            let nrm = [-dir[1], dir[0]];
            let sd = |p: [f32; 2]| (p[0] - a[0]) * nrm[0] + (p[1] - a[1]) * nrm[1];
            let mut next: Vec<Vec<[f32; 2]>> = Vec::new();
            for part in parts.drain(..) {
                let straddles =
                    part.iter().any(|&p| sd(p) < -margin) && part.iter().any(|&p| sd(p) > margin);
                let mut touch = f32::MAX;
                if straddles {
                    for k in 0..part.len() {
                        touch = touch
                            .min(seg_seg_dist(a, e, part[k], part[(k + 1) % part.len()]));
                    }
                }
                if !straddles || touch > margin + 2.0 {
                    next.push(part);
                    continue;
                }
                let lo = clip_half(&part, a, nrm, |s| s <= -margin);
                let hi = clip_half(&part, a, nrm, |s| s >= margin);
                let ok_lo = lo.len() >= 3 && poly_area(&lo) > 40.0;
                let ok_hi = hi.len() >= 3 && poly_area(&hi) > 40.0;
                if ok_lo {
                    next.push(lo);
                }
                if ok_hi {
                    next.push(hi);
                }
                if !ok_lo && !ok_hi {
                    // the cut left nothing viable: keep the part as it was
                    next.push(part);
                }
            }
            parts = next;
        }
    }
    parts.retain(|p| {
        let mut u0 = f32::MAX;
        let mut u1 = f32::MIN;
        let mut v0 = f32::MAX;
        let mut v1 = f32::MIN;
        for q in p {
            u0 = u0.min(q[0]);
            u1 = u1.max(q[0]);
            v0 = v0.min(q[1]);
            v1 = v1.max(q[1]);
        }
        (u1 - u0) >= 6.0 && (v1 - v0) >= 6.0 && poly_area(p) >= 250.0
    });
    if parts.is_empty() {
        parts.push(outline.to_vec());
    }
    parts
}

/// Direction of the polyline segment nearest to p.
fn dir_at(pts: &[[f32; 2]], p: [f32; 2]) -> [f32; 2] {
    let mut best = (f32::MAX, [1.0f32, 0.0]);
    for k in 0..pts.len().saturating_sub(1) {
        let a = pts[k];
        let ab = vsub(pts[k + 1], a);
        let l2 = ab[0] * ab[0] + ab[1] * ab[1];
        if l2 < 1e-9 {
            continue;
        }
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0);
        let q = vadd(a, vmul(ab, t));
        let d = vlen(vsub(p, q));
        if d < best.0 {
            best = (d, vunit(ab));
        }
    }
    best.1
}

/// Point p pulled back toward q so it stays within `max` metres of q along d.
fn clamp_along(q: [f32; 2], d: [f32; 2], p: [f32; 2], max: f32) -> [f32; 2] {
    let r = vsub(p, q);
    let t = r[0] * d[0] + r[1] * d[1];
    let tc = t.clamp(-max, max);
    vadd(q, vmul(d, tc))
}

/// Sort + union intervals closer than `gap`.
fn merge_intervals(mut v: Vec<(f32, f32)>, gap: f32) -> Vec<(f32, f32)> {
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<(f32, f32)> = Vec::new();
    for (a, b) in v {
        if let Some(l) = out.last_mut() {
            if a <= l.1 + gap {
                l.1 = l.1.max(b);
                continue;
            }
        }
        out.push((a, b));
    }
    out
}

/// The walk-line of one way crosses the roadway edges of the other: between
/// the two intersection points the kerb + sidewalk of the first way vanish.
fn emit_chords(
    own: &JWay,
    other: &JWay,
    q: [f32; 2],
    do_: [f32; 2],
    db_: [f32; 2],
    out: &mut Vec<([f32; 2], [f32; 2])>,
) {
    for side in [1.0f32, -1.0] {
        let no = vperp(do_);
        let wl = vadd(q, vmul(no, side * own.hw));
        let nb = vperp(db_);
        let e1 = vadd(q, vmul(nb, other.hr));
        let e2 = vsub(q, vmul(nb, other.hr));
        if let (Some(p1), Some(p2)) = (
            line_int2(wl, do_, e1, db_),
            line_int2(wl, do_, e2, db_),
        ) {
            out.push((clamp_along(q, do_, p1, 27.0), clamp_along(q, do_, p2, 27.0)));
        }
    }
}

/// For every pair of ways (at least one a drawn chain) the walk lines of
/// the grid ways are cut where the other's roadway edges cross them: the
/// kerbs, sidewalks and furniture of the main street / lanes break there.
/// Nearby interruptions merge into one junction.  The drawn chains keep
/// clear of the network geometrically inside `build_roads`.
pub fn net_cuts(p: &Params, chains: &[Vec<[f32; 2]>]) -> NetCuts {
    let (hr, hw) = (3.5f32, 6.5f32);
    let mut cuts = NetCuts {
        main: Vec::new(),
        lanes: vec![Vec::new(); p.cross.len()],
        events: 0,
        attaches: 0,
    };
    let main_pts = [[p.main_t0, 0.0], [p.main_t1, 0.0]];
    let mut lane_pts: Vec<[[f32; 2]; 2]> = Vec::new();
    for &(uc, _cr, _cw) in &p.cross {
        lane_pts.push([[uc, 3.5], [uc, p.cross_half.max(3.6)]]);
        lane_pts.push([[uc, -p.cross_half.max(3.6)], [uc, -3.5]]);
    }
    let mut ways: Vec<JWay> = Vec::new();
    for (ci, ch) in chains.iter().enumerate() {
        if ch.len() >= 2 {
            ways.push(JWay { pts: ch, hr, hw, kind: WKind::Chain(ci) });
        }
    }
    if p.main_t1 > p.main_t0 + 1.0 {
        ways.push(JWay { pts: &main_pts, hr, hw, kind: WKind::Main });
    }
    for k in 0..lane_pts.len() {
        let lw = p.cross[k / 2];
        ways.push(JWay { pts: &lane_pts[k], hr: lw.1, hw: lw.2, kind: WKind::Lane(k / 2) });
    }
    // self-crossings of a chain: same enumeration the mesh's junction pass
    // uses, so the counters match what build_roads will actually draw
    for w in ways.iter() {
        if !matches!(w.kind, WKind::Chain(_)) {
            continue;
        }
        for s in 0..w.pts.len() - 1 {
            for t in (s + 2)..w.pts.len() - 1 {
                if seg_int(w.pts[s], w.pts[s + 1], w.pts[t], w.pts[t + 1]).is_some() {
                    let di = vunit(vsub(w.pts[s + 1], w.pts[s]));
                    let dj = vunit(vsub(w.pts[t + 1], w.pts[t]));
                    if vdot(di, dj).abs() <= 0.995 {
                        cuts.events += 1;
                    }
                }
            }
        }
    }
    for i in 0..ways.len() {
        for j in (i + 1)..ways.len() {
            if !matches!(ways[i].kind, WKind::Chain(_)) && !matches!(ways[j].kind, WKind::Chain(_)) {
                continue;
            }
            let (a, b) = (&ways[i], &ways[j]);
            let mut events: Vec<[f32; 2]> = Vec::new();
            for t in 0..b.pts.len() - 1 {
                for s in 0..a.pts.len() - 1 {
                    if let Some((q, _ta, _tb)) = seg_int(a.pts[s], a.pts[s + 1], b.pts[t], b.pts[t + 1]) {
                        events.push(q);
                    }
                }
            }
            for (w, o) in [(i, j), (j, i)] {
                if !matches!(ways[w].kind, WKind::Chain(_)) {
                    continue;
                }
                let endn = ways[w].pts.len() - 1;
                for k in [0usize, endn] {
                    let ep = ways[w].pts[k];
                    let (np, d) = near_poly(ways[o].pts, ep);
                    if d <= 0.45 {
                        cuts.attaches += 1;
                        events.push(np);
                    }
                }
            }
            cuts.events += events.len();
            for q in &events {
                let da = dir_at(a.pts, *q);
                let db = dir_at(b.pts, *q);
                for (own, other, d1, d2) in [(a, b, da, db), (b, a, db, da)] {
                    let mut ch = Vec::new();
                    emit_chords(own, other, *q, d1, d2, &mut ch);
                    match own.kind {
                        WKind::Chain(_) => {}
                        WKind::Main => {
                            // both sides cut over one interval (the grid
                            // breaks kerb + walk across the whole junction)
                            let mut lo = f32::MAX;
                            let mut hi = f32::MIN;
                            for (q1, q2) in ch {
                                lo = lo.min(q1[0].min(q2[0]));
                                hi = hi.max(q1[0].max(q2[0]));
                            }
                            if lo < hi {
                                cuts.main.push((lo - 0.35, hi + 0.35));
                            }
                        }
                        WKind::Lane(li) => {
                            let mut lo = f32::MAX;
                            let mut hi = f32::MIN;
                            for (q1, q2) in ch {
                                lo = lo.min(q1[1].min(q2[1]));
                                hi = hi.max(q1[1].max(q2[1]));
                            }
                            if lo < hi {
                                cuts.lanes[li].push((lo - 0.35, hi + 0.35));
                            }
                        }
                    }
                }
            }
        }
    }
    cuts.main = merge_intervals(cuts.main, 0.5);
    for l in cuts.lanes.iter_mut() {
        *l = merge_intervals(std::mem::take(l), 0.5);
    }
    cuts
}

/// Editor-drawn road chains: the street cross-section swept along a
/// corner-filleted path, a dashed centre line, and street furniture along
/// the sidewalks.  Free ends keep only a small stub.
pub fn build_roads(
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
    chains: &[Vec<[f32; 2]>],
    net: &Params,
) -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>) {
    build_roads_ext(h_at, anchor, mn, eu, nvec, chains, &[], net)
}

// ---------------------------------------------------------------- junctions --
//
// A junction is emitted wherever two drawn roads cross (or a road crosses
// itself) and wherever a free end lands on another way.  For every corner
// sector between two arms the roadway pocket is filled in, the kerb line is
// rounded with a fillet arc and the two sidewalks are joined corner-to-corner
// with drop skirts; zebra crossings with stop lines go on every qualifying
// arm and traffic lights stand on the right side of each signalised approach.

struct JEv {
    q: [f32; 2],
    arms: Vec<([f32; 2], f32, f32)>, // unit dir away from q, kerb half-width, walk half-width
}

fn ev_key(i: usize, j: usize) -> u64 {
    if i <= j {
        ((i as u64) << 32) | j as u64
    } else {
        ((j as u64) << 32) | i as u64
    }
}

fn push_ev(
    evs: &mut Vec<JEv>,
    keys: &mut Vec<(u64, [f32; 2])>,
    key: u64,
    q: [f32; 2],
    arms: Vec<([f32; 2], f32, f32)>,
) {
    if keys.iter().any(|&(k, qq)| k == key && vlen(vsub(qq, q)) < 2.5) {
        return;
    }
    keys.push((key, q));
    evs.push(JEv { q, arms });
}

/// Top-facing triangle: winding forced to match the roadway slabs.
fn tri_j(b: &mut Builder, mat: u32, p: [[f32; 3]; 3], uv: [[f32; 2]; 3]) {
    let s = (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]);
    if s >= 0.0 {
        b.tri(mat, [p[0], p[1], p[2]], [uv[0], uv[1], uv[2]]);
    } else {
        b.tri(mat, [p[0], p[2], p[1]], [uv[0], uv[2], uv[1]]);
    }
}

/// Top-facing quad: same winding rule as `tri_j`.
fn top_quad_j(b: &mut Builder, mat: u32, mut p: [[f32; 3]; 4], mut uv: [[f32; 2]; 4]) {
    let s = (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]);
    if s < 0.0 {
        p = [p[3], p[2], p[1], p[0]];
        uv = [uv[3], uv[2], uv[1], uv[0]];
    }
    b.quad(mat, p, uv);
}

/// Vertical quad whose horizontal normal faces `out`.
fn face_quad_j(b: &mut Builder, mat: u32, mut p: [[f32; 3]; 4], mut uv: [[f32; 2]; 4], out: [f32; 2]) {
    let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];
    if nx * out[0] + nz * out[1] < 0.0 {
        p = [p[3], p[2], p[1], p[0]];
        uv = [uv[3], uv[2], uv[1], uv[0]];
    }
    b.quad(mat, p, uv);
}

/// Ear-clipping triangulation of a simple polygon (plan coords).
fn poly_tris(poly: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let mut pts: Vec<[f32; 2]> = Vec::with_capacity(poly.len());
    for &p in poly {
        if pts.last().map(|q: &[f32; 2]| vlen(vsub(*q, p)) > 1e-4).unwrap_or(true) {
            pts.push(p);
        }
    }
    while pts.len() > 1 && vlen(vsub(pts[0], pts[pts.len() - 1])) < 1e-4 {
        pts.pop();
    }
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let area: f32 = (0..n)
        .map(|i| {
            let a = pts[i];
            let b = pts[(i + 1) % n];
            a[0] * b[1] - a[1] * b[0]
        })
        .sum();
    let mut idx: Vec<usize> = (0..n).collect();
    if area < 0.0 {
        idx.reverse();
    }
    let mut out = Vec::new();
    let mut guard = 0;
    while idx.len() > 3 && guard < 512 {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let ia = idx[(i + m - 1) % m];
            let ib = idx[i];
            let ic = idx[(i + 1) % m];
            let (a, b, c) = (pts[ia], pts[ib], pts[ic]);
            let cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cr <= 1e-7 {
                continue;
            }
            let mut ok = true;
            for &k in &idx {
                if k == ia || k == ib || k == ic {
                    continue;
                }
                let p = pts[k];
                let d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
                let d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]);
                let d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0]);
                if d1 >= -1e-6 && d2 >= -1e-6 && d3 >= -1e-6 {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push([ia, ib, ic]);
                idx.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

/// Roadway top height at a plan point: the highest ground across the slab.
fn slab_top(ctx: &Ctx, p: [f32; 2], dir: [f32; 2], hr: f32) -> f32 {
    let n = vperp(dir);
    let mut h = ctx.hh(p[0], p[1]);
    h = h.max(ctx.hh(p[0] + n[0] * hr, p[1] + n[1] * hr));
    h = h.max(ctx.hh(p[0] - n[0] * hr, p[1] - n[1] * hr));
    h + ROAD_H
}

/// Traffic light: cast-iron pole with a three-lens head facing `face`.
fn traffic_light(b: &mut Builder, ctx: &Ctx, base: [f32; 2], z: f32, face: [f32; 2]) {
    let f = vunit(face);
    let s = vperp(f);
    if std::env::var("KOLOMNA_J_DEBUG").is_ok() {
        eprintln!("  light at ({:.1},{:.1}) face ({:.2},{:.2})", base[0], base[1], f[0], f[1]);
    }
    let lp = |a: f32, c: f32| -> [f32; 2] { vadd(base, vadd(vmul(f, a), vmul(s, c))) };
    cyl(b, ctx, base, 0.062, 0.05, z - 0.35, z + 3.22, 8, M_IRON, 1.0);
    let (bd, bw) = (0.095f32, 0.155f32);
    obb(b, ctx, lp(-bd, 0.0), lp(bd, 0.0), bw * 2.0, z + 2.30, z + 3.16, M_IRON, 0.35);
    for (k, lz) in [(0usize, 2.98f32), (1, 2.73), (2, 2.48)] {
        let vc = 1.0 - (k as f32 * 64.0 + 32.0) / 192.0;
        let lc = lp(bd + 0.016, 0.0);
        let c1 = vadd(lc, vmul(s, -0.11));
        let c2 = vadd(lc, vmul(s, 0.11));
        b.quad(
            M_LIGHTS,
            [
                ctx.pt(c1[0], c1[1], z + lz - 0.11),
                ctx.pt(c2[0], c2[1], z + lz - 0.11),
                ctx.pt(c2[0], c2[1], z + lz + 0.11),
                ctx.pt(c1[0], c1[1], z + lz + 0.11),
            ],
            [
                [0.125, vc - 0.125],
                [0.875, vc - 0.125],
                [0.875, vc + 0.125],
                [0.125, vc + 0.125],
            ],
        );
    }
}

/// Zebra crossing and stop line on one arm of a junction.
fn zebra_stop(b: &mut Builder, ctx: &Ctx, ev: &JEv, i: usize) {
    let (du, hru, _hwu) = ev.arms[i];
    if hru < 3.0 {
        return;
    }
    let mut bstop = 0.0f32;
    for (k, &(dw, hrw, _)) in ev.arms.iter().enumerate() {
        if k == i {
            continue;
        }
        let cr = (du[0] * dw[1] - du[1] * dw[0]).abs();
        if cr < 0.2 {
            continue;
        }
        bstop = bstop.max(hrw / cr.max(0.34));
    }
    if bstop < 0.5 {
        return;
    }
    let s_z = (bstop + 1.7).min(14.0);
    if std::env::var("KOLOMNA_J_DEBUG").is_ok() {
        eprintln!("  zebra arm ({:.2},{:.2}) s {:.1}..{:.1}", du[0], du[1], s_z - 2.0, s_z + 2.0);
    }
    let nrm = vperp(du);
    let zh = |p: [f32; 2]| -> f32 { slab_top(ctx, p, du, hru) + 0.055 };
    let pt = |s: f32, x: f32| -> [f32; 2] { vadd(ev.q, vadd(vmul(du, s), vmul(nrm, x))) };
    let mut x = -hru + 0.2;
    while x + 0.5 <= hru - 0.2 + 0.01 {
        let (q0, q1) = (pt(s_z - 2.0, x), pt(s_z + 2.0, x));
        let (q2, q3) = (pt(s_z + 2.0, x + 0.5), pt(s_z - 2.0, x + 0.5));
        top_quad_j(
            b,
            M_PAINT,
            [
                ctx.pt(q0[0], q0[1], zh(q0)),
                ctx.pt(q1[0], q1[1], zh(q1)),
                ctx.pt(q2[0], q2[1], zh(q2)),
                ctx.pt(q3[0], q3[1], zh(q3)),
            ],
            [
                [q0[0] / 2.0, q0[1] / 2.0],
                [q1[0] / 2.0, q1[1] / 2.0],
                [q2[0] / 2.0, q2[1] / 2.0],
                [q3[0] / 2.0, q3[1] / 2.0],
            ],
        );
        x += 1.0;
    }
    // stop line on the approach side (right of the inbound direction)
    if hru > 1.5 {
        let (t0, t1) = (s_z + 2.15, s_z + 2.5);
        let (q0, q1) = (pt(t0, -0.25), pt(t1, -0.25));
        let (q2, q3) = (pt(t1, -(hru - 0.25)), pt(t0, -(hru - 0.25)));
        top_quad_j(
            b,
            M_PAINT,
            [
                ctx.pt(q0[0], q0[1], zh(q0)),
                ctx.pt(q1[0], q1[1], zh(q1)),
                ctx.pt(q2[0], q2[1], zh(q2)),
                ctx.pt(q3[0], q3[1], zh(q3)),
            ],
            [[0.0, 0.0], [0.35, 0.0], [0.35, 1.0], [0.0, 1.0]],
        );
    }
    // the signal pole stands on the sidewalk at the end of the stop line,
    // just behind the zebra on the right of the inbound direction
    let spos = pt(s_z + 2.9, -(hru + 0.55));
    traffic_light(b, ctx, spos, slab_top(ctx, spos, du, hru) + KERB_H, du);
}

/// One corner of a junction: fillet arc between two kerb lines, pocket
/// asphalt, rounded kerb ring, sidewalk fill, signals.
#[allow(clippy::too_many_arguments)]
fn corner_patch(
    b: &mut Builder,
    ctx: &Ctx,
    q: [f32; 2],
    du: [f32; 2],
    hru: f32,
    hwu: f32,
    dv: [f32; 2],
    hrv: f32,
    hwv: f32,
    th: f32,
    jh: f32,
    others: &[[f32; 2]],
) {
    use std::f32::consts::{PI, TAU};
    let mut nu = vperp(du);
    if vdot(nu, dv) < 0.0 {
        nu = vmul(nu, -1.0);
    }
    let mut nv = vperp(dv);
    if vdot(nv, du) < 0.0 {
        nv = vmul(nv, -1.0);
    }
    let lu = vadd(q, vmul(nu, hru));
    let lv = vadd(q, vmul(nv, hrv));
    let Some(pc) = line_int2(lu, du, lv, dv) else {
        return;
    };
    let r = (14.0 * (th * 0.5).tan()).clamp(1.5, 7.0);
    let cu = vadd(lu, vmul(nu, r));
    let cv = vadd(lv, vmul(nv, r));
    let Some(c) = line_int2(cu, du, cv, dv) else {
        return;
    };
    // a junction owns a 10.5 m disk — the same reach its kerb cuts use:
    // this fillet must never intrude into a neighbouring junction's zone,
    // where the two patches would double up and the sector gap of one
    // junction would get rounded by the other's arc
    if others.iter().any(|&o| vlen(vsub(c, o)) < 10.5) {
        return;
    }
    let tu = vsub(c, vmul(nu, r));
    let tv = vsub(c, vmul(nv, r));
    let a0 = (tu[1] - c[1]).atan2(tu[0] - c[0]);
    let a1 = (tv[1] - c[1]).atan2(tv[0] - c[0]);
    let mut da = a1 - a0;
    while da > PI {
        da -= TAU;
    }
    while da < -PI {
        da += TAU;
    }
    if da.abs() < 1e-3 {
        return;
    }
    {
        let am = a0 + da * 0.5;
        let midr = [c[0] + am.cos() * r, c[1] + am.sin() * r];
        if vdot(vsub(midr, c), vsub(pc, c)) < 0.0 {
            da = if da > 0.0 { da - TAU } else { da + TAU };
        }
    }
    let steps = ((da.abs() / (PI / 14.0)).ceil() as usize).clamp(4, 22);
    let arc = |k: usize| -> f32 { a0 + da * (k as f32 / steps as f32) };
    if std::env::var("KOLOMNA_J_DEBUG").is_ok() {
        eprintln!(
            "  corner th={:.0}deg r={:.2} c=({:.1},{:.1}) pc=({:.1},{:.1})",
            th.to_degrees(),
            r,
            c[0],
            c[1],
            pc[0],
            pc[1]
        );
    }
    let rw = (r - KERB_W).max(0.25);
    let hw_t = jh + KERB_H - 0.002;
    let tail = 6.0f32;
    // ---- asphalt pocket: fan from the sharp corner onto the fillet arc
    for k in 0..steps {
        let (aa, ab) = (arc(k), arc(k + 1));
        let (pa, pb) = (
            [c[0] + aa.cos() * r, c[1] + aa.sin() * r],
            [c[0] + ab.cos() * r, c[1] + ab.sin() * r],
        );
        tri_j(
            b,
            M_ASPHALT,
            [
                ctx.pt(pc[0], pc[1], jh),
                ctx.pt(pa[0], pa[1], jh),
                ctx.pt(pb[0], pb[1], jh),
            ],
            [
                [pc[0] / 4.0, pc[1] / 4.0],
                [pa[0] / 4.0, pa[1] / 4.0],
                [pb[0] / 4.0, pb[1] / 4.0],
            ],
        );
    }
    // ---- rounded kerb ring [rw, r] and its face down to the local ground
    for k in 0..steps {
        let (aa, ab) = (arc(k), arc(k + 1));
        let (oa, ob) = (aa.cos(), aa.sin());
        let (oa1, ob1) = (ab.cos(), ab.sin());
        let q0 = [c[0] + oa * r, c[1] + ob * r];
        let q1 = [c[0] + oa1 * r, c[1] + ob1 * r];
        let s0 = [c[0] + oa * rw, c[1] + ob * rw];
        let s1 = [c[0] + oa1 * rw, c[1] + ob1 * rw];
        top_quad_j(
            b,
            M_KERB,
            [
                ctx.pt(s0[0], s0[1], hw_t),
                ctx.pt(s1[0], s1[1], hw_t),
                ctx.pt(q1[0], q1[1], hw_t),
                ctx.pt(q0[0], q0[1], hw_t),
            ],
            [
                [s0[0] / 2.0, s0[1] / 2.0],
                [s1[0] / 2.0, s1[1] / 2.0],
                [q1[0] / 2.0, q1[1] / 2.0],
                [q0[0] / 2.0, q0[1] / 2.0],
            ],
        );
        let z0 = ctx.hh(q0[0], q0[1]) - 0.5;
        let z1 = ctx.hh(q1[0], q1[1]) - 0.5;
        let mid = [(q0[0] + q1[0]) * 0.5 - c[0], (q0[1] + q1[1]) * 0.5 - c[1]];
        face_quad_j(
            b,
            M_KERB,
            [
                ctx.pt(q0[0], q0[1], hw_t),
                ctx.pt(q1[0], q1[1], hw_t),
                ctx.pt(q1[0], q1[1], z1),
                ctx.pt(q0[0], q0[1], z0),
            ],
            [
                [k as f32 * 0.4, 1.0],
                [(k + 1) as f32 * 0.4, 1.0],
                [(k + 1) as f32 * 0.4, 0.0],
                [k as f32 * 0.4, 0.0],
            ],
            vunit(mid),
        );
    }
    // ---- sidewalk fill: arc at rw joined corner-to-corner with both walks
    let tu2 = vadd(c, vmul(nu, -rw));
    let tv2 = vadd(c, vmul(nv, -rw));
    let p1b = vadd(tu2, vmul(du, tail));
    let p2b = vadd(tv2, vmul(dv, tail));
    let p1 = vadd(p1b, vmul(nu, (hwu - hru - KERB_W).max(0.15)));
    let p2 = vadd(p2b, vmul(nv, (hwv - hrv - KERB_W).max(0.15)));
    let Some(p3) = line_int2(
        vadd(lu, vmul(nu, hwu - hru)),
        du,
        vadd(lv, vmul(nv, hwv - hrv)),
        dv,
    ) else {
        return;
    };
    let mut poly: Vec<[f32; 2]> = Vec::with_capacity(steps + 6);
    poly.push(tu2);
    for k in 1..steps {
        let a = arc(k);
        poly.push([c[0] + a.cos() * rw, c[1] + a.sin() * rw]);
    }
    poly.push(tv2);
    poly.push(p2b);
    poly.push(p2);
    poly.push(p3);
    poly.push(p1);
    poly.push(p1b);
    for t in poly_tris(&poly) {
        let (a, bp, cp) = (poly[t[0]], poly[t[1]], poly[t[2]]);
        tri_j(
            b,
            M_WALK,
            [
                ctx.pt(a[0], a[1], hw_t),
                ctx.pt(bp[0], bp[1], hw_t),
                ctx.pt(cp[0], cp[1], hw_t),
            ],
            [
                [a[0] / 2.0, a[1] / 2.0],
                [bp[0] / 2.0, bp[1] / 2.0],
                [cp[0] / 2.0, cp[1] / 2.0],
            ],
        );
    }
    // skirt the outer boundary edges down to the ground
    let am = a0 + da * 0.5;
    let refp = [c[0] + am.cos() * (rw * 0.6), c[1] + am.sin() * (rw * 0.6)];
    for k in steps..poly.len() {
        let a = poly[k];
        let bp = poly[(k + 1) % poly.len()];
        let e = vsub(bp, a);
        if vlen(e) < 0.05 {
            continue;
        }
        let nrm = vperp(vunit(e));
        let mid = [(a[0] + bp[0]) * 0.5, (a[1] + bp[1]) * 0.5];
        let outward = if vdot(nrm, vsub(mid, refp)) > 0.0 {
            nrm
        } else {
            vmul(nrm, -1.0)
        };
        let (z0, z1) = (ctx.hh(a[0], a[1]) - 0.5, ctx.hh(bp[0], bp[1]) - 0.5);
        face_quad_j(
            b,
            M_WALK,
            [
                ctx.pt(a[0], a[1], hw_t),
                ctx.pt(bp[0], bp[1], hw_t),
                ctx.pt(bp[0], bp[1], z1),
                ctx.pt(a[0], a[1], z0),
            ],
            [
                [a[0] / 2.0, 0.9],
                [bp[0] / 2.0, 0.9],
                [bp[0] / 2.0, 0.0],
                [a[0] / 2.0, 0.0],
            ],
            outward,
        );
    }
}

/// All corner corners + markings of one junction event.
fn junction_patch(b: &mut Builder, ctx: &Ctx, ev: &JEv, others: &[[f32; 2]]) {
    if ev.arms.len() < 2 {
        return;
    }
    let q = ev.q;
    let mut jh = f32::MIN;
    for &(d, hrw, hww) in &ev.arms {
        for t in [hrw, hww] {
            for s in [-t, t] {
                let p = [q[0] + d[0] * s, q[1] + d[1] * s];
                jh = jh.max(ctx.hh(p[0], p[1]));
            }
        }
    }
    let jh = jh + ROAD_H;
    let mut idx: Vec<usize> = (0..ev.arms.len()).collect();
    idx.sort_by(|&x, &y| {
        let (a, c) = (ev.arms[x].0, ev.arms[y].0);
        a[1].atan2(a[0])
            .partial_cmp(&c[1].atan2(c[0]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = idx.len();
    if std::env::var("KOLOMNA_J_DEBUG").is_ok() {
        eprintln!("junction at ({:.1},{:.1}): {n} arms, floor {jh:.2}", q[0], q[1]);
    }
    for k in 0..n {
        let iu = idx[k];
        let iv = idx[(k + 1) % n];
        let (du, hru, hwu) = ev.arms[iu];
        let (dv, hrv, hwv) = ev.arms[iv];
        let cross = du[0] * dv[1] - du[1] * dv[0];
        let dotp = du[0] * dv[0] + du[1] * dv[1];
        let mut th = cross.atan2(dotp);
        if th < 0.0 {
            th += std::f32::consts::TAU;
        }
        if th > std::f32::consts::PI - 0.09 || th < 0.14 {
            continue;
        }
        corner_patch(b, ctx, q, du, hru, hwu, dv, hrv, hwv, th, jh, others);
    }
    for k in 0..n {
        zebra_stop(b, ctx, ev, idx[k]);
    }
}

/// Split a freshly drawn chain where it merely re-covers existing ways:
/// samples within `DUP_D` of an existing way that run parallel to it
/// (|dot| > 0.9) are dropped; the remaining runs are returned as new chains.
/// Also returns the total skipped length (metres).
pub fn split_duplicate(
    chain: &[[f32; 2]],
    roads: &[Vec<[f32; 2]>],
    net: &Params,
) -> (Vec<Vec<[f32; 2]>>, f32) {
    const DUP_D: f32 = 2.4;
    const DUP_PAR: f32 = 0.9;
    if chain.len() < 2 {
        return (Vec::new(), 0.0);
    }
    let mut main_pts = [[0.0f32, 0.0], [0.0, 0.0]];
    if net.main_t1 > net.main_t0 + 1.0 {
        main_pts = [[net.main_t0, 0.0], [net.main_t1, 0.0]];
    }
    let mut lane_pts: Vec<[[f32; 2]; 2]> = Vec::new();
    for &(uc, _cr, _cw) in &net.cross {
        lane_pts.push([[uc, 3.5], [uc, net.cross_half.max(3.6)]]);
        lane_pts.push([[uc, -net.cross_half.max(3.6)], [uc, -3.5]]);
    }
    let mut ways: Vec<&[[f32; 2]]> = Vec::new();
    for r in roads {
        if r.len() >= 2 {
            ways.push(r);
        }
    }
    ways.push(&main_pts);
    for lp in &lane_pts {
        ways.push(lp);
    }
    // resample the chain at 2 m for the coverage classification; the kept
    // runs are later sliced out of the ORIGINAL chain so corner vertices
    // survive (a resampled run would chamfer every turn)
    let step = 2.0f32;
    let mut pts: Vec<[f32; 2]> = vec![chain[0]];
    let mut dirs: Vec<[f32; 2]> = vec![[0.0, 0.0]];
    let mut samps: Vec<f32> = vec![0.0];
    let mut total = 0.0f32;
    for i in 1..chain.len() {
        total += vlen(vsub(chain[i], chain[i - 1]));
    }
    {
        let mut base = 0.0f32;
        let mut carry = 0.0f32;
        for k in 0..chain.len() - 1 {
            let (a, e) = (chain[k], chain[k + 1]);
            let d = vsub(e, a);
            let l = vlen(d);
            if l < 1e-4 {
                continue;
            }
            let u = vunit(d);
            let mut s = step - carry;
            while s < l {
                pts.push(vadd(a, vmul(u, s)));
                dirs.push(u);
                samps.push(base + s);
                s += step;
            }
            base += l;
            carry = (carry + l) % step;
        }
        let last = chain[chain.len() - 1];
        if vlen(vsub(*pts.last().unwrap(), last)) > 0.4 {
            pts.push(last);
            dirs.push(vunit(vsub(last, chain[chain.len() - 2])));
            samps.push(total);
        } else if pts.len() >= 2 {
            let l = pts.len() - 1;
            pts[l] = last;
            samps[l] = total;
        }
    }
    // classify samples: covered by an existing way or by an earlier accepted stretch
    // (the same-chain check needs real arc separation — two neighbours 2 m
    // apart are the same ribbon, not a doubled-back one)
    let mut covered = vec![false; pts.len()];
    let mut prev: Vec<([f32; 2], [f32; 2], f32)> = Vec::new();
    for k in 0..pts.len() {
        let p = pts[k];
        let d = dirs[k];
        let mut cov = false;
        for w in &ways {
            let (np, dd) = near_poly(w, p);
            if dd < DUP_D && vdot(d, dir_at(w, np)).abs() > DUP_PAR {
                cov = true;
                break;
            }
        }
        if !cov {
            for (pp, pd, ps) in &prev {
                if vlen(vsub(*pp, p)) < DUP_D
                    && vdot(d, *pd).abs() > DUP_PAR
                    && (samps[k] - ps).abs() > 6.0
                {
                    cov = true;
                    break;
                }
            }
        }
        if !cov {
            prev.push((p, d, samps[k]));
        }
        covered[k] = cov;
    }
    // bridge hairline holes so a run is not split by a single noisy sample
    for k in 1..covered.len().saturating_sub(1) {
        if covered[k] && !covered[k - 1] && !covered[k + 1] {
            covered[k] = false;
        }
    }
    // point at arc length s along the source chain
    fn pt_at(chain: &[[f32; 2]], s: f32) -> [f32; 2] {
        let mut base = 0.0f32;
        for i in 0..chain.len().saturating_sub(1) {
            let l = vlen(vsub(chain[i + 1], chain[i]));
            if base + l >= s || i + 2 == chain.len() {
                let f = if l > 1e-4 {
                    ((s - base) / l).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                return [
                    chain[i][0] + (chain[i + 1][0] - chain[i][0]) * f,
                    chain[i][1] + (chain[i + 1][1] - chain[i][1]) * f,
                ];
            }
            base += l;
        }
        chain[chain.len() - 1]
    }
    let mut out: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut skipped = 0.0f32;
    let mut k = 0;
    while k < pts.len() {
        if covered[k] {
            skipped += step;
            k += 1;
            continue;
        }
        let start = k;
        while k < pts.len() && !covered[k] {
            k += 1;
        }
        // the run window along the ORIGINAL chain: from the midpoint before
        // its first sample to the midpoint after its last one — the original
        // corner vertices live inside and are kept verbatim
        let s_a = if start == 0 {
            0.0
        } else {
            0.5 * (samps[start - 1] + samps[start])
        };
        let s_b = if k >= pts.len() {
            total
        } else {
            0.5 * (samps[k - 1] + samps[k])
        };
        let mut v: Vec<[f32; 2]> = vec![pt_at(chain, s_a)];
        {
            let mut base = 0.0f32;
            for i in 1..chain.len().saturating_sub(1) {
                base += vlen(vsub(chain[i], chain[i - 1]));
                if base > s_a + 0.05 && base < s_b - 0.05 {
                    v.push(chain[i]);
                }
            }
        }
        let e = pt_at(chain, s_b);
        if v
            .last()
            .map(|q| vlen(vsub(*q, e)) > 0.05)
            .unwrap_or(true)
        {
            v.push(e);
        }
        let l = {
            let mut s = 0.0f32;
            for w in v.windows(2) {
                s += vlen(vsub(w[1], w[0]));
            }
            s
        };
        if v.len() >= 2 && l >= 2.5 {
            out.push(v);
        } else {
            skipped += l;
        }
    }
    (out, skipped)
}

/// [`build_roads`] + extra way clearers: polylines the given chains must keep
/// clear of (junctions trim kerbs/walks, free ends extend T-attachments)
/// WITHOUT being triangulated themselves — the live stroke preview passes the
/// committed roads here so every cursor move only rebuilds the stroke.
pub fn build_roads_ext(
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
    chains: &[Vec<[f32; 2]>],
    extra: &[Vec<[f32; 2]>],
    net: &Params,
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

    let mut main_pts = [[0.0f32, 0.0], [0.0, 0.0]];
    if net.main_t1 > net.main_t0 + 1.0 {
        main_pts = [[net.main_t0, 0.0], [net.main_t1, 0.0]];
    }
    let mut lane_pts: Vec<[[f32; 2]; 2]> = Vec::new();
    for &(uc, _cr, _cw) in &net.cross {
        lane_pts.push([[uc, 3.5], [uc, net.cross_half.max(3.6)]]);
        lane_pts.push([[uc, -net.cross_half.max(3.6)], [uc, -3.5]]);
    }
    // ---- junction events: every crossing between ways where at least one
    // side is a drawn chain (crossings, chain×chain, self-crossings, T-hits)
    struct Way<'a> {
        pts: &'a [[f32; 2]],
        hr: f32,
        hw: f32,
        chain: bool,
    }
    let mut ways_j: Vec<Way> = Vec::new();
    for ch in chains.iter() {
        if ch.len() >= 2 {
            ways_j.push(Way {
                pts: ch,
                hr,
                hw,
                chain: true,
            });
        }
    }
    for ch in extra {
        if ch.len() >= 2 {
            ways_j.push(Way {
                pts: ch,
                hr,
                hw,
                chain: false,
            });
        }
    }
    if net.main_t1 > net.main_t0 + 1.0 {
        ways_j.push(Way {
            pts: &main_pts,
            hr,
            hw,
            chain: false,
        });
    }
    for (k, lp) in lane_pts.iter().enumerate() {
        let lw = net.cross[k / 2];
        ways_j.push(Way {
            pts: lp,
            hr: lw.1,
            hw: lw.2,
            chain: false,
        });
    }
    let mut evs: Vec<JEv> = Vec::new();
    let mut ev_keys: Vec<(u64, [f32; 2])> = Vec::new();
    for i in 0..ways_j.len() {
        for j in i + 1..ways_j.len() {
            if !ways_j[i].chain && !ways_j[j].chain {
                continue;
            }
            let (wa, wb) = (&ways_j[i], &ways_j[j]);
            for s in 0..wa.pts.len().saturating_sub(1) {
                for t in 0..wb.pts.len().saturating_sub(1) {
                    if let Some((q, _ta, _tb)) =
                        seg_int(wa.pts[s], wa.pts[s + 1], wb.pts[t], wb.pts[t + 1])
                    {
                        let di = vunit(vsub(wa.pts[s + 1], wa.pts[s]));
                        let dj = vunit(vsub(wb.pts[t + 1], wb.pts[t]));
                        push_ev(
                            &mut evs,
                            &mut ev_keys,
                            ev_key(i, j),
                            q,
                            vec![
                                (di, wa.hr, wa.hw),
                                (vmul(di, -1.0), wa.hr, wa.hw),
                                (dj, wb.hr, wb.hw),
                                (vmul(dj, -1.0), wb.hr, wb.hw),
                            ],
                        );
                    }
                }
            }
        }
    }
    // self-crossings of a chain
    for i in 0..ways_j.len() {
        if !ways_j[i].chain {
            continue;
        }
        let wa = &ways_j[i];
        for s in 0..wa.pts.len().saturating_sub(1) {
            for t in (s + 2)..wa.pts.len().saturating_sub(1) {
                if let Some((q, _ta, _tb)) =
                    seg_int(wa.pts[s], wa.pts[s + 1], wa.pts[t], wa.pts[t + 1])
                {
                    let di = vunit(vsub(wa.pts[s + 1], wa.pts[s]));
                    let dj = vunit(vsub(wa.pts[t + 1], wa.pts[t]));
                    if vdot(di, dj).abs() > 0.995 {
                        continue;
                    }
                    push_ev(
                        &mut evs,
                        &mut ev_keys,
                        ev_key(i, i),
                        q,
                        vec![
                            (di, wa.hr, wa.hw),
                            (vmul(di, -1.0), wa.hr, wa.hw),
                            (dj, wa.hr, wa.hw),
                            (vmul(dj, -1.0), wa.hr, wa.hw),
                        ],
                    );
                }
            }
        }
    }
    // free ends landing on another way join as three-arm junctions
    for i in 0..ways_j.len() {
        if !ways_j[i].chain {
            continue;
        }
        let wa = &ways_j[i];
        let endn = wa.pts.len() - 1;
        for isp in [true, false] {
            let ep = wa.pts[if isp { 0 } else { endn }];
            for (j, wb) in ways_j.iter().enumerate() {
                if j == i {
                    continue;
                }
                let (np, d) = near_poly(wb.pts, ep);
                if d > 0.45 {
                    continue;
                }
                let dird = if isp {
                    vunit(vsub(wa.pts[1], wa.pts[0]))
                } else {
                    vunit(vsub(wa.pts[endn], wa.pts[endn - 1]))
                };
                let db = dir_at(wb.pts, np);
                push_ev(
                    &mut evs,
                    &mut ev_keys,
                    ev_key(i, j),
                    np,
                    vec![
                        (dird, wa.hr, wa.hw),
                        (db, wb.hr, wb.hw),
                        (vmul(db, -1.0), wb.hr, wb.hw),
                    ],
                );
            }
        }
    }
    for (ci, ch) in chains.iter().enumerate() {
        if ch.len() < 2 {
            continue;
        }
        // the ways this chain must keep clear of: every other chain, the
        // main street and the cross lanes (raw polylines + half-widths)
        let mut clearers: Vec<(&[[f32; 2]], f32, f32)> = Vec::new();
        for (oj, och) in chains.iter().enumerate() {
            if oj != ci && och.len() >= 2 {
                clearers.push((och, hr, hw));
            }
        }
        for och in extra {
            if och.len() >= 2 {
                clearers.push((och, hr, hw));
            }
        }
        if net.main_t1 > net.main_t0 + 1.0 {
            clearers.push((&main_pts, hr, hw));
        }
        for (k, lp) in lane_pts.iter().enumerate() {
            let lw = net.cross[k / 2];
            clearers.push((lp, lw.1, lw.2));
        }
        // where this chain crosses ITSELF the other pass must clear the
        // sidewalk bands exactly like another way would: a chain is not in
        // its own clearers, so collect its self-crossing points here
        let mut self_x: Vec<[f32; 2]> = Vec::new();
        for s in 0..ch.len().saturating_sub(1) {
            for t in (s + 2)..ch.len().saturating_sub(1) {
                if let Some((q, _ta, _tb)) =
                    seg_int(ch[s], ch[s + 1], ch[t], ch[t + 1])
                {
                    let di = vunit(vsub(ch[s + 1], ch[s]));
                    let dj = vunit(vsub(ch[t + 1], ch[t]));
                    if vdot(di, dj).abs() > 0.995 {
                        continue;
                    }
                    self_x.push(q);
                }
            }
        }
        // T-attachments: a free end sitting on another way extends the slab
        // across that roadway (the point is appended before smoothing)
        let mut raw: Vec<[f32; 2]> = ch.to_vec();
        {
            let endn = raw.len() - 1;
            let mut ext = [0.0f32; 2];
            for isp in [true, false] {
                let k = if isp { 0 } else { endn };
                let ep = raw[k];
                for &(opts, ohr, _ohw) in &clearers {
                    let (np, d) = near_poly(opts, ep);
                    if d > 0.45 {
                        continue;
                    }
                    let dird = if isp {
                        vmul(vunit(vsub(raw[1], raw[0])), -1.0)
                    } else {
                        vunit(vsub(raw[endn], raw[endn - 1]))
                    };
                    let db = dir_at(opts, np);
                    let sn = (dird[0] * db[1] - dird[1] * db[0]).abs().max(0.34);
                    let e = ((ohr + 0.3) / sn).clamp(2.0, 16.0);
                    if isp {
                        ext[0] = ext[0].max(e);
                    } else {
                        ext[1] = ext[1].max(e);
                    }
                }
            }
            if ext[0] > 0.05 {
                let d0 = vunit(vsub(raw[1], raw[0]));
                raw.insert(0, vsub(raw[0], vmul(d0, ext[0])));
            }
            if ext[1] > 0.05 {
                let n0 = raw.len();
                let d1 = vunit(vsub(raw[n0 - 1], raw[n0 - 2]));
                let e1 = vadd(raw[n0 - 1], vmul(d1, ext[1]));
                raw.push(e1);
            }
        }
        let (path, corners) = smooth_chain(&raw, 9.0);
        if path.len() < 2 {
            continue;
        }
        let stns = path_stations(&ctx, &path, hr, hw);
        if stns.len() < 2 {
            continue;
        }
        let total = stns[stns.len() - 1].s;
        // a sidewalk band (kerb + walk of one side) dies where it would run
        // into another way's corridor; the roadway slab never dies.
        // A free END of a way blocks its own mouth corridor beyond the end
        // (the extension slab that crosses the attached way) plus a short
        // backsplash; the mouth FLANKS are waived so the attached way's
        // sidewalk is not undercut to either side of a T-attachment.
        let blocked = |p: [f32; 2], m: f32| -> bool {
            clearers.iter().any(|&(opts, _ohr, ohw)| {
                let r = ohw + m;
                let (q, d) = near_poly(opts, p);
                if d >= r {
                    return false;
                }
                let n = opts.len();
                let behind = if n >= 2 && vlen(vsub(q, opts[0])) < 1e-3 {
                    let i0 = vunit(vsub(opts[1], opts[0]));
                    let lat = vdot(vsub(p, opts[0]), vperp(i0)).abs();
                    vdot(vsub(p, opts[0]), i0) < -1.6 && lat > 3.8
                } else if n >= 2 && vlen(vsub(q, opts[n - 1])) < 1e-3 {
                    let i0 = vunit(vsub(opts[n - 2], opts[n - 1]));
                    let lat = vdot(vsub(p, opts[n - 1]), vperp(i0)).abs();
                    vdot(vsub(p, opts[n - 1]), i0) < -1.6 && lat > 3.8
                } else {
                    false
                };
                !behind
            })
        };
        let band_hit = |s0: &St, s1: &St, x0: f32, x1: f32| -> bool {
            for (st, x) in [(s0, x0), (s0, x1), (s1, x0), (s1, x1)] {
                let q = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
                // the margin leaves room for the junction fillet arcs
                if blocked(q, 4.0) {
                    return true;
                }
                // ...or into a corridor of another pass of this SAME chain
                // (a chain is never in its own clearers, so its sidewalk
                // bands must be cut at its self-crossings explicitly)
                if self_x.iter().any(|&qx| vlen(vsub(qx, q)) < hw + 4.0) {
                    return true;
                }
            }
            false
        };
        // never redraw roadway that already exists: pairs running parallel
        // (|dot| > 0.9) within 2.4 m of another way, or of a stretch of this
        // same chain at least 6 m away along the ribbon (arc separation — the
        // neighbour pair 1.5 m back is the same ribbon, not a doubled one),
        // are skipped entirely
        let mut dup_prev: Vec<([f32; 2], [f32; 2], f32)> = Vec::new();
        for j in 0..stns.len() - 1 {
            let (a, e) = (&stns[j], &stns[j + 1]);
            let st3 = |st: &St, x: f32, z: f32| -> [f32; 3] {
                let q = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
                ctx.pt(q[0], q[1], z)
            };
            let mid = [(a.p[0] + e.p[0]) * 0.5, (a.p[1] + e.p[1]) * 0.5];
            let s_mid = (a.s + e.s) * 0.5;
            let pdir = vunit(vsub(e.p, a.p));
            // never blank the pavement right around a junction: the pocket
            // and fillet patches need the slab under them (a shallow-angle
            // crossing would otherwise have its centre eaten by the
            // parallel-duplicate guard)
            let near_junc = evs.iter().any(|ev| vlen(vsub(ev.q, mid)) < 9.0);
            let mut dup = false;
            for &(pp, pd, ps) in &dup_prev {
                if !near_junc
                    && vlen(vsub(pp, mid)) < 2.4
                    && vdot(pdir, pd).abs() > 0.9
                    && (s_mid - ps).abs() > 6.0
                {
                    dup = true;
                    break;
                }
            }
            if !dup {
                for &(opts, _ohr, _ohw) in &clearers {
                    let (np, d) = near_poly(opts, mid);
                    if !near_junc && d < 2.4 && vdot(pdir, dir_at(opts, np)).abs() > 0.9 {
                        dup = true;
                        break;
                    }
                }
            }
            if dup {
                continue;
            }
            dup_prev.push((mid, pdir, s_mid));
            let kp = !band_hit(a, e, hr, hw);
            let kn = !band_hit(a, e, -hw, -hr);
            // roadway slab
            quad_up(
                &mut b,
                M_ASPHALT,
                [st3(a, -hr, a.h0), st3(e, -hr, e.h0), st3(e, hr, e.h0), st3(a, hr, a.h0)],
                [[a.s / 4.0, 0.0], [e.s / 4.0, 0.0], [e.s / 4.0, 1.0], [a.s / 4.0, 1.0]],
            );
            // kerb tops (they stop at every junction)
            if kp {
                quad_up(
                    &mut b,
                    M_KERB,
                    [st3(a, hr, a.h0 + KERB_H), st3(e, hr, e.h0 + KERB_H), st3(e, hr + KERB_W, e.h0 + KERB_H), st3(a, hr + KERB_W, a.h0 + KERB_H)],
                    [[a.s / 2.0, 0.0], [e.s / 2.0, 0.0], [e.s / 2.0, 1.0], [a.s / 2.0, 1.0]],
                );
            }
            if kn {
                quad_up(
                    &mut b,
                    M_KERB,
                    [st3(a, -hr - KERB_W, a.h0 + KERB_H), st3(e, -hr - KERB_W, e.h0 + KERB_H), st3(e, -hr, e.h0 + KERB_H), st3(a, -hr, a.h0 + KERB_H)],
                    [[a.s / 2.0, 0.0], [e.s / 2.0, 0.0], [e.s / 2.0, 1.0], [a.s / 2.0, 1.0]],
                );
            }
            // kerb faces toward the roadway
            for sx in [hr, -hr] {
                if (sx > 0.0 && !kp) || (sx < 0.0 && !kn) {
                    continue;
                }
                let p = [
                    st3(a, sx, a.h0 + KERB_H),
                    st3(e, sx, e.h0 + KERB_H),
                    st3(e, sx, e.hn - 0.5),
                    st3(a, sx, a.hn - 0.5),
                ];
                let uv = [[a.s / 2.0, 1.0], [e.s / 2.0, 1.0], [e.s / 2.0, 0.0], [a.s / 2.0, 0.0]];
                if sx < 0.0 {
                    b.quad(M_KERB, [p[1], p[0], p[3], p[2]], [uv[1], uv[0], uv[3], uv[2]]);
                } else {
                    b.quad(M_KERB, p, uv);
                }
            }
            // sidewalks, flush with the kerb tops
            if kp {
                quad_up(
                    &mut b,
                    M_WALK,
                    [st3(a, hr + KERB_W, a.h0 + KERB_H), st3(e, hr + KERB_W, e.h0 + KERB_H), st3(e, hw, e.h0 + KERB_H), st3(a, hw, a.h0 + KERB_H)],
                    [[a.s / 2.0, 0.0], [e.s / 2.0, 0.0], [e.s / 2.0, 1.0], [a.s / 2.0, 1.0]],
                );
            }
            if kn {
                quad_up(
                    &mut b,
                    M_WALK,
                    [st3(a, -hw, a.h0 + KERB_H), st3(e, -hw, e.h0 + KERB_H), st3(e, -hr - KERB_W, e.h0 + KERB_H), st3(a, -hr - KERB_W, a.h0 + KERB_H)],
                    [[a.s / 2.0, 0.0], [e.s / 2.0, 0.0], [e.s / 2.0, 1.0], [a.s / 2.0, 1.0]],
                );
            }
            // outer edges drop to their own local ground
            for sx in [hw, -hw] {
                if (sx > 0.0 && !kp) || (sx < 0.0 && !kn) {
                    continue;
                }
                let gb = |s: &St| {
                    let q = [s.p[0] + s.n[0] * sx, s.p[1] + s.n[1] * sx];
                    ctx.hh(q[0], q[1]) - 0.5
                };
                let p = [
                    st3(a, sx, a.h0 + KERB_H),
                    st3(e, sx, e.h0 + KERB_H),
                    st3(e, sx, gb(e)),
                    st3(a, sx, gb(a)),
                ];
                let uv = [[a.s / 2.0, 0.9], [e.s / 2.0, 0.9], [e.s / 2.0, 0.0], [a.s / 2.0, 0.0]];
                if sx < 0.0 {
                    b.quad(M_WALK, [p[1], p[0], p[3], p[2]], [uv[1], uv[0], uv[3], uv[2]]);
                } else {
                    b.quad(M_WALK, p, uv);
                }
            }
        }
        // dashed centre line follows the smoothed line
        let p3 = |st: &St, x: f32| -> [f32; 3] {
            let q = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
            ctx.pt(q[0], q[1], st.h0 + 0.055)
        };
        let mut sd = 3.0f32;
        while sd + 3.0 < total - 2.0 {
            let steps = 3;
            for k in 0..steps {
                let s0 = sd + 3.0 * k as f32 / steps as f32;
                let s1 = sd + 3.0 * (k + 1) as f32 / steps as f32;
                let a = st_at(&stns, s0);
                let e = st_at(&stns, s1);
                let mid = [(a.p[0] + e.p[0]) * 0.5, (a.p[1] + e.p[1]) * 0.5];
                if blocked(mid, 2.0) {
                    continue;
                }
                if self_x.iter().any(|&qx| vlen(vsub(qx, mid)) < hw + 2.0) {
                    continue;
                }
                quad_up(
                    &mut b,
                    M_PAINT,
                    [p3(&a, -0.13), p3(&e, -0.13), p3(&e, 0.13), p3(&a, 0.13)],
                    [[a.s / 2.0, 0.0], [e.s / 2.0, 0.0], [e.s / 2.0, 0.2], [a.s / 2.0, 0.2]],
                );
            }
            sd += 6.0;
        }
        // street lamps / trees / bushes / signs along this road
        let get = |s: f32, x: f32| -> ([f32; 2], f32, [f32; 2], [f32; 2]) {
            let st = st_at(&stns, s);
            let pos = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
            let along = [-st.n[1], st.n[0]];
            (pos, st.h0 + KERB_H, along, st.n)
        };
        street_furniture(
            &mut b,
            &ctx,
            0x5700 + ci as u32,
            hr,
            hw,
            total,
            &get,
            &corners,
            &|q: [f32; 2]| blocked(q, 4.8) || self_x.iter().any(|&qx| vlen(vsub(qx, q)) < hw + 4.8),
        );
    }
    // junction patches: pocket asphalt, fillet kerbs, sidewalk fills, zebra
    // crossings with stop lines and traffic lights on every crossing
    for ev in &evs {
        let others: Vec<[f32; 2]> = evs
            .iter()
            .filter(|o| !std::ptr::eq(*o, ev))
            .map(|o| o.q)
            .collect();
        junction_patch(&mut b, &ctx, ev, &others);
    }
    (b.finish(13), textures())
}


/// Traffic-light lenses: three vertical zones (red / amber / green discs on
/// a near-black housing face); lens quads sample the centre of their zone.
fn tex_lights() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(64 * 192 * 4);
    for y in 0..192u32 {
        for x in 0..64u32 {
            let zone = y / 64; // 0 red (top), 1 amber, 2 green
            let cy = (zone * 64) as i32 + 32;
            let dx = x as i32 - 32;
            let dy = y as i32 - cy;
            let r2 = dx * dx + dy * dy;
            let bg = 24 + (hash2(x + 900, y + 901) % 7) as i32 - 3;
            if r2 < 26 * 26 {
                let t = (r2 as f32 / (26.0 * 26.0)).sqrt();
                let bright = (100.0 - t * 55.0) as i32;
                let (base, hi) = match zone {
                    0 => ((196, 44, 36), (255, 96, 78)),
                    1 => ((216, 156, 34), (255, 208, 96)),
                    _ => ((52, 168, 84), (128, 235, 140)),
                };
                push(
                    &mut v,
                    base.0 * bright / 100 + hi.0 * (100 - bright) / 100,
                    base.1 * bright / 100 + hi.1 * (100 - bright) / 100,
                    base.2 * bright / 100 + hi.2 * (100 - bright) / 100,
                );
            } else if r2 < 30 * 30 {
                push(&mut v, 12, 13, 15);
            } else {
                push(&mut v, bg, bg + 1, bg + 3);
            }
        }
    }
    (64, 192, v)
}

/// Cast iron: near-black with a cold sheen and fine grain.
fn tex_iron() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 31, y + 77) % 15) as i32 - 7;
            let sheen = if hash2(x / 32, y) % 23 == 0 { 10 } else { 0 };
            let base = 48 + n + sheen;
            push(&mut v, base, base, base + 6);
        }
    }
    (256, 256, v)
}

/// Milk glass of the lantern panes: warm, faint swirl.
fn tex_glass() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 400, y + 21) % 17) as i32 - 8;
            push(&mut v, 246 + n, 232 + n, 196 + n / 2);
        }
    }
    (256, 256, v)
}

/// Plane-tree bark: warm grey patches over brown.
fn tex_bark() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 700, y + 120) % 21) as i32 - 10;
            let patch = hash2(x / 24, y / 34) % 5 == 0;
            let (r, g, b) = if patch { (138, 126, 108) } else { (96, 76, 58) };
            push(&mut v, r + n, g + n, b + n);
        }
    }
    (256, 256, v)
}

/// Summer foliage: mid green with denser clumps.
fn tex_leaf() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let clump = (hash2(x / 18, y / 18) % 31) as i32 - 15;
            let n = (hash2(x + 90, y + 90) % 13) as i32 - 6;
            push(&mut v, 68 + clump + n, 104 + clump + n, 52 + (clump / 2) + n);
        }
    }
    (256, 256, v)
}

/// Lighter foliage for young shoots.
fn tex_leaf2() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let clump = (hash2(x / 15, y / 15) % 27) as i32 - 13;
            let n = (hash2(x + 44, y + 61) % 11) as i32 - 5;
            push(&mut v, 92 + clump + n, 128 + clump + n, 64 + (clump / 2) + n);
        }
    }
    (256, 256, v)
}

/// Enamel sign blue.
fn tex_sign() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 55, y + 205) % 11) as i32 - 5;
            push(&mut v, 30 + n, 66 + n, 140 + n);
        }
    }
    (256, 256, v)
}

/// Painted wood (bench slats): fir green with grain lines.
fn tex_wood() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 12, y + 340) % 9) as i32 - 4;
            let grain = if (x + (hash2(x / 8, y) % 5) as u32) % 32 < 2 { -14 } else { 0 };
            push(&mut v, 60 + n + grain, 96 + n + grain, 66 + n + grain);
        }
    }
    (256, 256, v)
}

/// Sign enamel white.
fn tex_white() -> (u32, u32, Vec<u8>) {
    let mut v = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let n = (hash2(x + 800, y + 811) % 9) as i32 - 4;
            push(&mut v, 236 + n, 236 + n, 230 + n);
        }
    }
    (256, 256, v)
}

pub fn textures() -> Vec<(u32, u32, Vec<u8>)> {
    // the 12 procedural road textures never change: generate them once per
    // process and hand out clones (~3 MB memcpy) — the editor rebuilds the
    // roads mesh on every cursor move and used to pay the full noise
    // generation (~80 ms) each time
    static CACHE: std::sync::OnceLock<Vec<(u32, u32, Vec<u8>)>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            vec![
                tex_asphalt(),
        tex_kerb(),
        tex_walk(),
        tex_paint(),
        tex_iron(),
        tex_glass(),
        tex_bark(),
        tex_leaf(),
        tex_leaf2(),
        tex_sign(),
                tex_wood(),
                tex_white(),
                tex_lights(),
            ]
        })
        .clone()
}
