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

    (b.finish(12), textures())
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

/// Street tree: trunk plus two or three jittered crowns (three habits).
fn st_tree(b: &mut Builder, ctx: &Ctx, p: [f32; 2], z: f32, s: f32, v: u32) {
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
                        let face = if side > 0.0 { vmul(outw, -1.0) } else { outw };
                        st_bench(b, ctx, bp, bz, face);
                    }
                }
                1 => st_tree(b, ctx, p, z, 0.80 + ((j >> 20) & 7) as f32 * 0.035, variant),
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

    for (ci, ch) in chains.iter().enumerate() {
        if ch.len() < 2 {
            continue;
        }
        // the ways this chain must keep clear of: every other chain, the
        // main street and the cross lanes (raw polylines + half-widths)
        let mut main_pts = [[0.0f32, 0.0], [0.0, 0.0]];
        if net.main_t1 > net.main_t0 + 1.0 {
            main_pts = [[net.main_t0, 0.0], [net.main_t1, 0.0]];
        }
        let mut lane_pts: Vec<[[f32; 2]; 2]> = Vec::new();
        for &(uc, _cr, _cw) in &net.cross {
            lane_pts.push([[uc, 3.5], [uc, net.cross_half.max(3.6)]]);
            lane_pts.push([[uc, -net.cross_half.max(3.6)], [uc, -3.5]]);
        }
        let mut clearers: Vec<(&[[f32; 2]], f32, f32)> = Vec::new();
        for (oj, och) in chains.iter().enumerate() {
            if oj != ci && och.len() >= 2 {
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
        // into another way's corridor; the roadway slab never dies
        let blocked = |p: [f32; 2], m: f32| -> bool {
            clearers
                .iter()
                .any(|&(opts, _ohr, ohw)| near_poly(opts, p).1 < ohw + m)
        };
        let band_hit = |s0: &St, s1: &St, x0: f32, x1: f32| -> bool {
            for (st, x) in [(s0, x0), (s0, x1), (s1, x0), (s1, x1)] {
                let q = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
                if blocked(q, 0.0) {
                    return true;
                }
            }
            false
        };
        for j in 0..stns.len() - 1 {
            let (a, e) = (&stns[j], &stns[j + 1]);
            let st3 = |st: &St, x: f32, z: f32| -> [f32; 3] {
                let q = [st.p[0] + st.n[0] * x, st.p[1] + st.n[1] * x];
                ctx.pt(q[0], q[1], z)
            };
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
            &|q: [f32; 2]| blocked(q, 0.6),
        );
    }
    (b.finish(12), textures())
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

fn textures() -> Vec<(u32, u32, Vec<u8>)> {
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
    ]
}
