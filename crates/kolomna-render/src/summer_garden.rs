//! Procedurally generated Summer Garden (Летний сад) in the Roman manner:
//! a fence of square granite columns and an iron grille with gilded spears
//! (after Fельтен/Шарлемань) around an irregular polygon, with gates, a grid
//! of gravel alleys clipped to the shape, bosquets with clipped hedges,
//! fountains, marble busts, statues, big vases, benches, lanterns and urns —
//! everything conforming to the smoothed terrain.
//!
//! Plan-frame coordinates: `u` runs along the main street, `v` across it —
//! the same frame the village and the streets use: `world = anchor*16 +
//! eu*u + nvec*v`, heights in metres above `mn`.

use crate::model::ModelMesh;
use crate::procgen::{Builder, Rng};

pub const M_GRANITE: u32 = 0;
pub const M_MARBLE: u32 = 1;
pub const M_GRAVEL: u32 = 2;
pub const M_WATER: u32 = 3;
pub const M_JET: u32 = 4;
pub const M_LEAF: u32 = 5;
pub const M_HEDGE: u32 = 6;
pub const M_BARK: u32 = 7;
pub const M_IRON: u32 = 8;
pub const M_BENCH: u32 = 9;
pub const M_LAMP: u32 = 10;
pub const M_GOLD: u32 = 11;
pub const M_GRILLE: u32 = 12;
pub const M_GATE: u32 = 13;
pub const M_SHEET: u32 = 14;
const N_MATS: u32 = 15;

const SINK: f32 = 0.7; // buried depth of bases so slopes never show gaps
const PAR_H: f32 = 0.85; // granite parapet above the local ground
const PAR_W: f32 = 0.45; // parapet thickness
const COL_SPAN: f32 = 4.6; // fence column spacing
const COL_H: f32 = 2.35; // granite column above the parapet
const GRILLE_H: f32 = 2.05; // iron grille above the parapet
const RING_OFF: f32 = 4.0; // perimeter alley: distance from the fence
const RING_HW: f32 = 1.35;
const PATH_HW: f32 = 1.5; // cross alley half-width
const ALLEY_HW: f32 = 2.2; // main alley half-width

pub struct Params {
    /// park centre in plan coords
    pub centre: [f32; 2],
    pub ru: f32,
    pub rv: f32,
    pub seed: u64,
}

struct Ctx<'a> {
    h_at: &'a dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
}

impl Ctx<'_> {
    fn wm(&self, u: f32, v: f32) -> [f32; 2] {
        [
            self.eu[0] * u + self.nvec[0] * v,
            self.eu[1] * u + self.nvec[1] * v,
        ]
    }
    fn hh(&self, u: f32, v: f32) -> f32 {
        let w = self.wm(u, v);
        (self.h_at)(
            self.anchor[0] + w[0] / 16.0,
            self.anchor[1] + w[1] / 16.0,
        )
    }
    fn pt(&self, u: f32, v: f32, z: f32) -> [f32; 3] {
        let w = self.wm(u, v);
        [
            self.anchor[0] * 16.0 + w[0],
            z - self.mn,
            self.anchor[1] * 16.0 + w[1],
        ]
    }
}

// ------------------------------------------------------------ 2d helpers ----

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}
fn mul(a: [f32; 2], k: f32) -> [f32; 2] {
    [a[0] * k, a[1] * k]
}
fn perp(d: [f32; 2]) -> [f32; 2] {
    [-d[1], d[0]]
}
fn unit(d: [f32; 2]) -> [f32; 2] {
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-5);
    [d[0] / l, d[1] / l]
}

fn point_in_poly(poly: &[[f32; 2]], p: [f32; 2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Offset every edge inward by `m` (polygon wound CCW) and intersect the
/// neighbouring offset lines — the shrunk outline used for inside checks
/// and for the perimeter alley.
fn shrink(poly: &[[f32; 2]], m: f32) -> Vec<[f32; 2]> {
    let n = poly.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let p0 = poly[(k + n - 1) % n];
        let p1 = poly[k];
        let p2 = poly[(k + 1) % n];
        let d0 = unit(sub(p1, p0));
        let d1 = unit(sub(p2, p1));
        let q0 = add(p1, mul(perp(d0), m));
        let q1 = add(p1, mul(perp(d1), m));
        out.push(line_int(q0, d0, q1, d1));
    }
    out
}

fn line_int(p0: [f32; 2], d0: [f32; 2], p1: [f32; 2], d1: [f32; 2]) -> [f32; 2] {
    let den = d0[0] * d1[1] - d0[1] * d1[0];
    if den.abs() < 1e-4 {
        return p0;
    }
    let t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den;
    [p0[0] + d0[0] * t, p0[1] + d0[1] * t]
}

/// Intervals of `t ∈ [t0, t1]` inside the polygon along the line u = c (when
/// `axis_v`) or v = c (when !`axis_v`).
fn line_inside(poly: &[[f32; 2]], axis_v: bool, c: f32, t0: f32, t1: f32) -> Vec<(f32, f32)> {
    let n = poly.len();
    let mut ts: Vec<f32> = Vec::new();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let (ca, cb) = if axis_v { (a[1], b[1]) } else { (a[0], b[0]) };
        if ca == cb {
            continue;
        }
        if (ca - c) * (cb - c) <= 0.0 {
            let f = (c - ca) / (cb - ca);
            let t = if axis_v {
                a[0] + (b[0] - a[0]) * f
            } else {
                a[1] + (b[1] - a[1]) * f
            };
            ts.push(t);
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < ts.len() {
        let (a, b) = (ts[i].max(t0), ts[i + 1].min(t1));
        if b > a + 0.5 {
            let mid = (a + b) * 0.5;
            let p = if axis_v { [mid, c] } else { [c, mid] };
            if point_in_poly(poly, p) {
                out.push((a, b));
            }
        }
        i += 2;
    }
    out
}

// -------------------------------------------------------- mesh utilities ----

/// A gravel strip between plan points p0->p1, half-width `hw`, conforming
/// to the terrain: top slab at +0.12 with skirts and end caps sunk 0.6.
fn strip(b: &mut Builder, ctx: &Ctx, p0: [f32; 2], p1: [f32; 2], hw: f32) {
    let d = sub(p1, p0);
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if l < 0.4 {
        return;
    }
    let dir = unit(d);
    let n = perp(dir);
    let steps = (l / 2.0).ceil().max(1.0) as usize;
    let pt = |t: f32, s: f32, dz: f32| -> [f32; 3] {
        let p = add(p0, add(mul(dir, t), mul(n, s)));
        ctx.pt(p[0], p[1], ctx.hh(p[0], p[1]) + 0.12 + dz)
    };
    for k in 0..steps {
        let t0 = k as f32 * l / steps as f32;
        let t1 = (k + 1) as f32 * l / steps as f32;
        let ul = t0 / 2.5;
        let uh = t1 / 2.5;
        // top
        b.quad(
            M_GRAVEL,
            [pt(t0, -hw, 0.0), pt(t0, hw, 0.0), pt(t1, hw, 0.0), pt(t1, -hw, 0.0)],
            [[ul, 0.0], [ul, 2.0 * hw / 2.5], [uh, 2.0 * hw / 2.5], [uh, 0.0]],
        );
        // side skirts
        b.quad(
            M_GRAVEL,
            [pt(t0, -hw, 0.0), pt(t1, -hw, 0.0), pt(t1, -hw, -0.6), pt(t0, -hw, -0.6)],
            [[ul, 0.0], [uh, 0.0], [uh, 0.24], [ul, 0.24]],
        );
        b.quad(
            M_GRAVEL,
            [pt(t1, hw, 0.0), pt(t0, hw, 0.0), pt(t0, hw, -0.6), pt(t1, hw, -0.6)],
            [[uh, 0.0], [ul, 0.0], [ul, 0.24], [uh, 0.24]],
        );
    }
    // end caps
    b.quad(
        M_GRAVEL,
        [pt(0.0, -hw, 0.0), pt(0.0, hw, 0.0), pt(0.0, hw, -0.6), pt(0.0, -hw, -0.6)],
        [[0.0, 0.0], [2.0 * hw / 2.5, 0.0], [2.0 * hw / 2.5, 0.24], [0.0, 0.24]],
    );
    b.quad(
        M_GRAVEL,
        [pt(l, hw, 0.0), pt(l, -hw, 0.0), pt(l, -hw, -0.6), pt(l, hw, -0.6)],
        [[0.0, 0.0], [2.0 * hw / 2.5, 0.0], [2.0 * hw / 2.5, 0.24], [0.0, 0.24]],
    );
}

/// Oriented box in plan: axis a->e, width `wid`, from z0 to z1 (absolute
/// altitudes), all faces + top. UVs in metres of `uvm`.
#[allow(clippy::too_many_arguments)]
fn obb(b: &mut Builder, ctx: &Ctx, a: [f32; 2], e: [f32; 2], wid: f32, z0: f32, z1: f32, mat: u32, uvm: f32) {
    let d = sub(e, a);
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(0.02);
    let dir = unit(d);
    let n = perp(dir);
    let w = wid * 0.5;
    let p = |t: f32, s: f32, z: f32| -> [f32; 3] {
        let q = add(a, add(mul(dir, t), mul(n, s)));
        ctx.pt(q[0], q[1], z)
    };
    let ul = l / uvm;
    let uw = wid / uvm;
    let vz0 = 0.0;
    let vz1 = (z1 - z0) / uvm;
    // +n side
    b.quad(
        mat,
        [p(0.0, w, z0), p(l, w, z0), p(l, w, z1), p(0.0, w, z1)],
        [[0.0, vz0], [ul, vz0], [ul, vz1], [0.0, vz1]],
    );
    // -n side
    b.quad(
        mat,
        [p(l, -w, z0), p(0.0, -w, z0), p(0.0, -w, z1), p(l, -w, z1)],
        [[0.0, vz0], [ul, vz0], [ul, vz1], [0.0, vz1]],
    );
    // start cap (faces -d)
    b.quad(
        mat,
        [p(0.0, -w, z0), p(0.0, w, z0), p(0.0, w, z1), p(0.0, -w, z1)],
        [[0.0, vz0], [uw, vz0], [uw, vz1], [0.0, vz1]],
    );
    // end cap (faces +d)
    b.quad(
        mat,
        [p(l, w, z0), p(l, -w, z0), p(l, -w, z1), p(l, w, z1)],
        [[0.0, vz0], [uw, vz0], [uw, vz1], [0.0, vz1]],
    );
    // top (up order)
    b.quad(
        mat,
        [p(0.0, -w, z1), p(0.0, w, z1), p(l, w, z1), p(l, -w, z1)],
        [[0.0, 0.0], [uw, 0.0], [uw, ul], [0.0, ul]],
    );
}

/// Tapered cylinder (vertical) at plan point p: r0 at z0, r1 at z1, plus a
/// top cap when r1 > 0.03. CCW seen from above.
#[allow(clippy::too_many_arguments)]
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
        b.quad(
            mat,
            [ring(r0, z0, j), ring(r1, z1, j), ring(r1, z1, j1), ring(r0, z0, j1)],
            [[u0, 0.0], [u0, hz], [u1, hz], [u1, 0.0]],
        );
    }
    if r1 > 0.03 {
        let c = ctx.pt(p[0], p[1], z1);
        for j in 0..sides {
            let j1 = (j + 1) % sides;
            b.tri(
                mat,
                [c, ring(r1, z1, j), ring(r1, z1, j1)],
                [
                    [r1 / uvm, r1 / uvm],
                    [0.0, 0.0],
                    [std::f32::consts::TAU * r1 / uvm, 0.0],
                ],
            );
        }
    }
}

/// A tapered cylinder whose vertices carry an animation channel: weight 0
/// at the bottom ring, ~1 at the top (`base` 2.0 = jet, 3.0 = falling
/// sheet), so the shader bends the column more towards its tip. The top
/// cap inherits the top weight.
#[allow(clippy::too_many_arguments)]
fn jet_cyl(b: &mut Builder, ctx: &Ctx, p: [f32; 2], r0: f32, r1: f32, z0: f32, z1: f32, sides: usize, mat: u32, uvm: f32, base: f32) {
    let ring = |r: f32, z: f32, j: usize| -> [f32; 3] {
        let a = j as f32 / sides as f32 * std::f32::consts::TAU;
        let q = [p[0] + a.cos() * r, p[1] + a.sin() * r];
        ctx.pt(q[0], q[1], z)
    };
    let w_top = 0.95f32;
    for j in 0..sides {
        let j1 = (j + 1) % sides;
        let rr = r0.max(r1);
        let u0 = j as f32 / sides as f32 * (std::f32::consts::TAU * rr / uvm);
        let u1 = (j + 1) as f32 / sides as f32 * (std::f32::consts::TAU * rr / uvm);
        let hz = (z1 - z0) / uvm;
        let an = [base, base + w_top, base + w_top, base];
        b.quad_anim(
            mat,
            [ring(r0, z0, j), ring(r1, z1, j), ring(r1, z1, j1), ring(r0, z0, j1)],
            [[u0, 0.0], [u0, hz], [u1, hz], [u1, 0.0]],
            an,
        );
    }
    if r1 > 0.03 {
        let c = ctx.pt(p[0], p[1], z1);
        b.anim(base + w_top);
        for j in 0..sides {
            let j1 = (j + 1) % sides;
            b.tri(
                mat,
                [c, ring(r1, z1, j), ring(r1, z1, j1)],
                [
                    [r1 / uvm, r1 / uvm],
                    [0.0, 0.0],
                    [std::f32::consts::TAU * r1 / uvm, 0.0],
                ],
            );
        }
        b.anim(0.0);
    }
}

/// Churned water where a jet or a sheet lands: a flat shimmering disc.
fn splash(b: &mut Builder, ctx: &Ctx, p: [f32; 2], r: f32, z: f32) {
    b.anim(1.6);
    let c = ctx.pt(p[0], p[1], z);
    let rim = |j: usize| -> [f32; 3] {
        let a = j as f32 / 9.0 * std::f32::consts::TAU;
        ctx.pt(p[0] + a.cos() * r, p[1] + a.sin() * r, z)
    };
    for j in 0..9 {
        let j1 = (j + 1) % 9;
        b.tri(M_JET, [c, rim(j), rim(j1)], [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]);
    }
    b.anim(0.0);
}

/// A leafy blob (tree crown): 4 latitude rings + poles, jittered.
fn blob(b: &mut Builder, ctx: &Ctx, c: [f32; 3], r: f32, mat: u32, uvm: f32, rng: &mut Rng) {
    let lats = [-1.0f32, -0.33, 0.33, 1.0];
    let lons = 7;
    let ring = |lat: f32, j: usize, rng: &mut Rng| -> [f32; 3] {
        let a = j as f32 / lons as f32 * std::f32::consts::TAU;
        let rr = r * lat.cos() * rng.range(0.9, 1.1);
        [
            c[0] + a.cos() * rr,
            c[1] + lat.sin() * r * rng.range(0.92, 1.05),
            c[2] + a.sin() * rr,
        ]
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
            b.quad(
                mat,
                [a[j], e[j], e[j1], a[j1]],
                [uv(j, k), uv(j, k + 1), uv(j1, k + 2), uv(j, k + 2)],
            );
        }
    }
    // poles
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

// ------------------------------------------------------------ furniture ----

fn tree(b: &mut Builder, ctx: &Ctx, u: f32, v: f32, s: f32, rng: &mut Rng) {
    let h = ctx.hh(u, v);
    cyl(b, ctx, [u, v], 0.26 * s, 0.17 * s, h - 0.2, h + 2.3 * s, 7, M_BARK, 1.6);
    blob(b, ctx, ctx.pt(u, v, h + 3.1 * s), 1.55 * s, M_LEAF, 3.0, rng);
    let dx = rng.range(-0.5, 0.5) * s;
    let dz = rng.range(-0.5, 0.5) * s;
    blob(b, ctx, ctx.pt(u + dx, v + dz, h + 4.15 * s), 1.15 * s, M_LEAF, 3.0, rng);
}

fn hedge(b: &mut Builder, ctx: &Ctx, a: [f32; 2], e: [f32; 2], hgt: f32) {
    let l = {
        let d = sub(e, a);
        (d[0] * d[0] + d[1] * d[1]).sqrt()
    };
    if l < 0.5 {
        return;
    }
    let hm = hgt * 0.5;
    let h0 = ctx.hh(a[0], a[1]).min(ctx.hh(e[0], e[1]));
    let h1 = ctx.hh(a[0], a[1]).max(ctx.hh(e[0], e[1])) + hgt;
    // block with top and sides, slight taper ignored
    obb(b, ctx, a, e, 0.55, h0 - 0.3, h1, M_HEDGE, 1.4);
    // rounded crest: a thin top box overlapping
    let _ = hm;
}

fn bench(b: &mut Builder, ctx: &Ctx, p: [f32; 2], face: [f32; 2]) {
    let t = perp(face);
    let h = ctx.hh(p[0], p[1]);
    let q = |t_off: f32, f_off: f32| -> [f32; 2] { add(p, add(mul(t, t_off), mul(face, f_off))) };
    // legs (iron)
    for s in [-0.82f32, 0.82] {
        obb(b, ctx, q(s - 0.03, -0.24), q(s + 0.03, 0.24), 0.5, h - 0.2, h + 0.44, M_IRON, 0.8);
        // back posts
        obb(b, ctx, q(s - 0.03, 0.20), q(s + 0.03, 0.26), 0.1, h + 0.44, h + 1.0, M_IRON, 0.8);
    }
    // seat slats (painted wood)
    for f in [-0.19f32, -0.05, 0.09] {
        obb(b, ctx, q(-0.9, f - 0.055), q(0.9, f + 0.055), 0.11, h + 0.44, h + 0.49, M_BENCH, 1.2);
    }
    // back slats
    for (z0, z1) in [(0.68f32, 0.78f32), (0.86, 0.96)] {
        obb(b, ctx, q(-0.9, 0.235), q(0.9, 0.295), 0.09, h + z0, h + z1, M_BENCH, 1.2);
    }
}

fn urn(b: &mut Builder, ctx: &Ctx, p: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    obb(b, ctx, [p[0] - 0.14, p[1]], [p[0] + 0.14, p[1]], 0.28, h - 0.15, h + 0.30, M_IRON, 0.8);
    cyl(b, ctx, p, 0.26, 0.30, h + 0.30, h + 0.34, 8, M_IRON, 0.8);
    cyl(b, ctx, p, 0.30, 0.21, h + 0.34, h + 0.66, 8, M_IRON, 0.8);
    cyl(b, ctx, p, 0.24, 0.26, h + 0.66, h + 0.76, 8, M_IRON, 0.8);
}

fn lantern(b: &mut Builder, ctx: &Ctx, p: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    obb(b, ctx, [p[0] - 0.19, p[1]], [p[0] + 0.19, p[1]], 0.38, h - 0.25, h + 0.42, M_GRANITE, 0.9);
    cyl(b, ctx, p, 0.10, 0.07, h + 0.42, h + 2.45, 8, M_IRON, 1.0);
    cyl(b, ctx, p, 0.13, 0.11, h + 2.45, h + 2.52, 8, M_IRON, 1.0);
    // glazed body
    let g = |j: usize| -> [f32; 3] {
        let a = j as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let q = [p[0] + a.cos() * 0.17, p[1] + a.sin() * 0.17];
        [q[0], q[1], 0.0] // filled below
    };
    let _ = g;
    for j in 0..4 {
        let a0 = j as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let a1 = (j + 1) as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let q0 = [p[0] + a0.cos() * 0.17, p[1] + a0.sin() * 0.17];
        let q1 = [p[0] + a1.cos() * 0.17, p[1] + a1.sin() * 0.17];
        b.quad(
            M_LAMP,
            [
                ctx.pt(q0[0], q0[1], h + 2.52),
                ctx.pt(q1[0], q1[1], h + 2.52),
                ctx.pt(q1[0], q1[1], h + 2.95),
                ctx.pt(q0[0], q0[1], h + 2.95),
            ],
            [[0.0, 0.0], [0.34, 0.0], [0.34, 0.43], [0.0, 0.43]],
        );
    }
    // roof + finial
    cyl(b, ctx, p, 0.24, 0.05, h + 2.95, h + 3.16, 4, M_IRON, 1.0);
    blob(b, ctx, ctx.pt(p[0], p[1], h + 3.2), 0.06, M_IRON, 0.5, &mut Rng::new(7));
}

fn bust(b: &mut Builder, ctx: &Ctx, p: [f32; 2], face: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    let t = perp(face);
    let q = |to: f32, fo: f32| -> [f32; 2] { add(p, add(mul(t, to), mul(face, fo))) };
    obb(b, ctx, q(-0.26, -0.26), q(0.26, 0.26), 0.52, h - 0.25, h + 1.15, M_GRANITE, 1.1);
    obb(b, ctx, q(-0.30, -0.30), q(0.30, 0.30), 0.60, h + 1.15, h + 1.27, M_GRANITE, 1.1);
    obb(b, ctx, q(-0.13, -0.13), q(0.13, 0.13), 0.26, h + 1.27, h + 1.35, M_MARBLE, 0.9);
    // chest + head
    obb(b, ctx, q(-0.25, -0.15), q(0.25, 0.15), 0.30, h + 1.35, h + 1.79, M_MARBLE, 0.9);
    cyl(b, ctx, [p[0], p[1]], 0.075, 0.075, h + 1.79, h + 1.87, 6, M_MARBLE, 0.6);
    let hd = ctx.pt(p[0], p[1], h + 1.99);
    blob(b, ctx, hd, 0.125, M_MARBLE, 0.6, &mut Rng::new(3));
}

fn statue(b: &mut Builder, ctx: &Ctx, p: [f32; 2], face: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    let t = perp(face);
    let q = |to: f32, fo: f32| -> [f32; 2] { add(p, add(mul(t, to), mul(face, fo))) };
    obb(b, ctx, q(-0.33, -0.33), q(0.33, 0.33), 0.66, h - 0.25, h + 0.85, M_GRANITE, 1.1);
    obb(b, ctx, q(-0.36, -0.36), q(0.36, 0.36), 0.72, h + 0.85, h + 0.97, M_GRANITE, 1.1);
    obb(b, ctx, q(-0.15, -0.12), q(0.15, 0.12), 0.24, h + 0.97, h + 1.62, M_MARBLE, 1.0);
    obb(b, ctx, q(-0.20, -0.14), q(0.20, 0.14), 0.28, h + 1.62, h + 2.18, M_MARBLE, 1.0);
    for s in [-0.26f32, 0.26] {
        obb(b, ctx, q(s - 0.05, -0.05), q(s + 0.05, 0.05), 0.11, h + 1.58, h + 2.10, M_MARBLE, 0.8);
    }
    cyl(b, ctx, [p[0], p[1]], 0.07, 0.07, h + 2.18, h + 2.26, 6, M_MARBLE, 0.6);
    let hd = ctx.pt(p[0], p[1], h + 2.38);
    blob(b, ctx, hd, 0.13, M_MARBLE, 0.6, &mut Rng::new(5));
}

fn big_vase(b: &mut Builder, ctx: &Ctx, p: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    obb(b, ctx, [p[0] - 0.31, p[1]], [p[0] + 0.31, p[1]], 0.62, h - 0.3, h + 0.95, M_GRANITE, 1.1);
    obb(b, ctx, [p[0] - 0.34, p[1]], [p[0] + 0.34, p[1]], 0.68, h + 0.95, h + 1.03, M_GRANITE, 1.1);
    cyl(b, ctx, p, 0.30, 0.24, h + 1.03, h + 1.12, 10, M_MARBLE, 1.0);
    cyl(b, ctx, p, 0.14, 0.15, h + 1.12, h + 1.50, 10, M_MARBLE, 1.0);
    cyl(b, ctx, p, 0.20, 0.45, h + 1.50, h + 1.82, 12, M_MARBLE, 1.2);
    cyl(b, ctx, p, 0.45, 0.45, h + 1.82, h + 2.06, 12, M_MARBLE, 1.2);
    cyl(b, ctx, p, 0.45, 0.30, h + 2.06, h + 2.26, 12, M_MARBLE, 1.2);
    cyl(b, ctx, p, 0.28, 0.34, h + 2.26, h + 2.40, 10, M_MARBLE, 1.0);
    cyl(b, ctx, p, 0.42, 0.40, h + 2.40, h + 2.52, 12, M_MARBLE, 1.2);
}

/// A round basin: outer wall, coping, water disk. Returns the water top.
#[allow(clippy::too_many_arguments)]
fn basin(b: &mut Builder, ctx: &Ctx, p: [f32; 2], r_out: f32, wall_h: f32, segs: usize, mat: u32) -> f32 {
    let h = ctx.hh(p[0], p[1]);
    let r_in = r_out - 0.42;
    let z_bot = h - 0.4;
    let z_top = h + wall_h;
    let zw = z_top - 0.22;
    let pt = |j: usize, r: f32, z: f32| -> [f32; 3] {
        let a = j as f32 / segs as f32 * std::f32::consts::TAU;
        ctx.pt(p[0] + a.cos() * r, p[1] + a.sin() * r, z)
    };
    for j in 0..segs {
        let j1 = (j + 1) % segs;
        // outer wall
        b.quad(
            mat,
            [pt(j, r_out, z_bot), pt(j, r_out, z_top), pt(j1, r_out, z_top), pt(j1, r_out, z_bot)],
            [[0.0, 0.0], [0.0, (z_top - z_bot) / 1.2], [r_out / (segs as f32) * 3.0, (z_top - z_bot) / 1.2], [r_out / (segs as f32) * 3.0, 0.0]],
        );
        // coping top
        b.quad(
            mat,
            [pt(j1, r_out, z_top), pt(j, r_out, z_top), pt(j, r_in, z_top), pt(j1, r_in, z_top)],
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],
        );
        // inner wall down to below the water line
        b.quad(
            mat,
            [pt(j, r_in, z_top - 0.06), pt(j1, r_in, z_top - 0.06), pt(j1, r_in, zw - 0.3), pt(j, r_in, zw - 0.3)],
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.5], [0.0, 0.5]],
        );
    }
    // water disk (animated: ripples + slow drift)
    b.anim(1.0);
    let c = ctx.pt(p[0], p[1], zw);
    for j in 0..segs {
        let j1 = (j + 1) % segs;
        b.tri(
            M_WATER,
            [c, pt(j, r_in - 0.02, zw), pt(j1, r_in - 0.02, zw)],
            [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]],
        );
    }
    b.anim(0.0);
    zw
}

fn fountain_big(b: &mut Builder, ctx: &Ctx, p: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    let zw = basin(b, ctx, p, 4.6, 0.62, 16, M_GRANITE);
    // pedestal + tiered dish
    cyl(b, ctx, p, 0.85, 0.55, zw, h + 1.35, 12, M_GRANITE, 1.4);
    cyl(b, ctx, p, 0.35, 2.0, h + 1.35, h + 1.92, 14, M_MARBLE, 1.6);
    cyl(b, ctx, p, 2.0, 2.0, h + 1.92, h + 2.0, 14, M_MARBLE, 1.6);
    // tier 1 water
    b.anim(1.0);
    let c1 = ctx.pt(p[0], p[1], h + 1.95);
    for j in 0..14 {
        let j1 = (j + 1) % 14;
        let a = |j: usize| -> [f32; 3] {
            let a = j as f32 / 14.0 * std::f32::consts::TAU;
            ctx.pt(p[0] + a.cos() * 1.9, p[1] + a.sin() * 1.9, h + 1.95)
        };
        b.tri(M_WATER, [c1, a(j), a(j1)], [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]);
    }
    b.anim(0.0);
    // the falling veil from the tier-1 rim + churn where it lands
    jet_cyl(b, ctx, p, 1.96, 1.96, h + 0.52, h + 1.88, 16, M_SHEET, 2.6, 3.0);
    splash(b, ctx, p, 2.12, zw + 0.02);
    cyl(b, ctx, p, 0.24, 0.20, h + 1.95, h + 2.62, 10, M_MARBLE, 1.2);
    cyl(b, ctx, p, 0.20, 1.05, h + 2.62, h + 3.06, 12, M_MARBLE, 1.2);
    // the tier-2 veil falls onto the tier-1 water
    jet_cyl(b, ctx, p, 1.01, 1.01, h + 1.98, h + 3.00, 12, M_SHEET, 1.6, 3.0);
    splash(b, ctx, p, 1.18, h + 1.97);
    b.anim(1.0);
    let c2 = ctx.pt(p[0], p[1], h + 3.1);
    for j in 0..12 {
        let j1 = (j + 1) % 12;
        let a = |j: usize| -> [f32; 3] {
            let a = j as f32 / 12.0 * std::f32::consts::TAU;
            ctx.pt(p[0] + a.cos() * 1.0, p[1] + a.sin() * 1.0, h + 3.1)
        };
        b.tri(M_WATER, [c2, a(j), a(j1)], [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]);
    }
    b.anim(0.0);
    blob(b, ctx, ctx.pt(p[0], p[1], h + 3.28), 0.17, M_GOLD, 0.6, &mut Rng::new(11));
    // central jet + four side jets off the lower tier (animated columns)
    jet_cyl(b, ctx, p, 0.10, 0.26, h + 3.2, h + 5.3, 8, M_JET, 1.2, 2.0);
    splash(b, ctx, p, 0.46, h + 3.13);
    for j in 0..4 {
        let a = j as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let q = [p[0] + a.cos() * 1.35, p[1] + a.sin() * 1.35];
        jet_cyl(b, ctx, q, 0.08, 0.20, h + 1.95, h + 2.95, 6, M_JET, 1.0, 2.0);
        splash(b, ctx, q, 0.30, h + 1.97);
    }
    // four granite steps at the cardinals
    for j in 0..4 {
        let a = j as f32 / 4.0 * std::f32::consts::TAU;
        let d = [a.cos(), a.sin()];
        let p0 = add(p, mul(d, 4.55));
        let p1 = add(p, mul(d, 5.25));
        let n = perp(d);
        let l = add(p0, mul(n, 1.3));
        let r = add(p0, mul(n, -1.3));
        obb(b, ctx, add(l, mul(d, 0.0)), add(l, mul(d, 0.68)), 2.6, h - 0.4, h + 0.30, M_GRANITE, 1.2);
        let l2 = add(p1, mul(n, 1.3));
        obb(b, ctx, l2, add(l2, mul(d, 0.68)), 2.6, h - 0.4, h + 0.15, M_GRANITE, 1.2);
        let _ = r;
    }
}

fn fountain_small(b: &mut Builder, ctx: &Ctx, p: [f32; 2]) {
    let h = ctx.hh(p[0], p[1]);
    let zw = basin(b, ctx, p, 1.75, 0.46, 10, M_GRANITE);
    cyl(b, ctx, p, 0.34, 0.22, zw, h + 1.05, 8, M_GRANITE, 0.9);
    cyl(b, ctx, p, 0.16, 0.72, h + 1.05, h + 1.32, 10, M_MARBLE, 1.0);
    b.anim(1.0);
    let c = ctx.pt(p[0], p[1], h + 1.35);
    for j in 0..10 {
        let j1 = (j + 1) % 10;
        let a = |j: usize| -> [f32; 3] {
            let a = j as f32 / 10.0 * std::f32::consts::TAU;
            ctx.pt(p[0] + a.cos() * 0.66, p[1] + a.sin() * 0.66, h + 1.35)
        };
        b.tri(M_WATER, [c, a(j), a(j1)], [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]);
    }
    b.anim(0.0);
    blob(b, ctx, ctx.pt(p[0], p[1], h + 1.45), 0.10, M_GOLD, 0.5, &mut Rng::new(9));
    jet_cyl(b, ctx, p, 0.07, 0.17, h + 1.45, h + 2.55, 6, M_JET, 1.0, 2.0);
    splash(b, ctx, p, 0.30, h + 1.37);
}

fn gravel_disk(b: &mut Builder, ctx: &Ctx, p: [f32; 2], r: f32, segs: usize) {
    let pt = |j: usize, z: f32| -> [f32; 3] {
        let a = j as f32 / segs as f32 * std::f32::consts::TAU;
        let q = [p[0] + a.cos() * r, p[1] + a.sin() * r];
        ctx.pt(q[0], q[1], ctx.hh(q[0], q[1]) + z)
    };
    let c = ctx.pt(p[0], p[1], ctx.hh(p[0], p[1]) + 0.13);
    for j in 0..segs {
        let j1 = (j + 1) % segs;
        b.tri(M_GRAVEL, [c, pt(j, 0.13), pt(j1, 0.13)], [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]);
        // rim skirt
        b.quad(
            M_GRAVEL,
            [pt(j1, 0.13), pt(j, 0.13), pt(j, -0.25), pt(j1, -0.25)],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.35], [0.0, 0.35]],
        );
    }
}

// ------------------------------------------------------------------ fence ----

#[derive(Clone, Copy)]
struct Gate {
    t: f32,    // centre along the edge
    w: f32,    // opening width
    tall: f32, // pillar extra height
}

#[allow(clippy::too_many_arguments)]
fn fence_edge(b: &mut Builder, ctx: &Ctx, a: [f32; 2], e: [f32; 2], gate: Option<&Gate>, rng: &mut Rng) {
    let d = sub(e, a);
    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if l < 1.0 {
        return;
    }
    let dir = unit(d);
    let ext = perp(dir); // exterior (polygon is CCW: +perp is outward here since edges wind CCW and we fence outward of the interior... kept sign fixed below)
    let at = |t: f32| -> [f32; 2] { add(a, mul(dir, t)) };
    let par_top = |t: f32| -> f32 { ctx.hh(at(t)[0], at(t)[1]) + PAR_H };

    // ---- granite parapet: stepped blocks every ~2 m
    let m = (l / 2.0).ceil().max(1.0) as usize;
    for k in 0..m {
        let t0 = k as f32 * l / m as f32;
        let t1 = (k + 1) as f32 * l / m as f32;
        let z0 = ctx.hh(at(t0)[0], at(t0)[1]).min(ctx.hh(at(t1)[0], at(t1)[1])) - SINK;
        let z1 = ctx.hh(at(t0)[0], at(t0)[1]).max(ctx.hh(at(t1)[0], at(t1)[1])) + PAR_H;
        obb(b, ctx, at(t0), at(t1), PAR_W, z0, z1, M_GRANITE, 1.3);
    }

    // ---- columns on a fixed grid
    let ns = ((l / COL_SPAN).round() as usize).max(2);
    for i in 0..=ns {
        let t = i as f32 * l / ns as f32;
        if let Some(g) = gate {
            if (t - g.t).abs() < g.w * 0.5 + 0.55 {
                continue;
            }
        }
        let c = at(t);
        let z0 = par_top(t) - 0.12;
        let z1 = par_top(t) + COL_H;
        obb(b, ctx, add(c, mul(dir, -0.275)), add(c, mul(dir, 0.275)), 0.55, z0, z1, M_GRANITE, 1.2);
        obb(b, ctx, add(c, mul(dir, -0.33)), add(c, mul(dir, 0.33)), 0.66, z1 - 0.10, z1 + 0.10, M_GRANITE, 1.2);
    }

    // ---- iron grille between neighbouring columns (skipping the gate)
    let mut nodes: Vec<f32> = Vec::new();
    for i in 0..=ns {
        let t = i as f32 * l / ns as f32;
        if let Some(g) = gate {
            if (t - g.t).abs() < g.w * 0.5 + 0.55 {
                continue;
            }
        }
        nodes.push(t);
    }
    for w2 in nodes.windows(2) {
        let (t0, t1) = (w2[0], w2[1]);
        if let Some(g) = gate {
            let mid = (t0 + t1) * 0.5;
            if (mid - g.t).abs() < g.w * 0.5 + 0.1 {
                continue;
            }
        }
        let z0 = par_top(t0).max(par_top(t1)) + 0.04;
        let span = t1 - t0;
        let p0 = ctx.pt(at(t0)[0], at(t0)[1], z0);
        let p1 = ctx.pt(at(t1)[0], at(t1)[1], z0);
        let p2 = ctx.pt(at(t1)[0], at(t1)[1], z0 + GRILLE_H);
        let p3 = ctx.pt(at(t0)[0], at(t0)[1], z0 + GRILLE_H);
        b.quad(
            M_GRILLE,
            [p0, p1, p2, p3],
            [[0.0, 0.0], [span / COL_SPAN, 0.0], [span / COL_SPAN, 1.0], [0.0, 1.0]],
        );
        // top + bottom rails
        obb(
            b, ctx, at(t0), at(t1), 0.10,
            z0 + GRILLE_H - 0.02, z0 + GRILLE_H + 0.08, M_IRON, 1.0,
        );
        obb(b, ctx, at(t0), at(t1), 0.08, z0 - 0.02, z0 + 0.07, M_IRON, 1.0);
    }
    let _ = ext;

    // ---- the gate itself
    if let Some(g) = gate {
        let g0 = g.t - g.w * 0.5;
        let g1 = g.t + g.w * 0.5;
        let ztop = par_top(g.t) + COL_H + g.tall;
        // pillars
        for gt in [g0, g1] {
            let c = at(gt);
            let z0 = par_top(gt) - SINK * 0.7;
            let z1 = par_top(gt) + COL_H + g.tall;
            obb(b, ctx, add(c, mul(dir, -0.31)), add(c, mul(dir, 0.31)), 0.62, z0, z1, M_GRANITE, 1.2);
            obb(b, ctx, add(c, mul(dir, -0.37)), add(c, mul(dir, 0.37)), 0.74, z1 - 0.10, z1 + 0.12, M_GRANITE, 1.2);
            blob(b, ctx, ctx.pt(c[0], c[1], z1 + 0.32), 0.20, M_GOLD, 0.6, rng);
        }
        // leaves (closed) above the parapet
        let zl0 = par_top(g.t) + 0.06;
        let zl1 = par_top(g.t) + 2.62;
        for (ta, te) in [(g0 + 0.05, g.t - 0.015), (g.t + 0.015, g1 - 0.05)] {
            let pa = ctx.pt(at(ta)[0], at(ta)[1], zl0);
            let pb = ctx.pt(at(te)[0], at(te)[1], zl0);
            let pc = ctx.pt(at(te)[0], at(te)[1], zl1);
            let pd = ctx.pt(at(ta)[0], at(ta)[1], zl1);
            let uk = (te - ta) / 1.2;
            b.quad(M_GATE, [pa, pb, pc, pd], [[0.0, 0.0], [uk, 0.0], [uk, 1.0], [0.0, 1.0]]);
        }
        // top rail + gilded crest over the meeting stile
        obb(b, ctx, at(g0 + 0.02), at(g1 - 0.02), 0.12, zl1 - 0.02, zl1 + 0.10, M_IRON, 1.0);
        let cg = at(g.t);
        obb(b, ctx, add(cg, mul(dir, -0.55)), add(cg, mul(dir, 0.55)), 0.10, zl1 + 0.1, zl1 + 0.62, M_GOLD, 0.9);
        blob(b, ctx, ctx.pt(cg[0], cg[1], zl1 + 0.74), 0.11, M_GOLD, 0.5, rng);
        let _ = ztop;
    }
}

// ------------------------------------------------------------------- build ---

pub fn build_garden(
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    mn: f32,
    eu: [f32; 2],
    nvec: [f32; 2],
    p: &Params,
) -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>) {
    let ctx = Ctx { h_at, anchor, mn, eu, nvec };
    let mut b = Builder::new();
    let mut rng = Rng::new(p.seed);
    let c = p.centre;

    // ---- the irregular outline: 9 jittered vertices around the ellipse
    let nv = 9;
    let mut poly: Vec<[f32; 2]> = Vec::with_capacity(nv);
    for k in 0..nv {
        let a = k as f32 / nv as f32 * std::f32::consts::TAU + rng.range(-0.05, 0.05);
        let ru = p.ru * rng.range(0.80, 1.14);
        let rv = p.rv * rng.range(0.80, 1.14);
        poly.push([c[0] + a.cos() * ru, c[1] + a.sin() * rv]);
    }
    let shrunk6 = shrink(&poly, 6.0);
    let shrunk15 = shrink(&poly, 1.5);

    // ---- gates: nearest edges to the west and north targets
    let target_main = [c[0] - p.ru, c[1]];
    let target_north = [c[0], c[1] + p.rv];
    let mut gates: Vec<Option<Gate>> = vec![None; poly.len()];
    let mut best = (f32::MAX, 0usize, 0.5f32);
    for k in 0..poly.len() {
        let a = poly[k];
        let e = poly[(k + 1) % poly.len()];
        let mid = mul(add(a, e), 0.5);
        let dd = sub(target_main, mid);
        let d2 = dd[0] * dd[0] + dd[1] * dd[1];
        if d2 < best.0 {
            best = (d2, k, 0.5);
        }
    }
    let gk = best.1;
    {
        let a = poly[gk];
        let e = poly[(gk + 1) % poly.len()];
        let d = unit(sub(e, a));
        let l = {
            let v = sub(e, a);
            (v[0] * v[0] + v[1] * v[1]).sqrt()
        };
        let tg = sub(target_main, a);
        let t = (tg[0] * d[0] + tg[1] * d[1]).clamp(l * 0.3, l * 0.7);
        gates[gk] = Some(Gate { t, w: 5.2, tall: 1.15 });
    }
    let mut bestn = (f32::MAX, 0usize);
    for k in 0..poly.len() {
        if k == gk {
            continue;
        }
        let a = poly[k];
        let e = poly[(k + 1) % poly.len()];
        let mid = mul(add(a, e), 0.5);
        let dd = sub(target_north, mid);
        let d2 = dd[0] * dd[0] + dd[1] * dd[1];
        if d2 < bestn.0 {
            bestn = (d2, k);
        }
    }
    {
        let a = poly[bestn.1];
        let e = poly[(bestn.1 + 1) % poly.len()];
        let d = unit(sub(e, a));
        let l = {
            let v = sub(e, a);
            (v[0] * v[0] + v[1] * v[1]).sqrt()
        };
        let tg = sub(target_north, a);
        let t = (tg[0] * d[0] + tg[1] * d[1]).clamp(l * 0.3, l * 0.7);
        gates[bestn.1] = Some(Gate { t, w: 3.8, tall: 0.85 });
    }

    // ---- fence around the whole outline
    for k in 0..poly.len() {
        let a = poly[k];
        let e = poly[(k + 1) % poly.len()];
        fence_edge(&mut b, &ctx, a, e, gates[k].as_ref(), &mut rng);
    }

    // ---- gravel aprons outside the gates (toward the village and beyond)
    for k in 0..poly.len() {
        if let Some(g) = &gates[k] {
            let a = poly[k];
            let e = poly[(k + 1) % poly.len()];
            let d = unit(sub(e, a));
            let dir = d;
            let outn = mul(perp(dir), -1.0); // exterior side (polygon CCW)
            let gp = add(a, mul(dir, g.t));
            let len = if g.w > 4.5 { 13.0 } else { 8.0 };
            let q0 = add(gp, mul(outn, 0.3));
            let q1 = add(gp, mul(outn, len));
            strip(&mut b, &ctx, q0, q1, 2.4);
        }
    }

    // ---- perimeter alley along the shrunk outline
    let ring = shrink(&poly, RING_OFF);
    for k in 0..ring.len() {
        strip(&mut b, &ctx, ring[k], ring[(k + 1) % ring.len()], RING_HW);
    }

    // ---- main alley + cross alleys inside
    let main_iv = line_inside(&poly, true, c[1], c[0] - p.ru * 1.3, c[0] + p.ru * 1.3);
    let main_span = main_iv.first().copied().unwrap_or((c[0] - 10.0, c[0] + 10.0));
    let main_a = main_span.0 + 3.5;
    let main_e = main_span.1 - 3.5;
    let u_mid = (main_a + main_e) * 0.5;
    strip(&mut b, &ctx, [main_a, c[1]], [main_e, c[1]], ALLEY_HW);

    let mut v_lines: Vec<f32> = Vec::new();
    for dv in [-36.0f32, -18.0, 18.0, 36.0] {
        v_lines.push(c[1] + dv);
        for (a, e) in line_inside(&poly, true, c[1] + dv, c[0] - p.ru * 1.3, c[0] + p.ru * 1.3) {
            if e - a > 4.0 {
                strip(&mut b, &ctx, [a + 1.0, c[1] + dv], [e - 1.0, c[1] + dv], PATH_HW);
            }
        }
    }
    let mut u_lines: Vec<f32> = Vec::new();
    for du in [-44.0f32, -22.0, 22.0, 44.0] {
        u_lines.push(u_mid + du);
        for (a, e) in line_inside(&poly, false, u_mid + du, c[1] - p.rv * 1.3, c[1] + p.rv * 1.3) {
            if e - a > 4.0 {
                strip(&mut b, &ctx, [u_mid + du, a + 1.0], [u_mid + du, e - 1.0], PATH_HW);
            }
        }
    }

    // ---- the central fountain clearing
    gravel_disk(&mut b, &ctx, [u_mid, c[1]], 6.4, 18);
    fountain_big(&mut b, &ctx, [u_mid, c[1]]);
    for j in 0..4 {
        let a = j as f32 / 4.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_4;
        let pv = [u_mid + a.cos() * 7.4, c[1] + a.sin() * 7.4];
        big_vase(&mut b, &ctx, pv);
        let pl = [u_mid + a.cos() * 8.8, c[1] + a.sin() * 8.8];
        lantern(&mut b, &ctx, pl);
    }

    // ---- busts along the main alley (the famous alley of busts)
    let mut k = 0;
    let mut ub = main_a + 9.0;
    while ub < main_e - 9.0 {
        if (ub - u_mid).abs() > 8.5 {
            let side = if k % 2 == 0 { 1.0 } else { -1.0 };
            bust(&mut b, &ctx, [ub, c[1] + side * 3.6], [0.0, -side]);
        }
        ub += 8.5;
        k += 1;
    }

    // ---- trees + lanterns alternating along the main alley
    let mut u0 = main_a + 6.0;
    while u0 < main_e - 6.0 {
        if (u0 - u_mid).abs() > 8.5 {
            for s in [-5.3f32, 5.3] {
                if u0 <= main_e - 6.0 {
                    tree(&mut b, &ctx, u0, c[1] + s, rng.range(0.85, 1.25), &mut rng);
                }
                if u0 + 2.75 < main_e - 6.0 {
                    lantern(&mut b, &ctx, [u0 + 2.75, c[1] + s]);
                }
            }
        }
        u0 += 5.5;
    }

    // ---- statues flanking the main alley
    for du in [-30.0f32, -14.0, 14.0, 30.0] {
        let s = if du < 0.0 { -1.0 } else { 1.0 };
        let p2 = [u_mid + du, c[1] + s * 4.4];
        let d2 = sub([u_mid, c[1]], p2);
        statue(&mut b, &ctx, p2, unit(d2));
    }

    // ---- benches along the main alley, facing it, with urns
    let mut ub = main_a + 12.0;
    while ub < main_e - 12.0 {
        if (ub - u_mid).abs() > 9.5 {
            for s in [-1.0f32, 1.0] {
                let p2 = [ub, c[1] + s * 4.15];
                bench(&mut b, &ctx, p2, [0.0, -s]);
                urn(&mut b, &ctx, [ub - 1.15, c[1] + s * 4.15]);
                urn(&mut b, &ctx, [ub + 1.15, c[1] + s * 4.15]);
            }
        }
        ub += 11.0;
    }

    // ---- lanterns at every alley crossing
    for &vl in &v_lines {
        for &ul in &u_lines {
            if main_a + 4.0 < ul && ul < main_e - 4.0 {
                lantern(&mut b, &ctx, [ul + 2.6, vl + 2.6]);
            }
        }
    }
    // small fountains on the four crossings next to the centre
    for du in [-22.0f32, 22.0] {
        for dv in [-18.0f32, 18.0] {
            let pf = [u_mid + du, c[1] + dv];
            if (pf[0] - main_a > 5.0) && (main_e - pf[0] > 5.0) {
                // keep it off the paths: only when the crossing carries no path
                let _ = pf;
            }
            let _ = dv;
        }
    }

    // ---- benches + trees along the perimeter alley
    let ring_n = ring.len();
    let mut acc = 6.0f32;
    let mut seg_i = 0usize;
    loop {
        let a = ring[seg_i];
        let e = ring[(seg_i + 1) % ring_n];
        let d = sub(e, a);
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if acc <= l {
            let t = acc / l;
            let p2 = add(a, mul(d, t));
            let dir = unit(d);
            let inw = perp(dir);
            // bench just outside the ring path, facing the garden
            let pb = add(p2, mul(inw, -(RING_HW + 0.75)));
            if point_in_poly(&shrunk15, pb) {
                bench(&mut b, &ctx, pb, inw);
                urn(&mut b, &ctx, add(pb, add(mul(dir, 1.15), mul(inw, -0.4))));
                urn(&mut b, &ctx, add(pb, add(mul(dir, -1.15), mul(inw, -0.4))));
            }
            acc += 16.0;
        } else {
            acc -= l;
            seg_i += 1;
            if seg_i >= ring_n {
                break;
            }
        }
    }

    // ---- bosquets: the 4x4 grid of cells between the alleys
    for iu in -2..2i32 {
        for iv in -2..2i32 {
            let u0c = u_mid + iu as f32 * 22.0 + PATH_HW + 1.1;
            let u1c = u_mid + (iu + 1) as f32 * 22.0 - PATH_HW - 1.1;
            let v0c = c[1] + iv as f32 * 18.0 + PATH_HW + 1.1;
            let v1c = c[1] + (iv + 1) as f32 * 18.0 - PATH_HW - 1.1;
            if u1c - u0c < 8.0 || v1c - v0c < 7.0 {
                continue;
            }
            // the main-alley clearings have no bosquets
            if v0c < c[1] && v1c > c[1] {
                continue;
            }
            // fully inside the park, clear of the perimeter alley
            let corners = [[u0c, v0c], [u1c, v0c], [u0c, v1c], [u1c, v1c]];
            if !corners.iter().all(|q| point_in_poly(&shrunk6, *q)) {
                continue;
            }
            // clipped hedges around the cell
            let gap_side = rng.below(4) as usize;
            let sides = [
                ([u0c, v0c], [u1c, v0c]),
                ([u1c, v0c], [u1c, v1c]),
                ([u1c, v1c], [u0c, v1c]),
                ([u0c, v1c], [u0c, v0c]),
            ];
            for (si, (sa, se)) in sides.iter().enumerate() {
                let d = sub(*se, *sa);
                let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
                let nseg = (l / 2.9).ceil().max(1.0) as usize;
                let hgt = rng.range(1.02, 1.24);
                for q in 0..nseg {
                    let frac_mid = (q as f32 + 0.5) / nseg as f32;
                    if si == gap_side && q == nseg / 2 && frac_mid > 0.35 && frac_mid < 0.65 {
                        continue;
                    }
                    let ta = q as f32 / nseg as f32;
                    let te = (q + 1) as f32 / nseg as f32;
                    let a2 = add(*sa, mul(d, ta));
                    let e2 = add(*sa, mul(d, te));
                    hedge(&mut b, &ctx, a2, e2, hgt);
                }
            }
            // centre feature
            let cc = [u_mid + (iu as f32 + 0.5) * 22.0, c[1] + (iv as f32 + 0.5) * 18.0];
            let feature = rng.f();
            let mut keep_clear = f32::MAX;
            if feature < 0.30 {
                big_vase(&mut b, &ctx, cc);
                keep_clear = 3.0;
            } else if feature < 0.55 {
                fountain_small(&mut b, &ctx, cc);
                gravel_disk(&mut b, &ctx, cc, 3.0, 12);
                keep_clear = 4.0;
            } else if feature < 0.72 {
                let fd = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
                statue(&mut b, &ctx, cc, unit(fd));
                keep_clear = 3.0;
            }
            // trees inside the cell
            let cnt = 2 + rng.below(4) as usize;
            let mut placed: Vec<[f32; 2]> = Vec::new();
            let mut tries = 0;
            while placed.len() < cnt && tries < 40 {
                tries += 1;
                let q = [
                    rng.range(u0c + 1.6, u1c - 1.6),
                    rng.range(v0c + 1.6, v1c - 1.6),
                ];
                if (q[0] - cc[0]).hypot(q[1] - cc[1]) < keep_clear {
                    continue;
                }
                if placed.iter().any(|w| (w[0] - q[0]).hypot(w[1] - q[1]) < 4.0) {
                    continue;
                }
                placed.push(q);
                tree(&mut b, &ctx, q[0], q[1], rng.range(0.9, 1.3), &mut rng);
            }
        }
    }

    // ---- lanterns + vases inside the two gates
    for k in 0..poly.len() {
        if let Some(g) = &gates[k] {
            let a = poly[k];
            let e = poly[(k + 1) % poly.len()];
            let d = unit(sub(e, a));
            let inw = perp(d);
            let gp = add(a, mul(d, g.t));
            let pin = add(gp, mul(inw, 6.0));
            lantern(&mut b, &ctx, add(pin, mul(d, 3.2)));
            lantern(&mut b, &ctx, add(pin, mul(d, -3.2)));
            if g.w > 4.5 {
                let pv1 = add(gp, add(mul(inw, 7.5), mul(d, 4.6)));
                let pv2 = add(gp, add(mul(inw, 7.5), mul(d, -4.6)));
                big_vase(&mut b, &ctx, pv1);
                big_vase(&mut b, &ctx, pv2);
            }
        }
    }

    (b.finish(N_MATS), textures())
}

// ------------------------------------------------------------- textures ------

fn hash2(x: u32, y: u32) -> u32 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^ (h >> 16)
}

struct Img {
    px: Vec<u8>,
    w: usize,
    h: usize,
}

impl Img {
    fn new(w: usize, h: usize, c: [u8; 3], a: u8) -> Self {
        let mut px = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            px.extend_from_slice(&[c[0], c[1], c[2], a]);
        }
        Img { px, w, h }
    }
    fn set(&mut self, x: usize, y: usize, c: [u8; 3], a: u8) {
        let i = (y * self.w + x) * 4;
        self.px[i] = c[0];
        self.px[i + 1] = c[1];
        self.px[i + 2] = c[2];
        self.px[i + 3] = a;
    }
    fn px_at(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.w + x) * 4;
        [self.px[i], self.px[i + 1], self.px[i + 2]]
    }
    fn noise(&mut self, amp: i32, rng: &mut Rng) {
        for y in 0..self.h {
            for x in 0..self.w {
                let c = self.px_at(x, y);
                let n = rng.range(-(amp as f32), amp as f32) as i32;
                let f = |v: u8| (v as i32 + n).clamp(0, 255) as u8;
                self.set(x, y, [f(c[0]), f(c[1]), f(c[2])], 255);
            }
        }
    }
    fn speckle(&mut self, rng: &mut Rng, n: usize, dark: [u8; 3], light: [u8; 3]) {
        for _ in 0..n {
            let x = rng.below(self.w as u32) as usize;
            let y = rng.below(self.h as u32) as usize;
            let c = if rng.f() < 0.6 { dark } else { light };
            self.set(x, y, c, 255);
        }
    }
    fn line_h(&mut self, x0: usize, x1: usize, y: usize, c: [u8; 3], a: u8) {
        for x in x0..x1.min(self.w) {
            if y < self.h {
                self.set(x, y, c, a);
            }
        }
    }
    fn line_v(&mut self, x: usize, y0: usize, y1: usize, c: [u8; 3], a: u8) {
        for y in y0..y1.min(self.h) {
            if x < self.w {
                self.set(x, y, c, a);
            }
        }
    }
    fn flip_rows(&mut self) {
        let w = self.w;
        let h = self.h;
        for y in 0..h / 2 {
            for x in 0..w {
                let i0 = (y * w + x) * 4;
                let i1 = ((h - 1 - y) * w + x) * 4;
                for k in 0..4 {
                    self.px.swap(i0 + k, i1 + k);
                }
            }
        }
    }
    fn into(self) -> (u32, u32, Vec<u8>) {
        (self.w as u32, self.h as u32, self.px)
    }
}

const TS: usize = 256;

fn tex_granite(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [176, 172, 162], 255);
    im.noise(9, rng);
    im.speckle(rng, 380, [120, 116, 108], [212, 208, 200]);
    im.flip_rows();
    im.into()
}

fn tex_marble(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [232, 228, 219], 255);
    im.noise(5, rng);
    // faint grey veins
    for v in 0..5 {
        let mut x = rng.below(TS as u32) as i32;
        let mut y = 0i32;
        let drift = rng.range(-0.4, 0.4);
        while y < TS as i32 {
            x += (rng.range(-1.0, 1.0) + drift) as i32;
            if x >= 0 && x < TS as i32 {
                im.set(x as usize, y as usize, [196, 192, 186], 255);
            }
            y += 1;
            let _ = v;
        }
    }
    im.flip_rows();
    im.into()
}

fn tex_gravel(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [198, 180, 152], 255);
    im.noise(12, rng);
    im.speckle(rng, 2600, [150, 134, 112], [226, 212, 188]);
    im.flip_rows();
    im.into()
}

fn tex_water(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [72, 112, 130], 255);
    for y in 0..TS {
        if y % 9 == (y / 9) % 3 {
            let x0 = (hash2(0, y as u32) % 90) as usize;
            im.line_h(x0, (x0 + 60).min(TS), y, [104, 146, 162], 255);
        }
    }
    im.noise(6, rng);
    im.speckle(rng, 260, [176, 206, 214], [58, 92, 110]);
    im.flip_rows();
    im.into()
}

fn tex_jet(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [198, 224, 238], 255);
    for j in 0..40 {
        let x = (hash2(j, 3) % TS as u32) as usize;
        im.line_v(x, 0, TS, [234, 246, 251], 255);
    }
    im.noise(6, rng);
    im.speckle(rng, 320, [172, 204, 224], [244, 250, 253]);
    im.flip_rows();
    im.into()
}

fn tex_leaf(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [88, 112, 64], 255);
    im.noise(14, rng);
    im.speckle(rng, 3200, [54, 76, 40], [128, 150, 92]);
    im.flip_rows();
    im.into()
}

fn tex_hedge(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [70, 94, 56], 255);
    im.noise(11, rng);
    im.speckle(rng, 2600, [44, 62, 36], [106, 128, 76]);
    im.flip_rows();
    im.into()
}

fn tex_bark(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [106, 92, 78], 255);
    for j in 0..28 {
        let x = (hash2(j, 9) % TS as u32) as usize;
        im.line_v(x, 0, TS, [84, 72, 60], 255);
    }
    im.noise(7, rng);
    im.flip_rows();
    im.into()
}

fn tex_iron(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [56, 64, 58], 255);
    im.noise(8, rng);
    im.speckle(rng, 200, [36, 42, 38], [78, 88, 78]);
    im.flip_rows();
    im.into()
}

fn tex_bench(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [150, 168, 146], 255);
    im.noise(7, rng);
    for y in (0..TS).step_by(43) {
        im.line_h(0, TS, y, [116, 134, 112], 255);
    }
    im.flip_rows();
    im.into()
}

fn tex_lamp(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [244, 234, 198], 255);
    im.noise(6, rng);
    for j in 0..24 {
        let x = (hash2(j, 21) % TS as u32) as usize;
        im.line_v(x, 0, TS, [252, 246, 220], 255);
    }
    im.flip_rows();
    im.into()
}

fn tex_gold(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let mut im = Img::new(TS, TS, [212, 174, 74], 255);
    im.noise(9, rng);
    im.speckle(rng, 240, [168, 132, 48], [246, 216, 128]);
    im.flip_rows();
    im.into()
}

/// The Fельтен grille: on an alpha background, vertical iron rods with
/// gilded spear heads, a middle medallion and band rails.
fn tex_grille(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let iron = [44, 52, 46];
    let iron_l = [72, 82, 72];
    let gold = [216, 178, 74];
    let mut im = Img::new(TS, TS, [0, 0, 0], 0);
    // rods: 13 rods
    let rods = 13usize;
    for r in 0..rods {
        let x = ((r as f32 + 0.5) / rods as f32 * TS as f32) as usize;
        im.line_v(x, 20, TS - 10, iron, 255);
        im.line_v((x + 1).min(TS - 1), 20, TS - 10, iron_l, 255);
        // spear head
        for k in 0..7 {
            let yy = 20 - k;
            let half = (k / 2).max(0);
            for d in 0..=half {
                if x >= d && yy < TS {
                    im.set(x - d, yy, gold, 255);
                }
                if x + 1 + d < TS && yy < TS {
                    im.set(x + 1 + d, yy, gold, 255);
                }
            }
        }
        // gold rosette on every 2nd rod
        if r % 2 == 0 {
            let y = 60usize;
            for dy in 0..5 {
                for dx in 0..5 {
                    im.set((x + dx).min(TS - 1), (y + dy).min(TS - 1), gold, 255);
                }
            }
        }
    }
    // rails
    im.line_h(0, TS, 18, iron, 255);
    im.line_h(0, TS, 19, iron_l, 255);
    im.line_h(0, TS, 112, iron, 255);
    im.line_h(0, TS, 148, iron, 255);
    im.line_h(0, TS, 246, iron, 255);
    // medallion
    let (cx, cy) = (TS / 2, 130i32);
    for a in 0..720 {
        let a = a as f32 / 720.0 * std::f32::consts::TAU;
        for rr in [26.0f32, 27.0, 18.0] {
            let x = cx as f32 + a.cos() * rr;
            let y = cy as f32 + a.sin() * rr;
            if x >= 0.0 && y >= 0.0 && (x as usize) < TS && (y as usize) < TS {
                let c = if rr > 20.0 { gold } else { iron };
                im.set(x as usize, y as usize, c, 255);
            }
        }
    }
    im.noise(0, rng);
    im.flip_rows();
    im.into()
}

/// One gate leaf: denser rods, a gilded shield at the centre.
fn tex_gate(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let iron = [40, 48, 42];
    let iron_l = [66, 76, 66];
    let gold = [220, 182, 76];
    let mut im = Img::new(TS, TS, [0, 0, 0], 0);
    let rods = 21usize;
    for r in 0..rods {
        let x = ((r as f32 + 0.5) / rods as f32 * TS as f32) as usize;
        im.line_v(x, 24, TS - 12, iron, 255);
        im.line_v((x + 1).min(TS - 1), 24, TS - 12, iron_l, 255);
        for k in 0..8 {
            let yy = 24 - k;
            let half = (k / 2).max(0);
            for d in 0..=half {
                if x >= d && yy < TS {
                    im.set(x - d, yy, gold, 255);
                }
                if x + 1 + d < TS && yy < TS {
                    im.set(x + 1 + d, yy, gold, 255);
                }
            }
        }
    }
    // a solid frame + lower kick plate so the leaf reads as a structure
    for k in [0usize, 1, 2, TS - 3, TS - 2, TS - 1] {
        im.line_v(k, 12, 252, iron, 255);
    }
    for y in [12usize, 13, 14, 108, 109, 196, 197, 198, 248, 249, 250, 251, 252] {
        im.line_h(0, TS, y, iron, 255);
    }
    for y in 199..=252 {
        for x in 0..TS {
            if (x + y) % 4 < 2 {
                im.set(x, y, iron_l, 255);
            }
        }
    }
    // gilded shield at the centre
    let (cx, cy) = (TS as i32 / 2, 128i32);
    for dy in -40i32..=40 {
        let w2 = (1.0 - (dy as f32 / 40.0).abs().powi(2)).sqrt() * 30.0;
        for dx in -(w2 as i32)..=(w2 as i32) {
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && y >= 0 && (x as usize) < TS && (y as usize) < TS {
                let edge = dx.abs() > w2 as i32 - 3 || dy.abs() > 37;
                im.set(x as usize, y as usize, if edge { [176, 138, 52] } else { gold }, 255);
            }
        }
    }
    im.noise(0, rng);
    im.flip_rows();
    im.into()
}

/// A falling water sheet: vertical streams of pale blue water separated by
/// transparent gaps, so a cylinder of it reads as a ring of falling veils
/// (not a solid drum); the fragment shader scrolls it downwards.
fn tex_sheet(rng: &mut Rng) -> (u32, u32, Vec<u8>) {
    let water = [196, 222, 236];
    let water_l = [232, 244, 250];
    let mut im = Img::new(TS, TS, [0, 0, 0], 0);
    let mut x = 0usize;
    while x < TS {
        let w_ = 5 + (hash2(x as u32, 17) % 4) as usize; // stream width 5..8 px
        let gap = 2 + (hash2(x as u32, 31) % 3) as usize; // gap 2..4 px
        for xx in x..(x + w_).min(TS) {
            for y in 0..TS {
                let n = (hash2(xx as u32 * 7, y as u32 * 3) % 20) as i32 - 10;
                let c = if (xx - x) < 2 { water_l } else { water };
                let f = |v: u8| ((v as i32) + n).clamp(0, 255) as u8;
                im.set(xx, y, [f(c[0]), f(c[1]), f(c[2])], 255);
            }
        }
        // a few bright sparkle pixels inside the streams
        if hash2(x as u32, 3) % 2 == 0 {
            let y = (hash2(x as u32, 41) % TS as u32) as usize;
            im.set(x, y, water_l, 255);
        }
        x += w_ + gap;
    }
    im.flip_rows();
    im.into()
}

fn textures() -> Vec<(u32, u32, Vec<u8>)> {
    let mut rng = Rng::new(0x5EED_6A2D_5EED_6A2D);
    vec![
        tex_granite(&mut rng),
        tex_marble(&mut rng),
        tex_gravel(&mut rng),
        tex_water(&mut rng),
        tex_jet(&mut rng),
        tex_leaf(&mut rng),
        tex_hedge(&mut rng),
        tex_bark(&mut rng),
        tex_iron(&mut rng),
        tex_bench(&mut rng),
        tex_lamp(&mut rng),
        tex_gold(&mut rng),
        tex_grille(&mut rng),
        tex_gate(&mut rng),
        tex_sheet(&mut rng),
    ]
}
