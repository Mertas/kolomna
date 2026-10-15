//! Windowed (winit + wgpu surface) and headless (`--shot`) terrain rendering.
//!
//! Soft camera model:
//! - wheel = smooth zoom (exponential, eased toward a target)
//! - middle-button drag = smooth rotation (yaw around the grid centre; the
//!   shader applies the angle, so the mesh is never re-uploaded)
//! - Shift+middle / left / right drag = pan; the mouse near a window edge
//!   pans smoothly (edge scrolling, speed grows toward the edge)
//! - WASD/arrows = pan; Q/E = rotate 90°; R = reset; Esc = quit
//!
//! Every frame the eased camera state feeds the shared shader uniforms
//! (`center`/`scale`/`angle`/`time`); water shimmer animates via `time`.

use std::collections::HashSet;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use kolomna_render::model::{ModelMesh, ModelRenderer, PlacedModel};
use kolomna_render::{
    Params, TerrainRenderer, VIEW_H, VIEW_W, Vertex, ZSCALE, build_mesh, grid_side, smooth_field,
    write_png,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

/// keyboard pan speed (screen px per second)
const PAN_PX_PER_S: f32 = 480.0;
/// edge-scrolling speed at the window border (screen px per second)
const EDGE_MAX_PX_PER_S: f32 = 700.0;
/// edge-scrolling activation margin (px)
const EDGE_PX: f64 = 36.0;
const ZOOM_MIN: f32 = 0.12;
const ZOOM_MAX: f32 = 100.0;

// ---------------------------------------------------------------- scene ----

/// The editor's drawing: free road chains, four-click courtyard quads and
/// park outlines, all in plan coordinates (metres in the scene's eu/nvec
/// frame).  Session-only — nothing is persisted, by design.
#[derive(Default)]
pub struct UserPlan {
    /// road chains: click-to-click polylines
    pub roads: Vec<Vec<[f32; 2]>>,
    /// courtyards: >= 3 clicks, any polygon (CCW in plan)
    pub blocks: Vec<Vec<[f32; 2]>>,
    /// parks: >= 3 contour clicks (CCW in plan)
    pub parks: Vec<Vec<[f32; 2]>>,
}

/// Footprint of one placed house (plan frame, metres) — roads remove the
/// houses they cover and later blocks keep clear of those already standing.
pub type HouseQuad = [[f32; 2]; 4];

/// One built block: where its meshes live in `village` and the footprints of
/// its houses, so a road can later remove exactly the houses on its way.
pub struct BlkRec {
    pub blk: u64,
    /// outer plan ring, plan-frame metres (the shared frame of the roads)
    pub q: Vec<[f32; 2]>,
    pub usr: bool,
    pub entries: Vec<usize>,
    pub houses: Vec<HouseQuad>,
}

/// One drawn park: the outline as drawn plus the village entries of the
/// gardens its parts became after the road splits.
pub struct ParkRec {
    pub outline: Vec<[f32; 2]>,
    pub entries: Vec<usize>,
}

pub struct Scene {
    base: Vec<Vertex>,
    atlas: Vec<u8>,
    grid: f32,
    zspan: f32,
    /// terrain min altitude (metres) — feeds the shader's sand blend
    mn: f32,
    /// water surface level (metres)
    wl: f32,
    /// smoothed height grid (metres, water-clamped) — the exact 4 m field the
    /// render mesh is built from; every consumer samples it (roads, houses,
    /// gardens, picking, the camera raycast for the rotation pivot)
    sm: Vec<f32>,
    /// grid side length of `sm` (4 m nodes; `kf` maps 16 m grid cells into it)
    n: usize,
    /// coarse-cell (16 m) coordinates -> `sm` cell coordinates
    kf: f32,
    /// plan frame: cell coords of plan (0,0) and its orthonormal axes
    anchor: [f32; 2],
    eu: [f32; 2],
    nvec: [f32; 2],
    /// generated Petersburg tenements (mesh, textures, model origin, world rot)
    village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)>,
    /// index of the courtyard-boundary overlay inside `village`
    plan_idx: usize,
    /// index of the drawn-roads mesh inside `village` (editor commits)
    roads_idx: Option<usize>,
    /// index of the grid-streets mesh inside `village` (junction rebuilds)
    streets_idx: usize,
    /// the street-grid layout: junction trims and road snapping use it
    net: kolomna_render::streets::Params,
    /// built blocks: plan ring, house footprints and village entries
    blk_records: Vec<BlkRec>,
    /// drawn parks: outlines and the village entries of their parts
    park_records: Vec<ParkRec>,
    /// camera focus for `--shot` (the middle block's anchor house)
    focus: [f32; 3],
}

/// What one block build produces: the meshes (plinth + house per entry),
/// the plan rings for the boundary overlay (world metres), the street-layout
/// edge (generated blocks only), the size report, the --shot focus, the
/// plan-frame ring and the footprints of the houses that were built.
struct BlkBuilt {
    meshes: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)>,
    plan: (Vec<[f32; 2]>, Vec<[f32; 2]>),
    edge: Option<(f32, f32, f32)>,
    report: Vec<(f32, f32, f32)>,
    focus: Option<[f32; 3]>,
    q_plan: Vec<[f32; 2]>,
    houses: Vec<HouseQuad>,
}

/// Inset ring: intersections of the offset lines of neighbouring sides —
/// the courtyard corners (shared by the filler and the boundary overlay).
fn inset_ring(q: &[[f32; 2]], depth: f32) -> Vec<[f32; 2]> {
    let n = q.len();
    let side = |i: usize| -> ([f32; 2], [f32; 2]) {
        let a = q[i];
        let b = q[(i + 1) % n];
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
        let t = [d[0] / l, d[1] / l];
        (t, [-t[1], t[0]])
    };
    let mut vv = Vec::with_capacity(n);
    for i in 0..n {
        let ip = (i + n - 1) % n;
        let (t1, n1) = side(ip);
        let (t2, n2) = side(i);
        let p1 = [q[ip][0] + n1[0] * depth, q[ip][1] + n1[1] * depth];
        let p2 = [q[i][0] + n2[0] * depth, q[i][1] + n2[1] * depth];
        let den = t1[0] * t2[1] - t1[1] * t2[0];
        let den = if den.abs() < 1e-6 { 1e-6 } else { den };
        let u = ((p2[0] - p1[0]) * t2[1] - (p2[1] - p1[1]) * t2[0]) / den;
        vv.push([p1[0] + t1[0] * u, p1[1] + t1[1] * u]);
    }
    vv
}

/// One courtyard block: generated (blk < 10) or editor-drawn (q_user is
/// Some — the polygon arrives already CCW in the plan frame, any vertices
/// >= 3).  The filler inscribes corner houses, rows of rectangles and
/// closing trapezoids along the sides; every house is seated on the highest
/// ground under its footprint and gets a ground-following plinth.  Slots a
/// drawn road covers — or one an earlier block already occupies — are left
/// empty, so a road crossing a courtyard removes exactly the houses it hits.
#[allow(clippy::too_many_arguments)]
fn build_block(
    blk: u64,
    q_user: Option<&[[f32; 2]]>,
    h_at: &dyn Fn(f32, f32) -> f32,
    anchor: [f32; 2],
    eu: [f32; 2],
    nvec: [f32; 2],
    mn: f32,
    wl: f32,
    want_focus: bool,
    focus_blk: u64,
    roads: &[Vec<[f32; 2]>],
    avoid: &[HouseQuad],
) -> BlkBuilt {
    let mut village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)> = Vec::new();
    let mut report: Vec<(f32, f32, f32)> = Vec::new(); // w, d, fall
    let mut edge: Option<(f32, f32, f32)> = None;
    let mut focus: Option<[f32; 3]> = None;
    let mut houses: Vec<HouseQuad> = Vec::new();

    let depth = 12.0f32; // one depth for the whole block: rows close exactly
    let setback = 18.0f32; // front facade distance from the street line
    let eps = 0.025f32; // every house shrinks 2.5 cm toward its centre, so
                        // neighbour wall planes part by ~5 cm: no z-fighting,
                        // no visible gaps

    // bilinear map of a local rectangle (0..w, 0..d) onto the quad c
    let bilinear = |c: &[[f32; 2]; 4], w: f32, d: f32, lx: f32, lz: f32| -> [f32; 2] {
        let u = lx / w;
        let t = lz / d;
        [
            (1.0 - u) * (1.0 - t) * c[0][0]
                + u * (1.0 - t) * c[1][0]
                + (1.0 - u) * t * c[2][0]
                + u * t * c[3][0],
            (1.0 - u) * (1.0 - t) * c[0][1]
                + u * (1.0 - t) * c[1][1]
                + (1.0 - u) * t * c[2][1]
                + u * t * c[3][1],
        ]
    };
    // widths of a row: 12..24 m houses up to a 26 m closing piece, summing
    // exactly to `run` — no leftovers, no gaps
    let plan_widths = |rng: &mut kolomna_render::procgen::Rng, run: f32| -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        let mut rest = run;
        while rest > 26.0 {
            let w = loop {
                let w = [12.0f32, 16.0, 20.0, 24.0][rng.below(4) as usize];
                if rest - w >= 10.0 {
                    break w;
                }
            };
            out.push(w);
            rest -= w;
        }
        out.push(rest);
        out
    };

    let usr = q_user.is_some();
    let s: f32 = if usr { 1.0 } else if blk % 2 == 0 { 1.0 } else { -1.0 };
    let u_off = if usr { 0.0 } else { ((blk / 2) as f32 - 2.0) * 56.0 };
    let seed_b = if usr { 90_000 + blk * 131 } else { 9000 + blk * 131 };
    let mut rng = kolomna_render::procgen::Rng::new(seed_b + 7);
    // a separate stream for the roster dice, so the bay/arch roll can
    // never shift the layout the block planner rolls from `rng`
    let mut brng = kolomna_render::procgen::Rng::new(seed_b + 501);

    // the block plan: an irregular polygon (frame: a along the street, b
    // inward), every vertex jittered — never a neat rectangle.  A user
    // block (blk >= 10) arrives exactly as the editor drew it — any
    // polygon with >= 3 vertices, already CCW in this same plan frame.
    let mut q: Vec<[f32; 2]>;
    if usr {
        q = q_user.unwrap().to_vec();
    } else {
        let a_half = 20.0 + rng.below(5) as f32 * 0.5; // front half width 20–22 m
        let b_len = 44.0 + rng.below(13) as f32; // block depth 44–56 m
        q = vec![
            [u_off - a_half, setback],
            [u_off + a_half, setback],
            [u_off + a_half - 5.0, setback + b_len],
            [u_off - a_half + 5.0, setback + b_len],
        ];
        for p in q.iter_mut() {
            p[0] += (rng.f() - 0.5) * 9.0;
            p[1] += (rng.f() - 0.5) * 7.0;
        }
    }
    let n = q.len();
    if !usr {
        edge = Some((
            q[0][0].min(q[3][0]),
            q[1][0].max(q[2][0]),
            q[2][1].max(q[3][1]),
        ));
    }
    // side i runs q[i] -> q[i+1]: unit tangent, inward normal, length
    let side = |q: &[[f32; 2]], i: usize| -> ([f32; 2], [f32; 2], f32) {
        let a = q[i];
        let b = q[(i + 1) % q.len()];
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
        let t = [d[0] / l, d[1] / l];
        ([t[0], t[1]], [-t[1], t[0]], l)
    };
    let on_side = |q: &[[f32; 2]], i: usize, a: f32| -> [f32; 2] {
        let (t, _, _) = side(q, i);
        [q[i][0] + t[0] * a, q[i][1] + t[1] * a]
    };
    let lens: Vec<f32> = (0..n).map(|i| side(&q, i).2).collect();
    // courtyard corners: intersections of neighbouring inward lines
    let vv = inset_ring(&q, depth);
    // corner-house wings: [along the previous side (uB), along the next
    // side (uA)]; every side then keeps at least 6 m for its row
    let mut wings = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let ip = (i + n - 1) % n;
        // each side keeps >= 10 m for its own row: a wing never takes more
        // than half of (side - 10 m). The old one-sided cap left 4 m rows
        // and squeezed a 10 m house into them.
        // corner houses stay small (10–12 m): fat 11–16 m wings left the
        // street rows 8–14 m wide — too narrow for bay windows or a portico
        let cap0 = ((lens[ip] - 10.0) * 0.5).max(6.0);
        let cap1 = ((lens[i] - 10.0) * 0.5).max(6.0);
        wings[i][0] = (10.0 + rng.below(3) as f32).min(cap0).max(6.0);
        wings[i][1] = (10.0 + rng.below(3) as f32).min(cap1).max(6.0);
    }
    // the roads as drawn (smoothed) and the houses already standing:
    // both keep the filler from seating a house where they are
    let road_paths: Vec<Vec<[f32; 2]>> = roads
        .iter()
        .filter(|c| c.len() >= 2)
        .map(|c| kolomna_render::streets::smooth_path(c))
        .collect();
    // the parade (street) row faces the nearest road on drawn blocks;
    // the procedural gallery always parades on side 0 (its street side)
    let street_side = if usr {
        let mut best = (0usize, f32::INFINITY);
        for i in 0..n {
            let mid = on_side(&q, i, lens[i] * 0.5);
            let mut d = f32::INFINITY;
            for rp in &road_paths {
                let (_, dd) = kolomna_render::streets::near_poly(rp, mid);
                d = d.min(dd);
            }
            if d < best.1 {
                best = (i, d);
            }
        }
        best.0
    } else {
        0
    };

    // place one house: its plan quad c (block frame, metres) + requested
    // size; the mesh is generated, seated and mapped onto the quad.  A
    // slot the roadway covers — or one an earlier block already occupies —
    // is simply left empty: only the houses that fall on the road go.
    let mut place = |w_slot: f32,
                     d_slot: f32,
                     c: [[f32; 2]; 4],
                     arch: bool,
                     corner: bool,
                     roman: bool,
                     bay: bool,
                     seed: u64| {
        // the roads and the standing houses live in the plan frame; the
        // footprint c is in the block frame (b flipped by s on odd blocks)
        let cp: [[f32; 2]; 4] = [
            [c[0][0], s * c[0][1]],
            [c[1][0], s * c[1][1]],
            [c[2][0], s * c[2][1]],
            [c[3][0], s * c[3][1]],
        ];
        if !road_paths.is_empty() || !avoid.is_empty() {
            let mut hit = false;
            for rp in &road_paths {
                if quad_path_dist(&cp, rp) < ROAD_CUT {
                    hit = true;
                    break;
                }
            }
            if !hit {
                hit = avoid.iter().any(|a| quads_overlap(&cp, a));
            }
            if hit {
                return;
            }
        }
        houses.push(cp);
        let req_w = (w_slot / 2.0).round().clamp(5.0, 15.0) * 2.0;
        let req_d = (d_slot / 2.0).round().clamp(4.0, 9.0) * 2.0;
        let (mut mesh, texs, wa, da, archr) = kolomna_render::piter::generate_manor(
            seed,
            arch,
            Some((req_w, req_d)),
            corner,
            roman,
            bay,
        );
            // block frame -> world metres relative to the placement anchor
            let to_wm = |p: [f32; 2]| -> [f32; 2] {
                [
                    eu[0] * p[0] + nvec[0] * s * p[1],
                    eu[1] * p[0] + nvec[1] * s * p[1],
                ]
            };
            // pull the corners slightly toward the centre (see `eps`)
            let ctr = [
                (c[0][0] + c[1][0] + c[2][0] + c[3][0]) * 0.25,
                (c[0][1] + c[1][1] + c[2][1] + c[3][1]) * 0.25,
            ];
            let mut cq = c;
            for p in cq.iter_mut() {
                let dx_ = p[0] - ctr[0];
                let dz_ = p[1] - ctr[1];
                let l = (dx_ * dx_ + dz_ * dz_).sqrt().max(1e-4);
                p[0] -= dx_ / l * eps;
                p[1] -= dz_ / l * eps;
            }
            // seat on the highest ground under the footprint
            let mut h_max = f32::NEG_INFINITY;
            let mut h_min = f32::INFINITY;
            for (lx, lz) in [
                (0.0f32, 0.0f32),
                (wa, 0.0),
                (0.0, da),
                (wa, da),
                (wa * 0.5, da * 0.5),
            ] {
                let wm = to_wm(bilinear(&cq, wa, da, lx, lz));
                let h = h_at(anchor[0] + wm[0] / 16.0, anchor[1] + wm[1] / 16.0);
                h_max = h_max.max(h);
                h_min = h_min.min(h);
            }
            let seat = (h_max + 0.05).max(wl + 2.0);
            let low = |lx: f32, lz: f32| -> f32 {
                let wm = to_wm(bilinear(&cq, wa, da, lx, lz));
                h_at(anchor[0] + wm[0] / 16.0, anchor[1] + wm[1] / 16.0) - seat
            };
            let mut pl = kolomna_render::piter::manor_plinth(wa, da, &low, archr);
            let mut cw: [[f32; 2]; 4] = [[0.0; 2]; 4];
            for k in 0..4 {
                cw[k] = to_wm(cq[k]);
            }
            kolomna_render::piter::deform(&mut mesh, wa, da, cw);
            kolomna_render::piter::deform(&mut pl, wa, da, cw);
            let base_tex = texs[kolomna_render::piter::M_BASE as usize].clone();
            let at = [anchor[0], anchor[1], (seat - mn) * ZSCALE];
            if want_focus && blk == focus_blk && focus.is_none() {
                // aim the --shot view at THIS house's centre: `at` is the
                // shared plan anchor, the same point for every block
                let cw = to_wm(ctr);
                focus = Some([
                    anchor[0] + cw[0] / 16.0,
                    anchor[1] + cw[1] / 16.0,
                    (seat - mn) * ZSCALE,
                ]);
            }
            village.push((pl, vec![base_tex], at, 0.0));
            village.push((mesh, texs, at, 0.0));
            report.push((wa, da, h_max - h_min));
        };

    // rows along every side, between the corner houses
    for i in 0..n {
        let inx = (i + 1) % n;
        let a_start = wings[i][1];
        let run = lens[i] - wings[i][1] - wings[inx][0];
        if run < 6.0 {
            continue; // a short side belongs entirely to its corner houses
        }
        let widths = plan_widths(&mut rng, run);
        let k = widths.len();
        // The STREET row (side 0) must open at least one carriage arch
        // into the yard, and its widest facade gets the Roman portico
        // house. Both are chosen by width so they always fit.
        let mut archj = vec![false; k];
        let mut romanj = vec![false; k];
        let mut bayj = vec![false; k];
        if i == street_side {
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| widths[b].partial_cmp(&widths[a]).unwrap());
            // the roster keeps its deterministic core (the roman portico on
            // the widest facade of a roman block, the carriage arch of the
            // plain blocks) while the row mixes variants on top: a bay
            // block may carry TWO bay facades side by side, a plain block
            // occasionally gains one on a narrower piece — never the arch
            // candidate (a bay house carries no parade entrance)
            let kind = blk % 5;
            let rom = if kind == 0 || kind == 2 {
                order.iter().copied().find(|&j| widths[j] >= 14.0)
            } else {
                None
            };
            if let Some(r) = rom {
                romanj[r] = true;
            }
            let bays_ok = rom.is_none() && std::env::var("KOLOMNA_NOBAYS").is_err();
            if bays_ok && (kind == 1 || kind == 3) {
                for &j in order.iter().filter(|&&j| widths[j] >= 14.0).take(2) {
                    bayj[j] = true;
                }
            } else if bays_ok && kind == 4 && order.len() >= 2 && brng.f() < 0.35 {
                if let Some(b) = order[1..].iter().copied().find(|&j| widths[j] >= 14.0) {
                    bayj[b] = true;
                }
            }
            // a house with a carriage arch carries no parade entrance:
            // the arch goes only to a PLAIN street piece — a roman/bay
            // house never doubles as the arch house, and on k=1 parade
            // blocks the courtyard stays archless
            if rom.is_none() {
                if let Some(p) = order.iter().copied().find(|&j| widths[j] >= 12.0 && !bayj[j]) {
                    archj[p] = true;
                }
            }
            let ap: Vec<usize> = (0..k).filter(|&j| archj[j]).collect();
            let rp: Vec<usize> = (0..k).filter(|&j| romanj[j]).collect();
            let bp: Vec<usize> = (0..k).filter(|&j| bayj[j]).collect();
            println!("block {blk} street row: arches at {ap:?}, roman at {rp:?}, bays at {bp:?} (k={k})");
        }
        let (_, inn, _) = side(&q, i);
        let mut a0 = a_start;
        for j in 0..k {
            let a1 = a0 + widths[j];
            let p0 = on_side(&q, i, a0);
            let p1 = on_side(&q, i, a1);
            let mut c = [
                [p0[0] + inn[0] * depth, p0[1] + inn[1] * depth],
                [p1[0] + inn[0] * depth, p1[1] + inn[1] * depth],
                p0,
                p1,
            ];
            if j == 0 {
                c[0] = vv[i]; // slanted end on the corner house's face
            }
            if j == k - 1 {
                c[1] = vv[inx];
            }
            place(
                widths[j],
                depth,
                c,
                archj[j],
                false,
                romanj[j],
                bayj[j],
                seed_b + 1000 + 97 * i as u64 + 31 * j as u64,
            );
            a0 = a1;
        }
    }
    // corner houses: irregular quads, both street edges get parade walls
    for i in 0..n {
        let ip = (i + n - 1) % n;
        let (u_b, u_a) = (wings[i][0], wings[i][1]);
        let pl_pt = on_side(&q, ip, lens[ip] - u_b);
        let pf_pt = on_side(&q, i, u_a);
        let c = [vv[i], pl_pt, pf_pt, q[i]];
        place(u_a, u_b, c, false, true, false, false, seed_b + 500 + i as u64);
    }

    // keep the plan outlines for the boundary overlay drawn in the app
    let to_wm_b = |p: [f32; 2]| -> [f32; 2] {
        [
            eu[0] * p[0] + nvec[0] * s * p[1],
            eu[1] * p[0] + nvec[1] * s * p[1],
        ]
    };
    let plan_out = (
        q.iter().map(|&p| to_wm_b(p)).collect::<Vec<_>>(),
        vv.iter().map(|&p| to_wm_b(p)).collect::<Vec<_>>(),
    );
    // the ring in the plan frame (the frame the roads are drawn in)
    let q_plan: Vec<[f32; 2]> = q.iter().map(|p| [p[0], s * p[1]]).collect();
    BlkBuilt {
        meshes: village,
        plan: plan_out,
        edge,
        report,
        focus,
        q_plan,
        houses,
    }
}

// ------------------------------------------------------- city edit helpers --

/// Metres from a road centre line that a road clears of houses.
const ROAD_CUT: f32 = 6.4;
/// Fence distance from a road centre line for the parks a road splits.
const PARK_MARGIN: f32 = 6.9;

/// Min distance between two segments (standard clamped closest points).
fn seg_seg_dist(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> f32 {
    let d = |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let clamp01 = |v: f32| v.clamp(0.0, 1.0);
    let d1 = [a1[0] - a0[0], a1[1] - a0[1]];
    let d2 = [b1[0] - b0[0], b1[1] - b0[1]];
    let r = [a0[0] - b0[0], a0[1] - b0[1]];
    let a = d1[0] * d1[0] + d1[1] * d1[1];
    let e = d2[0] * d2[0] + d2[1] * d2[1];
    let f = d2[0] * r[0] + d2[1] * r[1];
    let (mut s, mut t);
    if a <= 1e-8 && e <= 1e-8 {
        return d(a0, b0);
    }
    if a <= 1e-8 {
        s = 0.0;
        t = clamp01(f / e);
    } else {
        let c = d1[0] * r[0] + d1[1] * r[1];
        if e <= 1e-8 {
            t = 0.0;
            s = clamp01(-c / a);
        } else {
            let b = d1[0] * d2[0] + d1[1] * d2[1];
            let den = a * e - b * b;
            s = if den.abs() > 1e-8 { clamp01((b * f - c * e) / den) } else { 0.0 };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = clamp01(-c / a);
            } else if t > 1.0 {
                t = 1.0;
                s = clamp01((b - c) / a);
            }
        }
    }
    let p = [a0[0] + d1[0] * s, a0[1] + d1[1] * s];
    let q = [b0[0] + d2[0] * t, b0[1] + d2[1] * t];
    d(p, q)
}

/// Distance from a house footprint (closed quad) to a road centre line.
fn quad_path_dist(quad: &[[f32; 2]; 4], path: &[[f32; 2]]) -> f32 {
    let mut best = f32::MAX;
    for k in 0..4 {
        let a = quad[k];
        let b = quad[(k + 1) % 4];
        for w in path.windows(2) {
            best = best.min(seg_seg_dist(a, b, w[0], w[1]));
        }
    }
    best
}

/// Two quads really overlap when they overlap by > 15 cm on every axis:
/// touching facades of neighbouring houses do not count.
fn quads_overlap(a: &[[f32; 2]; 4], b: &[[f32; 2]; 4]) -> bool {
    const MIN_DEPTH: f32 = 0.15;
    let proj = |poly: &[[f32; 2]; 4], n: [f32; 2]| -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for p in poly {
            let s = p[0] * n[0] + p[1] * n[1];
            lo = lo.min(s);
            hi = hi.max(s);
        }
        (lo, hi)
    };
    for poly in [a, b] {
        for k in 0..4 {
            let p0 = poly[k];
            let p1 = poly[(k + 1) % 4];
            let l = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt().max(1e-6);
            let n = [(p1[1] - p0[1]) / l, -(p1[0] - p0[0]) / l];
            let (al, ah) = proj(a, n);
            let (bl, bh) = proj(b, n);
            if ah.min(bh) - al.max(bl) <= MIN_DEPTH {
                return false; // a separating axis: no real overlap
            }
        }
    }
    true
}

fn point_in_ring(p: [f32; 2], ring: &[[f32; 2]]) -> bool {
    let mut inside = false;
    for k in 0..ring.len() {
        let a = ring[k];
        let b = ring[(k + 1) % ring.len()];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + (b[0] - a[0]) * t {
                inside = !inside;
            }
        }
    }
    inside
}

/// Does any drawn road reach this ring: cross it, end inside it, or run
/// close enough that a house at the ring could stand on the roadway?
fn ring_hits_roads(ring: &[[f32; 2]], paths: &[Vec<[f32; 2]>], reach: f32) -> bool {
    for path in paths {
        if path.len() < 2 {
            continue;
        }
        for w in path.windows(2) {
            for k in 0..ring.len() {
                let a = ring[k];
                let b = ring[(k + 1) % ring.len()];
                if seg_seg_dist(a, b, w[0], w[1]) < reach {
                    return true;
                }
            }
        }
        if point_in_ring(path[0], ring) || point_in_ring(path[path.len() - 1], ring) {
            return true;
        }
    }
    false
}

/// The drawn chains as the smoothed centre lines the mesh follows.
fn smooth_roads(chains: &[Vec<[f32; 2]>]) -> Vec<Vec<[f32; 2]>> {
    chains
        .iter()
        .filter(|c| c.len() >= 2)
        .map(|c| kolomna_render::streets::smooth_path(c))
        .collect()
}

/// A zero-area placeholder entry: fills a village slot without drawing
/// anything.  Rebuilt blocks and parks blank their stale entries and
/// append the fresh ones, so every other index stays valid.
fn blank_mesh() -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>) {
    let mut b = kolomna_render::procgen::Builder::new();
    b.tri(
        0,
        [
            [0.0, 0.0, -900.0],
            [0.0, 0.0, -900.0],
            [0.0, 0.0, -900.0],
        ],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    );
    (b.finish(1), vec![(1u32, 1u32, vec![0u8, 0, 0, 255])])
}

/// The two-colour palette of the courtyard-boundary ribbons (red outer,
/// cyan inner); the ribbon uv picks the half.
fn plan_palette() -> Vec<(u32, u32, Vec<u8>)> {
    let mut pal_px = Vec::with_capacity(256 * 4);
    for x in 0..256usize {
        let c = if x < 128 {
            [220u8, 60, 60]
        } else {
            [60, 200, 220]
        };
        pal_px.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    vec![(256u32, 1u32, pal_px)]
}

/// Ground ribbons along block outlines for the app (outer plan in red,
/// the courtyard hole in cyan).  Rings arrive in world metres.
fn overlay_ribbons(
    rings: &[(Vec<[f32; 2]>, Vec<[f32; 2]>)],
    anchor: [f32; 2],
    h_at: &dyn Fn(f32, f32) -> f32,
    mn: f32,
) -> ModelMesh {
    let mut bld = kolomna_render::procgen::Builder::new();
    for (outer, inner) in rings {
        for (ring, uv, off) in [
            (outer, [[0.25f32, 0.25f32]; 4], 0.9f32),
            (inner, [[0.75f32, 0.25f32]; 4], -0.6f32),
        ] {
            for i in 0..ring.len() {
                let p0 = ring[i];
                let p1 = ring[(i + 1) % ring.len()];
                let d = [p1[0] - p0[0], p1[1] - p0[1]];
                let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-3);
                let (tx, ty) = (d[0] / l, d[1] / l);
                // ccw ring: outward = (ty, -tx); outer lines go out, inner
                // ones into the hole
                let n_off = [ty * off, -tx * off];
                let perp = [-ty * 0.16, tx * 0.16]; // 32 cm wide ribbon
                let n_seg = (l / 1.5).ceil().max(1.0) as usize;
                for s in 0..n_seg {
                    let t0 = s as f32 / n_seg as f32;
                    let t1 = (s + 1) as f32 / n_seg as f32;
                    let a = [p0[0] + d[0] * t0 + n_off[0], p0[1] + d[1] * t0 + n_off[1]];
                    let e = [p0[0] + d[0] * t1 + n_off[0], p0[1] + d[1] * t1 + n_off[1]];
                    let hh = |p: [f32; 2]| -> f32 {
                        h_at(anchor[0] + p[0] / 16.0, anchor[1] + p[1] / 16.0) - mn + 0.35
                    };
                    let (ya, ye) = (hh(a), hh(e));
                    let ax = anchor[0] * 16.0 + a[0];
                    let az = anchor[1] * 16.0 + a[1];
                    let ex = anchor[0] * 16.0 + e[0];
                    let ez = anchor[1] * 16.0 + e[1];
                    bld.quad(
                        0,
                        [
                            [ax + perp[0], ya, az + perp[1]],
                            [ex + perp[0], ye, ez + perp[1]],
                            [ex - perp[0], ye, ez - perp[1]],
                            [ax - perp[0], ya, az - perp[1]],
                        ],
                        uv,
                    );
                }
            }
        }
    }
    bld.finish(1)
}

pub fn load_scene(user: &UserPlan, skip_proc: bool) -> Result<Scene, Box<dyn Error>> {
    // fully procedural terrain — nothing external involved
    let seed: u64 = std::env::var("KOLOMNA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_1957);
    let world = kolomna_render::terrain_gen::generate(seed);
    let terrain = world.coarse;
    let cn = grid_side(&terrain);
    if cn * cn != terrain.len() {
        return Err(format!("terrain grid is not square ({} values)", terrain.len()).into());
    }
    let mn = world.mn;
    let mx = world.mx;
    let wl = world.wl;
    println!(
        "terrain: {cn}x{cn} generated (seed {seed:#x}), altitude {mn:.1}..{mx:.1} m, water surface {wl:.2} m"
    );

    let atlas = kolomna_render::terrain_gen::terrain_textures();
    println!("atlas: 6 procedural textures -> 256x256 px layers");

    // (the terrain mesh is built after the tenement pads are levelled below)

    // ---- plan frame: the street runs west -> east along the river bank ---
    let anchor = [64.0f32, 64.0f32];
    println!(
        "city plan anchor at cell ({:.1}, {:.1}) — the street runs along +x",
        anchor[0], anchor[1]
    );
    let dx = 1.0f32;
    let dz = 0.0f32;

    // ---- terrain field + mesh (the houses adapt to it, not vice versa) ---
    // the render mesh samples the generator at 4 m so slopes stay smooth
    let base = build_mesh(&world.fine, mn, mx, wl);
    // The height field EVERY consumer samples (roads, houses, gardens, plan
    // rings, the camera raycast): the same 4 m grid `build_mesh` smooths and
    // renders — NOT the 16 m `world.coarse` copy, which misses metre-scale
    // relief. Roads posed on the coarse field sat up to ~2 m off the drawn
    // surface on hills: terrain punched through the asphalt (holes) and the
    // kerbs floated between section stations.
    let sm = smooth_field(&world.fine, wl);
    let n = grid_side(&sm);
    // coarse-cell (16 m grid) coordinates -> `sm` cell coordinates
    let kf = (n - 1) as f32 / (cn - 1) as f32;
    println!("mesh: {} triangles", base.len() / 3);

    // ---- two Petersburg courtyards (дворы-колодцы) beside the street ------
    // The PLAN comes first: each block is an irregular quad. Houses are then
    // inscribed along its sides with no gaps: four corner houses own the
    // corners (irregular quads, two street facades each), rows of rectangles
    // fill the sides, and the pieces next to a corner house are trapezoids
    // whose slanted ends lie exactly on the corner-house faces. Every mesh is
    // mapped onto its footprint quad by a bilinear transform, so the
    // rectangular generator output follows any four-sided plan. Houses sit on
    // the HIGHEST ground under their footprint and get a ground-following
    // plinth, so the slope neither crops walls nor leaves voids.
    let h_at = |x: f32, z: f32| -> f32 {
        sample_h(
            &sm,
            n,
            x.clamp(0.0, (cn - 1) as f32) * kf,
            z.clamp(0.0, (cn - 1) as f32) * kf,
        )
    };
    let nvec = [-dz, dx]; // road normal
    let eu = [dx, dz]; // along the street
    let mut village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)> = Vec::new();
    let mut roads_idx: Option<usize> = None;
    let mut report: Vec<(f32, f32, f32)> = Vec::new(); // w, d, fall

    // ten courtyards: a gallery of blocks along the street line, five columns
    // to a side, so several plan variants can be compared side by side
    let mut plan_pts: Vec<(Vec<[f32; 2]>, Vec<[f32; 2]>)> = Vec::new();
    let mut focus: Option<[f32; 3]> = None;
    // debug: aim the --shot view at an arbitrary plan point (u,v in metres)
    if let Ok(s) = std::env::var("KOLOMNA_FOCUS_PLAN") {
        if let Some((a, b)) = s.split_once(',') {
            if let (Ok(u), Ok(v)) = (a.trim().parse::<f32>(), b.trim().parse::<f32>()) {
                let w = [eu[0] * u + nvec[0] * v, eu[1] * u + nvec[1] * v];
                let hh = h_at(anchor[0] + w[0] / 16.0, anchor[1] + w[1] / 16.0);
                focus = Some([
                    anchor[0] + w[0] / 16.0,
                    anchor[1] + w[1] / 16.0,
                    (hh - mn) * ZSCALE,
                ]);
            }
        }
    }
    // which block anchors the --shot view (KOLOMNA_FOCUS_BLK overrides)
    let focus_blk: u64 = std::env::var("KOLOMNA_FOCUS_BLK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let mut blk_edges: Vec<(f32, f32, f32)> = Vec::new(); // per block: u_min, u_max, v_max
    let mut blk_records: Vec<BlkRec> = Vec::new();
    let mut park_records: Vec<ParkRec> = Vec::new();
    let n_proc = if skip_proc { 0 } else { 10 };
    // the houses standing so far in the plan frame: a courtyard drawn
    // over another adds its zone and builds only the missing houses
    let mut avoid: Vec<HouseQuad> = Vec::new();
    for blk in 0..(n_proc + user.blocks.len()) as u64 {
        let usr = blk >= n_proc as u64;
        let q_user = if usr {
            Some(user.blocks[(blk - n_proc as u64) as usize].as_slice())
        } else {
            None
        };
        let built = build_block(
            blk,
            q_user,
            &h_at,
            anchor,
            eu,
            nvec,
            mn,
            wl,
            focus.is_none(),
            focus_blk,
            &user.roads,
            &avoid,
        );
        let first = village.len();
        village.extend(built.meshes);
        plan_pts.push(built.plan);
        if let Some(e) = built.edge {
            blk_edges.push(e);
        }
        report.extend(built.report);
        if let Some(f) = built.focus {
            focus = Some(f);
        }
        if usr {
            avoid.extend_from_slice(&built.houses);
        }
        blk_records.push(BlkRec {
            blk,
            q: built.q_plan,
            usr,
            entries: (first..village.len()).collect(),
            houses: built.houses,
        });
    }
    // ---- procedural modern streets ---------------------------------------
    // One street along the block fronts, with cross lanes through the gaps
    // between block columns: asphalt roadway, granite kerbs, tiled
    // sidewalks, a dashed centre line and zebra crossings — all conforming
    // to the terrain. Pushed as one world-space mesh (like the overlay).
    let mut net = kolomna_render::streets::Params::default();
    let mut streets_idx = usize::MAX;
    if !skip_proc {
        let mut cols = [[f32::INFINITY, f32::NEG_INFINITY, 0.0f32]; 5];
        for (b, &(l, r, vm)) in blk_edges.iter().enumerate() {
            let col = b / 2;
            cols[col][0] = cols[col][0].min(l);
            cols[col][1] = cols[col][1].max(r);
            cols[col][2] = cols[col][2].max(vm);
        }
        // each cross lane is sized to the gap it runs through: 5.5 m wide
        // in a roomy gap, 4.5 m in a tight one, 3.8 m in a sliver
        let mut cross: Vec<(f32, f32, f32)> = Vec::new();
        let mut gaps: Vec<String> = Vec::new();
        for k in 0..4 {
            let (right, left) = (cols[k][1], cols[k + 1][0]);
            let g = left - right;
            gaps.push(format!("{g:.1}"));
            let uc = (right + left) * 0.5;
            if g >= 9.6 {
                cross.push((uc, 2.75, 4.7));
            } else if g >= 7.4 {
                cross.push((uc, 2.25, 3.6));
            } else if g >= 6.0 {
                cross.push((uc, 1.9, 2.95));
            }
        }
        println!("street gaps between columns: {}", gaps.join(", "));
        let main_t0 = cols.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min) - 8.0;
        let main_t1 = cols.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max) + 8.0;
        let cross_half = cols.iter().map(|c| c[2]).fold(0.0f32, f32::max) + 8.0;
        println!(
            "streets: main {main_t0:.0}..{main_t1:.0} m, {} cross lane(s) {:?}, reach {cross_half:.0} m",
            cross.len(),
            cross
                .iter()
                .map(|(c, cr, cw)| format!("({c:.0}: {:.1}m road)", cr * 2.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
        net = kolomna_render::streets::Params {
            main_t0,
            main_t1,
            cross_half,
            cross,
        };
        // the drawn roads are already known: their junctions cut the grid
        let cuts = kolomna_render::streets::net_cuts(&net, &user.roads);
        let (rmesh, rtexs) =
            kolomna_render::streets::build_streets(&h_at, anchor, mn, eu, nvec, &net, &cuts);
        streets_idx = village.len();
        village.push((rmesh, rtexs, [0.0, 0.0, 0.0], 0.0));

        // ---- the Summer Garden beyond the east end of the street ----------
        // an irregular polygon fenced with the Fельтен grille, with gates,
        // alleys, bosquets, fountains, busts, statues, vases, benches,
        // lanterns and urns — all procedural and conforming to the terrain
        let (gmesh, gtexs) = kolomna_render::summer_garden::build_garden(
            &h_at,
            anchor,
            mn,
            eu,
            nvec,
            &kolomna_render::summer_garden::Params {
                centre: [212.0, 0.0],
                ru: 60.0,
                rv: 46.0,
                seed: 0x5EED_6A2D,
                outline: Vec::new(),
            },
        );
        println!("summer garden: fence around an irregular outline, 2 gates");
        village.push((gmesh, gtexs, [0.0, 0.0, 0.0], 0.0));
    }

    // ---- the editor's drawings (roads / parks): built even when the
    // procedural city is skipped ("user drawings only" preview mode)
    {
        // ---- the editor's roads: free chains with the street cross-section
        // (junctions trim kerbs / walks / furniture and extend T-attachments)
        if !user.roads.is_empty() {
            let (rmesh, rtexs) = kolomna_render::streets::build_roads(
                &h_at, anchor, mn, eu, nvec, &user.roads, &net,
            );
            println!("drawn roads: {} chain(s)", user.roads.len());
            roads_idx = Some(village.len());
            village.push((rmesh, rtexs, [0.0, 0.0, 0.0], 0.0));
        }
        // ---- the editor's parks: the drawn contour is the outline; a
        // road running through a park cuts it into separate gardens, each
        // fenced on its own (a park drawn over a road splits the same way:
        // the road stays, the green parts around it remain)
        for (pi, park) in user.parks.iter().enumerate() {
            let parts = kolomna_render::streets::split_outline_by_chains(
                park,
                &user.roads,
                PARK_MARGIN,
            );
            let first = village.len();
            for (pj, part) in parts.iter().enumerate() {
                let (pmesh, ptexs) = kolomna_render::summer_garden::build_garden(
                    &h_at,
                    anchor,
                    mn,
                    eu,
                    nvec,
                    &kolomna_render::summer_garden::Params {
                        centre: [0.0, 0.0],
                        ru: 0.0,
                        rv: 0.0,
                        seed: 0xD00D_0000 + (pi * 16 + pj) as u64,
                        outline: part.clone(),
                    },
                );
                village.push((pmesh, ptexs, [0.0, 0.0, 0.0], 0.0));
            }
            println!(
                "drawn park {}: {} outline points -> {} garden(s)",
                pi + 1,
                park.len(),
                parts.len()
            );
            park_records.push(ParkRec {
                outline: park.clone(),
                entries: (first..village.len()).collect(),
            });
        }
    }

    for (i, (w, d, dh)) in report.iter().enumerate() {
        println!("tenement {i:02}: {w:.0}x{d:.0} m, slope {dh:.2} m");
    }
    println!("village: {} Petersburg tenements", report.len());

    // ---- courtyard boundary overlay ---------------------------------------
    // thin ground ribbons along every block's outer plan (red) and its inner
    // courtyard hole (cyan); the app toggles them with the button / key B.
    // Outer lines are pushed ~0.9 m out and inner ones ~0.6 m into the
    // courtyard so they stay visible on the ground instead of hiding under
    // the walls. One mesh, one 256x1 two-colour texture, uv picks the colour.
    let plan_idx = if plan_pts.is_empty() {
        usize::MAX
    } else {
        let i = village.len();
        village.push((
            overlay_ribbons(&plan_pts, anchor, &h_at, mn),
            plan_palette(),
            [0.0, 0.0, 0.0],
            0.0,
        ));
        i
    };
    let focus = focus.unwrap_or([anchor[0], anchor[1], 0.0]);

    Ok(Scene {
        base,
        atlas,
        grid: (cn - 1) as f32,
        zspan: (mx - mn) * ZSCALE,
        mn,
        wl,
        sm,
        n,
        kf,
        anchor,
        eu,
        nvec,
        village,
        plan_idx,
        roads_idx,
        streets_idx,
        net,
        blk_records,
        park_records,
        focus,
    })
}

/// Headless mode: render exactly one frame offscreen with the default camera
/// (`zoom` scales the framing; 1.0 = whole grid) and write it as a PNG.
pub fn shot(out: &str, zoom: f32) -> Result<(), Box<dyn Error>> {
    // hidden headless test hatch (KOLOMNA_PLAN_DEMO=1): a demo drawing so
    // the editor's generators can be exercised without a mouse — a road
    // chain, a courtyard quad and a park contour, all in plan coords
    let user = if std::env::var("KOLOMNA_PLAN_DEMO").is_ok() {
        UserPlan {
            roads: vec![
                vec![[-40.0, 130.0], [40.0, 150.0], [108.0, 125.0], [95.0, 55.0]],
                // a diagonal crossing the main street and the first chain:
                // junction test coverage inside the demo
                vec![[-30.0, -30.0], [30.0, 30.0]],
                // roads right through the drawn courtyards and the drawn
                // park: house clearing + garden splitting on show
                vec![[330.0, 30.0], [335.0, 130.0]],
                vec![[150.0, 60.0], [190.0, 210.0]],
                // a loop crossing itself: the self-crossing grows a full
                // four-arm junction (rounded corners, signals, zebras)
                vec![[-180.0, 40.0], [-120.0, 100.0], [-180.0, 150.0], [-250.0, 95.0], [-148.0, 41.0]],
                // a T: this stub's end lands on the first chain's leg
                vec![[60.0, 200.0], [80.5, 135.0]],
                // mostly re-covering the first chain: the duplicated part
                // must not be drawn again, only the fresh tail shows
                vec![[-40.5, 128.5], [40.5, 148.5], [95.0, 190.0]],
            ],
            blocks: vec![
                vec![[300.0, 60.0], [360.0, 55.0], [365.0, 105.0], [305.0, 110.0]],
                // a second courtyard overlapping the first: its zone is
                // added, the standing houses of the first are left alone
                vec![[320.0, 80.0], [385.0, 75.0], [390.0, 125.0], [325.0, 130.0]],
            ],
            parks: vec![vec![
                [150.0, 90.0],
                [220.0, 95.0],
                [235.0, 150.0],
                [200.0, 195.0],
                [140.0, 180.0],
                [125.0, 130.0],
            ]],
        }
    } else {
        UserPlan::default()
    };
    let only = std::env::var("KOLOMNA_ONLY_USER").is_ok();
    let scene = load_scene(&user, only)?;
    let mut placed = Vec::new();
    for (mesh, texs, pos, rot) in &scene.village {
        placed.push(PlacedModel {
            mesh,
            textures: texs,
            pos: *pos,
            rot: *rot,
            spin: 0.0,
        });
    }
    let tilt = std::env::var("KOLOMNA_TILT")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let angle = std::env::var("KOLOMNA_ANGLE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    // aim at the middle block's anchor house (the plan overlay sits last in
    // `village` and must not steal the focus)
    let p = scene.focus;
    // the shader rotates world points about the grid centre before the iso
    // projection — rotate the anchor the same way, or a frame with a non-zero
    // angle aims at the wrong place (empty ground).
    let c = scene.grid * 0.5;
    let (sa, ca) = angle.sin_cos();
    let d = [p[0] - c, p[1] - c];
    let rx = c + d[0] * ca - d[1] * sa;
    let rz = c + d[0] * sa + d[1] * ca;
    let view = Some((
        [
            (rx - rz) * 0.5,
            // + ~half the tenement height: keeps a ~17 m house centred
            (rx + rz) * (0.5 * tilt) - p[2] - 0.6,
        ],
        angle,
    ));
    let rgba = kolomna_render::model::render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &scene.atlas,
        &scene.base,
        scene.grid,
        scene.zspan,
        zoom,
        scene.mn,
        scene.wl,
        view,
        &placed,
        None,
        &[],
    )?;
    write_png(out, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {out}");
    Ok(())
}

// --------------------------------------------------------------- camera ----

struct Camera {
    /// camera target in iso screen space (the shader's `center` units)
    target: [f32; 2],
    zoom: f32,
    zoom_target: f32,
    rot: f32,
    rot_target: f32,
    /// last `rot` the pivot compensation was applied at
    rot_applied: f32,
    /// rotation pivot, cell coords (set under the cursor on middle-press)
    pivot: [f32; 2],
    /// grid side (cells) — pivot compensation needs it
    grid: f32,
    /// camera tilt = sin(elevation): 0.5 = 2:1 iso, 1.0 = top-down,
    /// 0.0 = straight ahead; vertical middle-drag drives it
    tilt: f32,
    tilt_target: f32,
    /// last `tilt` the pivot compensation was applied at
    tilt_applied: f32,
    /// smoothed pan velocity (screen px/s)
    vel: [f32; 2],
    /// seconds since start (water shimmer)
    time: f32,
}

impl Camera {
    fn new(grid: f32, zspan: f32) -> Self {
        Self {
            target: [0.0, grid * 0.25 - zspan * 0.5],
            zoom: 1.0,
            zoom_target: 1.0,
            rot: 0.0,
            rot_target: 0.0,
            rot_applied: 0.0,
            tilt: 0.5,
            tilt_target: 0.5,
            tilt_applied: 0.5,
            pivot: [grid * 0.5, grid * 0.5],
            grid,
            vel: [0.0; 2],
            time: 0.0,
        }
    }

    fn reset(&mut self, grid: f32, zspan: f32) {
        *self = Self::new(grid, zspan);
    }

    fn zoom_by(&mut self, factor: f32) {
        self.zoom_target = (self.zoom_target * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    }

    fn rotate_by(&mut self, delta: f32) {
        self.rot_target += delta;
    }

    /// Middle-drag up/down: 0.5 = the classic 2:1 iso view, up to 1.0
    /// (straight down) and down to 0.0 (level, houses seen from the side).
    fn tilt_by(&mut self, delta: f32) {
        self.tilt_target = (self.tilt_target + delta).clamp(0.0, 1.0);
    }

    /// Pan by a screen-space mouse delta (px).
    fn pan_by_px(&mut self, dx: f32, dy: f32, scale: f32) {
        let s = scale.max(1e-3);
        self.target[0] -= dx / s;
        self.target[1] -= dy / s;
    }

    /// Advance smoothing; `base_scale` is the zoom=1 world->screen scale.
    fn step(
        &mut self,
        dt: f32,
        keys: &HashSet<KeyCode>,
        cursor: Option<(f64, f64)>,
        win: (f64, f64),
        base_scale: f32,
    ) {
        // desired pan velocity: keyboard + window edges
        let mut desired = [0.0f32; 2];
        if keys.contains(&KeyCode::ArrowLeft) || keys.contains(&KeyCode::KeyA) {
            desired[0] -= PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowRight) || keys.contains(&KeyCode::KeyD) {
            desired[0] += PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowUp) || keys.contains(&KeyCode::KeyW) {
            desired[1] -= PAN_PX_PER_S;
        }
        if keys.contains(&KeyCode::ArrowDown) || keys.contains(&KeyCode::KeyS) {
            desired[1] += PAN_PX_PER_S;
        }
        if let Some((x, y)) = cursor {
            let (w, h) = win;
            let fx = if x < EDGE_PX {
                -((1.0 - x / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else if x > w - EDGE_PX {
                ((1.0 - (w - x) / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else {
                0.0
            };
            let fy = if y < EDGE_PX {
                -((1.0 - y / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else if y > h - EDGE_PX {
                ((1.0 - (h - y) / EDGE_PX) as f32) * EDGE_MAX_PX_PER_S
            } else {
                0.0
            };
            desired[0] += fx;
            desired[1] += fy;
        }
        let k = 1.0 - (-8.0 * dt).exp();
        self.vel[0] += (desired[0] - self.vel[0]) * k;
        self.vel[1] += (desired[1] - self.vel[1]) * k;
        let s = (base_scale * self.zoom).max(1e-3);
        self.target[0] += self.vel[0] * dt / s;
        self.target[1] += self.vel[1] * dt / s;

        // eased zoom and rotation
        let kz = 1.0 - (-13.0 * dt).exp();
        self.zoom += (self.zoom_target - self.zoom) * kz;
        let kr = 1.0 - (-14.0 * dt).exp();
        self.rot += (self.rot_target - self.rot) * kr;
        // keep the rotation pivot pinned under the cursor: rotating the world
        // about the map centre moves every point on screen; shifting `target`
        // by the pivot's own screen displacement cancels that exactly at the
        // pivot, so the ground under the mouse stays put while rotating
        if (self.rot - self.rot_applied).abs() > 1e-6 {
            let c0 = self.grid * 0.5;
            let d = [self.pivot[0] - c0, self.pivot[1] - c0];
            let (s0, c0c) = self.rot_applied.sin_cos();
            let (s1, c1) = self.rot.sin_cos();
            let v0 = [d[0] * c0c - d[1] * s0, d[0] * s0 + d[1] * c0c];
            let v1 = [d[0] * c1 - d[1] * s1, d[0] * s1 + d[1] * c1];
            let dd = [v1[0] - v0[0], v1[1] - v0[1]];
            self.target[0] += (dd[0] - dd[1]) * 0.5;
            // sy of a rotated delta is (dx + dy) * (0.5 * tilt): the hardcoded
            // 0.25 only matched tilt = 0.5, so at any other tilt the pivot
            // drifted away from the cursor while rotating.
            let t = self.tilt.clamp(0.05, 1.0);
            self.target[1] += (dd[0] + dd[1]) * (0.5 * t);
            self.rot_applied = self.rot;
        }
        // eased tilt; the view reclines about the ground point under the
        // SCREEN CENTRE (the OrbitControls convention: the focus point keeps
        // its screen spot). Anchoring on the click pivot instead made close
        // views swing hard — any offset from the cursor got stretched by
        // (x + y) * dtilt and the whole street flew off at high zoom.
        let kt = 1.0 - (-14.0 * dt).exp();
        self.tilt += (self.tilt_target - self.tilt) * kt;
        if (self.tilt - self.tilt_applied).abs() > 1e-6 {
            // sy = (x + y) * 0.5 * tilt - z  =>  the ground point held in the
            // screen centre has (x + y) * 0.5 = target[1] / tilt
            let t0 = self.tilt_applied.max(0.07);
            let p = (self.target[1] / t0).clamp(-2.0 * self.grid, 2.0 * self.grid);
            self.target[1] += p * (self.tilt - self.tilt_applied);
            self.tilt_applied = self.tilt;
        }
        // keep navigation over the map: clamp the pan centre (a little
        // overscroll so the edge cliff stays reachable, never off into void)
        let m = 6.0f32;
        self.target[0] = self.target[0].clamp(-m, self.grid + m);
        self.target[1] = self.target[1].clamp(-m, self.grid * self.tilt.max(0.2) + m);
        self.time += dt;
    }

    fn params(&self, w: f32, h: f32, grid: f32, base_scale: f32) -> Params {
        Params {
            center: self.target,
            scale: base_scale * self.zoom,
            hz: 1.0,
            half: [w / 2.0, h / 2.0],
            angle: self.rot,
            time: self.time,
            pivot: grid * 0.5,
            // the water plane height is filled by TerrainRenderer::set_params
            _pad0: 0.0,
            tilt: self.tilt,
            _pad2: 0.0,
        }
    }

    /// Point the rotation pivot at the terrain under a screen position:
    /// invert the iso projection, then walk the view ray onto the smoothed
    /// height field (the mesh is drawn from the same field, so the pivot
    /// lands where the user clicked).
    fn set_pivot_from_screen(
        &mut self,
        scene: &Scene,
        px: f64,
        py: f64,
        win: (f64, f64),
        base_scale: f32,
    ) {
        let scale = (base_scale * self.zoom).max(1e-3);
        let a = self.target[0] + (px as f32 - win.0 as f32 * 0.5) / scale;
        let b = self.target[1] + (py as f32 - win.1 as f32 * 0.5) / scale;
        // Iso ray: x = s + a, y = s - a and sy = (x + y) * (0.5 * tilt) - z,
        // so s = (b + z(s)) / tilt.  The old code hardcoded tilt = 0.5 (s =
        // 2*(b+z)) which put the pivot at the wrong map point for every other
        // tilt — the view then orbited around a phantom axis.
        let t = self.tilt.clamp(0.07, 1.0);
        let smax = 2.0 * self.grid;
        let mut s = (b / t).clamp(0.0, smax);
        for _ in 0..20 {
            let cx = (s + a).clamp(0.0, self.grid);
            let cy = (s - a).clamp(0.0, self.grid);
            let zmesh = (hx(scene, cx, cy) - scene.mn) * ZSCALE;
            s = ((b + zmesh) / t).clamp(0.0, smax);
        }
        // The ray inversion yields camera-rotated ("rp") coordinates, but the
        // pivot compensation in `step` needs a world point: undo the yaw here,
        // otherwise every rotation after the first orbits a wrong spot.
        let rp = [s + a, s - a];
        let c = self.grid * 0.5;
        let (sr, cr) = self.rot.sin_cos();
        let d = [rp[0] - c, rp[1] - c];
        self.pivot = [
            (c + d[0] * cr + d[1] * sr).clamp(0.0, self.grid),
            (c - d[0] * sr + d[1] * cr).clamp(0.0, self.grid),
        ];
    }
}

/// Bilinear sample of the smoothed height grid (metres) at cell coordinates.
fn sample_h(sm: &[f32], n: usize, cx: f32, cy: f32) -> f32 {
    let x = cx.clamp(0.0, (n - 1) as f32);
    let y = cy.clamp(0.0, (n - 1) as f32);
    let x0 = (x as usize).min(n.saturating_sub(2));
    let y0 = (y as usize).min(n.saturating_sub(2));
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let h00 = sm[y0 * n + x0];
    let h10 = sm[y0 * n + x0 + 1];
    let h01 = sm[(y0 + 1) * n + x0];
    let h11 = sm[(y0 + 1) * n + x0 + 1];
    (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fy) + (h01 * (1.0 - fx) + h11 * fx) * fy
}

/// Height at COARSE-cell (16 m scene grid) coordinates, sampled from the very
/// field the render mesh is built from (the 4 m grid in `Scene.sm`). The 16 m
/// `world.coarse` copy misses metre-scale relief — roads/houses posed on it
/// punched holes and floated on uneven ground.
fn hx(scene: &Scene, cx: f32, cy: f32) -> f32 {
    let b = (scene.n - 1) as f32 / scene.kf;
    sample_h(
        &scene.sm,
        scene.n,
        cx.clamp(0.0, b) * scene.kf,
        cy.clamp(0.0, b) * scene.kf,
    )
}

// ------------------------------------------------------------- renderer ----

fn make_targets(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let msaa = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("app-msaa"),
        size,
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("app-depth"),
        size,
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    (
        msaa.create_view(&wgpu::TextureViewDescriptor::default()),
        depth.create_view(&wgpu::TextureViewDescriptor::default()),
    )
}

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    terrain: TerrainRenderer,
    village: Vec<ModelRenderer>,
    /// index of the courtyard boundary overlay inside `village`
    plan_idx: usize,
    /// index of the drawn-roads mesh inside `village`
    roads_idx: Option<usize>,
    /// index of the live (in-progress) stretch of road inside `village`
    roads_live_idx: Option<usize>,
    /// the editor's in-progress sketch (drawn over the ground)
    preview: Option<ModelRenderer>,
    /// the animated snap marker under the block/park cursor
    snap_marker: Option<ModelRenderer>,
    /// whether the snap marker should be drawn this frame
    snap_marker_on: bool,
    ui: UiRenderer,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    fn new(window: Arc<Window>, scene: &Scene) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no graphics adapter found: {e}"))?;
        let info = adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("kolomna-app"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;

        let caps = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("window surface is not supported by the adapter")?;
        // prefer an sRGB surface format so on-screen colors match the offscreen reference
        if let Some(f) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = f;
        }
        println!(
            "surface: {}x{} {:?} ({:?})",
            config.width, config.height, config.format, config.present_mode
        );
        surface.configure(&device, &config);

        let terrain = TerrainRenderer::new(
            &device,
            &queue,
            config.format,
            4,
            &scene.atlas,
            &scene.base,
            scene.mn,
            scene.wl,
        );
        let village: Vec<ModelRenderer> = scene
            .village
            .iter()
            .map(|(mesh, texs, pos, rot)| {
                let r = ModelRenderer::new(&device, &queue, config.format, 4, mesh, texs);
                r.set_placement(&queue, *pos, *rot, 0.0);
                r
            })
            .collect();
        let (msaa_view, depth_view) = make_targets(&device, &config);
        let ui = UiRenderer::new(&device, &queue, config.format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            terrain,
            village,
            plan_idx: scene.plan_idx,
            roads_idx: scene.roads_idx,
            roads_live_idx: None,
            preview: None,
            snap_marker: None,
            snap_marker_on: false,
            ui,
            msaa_view,
            depth_view,
        })
    }

    /// The editor's in-progress sketch: a flat orange ribbon over the
    /// ground (None clears it).  The renderer is created once and then only
    /// its geometry is swapped, so the sketch can follow the cursor.
    fn set_preview(&mut self, mesh: Option<kolomna_render::model::ModelMesh>) {
        let Some(m) = mesh else {
            self.preview = None;
            return;
        };
        if let Some(r) = self.preview.as_mut() {
            r.set_mesh(&self.device, &m);
            return;
        }
        let tex: Vec<u8> = [255u8, 150, 40, 255]
            .iter()
            .copied()
            .cycle()
            .take(4 * 4 * 4)
            .collect();
        let r = ModelRenderer::new(
            &self.device,
            &self.queue,
            self.config.format,
            4,
            &m,
            &[(4u32, 4u32, tex)],
        );
        r.set_placement(&self.queue, [0.0, 0.0, 0.0], 0.0, 0.0);
        self.preview = Some(r);
    }

    /// The animated ground marker under the block/park cursor (None hides
    /// it): created lazily, then only re-placed — the spin and the breath
    /// run in the shader (anim 4).
    fn set_snap_marker(&mut self, at: Option<[f32; 3]>) {
        match at {
            Some(pos) => {
                if self.snap_marker.is_none() {
                    let mesh = snap_marker_mesh();
                    let tex: Vec<u8> = [255u8, 182, 52, 255]
                        .iter()
                        .copied()
                        .cycle()
                        .take(4 * 4 * 4)
                        .collect();
                    let r = ModelRenderer::new(
                        &self.device,
                        &self.queue,
                        self.config.format,
                        4,
                        &mesh,
                        &[(4u32, 4u32, tex)],
                    );
                    self.snap_marker = Some(r);
                }
                if let Some(r) = self.snap_marker.as_ref() {
                    r.set_placement(&self.queue, pos, 0.0, 0.0);
                }
                self.snap_marker_on = true;
            }
            None => self.snap_marker_on = false,
        }
    }

    /// Swap in a freshly generated village (the editor's commits): the
    /// device, surface and pipelines all stay — only the model renderers
    /// and the overlay index are rebuilt.  (Recreating the whole renderer
    /// would create a second surface on the same window, which wgpu 30
    /// rejects with "Invalid surface".)
    fn rebuild_village(&mut self, scene: &Scene) {
        self.village = scene
            .village
            .iter()
            .map(|(mesh, texs, pos, rot)| {
                let r = ModelRenderer::new(&self.device, &self.queue, self.config.format, 4, mesh, texs);
                r.set_placement(&self.queue, *pos, *rot, 0.0);
                r
            })
            .collect();
        self.plan_idx = scene.plan_idx;
        self.roads_idx = scene.roads_idx;
    }

    /// Append a freshly generated model to the drawn village (the editor's
    /// commits) — returns its index for later replacement.
    fn add_village_mesh(
        &mut self,
        mesh: kolomna_render::model::ModelMesh,
        texs: Vec<(u32, u32, Vec<u8>)>,
        pos: [f32; 3],
        rot: f32,
    ) -> usize {
        let r = ModelRenderer::new(&self.device, &self.queue, self.config.format, 4, &mesh, &texs);
        r.set_placement(&self.queue, pos, rot, 0.0);
        self.village.push(r);
        self.village.len() - 1
    }

    /// Swap one drawn entry — the roads mesh is rebuilt on every road
    /// commit, so it must not pile up.
    fn replace_village_mesh(
        &mut self,
        idx: usize,
        mesh: kolomna_render::model::ModelMesh,
        texs: Vec<(u32, u32, Vec<u8>)>,
    ) {
        if idx < self.village.len() {
            let r =
                ModelRenderer::new(&self.device, &self.queue, self.config.format, 4, &mesh, &texs);
            r.set_placement(&self.queue, [0.0, 0.0, 0.0], 0.0, 0.0);
            self.village[idx] = r;
        }
    }

    /// zoom=1 world->screen scale for the current window size.
    fn base_scale(&self, scene: &Scene) -> f32 {
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let vspan = scene.grid * 0.5 + scene.zspan;
        ((w - 60.0) / scene.grid).min((h - 60.0) / vspan)
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        let (msaa_view, depth_view) = make_targets(&self.device, &self.config);
        self.msaa_view = msaa_view;
        self.depth_view = depth_view;
    }

    fn render(
        &mut self,
        scene: &Scene,
        cam: &Camera,
        ui_state: &UiState,
    ) -> Result<(), Box<dyn Error>> {
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        let params = cam.params(w, h, scene.grid, self.base_scale(scene));
        self.terrain.set_params(&self.queue, &params);
        for r in &self.village {
            r.set_params(&self.queue, &params);
        }
        if let Some(p) = &self.preview {
            p.set_params(&self.queue, &params);
        }
        if let Some(m) = &self.snap_marker {
            m.set_params(&self.queue, &params);
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("surface validation error".into());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("app-enc"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("app-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&view),
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
                    view: &self.depth_view,
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
            self.terrain.draw(&mut pass);
            for (idx, r) in self.village.iter().enumerate() {
                if !ui_state.show_plans && idx == self.plan_idx {
                    continue; // courtyard boundaries hidden
                }
                r.draw(&mut pass);
            }
            if let Some(p) = &self.preview {
                p.draw(&mut pass);
            }
            if self.snap_marker_on {
                if let Some(m) = &self.snap_marker {
                    m.draw(&mut pass);
                }
            }
            self.ui.draw(&self.queue, &mut pass, w, h, ui_state);
        }
        self.queue.submit(Some(enc.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

// ------------------------------------------------------------- ui: toolbar ----
// A row of tool buttons plus a bottom hint bar, drawn as textured quads from
// a baked atlas (Pixelify Sans glyphs + pixel icons, see `ui_atlas`).  The
// face is a pixel font, so the sampler is NEAREST and the atlas is 1:1.

use crate::ui_atlas;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    col: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiUniform {
    screen: [f32; 2],
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum UiAction {
    TogglePlans,
    Tool(u8),
}

struct UiButton {
    action: UiAction,
    label: &'static str,
    icon: &'static str,
}

const UI_BUTTONS: [UiButton; 5] = [
    UiButton { action: UiAction::TogglePlans, label: "Границы", icon: "grid" },
    UiButton { action: UiAction::Tool(0), label: "Обзор", icon: "eye" },
    UiButton { action: UiAction::Tool(1), label: "Дороги", icon: "road" },
    UiButton { action: UiAction::Tool(2), label: "Кварталы", icon: "quarter" },
    UiButton { action: UiAction::Tool(3), label: "Парки", icon: "park" },
];
const UI_BTN_H: f32 = 36.0;
const UI_BTN_Y: f32 = 12.0;
const UI_BTN_PAD: f32 = 10.0;
const UI_ICON: f32 = 24.0;
const UI_GAP: f32 = 6.0;
/// the atlas white patch: flat-coloured quads sample it
const UI_WHITE: [f32; 4] = [0.0, 0.0, 4.0, 4.0];

/// Everything the toolbar needs from the app for one frame.
struct UiState<'a> {
    tool: u8,
    show_plans: bool,
    hint: &'a str,
    cursor: Option<(f64, f64)>,
}

/// sRGB intent -> linear, because the surface is an sRGB target and the blend
/// happens in linear space.
fn ui_srgb(c: [f32; 4]) -> [f32; 4] {
    [
        c[0].powf(2.2),
        c[1].powf(2.2),
        c[2].powf(2.2),
        c[3],
    ]
}

fn ui_glyph(ch: char) -> Option<&'static (char, [f32; 4], f32, f32, f32)> {
    ui_atlas::GLYPHS.iter().find(|g| g.0 == ch)
}

/// Display size of the baked face (the atlas itself is FONT_PX tall).
const UI_TEXT_PX: f32 = 24.0;

fn ui_k() -> f32 {
    UI_TEXT_PX / ui_atlas::FONT_PX
}

fn ui_text_width(s: &str) -> f32 {
    let k = ui_k();
    s.chars()
        .map(|c| ui_glyph(c).map(|g| g.2).unwrap_or(6.0))
        .sum::<f32>()
        * k
}

/// One rect [x, y, w, h] per button, physical px, laid left to right.
fn ui_button_rects() -> Vec<([f32; 4], &'static UiButton)> {
    let mut x = 12.0;
    let mut out = Vec::with_capacity(UI_BUTTONS.len());
    for b in &UI_BUTTONS {
        let w = UI_BTN_PAD + UI_ICON + UI_GAP + ui_text_width(b.label) + UI_BTN_PAD;
        out.push(([x, UI_BTN_Y, w, UI_BTN_H], b));
        x += w + UI_GAP;
    }
    out
}

fn ui_hit(r: [f32; 4], px: f64, py: f64) -> bool {
    let (px, py) = (px as f32, py as f32);
    px >= r[0] && px <= r[0] + r[2] && py >= r[1] && py <= r[1] + r[3]
}

/// Which button (if any) is under the cursor.
fn ui_action_at(px: f64, py: f64) -> Option<UiAction> {
    ui_button_rects()
        .into_iter()
        .find(|(r, _)| ui_hit(*r, px, py))
        .map(|(_, b)| b.action)
}

fn ui_quad(out: &mut Vec<UiVertex>, r: [f32; 4], uv: [f32; 4], col: [f32; 4]) {
    let (aw, ah) = (ui_atlas::ATLAS_W as f32, ui_atlas::ATLAS_H as f32);
    let (u0, v0) = (uv[0] / aw, uv[1] / ah);
    let (u1, v1) = ((uv[0] + uv[2]) / aw, (uv[1] + uv[3]) / ah);
    let (x, y, w, h) = (r[0], r[1], r[2], r[3]);
    let p = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]];
    let t = [[u0, v0], [u1, v0], [u0, v1], [u1, v1]];
    for i in [0usize, 1, 2, 2, 1, 3] {
        out.push(UiVertex { pos: p[i], uv: t[i], col });
    }
}

/// Draw one line of text with the baked metrics (atlas is FONT_PX tall,
/// shown at UI_TEXT_PX) and a soft drop shadow underneath.
fn ui_push_text(out: &mut Vec<UiVertex>, x: f32, top: f32, s: &str, col: [f32; 4]) {
    let k = ui_k();
    let shadow = ui_srgb([0.03, 0.04, 0.06, 0.55]);
    for pass in 0..2 {
        let (ox, oy, c) = if pass == 0 { (2.0, 2.0, shadow) } else { (0.0, 0.0, col) };
        let mut pen = x;
        for ch in s.chars() {
            match ui_glyph(ch) {
                Some(g) => {
                    if g.1[2] > 0.0 {
                        ui_quad(
                            out,
                            [pen + g.3 * k + ox, top + g.4 * k + oy, g.1[2] * k, g.1[3] * k],
                            g.1,
                            c,
                        );
                    }
                    pen += g.2 * k;
                }
                None => pen += 6.0 * k,
            }
        }
    }
}

fn ui_build(w: f32, h: f32, st: &UiState) -> Vec<UiVertex> {
    let mut v: Vec<UiVertex> = Vec::with_capacity(2048);
    for (r, b) in ui_button_rects() {
        let hover = st.cursor.map(|(cx, cy)| ui_hit(r, cx, cy)).unwrap_or(false);
        let active = match b.action {
            UiAction::Tool(t) => t == st.tool,
            UiAction::TogglePlans => st.show_plans,
        };
        let bg = if active {
            ui_srgb([0.271, 0.200, 0.086, 0.96])
        } else if hover {
            ui_srgb([0.235, 0.255, 0.310, 0.96])
        } else {
            ui_srgb([0.102, 0.114, 0.145, 0.88])
        };
        let edge = if active {
            ui_srgb([0.870, 0.665, 0.286, 1.0])
        } else if hover {
            ui_srgb([0.560, 0.610, 0.700, 1.0])
        } else {
            ui_srgb([0.280, 0.310, 0.380, 1.0])
        };
        // border ring + inner fill
        ui_quad(&mut v, r, UI_WHITE, edge);
        ui_quad(
            &mut v,
            [r[0] + 1.0, r[1] + 1.0, r[2] - 2.0, r[3] - 2.0],
            UI_WHITE,
            bg,
        );
        // icon
        let iuv = ui_atlas::ICONS
            .iter()
            .find(|(n, _)| *n == b.icon)
            .map(|(_, r)| *r)
            .unwrap_or([0.0, 0.0, 4.0, 4.0]);
        ui_quad(
            &mut v,
            [
                r[0] + UI_BTN_PAD,
                (r[1] + (r[3] - UI_ICON) / 2.0).round(),
                UI_ICON,
                UI_ICON,
            ],
            iuv,
            [1.0, 1.0, 1.0, 1.0],
        );
        // label
        let tx = r[0] + UI_BTN_PAD + UI_ICON + UI_GAP;
        let ty =
            (r[1] + (r[3] - (ui_atlas::FONT_ASC + ui_atlas::FONT_DESC) * ui_k()) / 2.0).round();
        let tc = if active {
            ui_srgb([0.980, 0.930, 0.780, 1.0])
        } else {
            ui_srgb([0.870, 0.880, 0.900, 1.0])
        };
        ui_push_text(&mut v, tx, ty, b.label, tc);
    }
    // bottom-left hint bar for the active tool
    if !st.hint.is_empty() {
        let tw = ui_text_width(st.hint);
        let (pad, bh) = (10.0, 36.0);
        let r = [12.0, h - bh - 14.0, tw + pad * 2.0, bh];
        ui_quad(&mut v, r, UI_WHITE, ui_srgb([0.070, 0.078, 0.098, 0.80]));
        ui_quad(
            &mut v,
            [r[0], r[1], 3.0, r[3]],
            UI_WHITE,
            ui_srgb([0.870, 0.665, 0.286, 0.95]),
        );
        let ty =
            (r[1] + (r[3] - (ui_atlas::FONT_ASC + ui_atlas::FONT_DESC) * ui_k()) / 2.0).round();
        ui_push_text(&mut v, r[0] + pad, ty, st.hint, ui_srgb([0.92, 0.92, 0.90, 1.0]));
    }
    v
}

/// Six painter-order quads per vertex list, no buffers: positions+uv+colour.
const UI_SHADER: &str = r#"
struct Uni {
    screen: vec2<f32>,
};
@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) col: vec4<f32>,
};

@vertex
fn vs_ui(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) col: vec4<f32>,
) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos.x / uni.screen.x * 2.0 - 1.0, 1.0 - pos.y / uni.screen.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    o.col = col;
    return o;
}

@fragment
fn fs_ui(in: VOut) -> @location(0) vec4<f32> {
    return in.col * textureSample(tex, smp, in.uv);
}
"#;

struct UiRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vb: wgpu::Buffer,
    vcap: usize,
}

impl UiRenderer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui-shader"),
            source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()),
        });
        // the baked atlas (font glyphs + icons), NEAREST: pixel-crisp
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ui-atlas"),
            size: wgpu::Extent3d {
                width: ui_atlas::ATLAS_W,
                height: ui_atlas::ATLAS_H,
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
            ui_atlas::ATLAS_RAW,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ui_atlas::ATLAS_W * 4),
                rows_per_image: Some(ui_atlas::ATLAS_H),
            },
            wgpu::Extent3d {
                width: ui_atlas::ATLAS_W,
                height: ui_atlas::ATLAS_H,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui-bgl"),
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
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui-uniform"),
            size: std::mem::size_of::<UiUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let vcap = 4096;
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui-vb"),
            size: (vcap * std::mem::size_of::<UiVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ui"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(vbl)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ui"),
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
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform,
            bind_group,
            vb,
            vcap,
        }
    }

    fn draw(&self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'_>, w: f32, h: f32, st: &UiState) {
        let verts = ui_build(w, h, st);
        assert!(verts.len() <= self.vcap, "ui vertex buffer overflow");
        let u = UiUniform { screen: [w, h] };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(&self.vb, 0, bytemuck::cast_slice(&verts));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vb.slice(..));
        pass.draw(0..verts.len() as u32, 0..1);
    }
}

fn ui_srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn ui_linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Bilinear atlas fetch in linear space - mirrors the GPU, which linearises
/// the sRGB texture before filtering and blends in linear space too.
fn ui_fetch(u: f32, v: f32) -> [f32; 4] {
    let x = u * ui_atlas::ATLAS_W as f32 - 0.5;
    let y = v * ui_atlas::ATLAS_H as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i32, y0 as i32);
    let texel = |x: i32, y: i32| -> [f32; 4] {
        let x = x.clamp(0, ui_atlas::ATLAS_W as i32 - 1) as usize;
        let y = y.clamp(0, ui_atlas::ATLAS_H as i32 - 1) as usize;
        let o = (y * ui_atlas::ATLAS_W as usize + x) * 4;
        let p = &ui_atlas::ATLAS_RAW[o..o + 4];
        [
            ui_srgb_to_linear(p[0] as f32 / 255.0),
            ui_srgb_to_linear(p[1] as f32 / 255.0),
            ui_srgb_to_linear(p[2] as f32 / 255.0),
            p[3] as f32 / 255.0,
        ]
    };
    let (c00, c10, c01, c11) = (
        texel(x0, y0),
        texel(x0 + 1, y0),
        texel(x0, y0 + 1),
        texel(x0 + 1, y0 + 1),
    );
    let mut o = [0.0f32; 4];
    for c in 0..4 {
        let a = c00[c] + (c10[c] - c00[c]) * fx;
        let b = c01[c] + (c11[c] - c01[c]) * fx;
        o[c] = a + (b - a) * fy;
    }
    o
}

/// KOLOMNA_UI_SHOT=<png>: rasterize the toolbar on the CPU (no window) so the
/// layout and the atlas can be checked from the dev loop.
pub fn ui_shot(path: &str) {
    let st = UiState {
        tool: 2,
        show_plans: true,
        hint: "КВАРТАЛЫ: ≥3 клика · Enter — готово · ПКМ — отмена",
        cursor: Some((100.0, 30.0)),
    };
    let (w, h) = (980u32, 220u32);
    let verts = ui_build(w as f32, h as f32, &st);
    let mut px = vec![0u8; (w * h * 4) as usize];
    for p in px.chunks_mut(4) {
        p.copy_from_slice(&[153, 184, 219, 255]); // sky, like in the app
    }
    for q in verts.chunks(6) {
        let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        let (mut u0, mut u1, mut v0, mut v1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for vtx in q {
            x0 = x0.min(vtx.pos[0]);
            x1 = x1.max(vtx.pos[0]);
            y0 = y0.min(vtx.pos[1]);
            y1 = y1.max(vtx.pos[1]);
            u0 = u0.min(vtx.uv[0]);
            u1 = u1.max(vtx.uv[0]);
            v0 = v0.min(vtx.uv[1]);
            v1 = v1.max(vtx.uv[1]);
        }
        let col = q[0].col;
        for yy in (y0.max(0.0) as u32)..(y1.min(h as f32) as u32) {
            for xx in (x0.max(0.0) as u32)..(x1.min(w as f32) as u32) {
                let fu = (xx as f32 + 0.5 - x0) / (x1 - x0).max(1e-6);
                let fv = (yy as f32 + 0.5 - y0) / (y1 - y0).max(1e-6);
                let u = u0 + fu * (u1 - u0);
                let v = v0 + fv * (v1 - v0);
                let t = ui_fetch(u, v);
                let a = t[3] * col[3];
                let i = ((yy * w + xx) * 4) as usize;
                for c in 0..3 {
                    let s = t[c] * col[c];
                    let dst = ui_srgb_to_linear(px[i + c] as f32 / 255.0);
                    px[i + c] = (ui_linear_to_srgb(s * a + dst * (1.0 - a)) * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    if let Err(e) = write_png(path, w, h, &px) {
        eprintln!("ui shot: {e}");
        return;
    }
    println!("ui shot saved: {path} ({w}x{h})");
}

// ------------------------------------------------------------ winit app ----


struct App {
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    camera: Option<Camera>,
    keys: HashSet<KeyCode>,
    shift: bool,
    cursor: Option<(f64, f64)>,
    win_size: (f64, f64),
    drag_rot: bool,
    drag_pan: bool,
    last_frame: Instant,
    title_acc: f32,
    fps_acc: f32,
    frames: u32,
    /// courtyard boundary overlay (button / key B)
    show_plans: bool,
    /// editor: 0 camera, 1 roads, 2 courtyards, 3 parks
    tool: u8,
    /// editor: the in-progress shape (plan coords)
    pts: Vec<[f32; 2]>,
    /// editor: everything committed this session
    user: UserPlan,
    /// last live-preview road point (throttles per-cursor rebuilds)
    last_live: Option<[f32; 2]>,
    /// last sketch cursor point of the block/park tools (throttle)
    last_shape: Option<[f32; 2]>,
    /// KOLOMNA_FRAMES: exit cleanly after N rendered frames (render smoke)
    frame_stop: Option<u32>,
    frames_done: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let title = "Kolomna — сгенерированный город".to_string();
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(VIEW_W, VIEW_H));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("error: failed to create window: {e}");
                std::process::exit(1);
            }
        };
        let scene = match load_scene(&self.user, false) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let renderer = match Renderer::new(window.clone(), &scene) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        let size = window.inner_size();
        self.win_size = (size.width as f64, size.height as f64);
        println!("Управление:");
        println!("  колесо               — мягкий зум");
        println!("  средняя кнопка       — мягкий поворот");
        println!("  Shift+средняя / ЛКМ / ПКМ — перенос карты");
        println!("  края окна            — навигация (карта едет, если мышь у края)");
        println!("  WASD/стрелки — перенос · Q/E — поворот 90° · R — сброс · Esc — выход");
        println!("  кнопки слева вверху — Границы · Камера · Дороги · Кварталы · Парки");
        println!("  РЕДАКТОР: 1 камера · 2 дороги · 3 кварталы · 4 парки");
        println!("            ДОРОГИ: ЛКМ — точка, дорога строится сразу (превью за мышью, снап к дорогам)");
        println!("                    ПКМ/Backspace — убрать точку · Esc — закончить дорогу · Esc ещё — выйти из режима");
        println!("            кварталы/парки: ЛКМ — точка · ПКМ — отменить фигуру · Enter — закончить · Esc — выйти");
        self.camera = Some(Camera::new(scene.grid, scene.zspan));
        self.scene = Some(scene);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.last_frame = Instant::now();
        self.update_title();

        // self-driving editor test (KOLOMNA_EDIT_TEST=1): draw and commit
        // in the real windowed path without a mouse, then exit.
        // KOLOMNA_EDIT_TEST=select / =draw stop right after the tool pick
        // (and the first clicks); KOLOMNA_FRAMES then renders live frames —
        // the synchronous suite itself never draws a frame.
        let st_mode = std::env::var("KOLOMNA_EDIT_TEST").unwrap_or_default();
        if st_mode == "select" || st_mode == "draw" {
            let (cw, ch_) = self.win_size;
            self.cursor = Some((cw * 0.5, ch_ * 0.5));
            self.set_tool(1);
            if st_mode == "draw" {
                for (fx, fy) in [(0.50, 0.50), (0.56, 0.46)] {
                    let (px, py) = (cw * fx, ch_ * fy);
                    self.cursor = Some((px, py));
                    if let Some(p) = self.screen_to_plan(px, py) {
                        let p = self.road_snap_point(p);
                        self.pts.push(p);
                        self.last_live = Some(p);
                        self.refresh_preview();
                    }
                }
                let (px, py) = (cw * 0.58, ch_ * 0.42);
                self.cursor = Some((px, py));
                if let Some(p) = self.road_cursor_pt() {
                    let far = self
                        .last_live
                        .map(|q| {
                            let dx = p[0] - q[0];
                            let dy = p[1] - q[1];
                            dx * dx + dy * dy > 0.36
                        })
                        .unwrap_or(true);
                    if far {
                        self.last_live = Some(p);
                        self.refresh_live_stroke();
                    }
                }
                for i in 0..5 {
                    let t0 = std::time::Instant::now();
                    self.refresh_live_stroke();
                    println!(
                        "timing: draw-mode live stroke #{i}: {:.1} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            println!(
                "selftest: {st_mode} phase done (tool {}, {} point(s))",
                self.tool,
                self.pts.len()
            );
            if self.frame_stop.is_none() {
                std::thread::sleep(std::time::Duration::from_millis(1200));
                std::process::exit(0);
            }
        } else if !st_mode.is_empty() {
            println!("selftest: road commit ...");
            self.set_tool(1);
            self.pts = vec![[-600.0, -600.0], [600.0, 600.0]];
            self.commit();
            println!("selftest: road commit OK");
            println!("selftest: road commit 2 (replacement path) ...");
            self.pts = vec![[-200.0, 200.0], [0.0, 260.0]];
            self.commit();
            println!("selftest: road commit 2 OK");
            println!("selftest: road commit 3 (T-attachment onto road 1) ...");
            self.pts = vec![[240.0, 30.0], [100.0, 100.0]];
            self.commit();
            println!("selftest: road commit 3 OK");
            {
                let np = self
                    .scene
                    .as_ref()
                    .map(|s| s.net.clone())
                    .unwrap_or_default();
                let cuts = kolomna_render::streets::net_cuts(&np, &self.user.roads);
                println!(
                    "selftest: junctions {} event(s), {} attachment(s), main cuts {:?}",
                    cuts.events, cuts.attaches, cuts.main
                );
                assert!(cuts.events > 0, "no junction events were produced");
                assert!(cuts.attaches > 0, "no T-attachments were produced");
                assert!(!cuts.main.is_empty(), "no main-street cuts were produced");
                // mesh-level check: no kerb / walk geometry may survive inside
                // the junction boxes, the asphalt union must cover them
                let scene = self.scene.as_ref().unwrap();
                let h_at = Self::scene_h_at(scene);
                let (mesh, _) = kolomna_render::streets::build_roads(
                    &h_at,
                    scene.anchor,
                    scene.mn,
                    scene.eu,
                    scene.nvec,
                    &self.user.roads,
                    &scene.net,
                );
                // the boxes cover the carriageway interiors: the corner
                // fillet rings legitimately curve to ~4.35 m from the centre,
                // so they sit outside; kerb-ish geometry within the roadway
                // square itself would be a leak
                let boxes = [(0.0f32, 0.0f32, 3.4f32), (100.0, 100.0, 3.4f32)];
                let mut bad = 0usize;
                let mut asph = 0usize;
                let mut dbg: Vec<(f32, f32, u32, f32, f32)> = Vec::new();
                let mut dbg_tri: Vec<[(f32, f32); 3]> = Vec::new();
                for part in &mesh.parts {
                    let kerbish = part.texture == 1 || part.texture == 2;
                    let is_a = part.texture == 0;
                    if !kerbish && !is_a {
                        continue;
                    }
                    let i0 = part.first_index as usize;
                    let i1 = i0 + part.index_count as usize;
                    for k in i0..i1 {
                        let vx = mesh.verts[mesh.indices[k] as usize].pos;
                        let w = [
                            vx[0] - scene.anchor[0] * 16.0,
                            vx[2] - scene.anchor[1] * 16.0,
                        ];
                        let uu = w[0] * scene.eu[0] + w[1] * scene.eu[1];
                        let vv = w[0] * scene.nvec[0] + w[1] * scene.nvec[1];
                        for &(bu, bv, bh) in &boxes {
                            if (uu - bu).abs() < bh && (vv - bv).abs() < bh {
                                if kerbish {
                                    bad += 1;
                                    if dbg.len() < 80 {
                                        dbg.push((uu, vv, part.texture, bu, bv));
                                        let t0 = i0 + ((k - i0) / 3) * 3;
                                        let mut tri = [(0.0f32, 0.0f32); 3];
                                        for (m, kk) in (t0..t0 + 3).enumerate() {
                                            let pv = mesh.verts[mesh.indices[kk] as usize].pos;
                                            let w2 = [
                                                pv[0] - scene.anchor[0] * 16.0,
                                                pv[2] - scene.anchor[1] * 16.0,
                                            ];
                                            tri[m] = (
                                                w2[0] * scene.eu[0] + w2[1] * scene.eu[1],
                                                w2[0] * scene.nvec[0] + w2[1] * scene.nvec[1],
                                            );
                                        }
                                        dbg_tri.push(tri);
                                    }
                                } else {
                                    asph += 1;
                                }
                            }
                        }
                    }
                }
                println!(
                    "selftest: junction boxes kerb/walk verts {}, asphalt verts {}",
                    bad, asph
                );
                if bad > 0 {
                    for (i, (uu, vv, tex, bu, bv)) in dbg.iter().enumerate() {
                        println!("badbox u={uu:7.2} v={vv:7.2} tex={tex} box=({bu},{bv})");
                        if let Some(t) = dbg_tri.get(i) {
                            println!(
                                "  tri ({:7.2},{:7.2}) ({:7.2},{:7.2}) ({:7.2},{:7.2})",
                                t[0].0, t[0].1, t[1].0, t[1].1, t[2].0, t[2].1
                            );
                        }
                    }
                }
                assert_eq!(bad, 0, "kerb/walk geometry inside a junction box");
                assert!(asph > 0, "no asphalt inside a junction box");
                // snap probe: a point near road 1 lands on its centreline
                let sp = self.road_snap_point([50.0, 52.0]);
                println!(
                    "selftest: snap probe (50.0, 52.0) -> ({:.2}, {:.2})",
                    sp[0], sp[1]
                );
                assert!(
                    (sp[1] - sp[0]).abs() < 0.3,
                    "snap did not land on road 1's centreline"
                );
            }
            println!("selftest: junctions OK");
            // a chain crossing itself must grow a junction of its own
            println!("selftest: self-crossing commit ...");
            let ev0 = {
                let np = self
                    .scene
                    .as_ref()
                    .map(|s| s.net.clone())
                    .unwrap_or_default();
                kolomna_render::streets::net_cuts(&np, &self.user.roads).events
            };
            self.set_tool(1);
            self.pts = vec![
                [-180.0, 40.0],
                [-120.0, 100.0],
                [-180.0, 150.0],
                [-250.0, 95.0],
                [-148.0, 41.0],
            ];
            self.commit();
            {
                let scene = self.scene.as_ref().unwrap();
                let np = scene.net.clone();
                let cuts = kolomna_render::streets::net_cuts(&np, &self.user.roads);
                println!(
                    "selftest: self-crossing junctions {ev0} -> {} event(s)",
                    cuts.events
                );
                assert!(
                    cuts.events > ev0,
                    "the self-crossing produced no new junction"
                );
                // mesh-level: the crossing carries traffic lights of its own
                let h_at = Self::scene_h_at(scene);
                let (mesh, _) = kolomna_render::streets::build_roads(
                    &h_at,
                    scene.anchor,
                    scene.mn,
                    scene.eu,
                    scene.nvec,
                    &self.user.roads,
                    &scene.net,
                );
                let mut near = 0usize;
                for part in &mesh.parts {
                    if part.texture != 12 {
                        continue;
                    }
                    let i0 = part.first_index as usize;
                    let i1 = i0 + part.index_count as usize;
                    for k in i0..i1 {
                        let vx = mesh.verts[mesh.indices[k] as usize].pos;
                        let w = [
                            vx[0] - scene.anchor[0] * 16.0,
                            vx[2] - scene.anchor[1] * 16.0,
                        ];
                        let uu = w[0] * scene.eu[0] + w[1] * scene.eu[1];
                        let vv = w[0] * scene.nvec[0] + w[1] * scene.nvec[1];
                        // the poles of a crossing this size sit along the
                        // legs a good deal out (≈19–23 m from the centre):
                        // we only need to see that THIS crossing's lights exist
                        if (uu + 168.3).abs() < 26.0 && (vv - 51.7).abs() < 26.0 {
                            near += 1;
                            break;
                        }
                    }
                }
                println!("selftest: self-crossing lights near the crossing: {near}");
                assert!(near > 0, "no traffic lights at the self-crossing junction");
            }
            println!("selftest: self-crossing OK");
            println!("selftest: courtyard commit ...");
            self.set_tool(2);
            self.pts = vec![[310.0, 60.0], [360.0, 55.0], [365.0, 105.0], [305.0, 110.0]];
            self.commit();
            println!("selftest: courtyard commit OK");
            println!("selftest: park commit ...");
            self.set_tool(3);
            self.pts = vec![
                [150.0, 90.0],
                [220.0, 95.0],
                [235.0, 150.0],
                [200.0, 195.0],
                [140.0, 180.0],
                [125.0, 130.0],
            ];
            self.commit();
            println!("selftest: park commit OK");
            // B5: a road across a drawn courtyard removes only its houses
            let rec_i = self.scene.as_ref().unwrap().blk_records.len() - 1;
            let before = self.scene.as_ref().unwrap().blk_records[rec_i].houses.len();
            self.set_tool(1);
            self.pts = vec![[330.0, 30.0], [335.0, 130.0]];
            self.commit();
            let after = self.scene.as_ref().unwrap().blk_records[rec_i].houses.len();
            println!("selftest: courtyard houses {before} -> {after} after a road through it");
            assert!(after < before, "the road through the courtyard removed no houses");
            assert!(after > 0, "the road through the courtyard flattened the whole block");
            // B8: a courtyard drawn over a courtyard refills only free slots
            self.set_tool(2);
            self.pts = vec![[320.0, 80.0], [385.0, 75.0], [390.0, 125.0], [325.0, 130.0]];
            self.commit();
            {
                let scene = self.scene.as_ref().unwrap();
                let nn = scene.blk_records.len();
                let (r1, r2) = (&scene.blk_records[nn - 2], &scene.blk_records[nn - 1]);
                let mut overlaps = 0usize;
                for a in &r1.houses {
                    for b in &r2.houses {
                        if quads_overlap(a, b) {
                            overlaps += 1;
                        }
                    }
                }
                println!(
                    "selftest: courtyard over courtyard: {} + {} houses, {overlaps} overlap(s)",
                    r1.houses.len(),
                    r2.houses.len()
                );
                assert_eq!(overlaps, 0, "stacked houses in the overlap of two courtyards");
                assert!(r2.houses.len() >= 4, "the second courtyard built nothing");
            }
            // B6: a road across a drawn park splits it into separate gardens
            let parts_before = self
                .scene
                .as_ref()
                .unwrap()
                .park_records
                .last()
                .unwrap()
                .entries
                .len();
            self.set_tool(1);
            self.pts = vec![[150.0, 60.0], [190.0, 210.0]];
            self.commit();
            let parts_after = self
                .scene
                .as_ref()
                .unwrap()
                .park_records
                .last()
                .unwrap()
                .entries
                .len();
            println!("selftest: park gardens {parts_before} -> {parts_after} after the road");
            assert!(
                parts_after > parts_before,
                "the road through the park did not split it"
            );
            // B9: a courtyard/park vertex snaps onto the kerb edge of a road
            {
                let sp = self.shape_snap_point([336.0, 90.0]);
                // nearest chain, not a fixed index: earlier sections may
                // insert more roads, and the probe must not go stale
                let (_, dd) = self
                    .user
                    .roads
                    .iter()
                    .map(|ch| kolomna_render::streets::near_poly(ch, sp))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                println!(
                    "selftest: shape snap probe (336.0, 90.0) -> ({:.2}, {:.2}), {dd:.2} m off the centreline",
                    sp[0], sp[1]
                );
                assert!(
                    (dd - 6.6).abs() < 1.2,
                    "shape snap did not land on the kerb edge"
                );
            }
            // B10: the snap lookup the sketch/marker both ride on: engages
            // near a road, stays off far away, and the marker seats on the
            // ground at the snapped point
            {
                let near = self.shape_snap_lookup([336.0, 90.0]);
                assert!(near.is_some(), "snap did not engage near the road");
                assert!(
                    self.shape_snap_lookup([700.0, 700.0]).is_none(),
                    "snap engaged far away from every road"
                );
                let q = near.unwrap();
                let scene = self.scene.as_ref().unwrap();
                let pos = self
                    .snap_marker_pos(q)
                    .expect("snap marker has no placement");
                let h = hx(scene, pos[0], pos[1]) - scene.mn;
                assert!(
                    (pos[2] - (h + 0.25) * ZSCALE).abs() < 1e-4,
                    "snap marker is not seated on the ground"
                );
                println!(
                    "selftest: snap marker at ({:.2}, {:.2}, {:.3}) — engaged at ({:.2}, {:.2})",
                    pos[0], pos[1], pos[2], q[0], q[1]
                );
            }
            // B11: the sketch rubber-band: the preview outline gains the
            // cursor vertex, snapped to the road when the cursor reaches
            // one, and the ground marker lights exactly with that snap
            {
                self.set_tool(2);
                self.pts = vec![[332.0, 78.0], [344.0, 78.0], [344.0, 94.0]];
                self.cursor = Some((self.win_size.0 * 0.5, self.win_size.1 * 0.5));
                let before = self.pts.len();
                let (pv, snap) = self.sketch_pts();
                assert!(
                    pv.len() >= before,
                    "the rubber band dropped the clicked points"
                );
                self.refresh_preview();
                let on = self
                    .renderer
                    .as_ref()
                    .map(|r| r.snap_marker_on)
                    .unwrap_or(false);
                assert_eq!(
                    on,
                    snap.is_some(),
                    "the snap marker's lit state disagrees with the sketch snap"
                );
                println!(
                    "selftest: sketch {before} clicked -> {} pt(s) with the cursor, snap {}, marker lit {on}",
                    pv.len(),
                    snap.is_some()
                );
                self.last_shape = None;
            }
            // timings: the live stroke preview must rebuild instantly
            {
                self.tool = 1;
                let nch = self.user.roads.len();
                if let Some(ch0) = self.user.roads.last().cloned() {
                    self.pts = vec![ch0[0], [ch0[0][0] + 40.0, ch0[0][1] + 40.0]];
                    self.cursor = Some((self.win_size.0 * 0.5, self.win_size.1 * 0.5));
                    self.last_live = None;
                    for i in 0..5 {
                        let t0 = std::time::Instant::now();
                        self.refresh_live_stroke();
                        println!(
                            "timing: live stroke #{i} over {nch} committed chain(s): {:.1} ms",
                            t0.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    let t0 = std::time::Instant::now();
                    self.refresh_roads_committed();
                    println!(
                        "timing: committed roads rebuild over {nch} chain(s): {:.1} ms",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                    self.pts.clear();
                    self.refresh_live_stroke();
                }
            }
            // toolbar dispatch: every button must answer at its centre
            for (r, b) in ui_button_rects() {
                let a = ui_action_at((r[0] + r[2] / 2.0) as f64, (r[1] + r[3] / 2.0) as f64);
                let ok = a == Some(b.action);
                println!(
                    "selftest: button «{}» -> {:?} {}",
                    b.label,
                    a,
                    if ok { "OK" } else { "FAIL" }
                );
                assert!(ok, "button {} not clickable", b.label);
            }
            if self.frame_stop.is_none() {
                std::thread::sleep(std::time::Duration::from_millis(2000));
                std::process::exit(0);
            }
            println!("selftest: suite done; rendering frames ...");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(mods) => {
                self.shift = mods.state().contains(ModifiersState::SHIFT);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Escape {
                                if self.tool == 1 {
                                    // roads: Esc ends the chain (the mode
                                    // stays); another Esc then exits
                                    if self.pts.len() >= 2 {
                                        self.commit_road_chain();
                                        self.after_road_commit();
                                    } else if !self.pts.is_empty() {
                                        self.pts.clear();
                                        self.refresh_preview();
                                    } else {
                                        self.tool = 0;
                                        self.refresh_preview();
                                    }
                                    return;
                                }
                                if self.tool != 0 || !self.pts.is_empty() {
                                    self.tool = 0;
                                    self.pts.clear();
                                    self.refresh_preview();
                                } else {
                                    event_loop.exit();
                                }
                                return;
                            }
                            if !event.repeat {
                                match code {
                                    KeyCode::Digit1 => self.set_tool(0),
                                    KeyCode::Digit2 => self.set_tool(1),
                                    KeyCode::Digit3 => self.set_tool(2),
                                    KeyCode::Digit4 => self.set_tool(3),
                                    KeyCode::Enter | KeyCode::NumpadEnter => self.commit(),
                                    KeyCode::Backspace => {
                                        if !self.pts.is_empty() {
                                            self.pts.pop();
                                            self.refresh_preview();
                                        }
                                    }
                                    _ => {}
                                }
                                if code == KeyCode::KeyB {
                                    self.show_plans = !self.show_plans;
                                }
                                if let Some(scene) = self.scene.as_ref() {
                                    if code == KeyCode::KeyR {
                                        if let Some(cam) = self.camera.as_mut() {
                                            cam.reset(scene.grid, scene.zspan);
                                        }
                                    }
                                }
                                if let Some(cam) = self.camera.as_mut() {
                                    match code {
                                        KeyCode::KeyQ => cam.rotate_by(-std::f32::consts::FRAC_PI_2),
                                        KeyCode::KeyE => cam.rotate_by(std::f32::consts::FRAC_PI_2),
                                        _ => {}
                                    }
                                }
                            }
                            self.keys.insert(code);
                        }
                        ElementState::Released => {
                            self.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(cam) = self.camera.as_mut() {
                    let factor = match delta {
                        MouseScrollDelta::LineDelta(_, y) => 1.12f32.powf(y),
                        MouseScrollDelta::PixelDelta(p) => (p.y as f32 * 0.002).exp(),
                    };
                    cam.zoom_by(factor);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Middle => {
                        if pressed && self.shift {
                            self.drag_pan = true;
                        } else {
                            self.drag_rot = pressed;
                            if pressed {
                                // rotation pivots around the terrain point
                                // under the cursor
                                let base_scale = self
                                    .renderer
                                    .as_ref()
                                    .zip(self.scene.as_ref())
                                    .map(|(r, s)| r.base_scale(s))
                                    .unwrap_or(1.0);
                                if let (Some(scene), Some(cam), Some((px, py))) = (
                                    self.scene.as_ref(),
                                    self.camera.as_mut(),
                                    self.cursor,
                                ) {
                                    cam.set_pivot_from_screen(
                                        scene,
                                        px,
                                        py,
                                        self.win_size,
                                        base_scale,
                                    );
                                }
                            }
                        }
                    }
                    MouseButton::Left | MouseButton::Right => {
                        if pressed && button == MouseButton::Left {
                            if let Some(a) = self.cursor.and_then(|(px, py)| ui_action_at(px, py)) {
                                match a {
                                    UiAction::TogglePlans => self.show_plans = !self.show_plans,
                                    UiAction::Tool(t) => self.set_tool(t),
                                }
                                return; // a button click is not a pan
                            }
                        }
                        // editor: left adds a point (roads snap to the
                        // nearest road), right drops the last one (roads) or
                        // cancels the shape (blocks, parks)
                        if self.tool != 0 {
                            if let Some((px, py)) = self.cursor {
                                if button == MouseButton::Left && pressed {
                                    if let Some(p) = self.screen_to_plan(px, py) {
                                        if self.tool == 1 {
                                            let p = self.road_snap_point(p);
                                            self.pts.push(p);
                                            self.last_live = Some(p);
                                            self.refresh_preview();
                                        } else {
                                            // courtyards and parks snap their
                                            // corners onto the kerb edge of a
                                            // road the cursor reaches
                                            let p = self.shape_snap_point(p);
                                            self.pts.push(p);
                                            self.refresh_preview();
                                        }
                                    }
                                } else if button == MouseButton::Right && pressed {
                                    if self.tool == 1 {
                                        if !self.pts.is_empty() {
                                            self.pts.pop();
                                            self.refresh_preview();
                                        }
                                    } else if !self.pts.is_empty() {
                                        self.pts.clear();
                                        self.refresh_preview();
                                    }
                                }
                            }
                            return; // no map panning while a tool is active
                        }
                        self.drag_pan = pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some(prev) = self.cursor {
                    let dx = (pos.0 - prev.0) as f32;
                    let dy = (pos.1 - prev.1) as f32;
                    if self.drag_rot && !self.shift {
                        if let Some(cam) = self.camera.as_mut() {
                            cam.rotate_by(dx * 0.005);
                            // vertical drag = camera tilt (top-down <-> level);
                            // the closer the camera, the softer the tilt — the
                            // on-screen motion of any point scales with (x + y),
                            // which explodes at high zoom (OrbitControls-style
                            // zoom-dependent damping).
                            let soft = (12.0 / cam.zoom).clamp(0.25, 1.0);
                            cam.tilt_by(dy as f32 * 0.0020 * soft);
                        }
                    }
                    if self.drag_pan || (self.drag_rot && self.shift) {
                        let scale = self
                            .renderer
                            .as_ref()
                            .zip(self.scene.as_ref())
                            .zip(self.camera.as_ref())
                            .map(|((r, s), c)| r.base_scale(s) * c.zoom)
                            .unwrap_or(1.0)
                            .max(1e-3);
                        if let Some(cam) = self.camera.as_mut() {
                            cam.pan_by_px(dx, dy, scale);
                        }
                    }
                }
                self.cursor = Some(pos);
                // roads: the live preview follows the cursor (throttled to
                // one rebuild per ~0.6 m of cursor travel)
                if self.tool == 1 && !self.pts.is_empty() {
                    if let Some(p) = self.road_cursor_pt() {
                        let far = self
                            .last_live
                            .map(|q| {
                                let dx = p[0] - q[0];
                                let dy = p[1] - q[1];
                                dx * dx + dy * dy > 0.36
                            })
                            .unwrap_or(true);
                        if far {
                            self.last_live = Some(p);
                            self.refresh_live_stroke();
                        }
                    }
                }
                // block/park sketch: the rubber band up to the cursor, its
                // road snap and the ground marker follow too (throttled like
                // the road stroke)
                if self.tool >= 2 {
                    if let Some((raw, _)) = self.shape_cursor() {
                        let far = self
                            .last_shape
                            .map(|q| {
                                let dx = raw[0] - q[0];
                                let dy = raw[1] - q[1];
                                dx * dx + dy * dy > 0.36
                            })
                            .unwrap_or(true);
                        if far {
                            self.last_shape = Some(raw);
                            self.refresh_preview();
                        }
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor = None;
                if self.tool >= 2 {
                    self.last_shape = None;
                    self.refresh_preview();
                }
            }
            WindowEvent::Resized(size) => {
                self.win_size = (size.width as f64, size.height as f64);
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                self.fps_acc += dt;
                self.frames += 1;
                let mut frame_ok = true;
                let ui_state = UiState {
                    tool: self.tool,
                    show_plans: self.show_plans,
                    hint: self.hint_text(),
                    cursor: self.cursor,
                };
                if let Some(scene) = self.scene.as_ref() {
                    if let (Some(cam), Some(renderer)) =
                        (self.camera.as_mut(), self.renderer.as_mut())
                    {
                        let base = renderer.base_scale(scene);
                        cam.step(dt, &self.keys, self.cursor, self.win_size, base);
                        if let Err(e) = renderer.render(scene, cam, &ui_state) {
                            eprintln!("error: {e}");
                            frame_ok = false;
                        }
                        // once per 0.5 s: window title with zoom/angle/fps
                        self.title_acc += dt;
                        if self.title_acc >= 0.5 {
                            self.title_acc = 0.0;
                            let fps = self.frames as f32 / self.fps_acc.max(1e-3);
                            self.fps_acc = 0.0;
                            self.frames = 0;
                            if let Some(w) = self.window.as_ref() {
                                let deg = cam.rot.to_degrees().round() as i32;
                                w.set_title(&format!(
                                    "Kolomna · {fps:.0} fps · зум ×{:.2} · {deg}°",
                                    cam.zoom
                                ));
                            }
                        }
                    }
                }
                if !frame_ok {
                    event_loop.exit();
                }
                self.frames_done += 1;
                if let Some(stop) = self.frame_stop {
                    if self.frames_done >= stop {
                        println!("selftest: frames rendered OK ({})", self.frames_done);
                        std::process::exit(0);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

/// The editor's in-progress sketch: a flat orange ribbon 0.30 m over the
/// ground with a marker at every point (one flat texture, no lighting
/// detail needed).
fn preview_mesh(scene: &Scene, pts: &[[f32; 2]]) -> Option<kolomna_render::model::ModelMesh> {
    if pts.is_empty() {
        return None;
    }
    let mut b = kolomna_render::procgen::Builder::new();
    let at = |u: f32, v: f32| -> [f32; 3] {
        let w = [
            scene.eu[0] * u + scene.nvec[0] * v,
            scene.eu[1] * u + scene.nvec[1] * v,
        ];
        let h = hx(
            scene,
            scene.anchor[0] + w[0] / 16.0,
            scene.anchor[1] + w[1] / 16.0,
        ) - scene.mn
            + 0.30;
        [
            scene.anchor[0] * 16.0 + w[0],
            h,
            scene.anchor[1] * 16.0 + w[1],
        ]
    };
    let quad_up = |b: &mut kolomna_render::procgen::Builder, mut p: [[f32; 3]; 4]| {
        let cz =
            (p[1][0] - p[0][0]) * (p[2][1] - p[1][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[1][0]);
        if cz < 0.0 {
            p = [p[3], p[2], p[1], p[0]];
        }
        b.quad(0, p, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    };
    for w in pts.windows(2) {
        let d = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-3);
        let dir = [d[0] / l, d[1] / l];
        let per = [-dir[1] * 0.7, dir[0] * 0.7];
        let p = [
            at(w[0][0] - per[0], w[0][1] - per[1]),
            at(w[1][0] - per[0], w[1][1] - per[1]),
            at(w[1][0] + per[0], w[1][1] + per[1]),
            at(w[0][0] + per[0], w[0][1] + per[1]),
        ];
        quad_up(&mut b, p);
    }
    let m = 1.6f32;
    for q in pts {
        let p = [
            at(q[0] - m, q[1] - m),
            at(q[0] + m, q[1] - m),
            at(q[0] + m, q[1] + m),
            at(q[0] - m, q[1] + m),
        ];
        quad_up(&mut b, p);
    }
    Some(b.finish(1))
}

/// The editor's snap marker: a broken ring around a centre dot, spun and
/// breathed by the model shader (anim 4) so the snap target reads as a
/// living thing on the ground.
fn snap_marker_mesh() -> kolomna_render::model::ModelMesh {
    let mut b = kolomna_render::procgen::Builder::new();
    b.anim(4.0);
    let tau = std::f32::consts::TAU;
    let (r0, r1) = (2.55f32, 3.05f32);
    for seg in 0..4 {
        let a0 = seg as f32 * tau / 4.0 + 0.38;
        let a1 = a0 + tau / 4.0 - 0.76;
        let n = 6usize;
        for k in 0..n {
            let t0 = a0 + (a1 - a0) * k as f32 / n as f32;
            let t1 = a0 + (a1 - a0) * (k + 1) as f32 / n as f32;
            let q = |r: f32, t: f32| -> [f32; 3] { [r * t.cos(), 0.0, r * t.sin()] };
            b.quad(
                0,
                [q(r0, t0), q(r0, t1), q(r1, t1), q(r1, t0)],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            );
        }
    }
    let rd = 0.52f32;
    for k in 0..8 {
        let t0 = k as f32 / 8.0 * tau;
        let t1 = (k + 1) as f32 / 8.0 * tau;
        b.tri(
            0,
            [
                [0.0, 0.0, 0.0],
                [rd * t1.cos(), 0.0, rd * t1.sin()],
                [rd * t0.cos(), 0.0, rd * t0.sin()],
            ],
            [[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]],
        );
    }
    b.finish(1)
}

/// Positive (CCW) winding in the plan frame; the block filler and the park
/// outline both expect it, so a right-to-left drawing works as well.
fn fix_ccw(pts: &mut [[f32; 2]]) {
    let mut ar = 0.0f32;
    for k in 0..pts.len() {
        let a = pts[k];
        let e = pts[(k + 1) % pts.len()];
        ar += a[0] * e[1] - e[0] * a[1];
    }
    if ar < 0.0 {
        pts.reverse();
    }
}

impl App {
    /// Plan coordinates (metres in the scene's eu/nvec frame) under a
    /// screen point: the same ray inversion the rotation pivot uses, then
    /// the cell -> plan transform.
    fn screen_to_plan(&self, px: f64, py: f64) -> Option<[f32; 2]> {
        let scene = self.scene.as_ref()?;
        let cam = self.camera.as_ref()?;
        let renderer = self.renderer.as_ref()?;
        let scale = (renderer.base_scale(scene) * cam.zoom).max(1e-3);
        let a = cam.target[0] + (px as f32 - self.win_size.0 as f32 * 0.5) / scale;
        let b = cam.target[1] + (py as f32 - self.win_size.1 as f32 * 0.5) / scale;
        let t = cam.tilt.clamp(0.07, 1.0);
        let smax = 2.0 * cam.grid;
        let mut s = (b / t).clamp(0.0, smax);
        for _ in 0..20 {
            let cx = (s + a).clamp(0.0, cam.grid);
            let cy = (s - a).clamp(0.0, cam.grid);
            let zmesh = (hx(scene, cx, cy) - scene.mn) * ZSCALE;
            s = ((b + zmesh) / t).clamp(0.0, smax);
        }
        let rp = [s + a, s - a];
        let c = cam.grid * 0.5;
        let (sr, cr) = cam.rot.sin_cos();
        let d = [rp[0] - c, rp[1] - c];
        let cell = [c + d[0] * cr + d[1] * sr, c - d[0] * sr + d[1] * cr];
        let rel = [
            (cell[0] - scene.anchor[0]) * 16.0,
            (cell[1] - scene.anchor[1]) * 16.0,
        ];
        Some([
            rel[0] * scene.eu[0] + rel[1] * scene.eu[1],
            rel[0] * scene.nvec[0] + rel[1] * scene.nvec[1],
        ])
    }

    /// Rebuild the sketch + refresh the title (called after every edit).
    /// Roads draw as real geometry (live preview); the other tools keep the
    /// orange ribbon sketch.
    fn refresh_preview(&mut self) {
        if self.tool == 1 {
            self.refresh_live_stroke();
            if let Some(r) = self.renderer.as_mut() {
                r.set_snap_marker(None);
            }
            self.update_title();
            return;
        }
        // block/park tools: the sketch carries the rubber band up to the
        // cursor, snapped to the nearest road when the cursor reaches one;
        // the animated ground marker shows where that snap lands
        let (pv, snap) = self.sketch_pts();
        let mesh = if let (Some(s), Some(_)) = (self.scene.as_ref(), self.renderer.as_ref()) {
            preview_mesh(s, &pv)
        } else {
            None
        };
        let marker = snap.and_then(|q| self.snap_marker_pos(q));
        if let Some(r) = self.renderer.as_mut() {
            r.set_preview(mesh);
            r.set_snap_marker(marker);
        }
        self.update_title();
    }

    /// Plan point of the cursor, snapped to the nearest road it reaches.
    fn road_cursor_pt(&self) -> Option<[f32; 2]> {
        let (px, py) = self.cursor?;
        let p = self.screen_to_plan(px, py)?;
        Some(self.road_snap_point(p))
    }

    /// Snap a plan point onto the nearest road centreline once the road
    /// corridors would touch — the reach grows with the road widths.
    /// Committed chains and the procedural grid both attract the point.
    fn road_snap_point(&self, p: [f32; 2]) -> [f32; 2] {
        let scene = match self.scene.as_ref() {
            Some(s) => s,
            None => return p,
        };
        let hw = 6.5f32;
        let mut best: Option<([f32; 2], f32)> = None;
        let consider = |q: [f32; 2], d: f32, reach: f32, best: &mut Option<([f32; 2], f32)>| {
            if d < reach && best.map(|b| d < b.1).unwrap_or(true) {
                *best = Some((q, d));
            }
        };
        for ch in &self.user.roads {
            if ch.len() >= 2 {
                let (q, d) = kolomna_render::streets::near_poly(ch, p);
                consider(q, d, hw + hw + 1.0, &mut best);
            }
        }
        let net = &scene.net;
        if net.main_t1 > net.main_t0 + 1.0 {
            let main = [[net.main_t0, 0.0], [net.main_t1, 0.0]];
            let (q, d) = kolomna_render::streets::near_poly(&main, p);
            consider(q, d, hw + hw + 1.0, &mut best);
        }
        for &(uc, _cr, cw) in &net.cross {
            for half in [
                [[uc, 3.5], [uc, net.cross_half]],
                [[uc, -net.cross_half], [uc, -3.5]],
            ] {
                let (q, d) = kolomna_render::streets::near_poly(&half, p);
                consider(q, d, hw + cw + 1.0, &mut best);
            }
        }
        best.map(|(q, _)| q).unwrap_or(p)
    }

    /// Snap a block/park vertex onto a road's kerb edge once the cursor
    /// comes within the road corridor: the vertex lands ~6.6 m off the
    /// centre line on the side the cursor is on, so a courtyard wall or a
    /// park fence stands right along the roadway.  `None` when no road is
    /// within reach — the sketch uses that to light up the snap marker.
    /// The procedural grid (main street, cross lanes) attracts the same way.
    fn shape_snap_lookup(&self, p: [f32; 2]) -> Option<[f32; 2]> {
        let reach = 10.0f32;
        let mut best: Option<([f32; 2], f32)> = None;
        let take = |q: [f32; 2], d: f32, best: &mut Option<([f32; 2], f32)>| {
            if d < reach && best.map(|b| d < b.1).unwrap_or(true) {
                *best = Some((q, d));
            }
        };
        for ch in &self.user.roads {
            if ch.len() >= 2 {
                let (q, d) = kolomna_render::streets::near_poly(ch, p);
                take(q, d, &mut best);
            }
        }
        if let Some(scene) = self.scene.as_ref() {
            let net = &scene.net;
            if net.main_t1 > net.main_t0 + 1.0 {
                let main = [[net.main_t0, 0.0], [net.main_t1, 0.0]];
                let (q, d) = kolomna_render::streets::near_poly(&main, p);
                take(q, d, &mut best);
            }
            for &(uc, _cr, _cw) in &net.cross {
                for half in [
                    [[uc, 3.5], [uc, net.cross_half]],
                    [[uc, -net.cross_half], [uc, -3.5]],
                ] {
                    let (q, d) = kolomna_render::streets::near_poly(&half, p);
                    take(q, d, &mut best);
                }
            }
        }
        let (q, d) = best?;
        // step off the centre line toward the cursor: the vertex
        // lands on the outer kerb, not mid-asphalt
        let dir = [p[0] - q[0], p[1] - q[1]];
        let l = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-4);
        if d < 0.5 {
            Some(q)
        } else {
            Some([q[0] + dir[0] / l * 6.6, q[1] + dir[1] / l * 6.6])
        }
    }

    /// The vertex actually committed: the snap when a road is in reach.
    fn shape_snap_point(&self, p: [f32; 2]) -> [f32; 2] {
        self.shape_snap_lookup(p).unwrap_or(p)
    }

    /// (raw, snapped) plan points under the cursor for the block/park
    /// tools: `snapped` is Some while the cursor reaches a road's kerb.
    fn shape_cursor(&self) -> Option<([f32; 2], Option<[f32; 2]>)> {
        let (px, py) = self.cursor?;
        let p = self.screen_to_plan(px, py)?;
        Some((p, self.shape_snap_lookup(p)))
    }

    /// The sketch outline the block/park preview draws: the clicked points
    /// plus the cursor vertex — snapped onto a road's kerb when the cursor
    /// reaches one (returns that snap too, for the ground marker).
    fn sketch_pts(&self) -> (Vec<[f32; 2]>, Option<[f32; 2]>) {
        let mut pv = self.pts.clone();
        let mut snap: Option<[f32; 2]> = None;
        if self.tool >= 2 {
            if let Some((raw, s)) = self.shape_cursor() {
                let q = s.unwrap_or(raw);
                if let Some(&l) = pv.last() {
                    let dx = q[0] - l[0];
                    let dy = q[1] - l[1];
                    if dx * dx + dy * dy > 0.09 {
                        pv.push(q);
                    }
                }
                snap = s;
            }
        }
        (pv, snap)
    }

    /// Placement of the snap marker at a plan point: world metres -> cell
    /// coordinates plus the scaled ground height (the village's own
    /// convention), lifted a touch over the ground.
    fn snap_marker_pos(&self, p: [f32; 2]) -> Option<[f32; 3]> {
        let scene = self.scene.as_ref()?;
        let w = [
            scene.eu[0] * p[0] + scene.nvec[0] * p[1],
            scene.eu[1] * p[0] + scene.nvec[1] * p[1],
        ];
        let cx = scene.anchor[0] + w[0] / 16.0;
        let cy = scene.anchor[1] + w[1] / 16.0;
        let h = hx(scene, cx, cy) - scene.mn;
        Some([cx, cy, (h + 0.25) * ZSCALE])
    }

    /// The chain being drawn right now: the clicked points + the rubber-band
    /// point under the cursor.
    fn live_stroke_pts(&self) -> Vec<[f32; 2]> {
        let mut ch = self.pts.clone();
        if self.tool == 1 && !ch.is_empty() {
            if let Some(p) = self.road_cursor_pt() {
                let l = ch[ch.len() - 1];
                let dx = p[0] - l[0];
                let dy = p[1] - l[1];
                if dx * dx + dy * dy > 0.09 {
                    ch.push(p);
                }
            }
        }
        ch
    }

    /// Full rebuild of the committed roads mesh (all settled chains) — the
    /// expensive path; runs at load and once per chain commit, never per
    /// cursor move.
    fn refresh_roads_committed(&mut self) {
        if self.user.roads.is_empty() {
            return;
        }
        let t_all = std::time::Instant::now();
        let chains = self.user.roads.clone();
        let (mesh, texs) = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            let h_at = Self::scene_h_at(scene);
            kolomna_render::streets::build_roads(
                &h_at,
                scene.anchor,
                scene.mn,
                scene.eu,
                scene.nvec,
                &chains,
                &scene.net,
            )
        };
        let nv = mesh.verts.len();
        let t_build = t_all.elapsed();
        if let Some(r) = self.renderer.as_mut() {
            match r.roads_idx {
                Some(ix) if ix < r.village.len() => {
                    r.village[ix].set_mesh(&r.device, &mesh);
                }
                _ => {
                    let ix = r.add_village_mesh(mesh, texs, [0.0, 0.0, 0.0], 0.0);
                    r.roads_idx = Some(ix);
                }
            }
        }
        if std::env::var("KOLOMNA_EDIT_TEST").is_ok() {
            let up = t_all.elapsed() - t_build;
            println!(
                "committed roads: mesh {:.1} ms + swap {:.1} ms ({} chain(s), {nv} verts)",
                t_build.as_secs_f64() * 1e3,
                up.as_secs_f64() * 1e3,
                chains.len()
            );
        }
    }

    /// Live rebuild of ONLY the stroke being drawn (the instant path): the
    /// committed roads ride along as extra clearers instead of being
    /// re-triangulated, and the renderer swaps geometry without touching its
    /// shader or textures (ModelRenderer::set_mesh).
    fn refresh_live_stroke(&mut self) {
        let stroke = self.live_stroke_pts();
        if stroke.len() < 2 {
            self.clear_live_stroke();
            return;
        }
        let t_all = std::time::Instant::now();
        let (mesh, texs) = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            let h_at = Self::scene_h_at(scene);
            kolomna_render::streets::build_roads_ext(
                &h_at,
                scene.anchor,
                scene.mn,
                scene.eu,
                scene.nvec,
                std::slice::from_ref(&stroke),
                &self.user.roads,
                &scene.net,
            )
        };
        let nv = mesh.verts.len();
        let t_build = t_all.elapsed();
        if let Some(r) = self.renderer.as_mut() {
            match r.roads_live_idx {
                Some(ix) if ix < r.village.len() => r.village[ix].set_mesh(&r.device, &mesh),
                _ => {
                    let ix = r.add_village_mesh(mesh, texs, [0.0, 0.0, 0.0], 0.0);
                    r.roads_live_idx = Some(ix);
                }
            }
        }
        if std::env::var("KOLOMNA_EDIT_TEST").is_ok() {
            let up = t_all.elapsed() - t_build;
            println!(
                "live stroke: mesh {:.1} ms + swap {:.1} ms ({} pt(s), {nv} verts)",
                t_build.as_secs_f64() * 1e3,
                up.as_secs_f64() * 1e3,
                stroke.len()
            );
        }
    }

    /// Blank the live stroke layer (nothing is being drawn).
    fn clear_live_stroke(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            if let Some(ix) = r.roads_live_idx {
                if ix < r.village.len() {
                    r.village[ix].set_mesh(&r.device, &kolomna_render::model::ModelMesh::empty());
                }
            }
        }
    }

    /// Once a chain is settled: rebuild the roads mesh, re-cut the grid
    /// streets at the junctions and sweep the courtyards and parks the
    /// fresh chain crosses (houses on the roadway go, gardens split).
    fn after_road_commit(&mut self) {
        self.last_live = None;
        self.refresh_roads_committed();
        self.clear_live_stroke();
        if self.chains_near_grid() {
            self.rebuild_streets();
        }
        if let Some(ch) = self.user.roads.last().cloned() {
            self.rebuild_blocks_on_roads(std::slice::from_ref(&ch));
            self.rebuild_parks_on_roads(std::slice::from_ref(&ch));
        }
        println!("дороги перестроены ({} цепочек)", self.user.roads.len());
    }

    /// Push the pending `pts` as a road chain, minus the stretches that
    /// merely re-cover roadway that already exists: those are dropped here
    /// (and skipped by the mesh builder) so a redrawn road never lands on
    /// top of the old one — only genuinely new pieces become roads.
    fn commit_road_chain(&mut self) {
        let (kept, skipped) = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            kolomna_render::streets::split_duplicate(&self.pts, &self.user.roads, &scene.net)
        };
        self.pts.clear();
        if skipped > 0.5 && std::env::var("KOLOMNA_EDIT_TEST").is_ok() {
            println!("road commit: {:.0} m уже существующей дороги пропущено", skipped);
        }
        for k in kept {
            if k.len() >= 2 {
                self.user.roads.push(k);
            }
        }
    }

    /// Entries in the renderer's model list; live edits append here while
    /// the CPU `scene.village` keeps the load-time list.
    fn village_len(&self) -> usize {
        self.renderer.as_ref().map(|r| r.village.len()).unwrap_or(0)
    }

    /// A fresh road sweeps the courtyards it reaches: every affected block
    /// regenerates with the roads in hand (the filler drops exactly the
    /// houses the roadway covers) and its old entries turn into blanks.
    fn rebuild_blocks_on_roads(&mut self, trigger: &[Vec<[f32; 2]>]) {
        let trig = smooth_roads(trigger);
        if trig.is_empty() {
            return;
        }
        let n_rec = self.scene.as_ref().map(|s| s.blk_records.len()).unwrap_or(0);
        for ri in 0..n_rec {
            let (needs, blk, q_user) = {
                let scene = match self.scene.as_ref() {
                    Some(s) => s,
                    None => return,
                };
                let rec = &scene.blk_records[ri];
                let mut hit = ring_hits_roads(&rec.q, &trig, ROAD_CUT);
                if !hit {
                    'outer: for h in &rec.houses {
                        for rp in &trig {
                            if quad_path_dist(h, rp) < ROAD_CUT {
                                hit = true;
                                break 'outer;
                            }
                        }
                    }
                }
                (
                    hit,
                    rec.blk,
                    if rec.usr { Some(rec.q.clone()) } else { None },
                )
            };
            if !needs {
                continue;
            }
            let t0 = Instant::now();
            let built = {
                let scene = match self.scene.as_ref() {
                    Some(s) => s,
                    None => return,
                };
                let h_at = Self::scene_h_at(scene);
                // houses of the OTHER drawn courtyards stay: only free
                // slots fill (the block's own houses must not block it)
                let avoid: Vec<HouseQuad> = scene
                    .blk_records
                    .iter()
                    .enumerate()
                    .filter(|(k, r)| *k != ri && r.usr)
                    .flat_map(|(_, r)| r.houses.iter().copied())
                    .collect();
                build_block(
                    blk,
                    q_user.as_deref(),
                    &h_at,
                    scene.anchor,
                    scene.eu,
                    scene.nvec,
                    scene.mn,
                    scene.wl,
                    false,
                    0,
                    &self.user.roads,
                    &avoid,
                )
            };
            let old = self
                .scene
                .as_ref()
                .map(|s| s.blk_records[ri].entries.clone())
                .unwrap_or_default();
            let n_new = built.meshes.len();
            let n_houses = built.houses.len();
            let first = self.village_len();
            if let Some(s) = self.scene.as_mut() {
                // live edits land in the renderer only: `scene.village` stays
                // the load-time list (its meshes are never clonable)
                let rec = &mut s.blk_records[ri];
                rec.entries = (first..first + n_new).collect();
                rec.houses = built.houses.clone();
            }
            if let Some(r) = self.renderer.as_mut() {
                for &ix in &old {
                    let (bm, bt) = blank_mesh();
                    r.replace_village_mesh(ix, bm, bt);
                }
                for (mesh, texs, pos, rot) in built.meshes {
                    r.add_village_mesh(mesh, texs, pos, rot);
                }
            }
            println!(
                "блок {blk} перестроен за {:.2} с ({n_houses} домов)",
                t0.elapsed().as_secs_f32()
            );
        }
    }

    /// A fresh road cuts the parks it reaches: every affected park is
    /// re-split against all the chains, its old gardens blanked and the
    /// new parts appended.
    fn rebuild_parks_on_roads(&mut self, trigger: &[Vec<[f32; 2]>]) {
        let trig = smooth_roads(trigger);
        if trig.is_empty() {
            return;
        }
        let n = self.scene.as_ref().map(|s| s.park_records.len()).unwrap_or(0);
        for pi in 0..n {
            let (needs, outline, old) = {
                let scene = match self.scene.as_ref() {
                    Some(s) => s,
                    None => return,
                };
                let rec = &scene.park_records[pi];
                (
                    ring_hits_roads(&rec.outline, &trig, PARK_MARGIN),
                    rec.outline.clone(),
                    rec.entries.clone(),
                )
            };
            if !needs {
                continue;
            }
            let parts = kolomna_render::streets::split_outline_by_chains(
                &outline,
                &self.user.roads,
                PARK_MARGIN,
            );
            let mut new_meshes = Vec::new();
            for (pj, part) in parts.iter().enumerate() {
                let (mesh, texs) = {
                    let scene = match self.scene.as_ref() {
                        Some(s) => s,
                        None => return,
                    };
                    let h_at = Self::scene_h_at(scene);
                    kolomna_render::summer_garden::build_garden(
                        &h_at,
                        scene.anchor,
                        scene.mn,
                        scene.eu,
                        scene.nvec,
                        &kolomna_render::summer_garden::Params {
                            centre: [0.0, 0.0],
                            ru: 0.0,
                            rv: 0.0,
                            seed: 0xD00D_0000 + (pi * 16 + pj) as u64,
                            outline: part.clone(),
                        },
                    )
                };
                new_meshes.push((mesh, texs));
            }
            let n_new = new_meshes.len();
            let first = self.village_len();
            if let Some(s) = self.scene.as_mut() {
                // renderer-only edits (see rebuild_blocks_on_roads)
                s.park_records[pi].entries = (first..first + n_new).collect();
            }
            if let Some(r) = self.renderer.as_mut() {
                for &ix in &old {
                    let (bm, bt) = blank_mesh();
                    r.replace_village_mesh(ix, bm, bt);
                }
                for (m, t) in new_meshes {
                    r.add_village_mesh(m, t, [0.0, 0.0, 0.0], 0.0);
                }
            }
            println!("парк {}: дорога разрезала — {n_new} сад(ов)", pi + 1);
        }
    }

    /// Rough test: does any drawn chain reach the procedural grid?
    fn chains_near_grid(&self) -> bool {
        let scene = match self.scene.as_ref() {
            Some(s) => s,
            None => return false,
        };
        let net = &scene.net;
        if net.main_t1 <= net.main_t0 + 1.0 && net.cross.is_empty() {
            return false;
        }
        for ch in &self.user.roads {
            for p in ch {
                if net.main_t1 > net.main_t0 + 1.0 && p[1].abs() < 14.0 {
                    return true;
                }
                for &(uc, _, _) in &net.cross {
                    if (p[0] - uc).abs() < 14.0 && p[1].abs() < net.cross_half + 14.0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Rebuild the grid streets with the current chains' junction cuts.
    fn rebuild_streets(&mut self) {
        let t0 = Instant::now();
        let ix = match self.scene.as_ref() {
            Some(s) if s.streets_idx != usize::MAX => s.streets_idx,
            _ => return,
        };
        let (mesh, texs) = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            let h_at = Self::scene_h_at(scene);
            let cuts = kolomna_render::streets::net_cuts(&scene.net, &self.user.roads);
            kolomna_render::streets::build_streets(
                &h_at,
                scene.anchor,
                scene.mn,
                scene.eu,
                scene.nvec,
                &scene.net,
                &cuts,
            )
        };
        if let Some(r) = self.renderer.as_mut() {
            r.replace_village_mesh(ix, mesh, texs);
        }
        println!(
            "улицы перестроены за {:.2} с (перекрёстки)",
            t0.elapsed().as_secs_f32()
        );
    }

    fn set_tool(&mut self, t: u8) {
        // leaving the road tool settles the chain that is being drawn
        if self.tool == 1 && t != 1 && self.pts.len() >= 2 {
            self.commit_road_chain();
            self.after_road_commit();
        }
        self.tool = t;
        self.pts.clear();
        self.last_live = None;
        if t != 1 {
            self.clear_live_stroke();
        }
        // pre-warm the live stroke layer when the roads tool is picked, so
        // the first stroke of the session does not pay the one-off shader /
        // texture cost mid-draw (it lands on the toolbar click instead)
        if t == 1 {
            if let Some(r) = self.renderer.as_mut() {
                if r.roads_live_idx.is_none() {
                    let texs = kolomna_render::streets::textures();
                    let n_slots = texs.len();
                    let ix = r.add_village_mesh(
                        kolomna_render::model::ModelMesh::empty_with_slots(n_slots),
                        texs,
                        [0.0, 0.0, 0.0],
                        0.0,
                    );
                    r.roads_live_idx = Some(ix);
                }
            }
        }
        self.refresh_preview();
    }

    /// The one-line hint shown both in the window title and the status bar.
    fn hint_text(&self) -> &'static str {
        match self.tool {
            1 => "ДОРОГИ: ЛКМ точка (снап к дорогам) · ПКМ убрать · Esc закончить",
            2 => "КВАРТАЛЫ: ≥3 клика (снап к дорогам) · Enter — готово · ПКМ — отмена",
            3 => "ПАРКИ: клики по контуру (снап к дорогам) · Enter — готово · ПКМ — отмена",
            _ => "ЛКМ — обзор · кнопки вверху: дороги · кварталы · парки · B — границы дворов",
        }
    }

    fn update_title(&self) {
        if let Some(w) = self.window.as_ref() {
            w.set_title(&format!("Kolomna — {}", self.hint_text()));
        }
    }

    /// Commit the in-progress shape: roads >= 2 points, courtyards exactly
    /// 4, parks >= 3 (and at least 20 m across).  Then the scene is rebuilt.
    fn commit(&mut self) {
        let mut made = false;
        match self.tool {
            1 => {
                // Enter settles the chain exactly like Esc (the mode stays)
                if self.pts.len() >= 2 {
                    self.commit_road_chain();
                    self.after_road_commit();
                } else {
                    println!("дорога: нужно хотя бы 2 точки (Esc — закончить)");
                }
                return;
            }
            2 => {
                if self.pts.len() >= 3 {
                    let mut q = self.pts.clone();
                    fix_ccw(&mut q);
                    self.user.blocks.push(q);
                    made = true;
                } else {
                    println!("квартал: нужно хотя бы 3 клика (сейчас {})", self.pts.len());
                }
            }
            3 => {
                if self.pts.len() >= 3 {
                    let mut q = self.pts.clone();
                    fix_ccw(&mut q);
                    let (mut u0, mut u1, mut v0, mut v1) =
                        (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                    for p in &q {
                        u0 = u0.min(p[0]);
                        u1 = u1.max(p[0]);
                        v0 = v0.min(p[1]);
                        v1 = v1.max(p[1]);
                    }
                    if (u1 - u0).min(v1 - v0) < 20.0 {
                        println!(
                            "парк: маловат ({:.0}x{:.0} м) — нужно хотя бы 20 м по каждой стороне",
                            u1 - u0,
                            v1 - v0
                        );
                    } else {
                        self.user.parks.push(q);
                        made = true;
                    }
                } else {
                    println!("парк: нужно хотя бы 3 точки контура");
                }
            }
            _ => {}
        }
        if made {
            self.pts.clear();
            self.refresh_preview();
            // every commit generates only its own addition
            match self.tool {
                3 => self.commit_park(),
                2 => self.commit_quarter(),
                _ => {}
            }
        }
    }

    /// The scene's cell-space height sampler (what the generators take).
    fn scene_h_at(scene: &Scene) -> impl Fn(f32, f32) -> f32 + '_ {
        move |x: f32, z: f32| hx(scene, x, z)
    }

    /// Append only the park that was just committed: the roads running
    /// through it cut it into separate gardens, each built and appended on
    /// its own (the same split a full reload would produce).
    fn commit_park(&mut self) {
        let t0 = Instant::now();
        let pi = self.user.parks.len() - 1;
        let parts = kolomna_render::streets::split_outline_by_chains(
            self.user.parks.last().unwrap(),
            &self.user.roads,
            PARK_MARGIN,
        );
        let mut new_meshes = Vec::new();
        for (pj, part) in parts.iter().enumerate() {
            let (mesh, texs) = {
                let scene = match self.scene.as_ref() {
                    Some(s) => s,
                    None => return,
                };
                let h_at = Self::scene_h_at(scene);
                kolomna_render::summer_garden::build_garden(
                    &h_at,
                    scene.anchor,
                    scene.mn,
                    scene.eu,
                    scene.nvec,
                    &kolomna_render::summer_garden::Params {
                        centre: [0.0, 0.0],
                        ru: 0.0,
                        rv: 0.0,
                        seed: 0xD00D_0000 + (pi * 16 + pj) as u64,
                        outline: part.clone(),
                    },
                )
            };
            new_meshes.push((mesh, texs));
        }
        let n_new = new_meshes.len();
        let first = self.village_len();
        if let Some(s) = self.scene.as_mut() {
            // renderer-only edits (see commit_quarter)
            s.park_records.push(ParkRec {
                outline: self.user.parks.last().unwrap().clone(),
                entries: (first..first + n_new).collect(),
            });
        }
        if let Some(r) = self.renderer.as_mut() {
            for (m, t) in new_meshes {
                r.add_village_mesh(m, t, [0.0, 0.0, 0.0], 0.0);
            }
        }
        println!(
            "парк добавлен за {:.2} с ({} сад(ов), всего парков {})",
            t0.elapsed().as_secs_f32(),
            n_new,
            self.user.parks.len()
        );
    }

    /// Append only the courtyard that was just committed: the block filler
    /// runs directly (no save parse, no scene scan) and its houses become
    /// regular village entries.  The block index matches the one a full
    /// reload would give it, so reloads reproduce the very same buildings.
    /// Roads already drawn cut through the fresh block and the houses of
    /// the other drawn courtyards are left standing: only free slots fill.
    fn commit_quarter(&mut self) {
        let t0 = Instant::now();
        let q = match self.user.blocks.last() {
            Some(q) => q.clone(),
            None => return,
        };
        let blk = 10 + (self.user.blocks.len() - 1) as u64;
        let built = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            let h_at = Self::scene_h_at(scene);
            let avoid: Vec<HouseQuad> = scene
                .blk_records
                .iter()
                .filter(|r| r.usr)
                .flat_map(|r| r.houses.iter().copied())
                .collect();
            build_block(
                blk,
                Some(&q),
                &h_at,
                scene.anchor,
                scene.eu,
                scene.nvec,
                scene.mn,
                scene.wl,
                false,
                0,
                &self.user.roads,
                &avoid,
            )
        };
        let n = built.meshes.len();
        let first = self.village_len();
        if let Some(s) = self.scene.as_mut() {
            // live edits land in the renderer only: `scene.village` stays the
            // load-time list (its meshes are never clonable)
            s.blk_records.push(BlkRec {
                blk,
                q: built.q_plan.clone(),
                usr: true,
                entries: (first..first + n).collect(),
                houses: built.houses.clone(),
            });
        }
        if let Some(r) = self.renderer.as_mut() {
            for (mesh, texs, pos, rot) in built.meshes {
                r.add_village_mesh(mesh, texs, pos, rot);
            }
        }
        println!(
            "квартал добавлен за {:.2} с ({} домов, всего кварталов {})",
            t0.elapsed().as_secs_f32(),
            n,
            self.user.blocks.len()
        );
    }
}

/// Interactive mode: open a window and run the event loop until the user quits.
pub fn run() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        scene: None,
        renderer: None,
        camera: None,
        keys: HashSet::new(),
        shift: false,
        cursor: None,
        win_size: (VIEW_W as f64, VIEW_H as f64),
        drag_rot: false,
        drag_pan: false,
        last_frame: Instant::now(),
        title_acc: 0.0,
        fps_acc: 0.0,
        frames: 0,
        show_plans: true,
        tool: 0,
        pts: Vec::new(),
        user: UserPlan::default(),
        last_live: None,
        last_shape: None,
        frame_stop: std::env::var("KOLOMNA_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok()),
        frames_done: 0,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
