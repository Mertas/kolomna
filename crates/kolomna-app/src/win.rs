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

use kolomna_render::assets::fsh_rgba_search;
use kolomna_render::model::{ModelMesh, ModelRenderer, PlacedModel};
use kolomna_render::road::{build_road_mesh, load_road_tiles, RoadMesh, RoadRenderer};
use kolomna_render::{
    Params, TerrainRenderer, VIEW_H, VIEW_W, Vertex, ZSCALE, build_mesh, grid_side, load_atlas,
    load_terrain, smooth_field, water_level, write_png,
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
    /// courtyards: exactly four clicks (CCW in plan)
    pub blocks: Vec<[[f32; 2]; 4]>,
    /// parks: >= 3 contour clicks (CCW in plan)
    pub parks: Vec<Vec<[f32; 2]>>,
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
    /// smoothed height grid (metres, water-clamped) — the exact field the
    /// mesh is built from; the camera raycasts it for the rotation pivot
    sm: Vec<f32>,
    /// grid side length of `sm`
    n: usize,
    /// plan frame: cell coords of plan (0,0) and its orthonormal axes
    anchor: [f32; 2],
    eu: [f32; 2],
    nvec: [f32; 2],
    /// road network from the save (16 m tiles)
    road_mesh: RoadMesh,
    road_textures: Vec<(u32, u32, Vec<u8>)>,
    /// generated Petersburg tenements (mesh, textures, model origin, world rot)
    village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)>,
    /// index of the courtyard-boundary overlay inside `village`
    plan_idx: usize,
    /// index of the drawn-roads mesh inside `village` (editor commits)
    roads_idx: Option<usize>,
    /// camera focus for `--shot` (the middle block's anchor house)
    focus: [f32; 3],
}

pub fn load_scene(
    city: &str,
    dat: &str,
    user: &UserPlan,
    skip_proc: bool,
) -> Result<Scene, Box<dyn Error>> {
    let mut terrain = load_terrain(city)?;
    let n = grid_side(&terrain);
    if n * n != terrain.len() {
        return Err(format!("terrain grid is not square ({} values)", terrain.len()).into());
    }
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    println!("terrain: {n}x{n} vertices, altitude {mn:.1}..{mx:.1} m, water surface {wl:.2} m");

    let atlas = load_atlas(dat)?;
    println!("atlas: 6 textures -> 256x256 px layers");

    // (the terrain mesh is built after the tenement pads are levelled below)

    // ---- road network from the save (one 16 m tile per record)
    let tiles = load_road_tiles(city)?;
    println!("roads: {} tiles", tiles.len());
    let road_mesh = build_road_mesh(&tiles, mn, wl);
    let mut road_textures = Vec::new();
    for id in &road_mesh.texture_ids {
        road_textures.push(fsh_rgba_search(dat, *id)?);
    }
    println!("road textures: {}", road_textures.len());

    // (The 3DMD house family is retired from the scene — the focus is now the
    // generated Petersburg tenements; the 3DMD parser stays for the examples.)
    let t = &tiles[tiles.len() / 2];
    let alt = (t.verts[0][1] + t.verts[2][1]) * 0.5;
    let anchor = [
        (t.verts[0][0] + t.verts[2][0]) * 0.5 / 16.0,
        (t.verts[0][2] + t.verts[2][2]) * 0.5 / 16.0 - 1.0,
    ];
    println!(
        "street anchor at cell ({:.1}, {:.1}), altitude {alt:.1} m",
        anchor[0], anchor[1]
    );
    let mut dx = t.verts[1][0] - t.verts[0][0];
    let mut dz = t.verts[1][2] - t.verts[0][2];
    let dl = (dx * dx + dz * dz).sqrt().max(1e-6);
    dx /= dl;
    dz /= dl;

    // ---- terrain field + mesh (the houses adapt to it, not vice versa) ----
    let base = build_mesh(&terrain, mn, mx, wl);
    let sm = smooth_field(&terrain, wl);
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
            x.clamp(0.0, (n - 1) as f32),
            z.clamp(0.0, (n - 1) as f32),
        )
    };
    let nvec = [-dz, dx]; // road normal
    let eu = [dx, dz]; // along the street
    let mut village: Vec<(ModelMesh, Vec<(u32, u32, Vec<u8>)>, [f32; 3], f32)> = Vec::new();
    let mut roads_idx: Option<usize> = None;
    let mut report: Vec<(f32, f32, f32)> = Vec::new(); // w, d, fall

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

    // ten courtyards: a gallery of blocks along the street line, five columns
    // to a side, so several plan variants can be compared side by side
    let mut plan_pts: Vec<([[f32; 2]; 4], [[f32; 2]; 4])> = Vec::new();
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
    let n_proc = if skip_proc { 0 } else { 10 };
    for blk in 0..(n_proc + user.blocks.len()) as u64 {
        let usr = blk >= n_proc as u64; // a courtyard drawn in the editor
        let s: f32 = if usr { 1.0 } else if blk % 2 == 0 { 1.0 } else { -1.0 };
        let u_off = if usr { 0.0 } else { ((blk / 2) as f32 - 2.0) * 56.0 };
        let seed_b = if usr { 90_000 + blk * 131 } else { 9000 + blk * 131 };
        let mut rng = kolomna_render::procgen::Rng::new(seed_b + 7);

        // the block plan: an irregular quad (frame: a along the street, b
        // inward), every vertex jittered — never a neat rectangle; the
        // footprint varies per block so the variants differ.  A user block
        // (blk >= 10) arrives exactly as the editor drew it — already CCW
        // in this same plan frame — so the very same path fills it.
        let mut q: [[f32; 2]; 4];
        if usr {
            q = user.blocks[(blk - n_proc as u64) as usize];
        } else {
            let a_half = 20.0 + rng.below(5) as f32 * 0.5; // front half width 20–22 m
            let b_len = 44.0 + rng.below(13) as f32; // block depth 44–56 m
            q = [
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
        if !usr {
            blk_edges.push((
                q[0][0].min(q[3][0]),
                q[1][0].max(q[2][0]),
                q[2][1].max(q[3][1]),
            ));
        }
        // side i runs q[i] -> q[i+1]: unit tangent, inward normal, length
        let side = |q: &[[f32; 2]; 4], i: usize| -> ([f32; 2], [f32; 2], f32) {
            let a = q[i];
            let b = q[(i + 1) % 4];
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-4);
            let t = [d[0] / l, d[1] / l];
            ([t[0], t[1]], [-t[1], t[0]], l)
        };
        let on_side = |q: &[[f32; 2]; 4], i: usize, a: f32| -> [f32; 2] {
            let (t, _, _) = side(q, i);
            [q[i][0] + t[0] * a, q[i][1] + t[1] * a]
        };
        let lens = [side(&q, 0).2, side(&q, 1).2, side(&q, 2).2, side(&q, 3).2];
        // courtyard corners: intersections of neighbouring inward lines
        let mut vv = [[0.0f32; 2]; 4];
        for i in 0..4 {
            let ip = (i + 3) % 4;
            let (t1, n1, _) = side(&q, ip);
            let (t2, n2, _) = side(&q, i);
            let p1 = [q[ip][0] + n1[0] * depth, q[ip][1] + n1[1] * depth];
            let p2 = [q[i][0] + n2[0] * depth, q[i][1] + n2[1] * depth];
            let den = t1[0] * t2[1] - t1[1] * t2[0];
            let den = if den.abs() < 1e-6 { 1e-6 } else { den };
            let u = ((p2[0] - p1[0]) * t2[1] - (p2[1] - p1[1]) * t2[0]) / den;
            vv[i] = [p1[0] + t1[0] * u, p1[1] + t1[1] * u];
        }
        // corner-house wings: [along the previous side (uB), along the next
        // side (uA)]; every side then keeps at least 10 m for its row
        let mut wings = [[0.0f32; 2]; 4];
        for i in 0..4 {
            let ip = (i + 3) % 4;
            // each side keeps >= 10 m for its own row: a wing never takes more
            // than half of (side - 10 m). The old one-sided cap left 4 m rows
            // and squeezed a 10 m house into them.
            // corner houses stay small (10–12 m): fat 11–16 m wings left the
            // street rows 8–14 m wide — too narrow for bay windows or a portico
            let cap0 = ((lens[ip] - 10.0) * 0.5).max(10.0);
            let cap1 = ((lens[i] - 10.0) * 0.5).max(10.0);
            wings[i][0] = (10.0 + rng.below(3) as f32).min(cap0).max(10.0);
            wings[i][1] = (10.0 + rng.below(3) as f32).min(cap1).max(10.0);
        }

        // place one house: its plan quad c (block frame, metres) + requested
        // size; the mesh is generated, seated and mapped onto the quad
        let mut place = |w_slot: f32,
                         d_slot: f32,
                         c: [[f32; 2]; 4],
                         arch: bool,
                         corner: bool,
                         roman: bool,
                         bay: bool,
                         seed: u64| {
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
                if blk == focus_blk && focus.is_none() {
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

        // rows along the four sides, between the corner houses
        for i in 0..4 {
            let inx = (i + 1) % 4;
            let a_start = wings[i][1];
            let run = lens[i] - wings[i][1] - wings[inx][0];
            let widths = plan_widths(&mut rng, run);
            let k = widths.len();
            // The STREET row (side 0) must open at least one carriage arch
            // into the yard, and its widest facade gets the Roman portico
            // house. Both are chosen by width so they always fit.
            let mut archj = vec![false; k];
            let mut romanj = vec![false; k];
            let mut bayj = vec![false; k];
            if i == 0 {
                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by(|&a, &b| widths[b].partial_cmp(&widths[a]).unwrap());
                // the gallery splits deterministically so the variants sit
                // side by side for comparison:
                // 0,2 roman portico · 1,3 bay windows · 4 plain = arch house
                let kind = blk % 5;
                let rom = if kind == 0 || kind == 2 {
                    order.iter().copied().find(|&j| widths[j] >= 14.0)
                } else {
                    None
                };
                if let Some(r) = rom {
                    romanj[r] = true;
                }
                if rom.is_none()
                    && (kind == 1 || kind == 3)
                    && std::env::var("KOLOMNA_NOBAYS").is_err()
                {
                    if let Some(b) = order.iter().copied().find(|&j| widths[j] >= 14.0) {
                        bayj[b] = true;
                    }
                }
                // a house with a carriage arch carries no parade entrance:
                // the arch goes only to a PLAIN street piece — a roman/bay
                // house never doubles as the arch house, and on k=1 parade
                // blocks the courtyard stays archless
                if rom.is_none() && !bayj.iter().any(|&b| b) {
                    if let Some(p) = order.iter().copied().find(|&j| widths[j] >= 12.0) {
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
        for i in 0..4 {
            let ip = (i + 3) % 4;
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
        plan_pts.push((
            [to_wm_b(q[0]), to_wm_b(q[1]), to_wm_b(q[2]), to_wm_b(q[3])],
            [to_wm_b(vv[0]), to_wm_b(vv[1]), to_wm_b(vv[2]), to_wm_b(vv[3])],
        ));
    }
    // ---- procedural modern streets ---------------------------------------
    // One street along the block fronts, with cross lanes through the gaps
    // between block columns: asphalt roadway, granite kerbs, tiled
    // sidewalks, a dashed centre line and zebra crossings — all conforming
    // to the terrain. Pushed as one world-space mesh (like the overlay).
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
        let (rmesh, rtexs) = kolomna_render::streets::build_streets(
            &h_at,
            anchor,
            mn,
            eu,
            nvec,
            &kolomna_render::streets::Params {
                main_t0,
                main_t1,
                cross_half,
                cross,
            },
        );
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

        // ---- the editor's roads: free chains with the street cross-section
        if !user.roads.is_empty() {
            let (rmesh, rtexs) =
                kolomna_render::streets::build_roads(&h_at, anchor, mn, eu, nvec, &user.roads);
            println!("drawn roads: {} chain(s)", user.roads.len());
            roads_idx = Some(village.len());
            village.push((rmesh, rtexs, [0.0, 0.0, 0.0], 0.0));
        }
        // ---- the editor's parks: the drawn contour is the outline
        for (pi, park) in user.parks.iter().enumerate() {
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
                    seed: 0xD00D_0000 + pi as u64,
                    outline: park.clone(),
                },
            );
            println!("drawn park {}: {} outline points", pi + 1, park.len());
            village.push((pmesh, ptexs, [0.0, 0.0, 0.0], 0.0));
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
    let mut pal_px = Vec::with_capacity(256 * 4);
    for x in 0..256usize {
        let c = if x < 128 { [220u8, 60, 60] } else { [60, 200, 220] };
        pal_px.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    let mut bld = kolomna_render::procgen::Builder::new();
    for (outer, inner) in &plan_pts {
        for (ring, uv, off) in [
            (outer, [[0.25f32, 0.25f32]; 4], 0.9f32),
            (inner, [[0.75f32, 0.25f32]; 4], -0.6f32),
        ] {
            for i in 0..4 {
                let p0 = ring[i];
                let p1 = ring[(i + 1) % 4];
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
    let plan_idx = if skip_proc {
        0
    } else {
        let i = village.len();
        village.push((bld.finish(1), vec![(256u32, 1u32, pal_px)], [0.0, 0.0, 0.0], 0.0));
        i
    };
    let focus = focus.unwrap_or([anchor[0], anchor[1], 0.0]);

    Ok(Scene {
        base,
        atlas,
        grid: (n - 1) as f32,
        zspan: (mx - mn) * ZSCALE,
        mn,
        wl,
        sm,
        n,
        anchor,
        eu,
        nvec,
        road_mesh,
        road_textures,
        village,
        plan_idx,
        roads_idx,
        focus,
    })
}

/// Headless mode: render exactly one frame offscreen with the default camera
/// (`zoom` scales the framing; 1.0 = whole grid) and write it as a PNG.
pub fn shot(out: &str, city: &str, dat: &str, zoom: f32) -> Result<(), Box<dyn Error>> {
    // hidden headless test hatch (KOLOMNA_PLAN_DEMO=1): a demo drawing so
    // the editor's generators can be exercised without a mouse — a road
    // chain, a courtyard quad and a park contour, all in plan coords
    let user = if std::env::var("KOLOMNA_PLAN_DEMO").is_ok() {
        UserPlan {
            roads: vec![vec![[-40.0, 130.0], [40.0, 150.0], [108.0, 125.0]]],
            blocks: vec![[[300.0, 60.0], [360.0, 55.0], [365.0, 105.0], [305.0, 110.0]]],
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
    let scene = load_scene(city, dat, &user, false)?;
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
        sm: &[f32],
        n: usize,
        mn: f32,
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
            let zmesh = (sample_h(sm, n, cx, cy) - mn) * ZSCALE;
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
    road: RoadRenderer,
    village: Vec<ModelRenderer>,
    /// index of the courtyard boundary overlay inside `village`
    plan_idx: usize,
    /// index of the drawn-roads mesh inside `village`
    roads_idx: Option<usize>,
    /// the editor's in-progress sketch (drawn over the ground)
    preview: Option<ModelRenderer>,
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
        let road = RoadRenderer::new(
            &device,
            &queue,
            config.format,
            4,
            &scene.road_mesh,
            &scene.road_textures,
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
        let ui = UiRenderer::new(&device, config.format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            terrain,
            road,
            village,
            plan_idx: scene.plan_idx,
            roads_idx: scene.roads_idx,
            preview: None,
            ui,
            msaa_view,
            depth_view,
        })
    }

    /// The editor's in-progress sketch: a flat orange ribbon over the
    /// ground (None clears it).
    fn set_preview(&mut self, mesh: Option<kolomna_render::model::ModelMesh>) {
        self.preview = mesh.map(|m| {
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
            r
        });
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
        show_plans: bool,
        hover_btn: bool,
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
                if !show_plans && idx == self.plan_idx {
                    continue; // courtyard boundaries hidden
                }
                r.draw(&mut pass);
            }
            if let Some(p) = &self.preview {
                p.draw(&mut pass);
            }
            self.ui.draw(&self.queue, &mut pass, w, h, hover_btn, show_plans);
        }
        self.queue.submit(Some(enc.finish()));
        self.queue.present(frame);
        Ok(())
    }
}

// --------------------------------------------------- courtyard ui button ----

/// Screen-space toggle button (top-left corner, physical px) that shows or
/// hides the courtyard boundary overlay; clicks compare against this rect.
const UI_BTN: [f32; 4] = [12.0, 12.0, 44.0, 44.0]; // x, y, w, h

fn ui_btn_hit(px: f64, py: f64) -> bool {
    px >= UI_BTN[0] as f64
        && px <= (UI_BTN[0] + UI_BTN[2]) as f64
        && py >= UI_BTN[1] as f64
        && py <= (UI_BTN[1] + UI_BTN[3]) as f64
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiUniform {
    rect: [f32; 4],
    screen: [f32; 2],
    flags: [f32; 2], // x: overlay visible, y: cursor over the button
}

/// Six painter-order rectangles (border, background, red icon frame + inset,
/// cyan icon frame + inset) built from `vertex_index` alone — no buffers.
/// The icon mirrors the overlay: two nested "courtyard" outlines.
const UI_SHADER: &str = r#"
struct Btn {
    rect: vec4<f32>,   // x, y, w, h in physical px
    screen: vec2<f32>, // surface px
    flags: vec2<f32>,  // x: overlay on, y: hover
};
@group(0) @binding(0) var<uniform> btn: Btn;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
};

@vertex
fn vs_ui(@builtin(vertex_index) vi: u32) -> VOut {
    let r: u32 = vi / 6u;
    let c: u32 = vi % 6u;
    var lx = 0.0;
    var ly = 0.0;
    var lw = 0.0;
    var lh = 0.0;
    if (r == 0u) { lx = 0.0; ly = 0.0; lw = btn.rect.z; lh = btn.rect.w; }
    else if (r == 1u) { lx = 2.0; ly = 2.0; lw = btn.rect.z - 4.0; lh = btn.rect.w - 4.0; }
    else if (r == 2u) { lx = 7.0; ly = 7.0; lw = 30.0; lh = 30.0; }
    else if (r == 3u) { lx = 10.0; ly = 10.0; lw = 24.0; lh = 24.0; }
    else if (r == 4u) { lx = 14.0; ly = 14.0; lw = 16.0; lh = 16.0; }
    else { lx = 17.0; ly = 17.0; lw = 10.0; lh = 10.0; }
    var ux = 0.0;
    var uy = 0.0;
    if (c == 0u) { ux = 0.0; uy = 0.0; }
    else if (c == 1u) { ux = 1.0; uy = 0.0; }
    else if (c == 2u) { ux = 0.0; uy = 1.0; }
    else if (c == 3u) { ux = 1.0; uy = 0.0; }
    else if (c == 4u) { ux = 1.0; uy = 1.0; }
    else { ux = 0.0; uy = 1.0; }
    let px = btn.rect.x + lx + ux * lw;
    let py = btn.rect.y + ly + uy * lh;
    let on = btn.flags.x;
    let hov = btn.flags.y;
    let bg = vec3<f32>(0.10 + 0.08 * hov, 0.12 + 0.09 * hov, 0.15 + 0.10 * hov);
    var col = vec4<f32>(bg, 0.92);
    if (r == 0u) {
        let g = 0.55 + 0.40 * hov;
        col = vec4<f32>(g, g, g + 0.03, 1.0);
    } else if (r == 2u) {
        col = vec4<f32>(select(vec3<f32>(0.45, 0.45, 0.48), vec3<f32>(0.92, 0.30, 0.32), on > 0.5), 1.0);
    } else if (r == 4u) {
        col = vec4<f32>(select(vec3<f32>(0.45, 0.45, 0.48), vec3<f32>(0.25, 0.82, 0.92), on > 0.5), 1.0);
    }
    var o: VOut;
    o.pos = vec4<f32>(px / btn.screen.x * 2.0 - 1.0, 1.0 - py / btn.screen.y * 2.0, 0.0, 1.0);
    o.col = col;
    return o;
}

@fragment
fn fs_ui(in: VOut) -> @location(0) vec4<f32> {
    return in.col;
}
"#;

struct UiRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl UiRenderer {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui-shader"),
            source: wgpu::ShaderSource::Wgsl(UI_SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ui"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
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
        }
    }

    fn draw(
        &self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        w: f32,
        h: f32,
        hover: bool,
        on: bool,
    ) {
        let u = UiUniform {
            rect: UI_BTN,
            screen: [w, h],
            flags: [if on { 1.0 } else { 0.0 }, if hover { 1.0 } else { 0.0 }],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..36, 0..1);
    }
}

// ------------------------------------------------------------ winit app ----


struct App {
    city: String,
    dat: String,
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
    hover_btn: bool,
    /// editor: 0 camera, 1 roads, 2 courtyards, 3 parks
    tool: u8,
    /// editor: the in-progress shape (plan coords)
    pts: Vec<[f32; 2]>,
    /// editor: everything committed this session
    user: UserPlan,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let title = format!(
            "Kolomna — {}",
            std::path::Path::new(&self.city)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.city.clone())
        );
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
        let scene = match load_scene(&self.city, &self.dat, &self.user, false) {
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
        println!("  B / кнопка слева вверху — границы дворов (вкл/выкл)");
        println!("  РЕДАКТОР: 1 камера · 2 дороги · 3 кварталы · 4 парки");
        println!("            ЛКМ — точка · ПКМ — отменить фигуру · Backspace — убрать точку");
        println!("            Enter — закончить (дороги: ≥2 точек, кварталы: 4, парки: ≥3) · Esc — выйти из режима");
        self.camera = Some(Camera::new(scene.grid, scene.zspan));
        self.scene = Some(scene);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.last_frame = Instant::now();
        self.update_title();

        // self-driving editor test (KOLOMNA_EDIT_TEST=1): draw and commit
        // in the real windowed path without a mouse, then exit
        if std::env::var("KOLOMNA_EDIT_TEST").is_ok() {
            println!("selftest: road commit ...");
            self.set_tool(1);
            self.pts = vec![[-600.0, -600.0], [600.0, 600.0]];
            self.commit();
            println!("selftest: road commit OK");
            println!("selftest: road commit 2 (replacement path) ...");
            self.pts = vec![[-200.0, 200.0], [0.0, 260.0]];
            self.commit();
            println!("selftest: road commit 2 OK");
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
            std::thread::sleep(std::time::Duration::from_millis(2000));
            std::process::exit(0);
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
                                        &scene.sm,
                                        scene.n,
                                        scene.mn,
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
                        if pressed
                            && button == MouseButton::Left
                            && self.cursor.map(|(px, py)| ui_btn_hit(px, py)).unwrap_or(false)
                        {
                            self.show_plans = !self.show_plans;
                            return; // a button click is not a pan
                        }
                        // editor: left adds a point, right cancels the shape
                        if self.tool != 0 {
                            if let Some((px, py)) = self.cursor {
                                if button == MouseButton::Left && pressed {
                                    if let Some(p) = self.screen_to_plan(px, py) {
                                        if !(self.tool == 2 && self.pts.len() >= 4) {
                                            self.pts.push(p);
                                            self.refresh_preview();
                                        }
                                    }
                                } else if button == MouseButton::Right && pressed {
                                    if !self.pts.is_empty() {
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
                self.hover_btn = ui_btn_hit(pos.0, pos.1);
                self.cursor = Some(pos);
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
                let show_plans = self.show_plans;
                let hover_btn = self.hover_btn;
                if let Some(scene) = self.scene.as_ref() {
                    if let (Some(cam), Some(renderer)) =
                        (self.camera.as_mut(), self.renderer.as_mut())
                    {
                        let base = renderer.base_scale(scene);
                        cam.step(dt, &self.keys, self.cursor, self.win_size, base);
                        if let Err(e) = renderer.render(scene, cam, show_plans, hover_btn) {
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
                                let stem = std::path::Path::new(&self.city)
                                    .file_stem()
                                    .map(|s| s.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                let deg = cam.rot.to_degrees().round() as i32;
                                w.set_title(&format!(
                                    "Kolomna — {stem} · {fps:.0} fps · зум ×{:.2} · {deg}°",
                                    cam.zoom
                                ));
                            }
                        }
                    }
                }
                if !frame_ok {
                    event_loop.exit();
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
        let h = sample_h(
            &scene.sm,
            scene.n,
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
            let zmesh = (sample_h(&scene.sm, scene.n, cx, cy) - scene.mn) * ZSCALE;
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
    fn refresh_preview(&mut self) {
        let mesh = if let (Some(s), Some(_)) = (self.scene.as_ref(), self.renderer.as_ref()) {
            preview_mesh(s, &self.pts)
        } else {
            None
        };
        if let Some(r) = self.renderer.as_mut() {
            r.set_preview(mesh);
        }
        self.update_title();
    }

    fn set_tool(&mut self, t: u8) {
        self.tool = t;
        self.pts.clear();
        self.refresh_preview();
    }

    fn update_title(&self) {
        let hint = match self.tool {
            1 => "ДОРОГИ: ЛКМ точка · Enter — готово · ПКМ — отмена",
            2 => "КВАРТАЛЫ: 4 клика · Enter — готово · ПКМ — отмена",
            3 => "ПАРКИ: клики по контуру · Enter — готово · ПКМ — отмена",
            _ => "1 камера · 2 дороги · 3 кварталы · 4 парки · B — границы дворов",
        };
        if let Some(w) = self.window.as_ref() {
            w.set_title(&format!("Kolomna — {hint}"));
        }
    }

    /// Commit the in-progress shape: roads >= 2 points, courtyards exactly
    /// 4, parks >= 3 (and at least 20 m across).  Then the scene is rebuilt.
    fn commit(&mut self) {
        let mut made = false;
        match self.tool {
            1 => {
                if self.pts.len() >= 2 {
                    self.user.roads.push(self.pts.clone());
                    made = true;
                } else {
                    println!("дорога: нужно хотя бы 2 точки");
                }
            }
            2 => {
                if self.pts.len() == 4 {
                    let mut q = self.pts.clone();
                    fix_ccw(&mut q);
                    self.user.blocks.push([q[0], q[1], q[2], q[3]]);
                    made = true;
                } else {
                    println!("квартал: нужно ровно 4 клика (сейчас {})", self.pts.len());
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
                1 => self.commit_roads(),
                3 => self.commit_park(),
                2 => self.commit_quarter(),
                _ => {}
            }
        }
    }

    /// The scene's cell-space height sampler (what the generators take).
    fn scene_h_at(scene: &Scene) -> impl Fn(f32, f32) -> f32 + '_ {
        let n = scene.n;
        move |x: f32, z: f32| {
            sample_h(&scene.sm, n, x.clamp(0.0, (n - 1) as f32), z.clamp(0.0, (n - 1) as f32))
        }
    }

    /// Rebuild only the drawn-roads mesh over the committed chains.
    fn commit_roads(&mut self) {
        let t0 = Instant::now();
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
                &self.user.roads,
            )
        };
        if let Some(r) = self.renderer.as_mut() {
            match r.roads_idx {
                Some(ix) => r.replace_village_mesh(ix, mesh, texs),
                None => {
                    let ix = r.add_village_mesh(mesh, texs, [0.0, 0.0, 0.0], 0.0);
                    r.roads_idx = Some(ix);
                }
            }
        }
        println!(
            "дороги перестроены за {:.2} с ({} цепочек)",
            t0.elapsed().as_secs_f32(),
            self.user.roads.len()
        );
    }

    /// Append only the park that was just committed.
    fn commit_park(&mut self) {
        let t0 = Instant::now();
        let (mesh, texs) = {
            let scene = match self.scene.as_ref() {
                Some(s) => s,
                None => return,
            };
            let park = match self.user.parks.last() {
                Some(p) => p,
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
                    seed: 0xD00D_0000 + self.user.parks.len() as u64,
                    outline: park.clone(),
                },
            )
        };
        if let Some(r) = self.renderer.as_mut() {
            r.add_village_mesh(mesh, texs, [0.0, 0.0, 0.0], 0.0);
        }
        println!(
            "парк добавлен за {:.2} с (всего парков {})",
            t0.elapsed().as_secs_f32(),
            self.user.parks.len()
        );
    }

    /// Append only the courtyard that was just committed: the scene builder
    /// runs with the procedural city skipped, so it generates nothing but
    /// the new block (the save parse stays, it is a fraction of a second).
    fn commit_quarter(&mut self) {
        let t0 = Instant::now();
        let q = match self.user.blocks.last() {
            Some(q) => *q,
            None => return,
        };
        let user = UserPlan {
            roads: Vec::new(),
            blocks: vec![q],
            parks: Vec::new(),
        };
        let scene = match load_scene(&self.city, &self.dat, &user, true) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("постройка квартала не удалась: {e}");
                return;
            }
        };
        let n = scene.village.len();
        if let Some(r) = self.renderer.as_mut() {
            for (mesh, texs, pos, rot) in scene.village {
                r.add_village_mesh(mesh, texs, pos, rot);
            }
        }
        println!(
            "квартал добавлен за {:.2} с ({} элементов, всего кварталов {})",
            t0.elapsed().as_secs_f32(),
            n,
            self.user.blocks.len()
        );
    }
}

/// Interactive mode: open a window and run the event loop until the user quits.
pub fn run(city: &str, dat: &str) -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        city: city.to_string(),
        dat: dat.to_string(),
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
        hover_btn: false,
        tool: 0,
        pts: Vec::new(),
        user: UserPlan::default(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
