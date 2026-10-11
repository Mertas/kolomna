//! Procedural building generator — Foundation-style: every house is assembled
//! from parameterised parts (blocks with floors, 2 m x 3 m tiled facades,
//! optional timber framing, gabled roofs, chimneys, a door tile) driven by a
//! seeded RNG, and painted with procedurally generated RGBA textures.
//!
//! The output is a `ModelMesh` in the project's model conventions
//! (metres, y-up, faces CCW outwards), so it renders through the existing
//! `ModelRenderer` with `textures()` feeding `mesh.texture_ids` slots in order.

use crate::model::{ModelMesh, ModelPart, ModelVertex};

/// Texture slots (indices into `textures()`).
pub const MAT_PLASTER: u32 = 0;
pub const MAT_TIMBER: u32 = 1;
pub const MAT_ROOF_TILE: u32 = 2;
pub const MAT_ROOF_SLATE: u32 = 3;
pub const MAT_STONE: u32 = 4;
pub const MAT_DOOR: u32 = 5;

// ------------------------------------------------------------------ rng ----

/// Small xorshift* RNG so each seed yields a reproducible house.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn f(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    pub fn below(&mut self, n: u32) -> u32 {
        (self.next() % n.max(1) as u64) as u32
    }
}

// -------------------------------------------------------------- builder ----

pub struct Builder {
    verts: Vec<ModelVertex>,
    indices: Vec<u32>,
    /// (material slot, first index, index count) — one segment per material run
    segs: Vec<(u32, u32, u32)>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            indices: Vec::new(),
            segs: Vec::new(),
        }
    }

    fn push_tri(&mut self, mat: u32, a: u32, b: u32, c: u32) {
        if self.segs.last().map(|s| s.0) != Some(mat) {
            self.segs.push((mat, self.indices.len() as u32, 0));
        }
        self.indices.extend_from_slice(&[a, b, c]);
        if let Some(s) = self.segs.last_mut() {
            s.2 += 3;
        }
    }

    /// Quad from 4 positions with per-corner UVs (CCW seen from outside).
    pub fn quad(&mut self, mat: u32, p: [[f32; 3]; 4], uv: [[f32; 2]; 4]) {
        let n = face_n(p[0], p[1], p[2]);
        let base = self.verts.len() as u32;
        for k in 0..4 {
            self.verts.push(ModelVertex { pos: p[k], uv: uv[k], nrm: n });
        }
        self.push_tri(mat, base, base + 1, base + 2);
        self.push_tri(mat, base, base + 2, base + 3);
    }

    pub fn tri(&mut self, mat: u32, p: [[f32; 3]; 3], uv: [[f32; 2]; 3]) {
        let n = face_n(p[0], p[1], p[2]);
        let base = self.verts.len() as u32;
        for k in 0..3 {
            self.verts.push(ModelVertex { pos: p[k], uv: uv[k], nrm: n });
        }
        self.push_tri(mat, base, base + 1, base + 2);
    }

    /// Debug helper: vertices lying beyond `z` (protrusions past a facade)
    pub fn verts_beyond(&self, z: f32) -> usize {
        self.verts.iter().filter(|v| v.pos[2] > z).count()
    }

    pub fn finish(self, n_tex: u32) -> ModelMesh {
        ModelMesh {
            verts: self.verts,
            indices: self.indices,
            parts: self
                .segs
                .iter()
                .filter(|s| s.2 > 0)
                .map(|&(m, f, c)| ModelPart {
                    first_index: f,
                    index_count: c,
                    texture: m,
                })
                .collect(),
            texture_ids: (0..n_tex).collect(),
        }
    }
}

/// Face normal in the renderer's mesh convention: model space is (x, y_up, z);
/// `push_submesh` stores (n.x, n.z, n.y), so do the same.
fn face_n(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
    [n[0] / l, n[2] / l, n[1] / l]
}

// ---------------------------------------------------------------- parts ----

/// One wall along `a -> c` (plan metres; pass the ring counter-clockwise so
/// faces point outwards), `floors` storeys high, 2 m x 3 m texture tiles per
/// storey. `door_tile` swaps one ground-floor tile for the door texture.
fn wall(b: &mut Builder, a: [f32; 2], c: [f32; 2], floors: u32, mat: u32, door_tile: Option<u32>) {
    let dx = c[0] - a[0];
    let dz = c[1] - a[1];
    let n = ((dx * dx + dz * dz).sqrt() / 2.0).round().max(1.0) as u32;
    for t in 0..n {
        let ta = t as f32 / n as f32;
        let tb = (t as f32 + 1.0) / n as f32;
        let ax = a[0] + dx * ta;
        let az = a[1] + dz * ta;
        let cx = a[0] + dx * tb;
        let cz = a[1] + dz * tb;
        for fl in 0..floors {
            let y0 = fl as f32 * 3.0;
            let y1 = y0 + 3.0;
            let v0 = (floors - fl) as f32; // v grows downwards, tiles per storey
            let v1 = v0 - 1.0;
            let (m, uv) = if door_tile == Some(t) && fl == 0 {
                (
                    MAT_DOOR,
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
                )
            } else {
                let u0 = t as f32;
                let u1 = u0 + 1.0;
                (mat, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
            };
            b.quad(
                m,
                [[ax, y0, az], [cx, y0, cz], [cx, y1, cz], [ax, y1, az]],
                uv,
            );
        }
    }
}

/// Gabled roof over a plan rect (metres); the ridge runs along the longer
/// axis. `rh` is the ridge height above the wall top.
fn roof(
    b: &mut Builder,
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    h: f32,
    rh: f32,
    mat_roof: u32,
    mat_wall: u32,
) {
    let (w, d) = (x1 - x0, z1 - z0);
    let ov = 0.28; // eave overhang
    if w >= d {
        let zm = (z0 + z1) * 0.5;
        let u1 = ((w + 2.0 * ov) / 2.0).ceil().max(1.0);
        let span = (((d * 0.5 + ov).powi(2) + (rh + 0.10).powi(2)).sqrt() / 2.0)
            .ceil()
            .max(1.0);
        // south slope (+z out)
        b.quad(
            mat_roof,
            [
                [x0 - ov, h - 0.10, z1 + ov],
                [x1 + ov, h - 0.10, z1 + ov],
                [x1 + ov, h + rh, zm],
                [x0 - ov, h + rh, zm],
            ],
            [[0.0, span], [u1, span], [u1, 0.0], [0.0, 0.0]],
        );
        // north slope (-z out)
        b.quad(
            mat_roof,
            [
                [x1 + ov, h - 0.10, z0 - ov],
                [x0 - ov, h - 0.10, z0 - ov],
                [x0 - ov, h + rh, zm],
                [x1 + ov, h + rh, zm],
            ],
            [[0.0, span], [u1, span], [u1, 0.0], [0.0, 0.0]],
        );
        // gable ends (wall texture continues up)
        b.tri(
            mat_wall,
            [[x0, h, z0], [x0, h, z1], [x0, h + rh, zm]],
            [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, rh / 3.0]],
        );
        b.tri(
            mat_wall,
            [[x1, h, z1], [x1, h, z0], [x1, h + rh, zm]],
            [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, rh / 3.0]],
        );
    } else {
        let xm = (x0 + x1) * 0.5;
        let u1 = ((d + 2.0 * ov) / 2.0).ceil().max(1.0);
        let span = (((w * 0.5 + ov).powi(2) + (rh + 0.10).powi(2)).sqrt() / 2.0)
            .ceil()
            .max(1.0);
        // west slope (-x out)
        b.quad(
            mat_roof,
            [
                [x0 - ov, h - 0.10, z0 - ov],
                [x0 - ov, h - 0.10, z1 + ov],
                [xm, h + rh, z1 + ov],
                [xm, h + rh, z0 - ov],
            ],
            [[0.0, span], [u1, span], [u1, 0.0], [0.0, 0.0]],
        );
        // east slope (+x out)
        b.quad(
            mat_roof,
            [
                [x1 + ov, h - 0.10, z1 + ov],
                [x1 + ov, h - 0.10, z0 - ov],
                [xm, h + rh, z0 - ov],
                [xm, h + rh, z1 + ov],
            ],
            [[0.0, span], [u1, span], [u1, 0.0], [0.0, 0.0]],
        );
        // gable ends (south + north)
        b.tri(
            mat_wall,
            [[x0, h, z1], [x1, h, z1], [xm, h + rh, z1]],
            [[0.0, 0.0], [w / 2.0, 0.0], [w / 4.0, rh / 3.0]],
        );
        b.tri(
            mat_wall,
            [[x1, h, z0], [x0, h, z0], [xm, h + rh, z0]],
            [[0.0, 0.0], [w / 2.0, 0.0], [w / 4.0, rh / 3.0]],
        );
    }
}

/// Square chimney on the ridge (0.42 m across, 0.9 m tall from `y0`).
fn chimney(b: &mut Builder, x: f32, z: f32, y0: f32, mat: u32) {
    let a = 0.28f32;
    let hgt = 1.05f32;
    let (x0, x1, z0, z1) = (x - a, x + a, z - a, z + a);
    let uv = [[0.0, 0.45], [0.2, 0.45], [0.2, 0.0], [0.0, 0.0]];
    b.quad(mat, [[x0, y0, z1], [x1, y0, z1], [x1, y0 + hgt, z1], [x0, y0 + hgt, z1]], uv);
    b.quad(mat, [[x1, y0, z0], [x0, y0, z0], [x0, y0 + hgt, z0], [x1, y0 + hgt, z0]], uv);
    b.quad(mat, [[x1, y0, z1], [x1, y0, z0], [x1, y0 + hgt, z0], [x1, y0 + hgt, z1]], uv);
    b.quad(mat, [[x0, y0, z0], [x0, y0, z1], [x0, y0 + hgt, z1], [x0, y0 + hgt, z0]], uv);
    b.quad(
        mat,
        [
            [x0, y0 + hgt, z0],
            [x0, y0 + hgt, z1],
            [x1, y0 + hgt, z1],
            [x1, y0 + hgt, z0],
        ],
        uv,
    );
    // crown (wider cap on top)
    let c = 0.37f32;
    let (cx0, cx1, cz0, cz1) = (x - c, x + c, z - c, z + c);
    let (y1, y2) = (y0 + hgt, y0 + hgt + 0.16);
    let uv2 = [[0.0, 0.12], [0.24, 0.12], [0.24, 0.0], [0.0, 0.0]];
    b.quad(mat, [[cx0, y1, cz1], [cx1, y1, cz1], [cx1, y2, cz1], [cx0, y2, cz1]], uv2);
    b.quad(mat, [[cx1, y1, cz0], [cx0, y1, cz0], [cx0, y2, cz0], [cx1, y2, cz0]], uv2);
    b.quad(mat, [[cx1, y1, cz1], [cx1, y1, cz0], [cx1, y2, cz0], [cx1, y2, cz1]], uv2);
    b.quad(mat, [[cx0, y1, cz0], [cx0, y1, cz1], [cx0, y2, cz1], [cx0, y2, cz0]], uv2);
    b.quad(mat, [[cx0, y2, cz0], [cx0, y2, cz1], [cx1, y2, cz1], [cx1, y2, cz0]], uv2);
}

/// Stone plinth (apron) under a building: a low slab poking ~15 cm out of the
/// walls, so the house reads as sitting into the ground instead of floating.
fn plinth(b: &mut Builder, x0: f32, x1: f32, z0: f32, z1: f32) {
    let p = 0.15f32;
    let (px0, px1, pz0, pz1) = (x0 - p, x1 + p, z0 - p, z1 + p);
    let (y0, y1) = (0.0f32, 0.17f32);
    let uw = ((x1 - x0 + 2.0 * p) / 2.0).max(1.0);
    let ud = ((z1 - z0 + 2.0 * p) / 2.0).max(1.0);
    // top (+y)
    b.quad(
        MAT_STONE,
        [
            [px0, y1, pz0],
            [px0, y1, pz1],
            [px1, y1, pz1],
            [px1, y1, pz0],
        ],
        [[0.0, 0.0], [0.0, ud], [uw, ud], [uw, 0.0]],
    );
    let uv = [[0.0, 0.1], [uw, 0.1], [uw, 0.0], [0.0, 0.0]];
    // south face (+z)
    b.quad(MAT_STONE, [[px0, y0, pz1], [px1, y0, pz1], [px1, y1, pz1], [px0, y1, pz1]], uv);
    // north face (-z)
    b.quad(MAT_STONE, [[px1, y0, pz0], [px0, y0, pz0], [px0, y1, pz0], [px1, y1, pz0]], uv);
    // east face (+x)
    b.quad(MAT_STONE, [[px1, y0, pz1], [px1, y0, pz0], [px1, y1, pz0], [px1, y1, pz1]], uv);
    // west face (-x)
    b.quad(MAT_STONE, [[px0, y0, pz0], [px0, y0, pz1], [px0, y1, pz1], [px0, y1, pz0]], uv);
}

// -------------------------------------------------------------- houses -----

/// Generate one house from a seed. Sizes are quantised to the 2 m facade tile
/// so window columns line up; the door sits on the south face.
pub fn generate_house(seed: u64) -> ModelMesh {
    let mut r = Rng::new(seed);
    let mut b = Builder::new();
    let timber = r.f() < 0.45;
    let wall_mat = if timber { MAT_TIMBER } else { MAT_PLASTER };
    let roof_mat = if r.f() < 0.6 { MAT_ROOF_TILE } else { MAT_ROOF_SLATE };
    let w = 4.0 + 2.0 * r.below(3) as f32; // 4 / 6 / 8 m
    let d = 4.0 + 2.0 * r.below(3) as f32;
    // small footprints get fewer storeys (no 4x4 towers)
    let floors = 1 + r.below(if w.min(d) <= 4.0 { 2 } else { 3 });
    let h = floors as f32 * 3.0;
    let rh = (w.min(d) * 0.5 * r.range(0.55, 0.9) + 0.25).max(1.1);

    // counter-clockwise ring: south, east, north, west (faces outwards)
    let tiles = (w / 2.0).round() as u32;
    let door = Some(r.below(tiles));
    wall(&mut b, [0.0, d], [w, d], floors, wall_mat, door);
    wall(&mut b, [w, d], [w, 0.0], floors, wall_mat, None);
    wall(&mut b, [w, 0.0], [0.0, 0.0], floors, wall_mat, None);
    wall(&mut b, [0.0, 0.0], [0.0, d], floors, wall_mat, None);
    roof(&mut b, 0.0, w, 0.0, d, h, rh, roof_mat, wall_mat);

    // stone plinth so the house sits into the ground
    plinth(&mut b, 0.0, w, 0.0, d);

    // chimneys sitting on the ridge
    let mut nch = 0;
    if r.f() < 0.7 {
        nch = 1;
        if r.f() < 0.35 {
            nch = 2;
        }
    }
    for _ in 0..nch {
        let y0 = h + rh - 0.45;
        if w >= d {
            chimney(&mut b, r.range(0.5, (w - 0.5).max(0.6)), d * 0.5, y0, MAT_STONE);
        } else {
            chimney(&mut b, w * 0.5, r.range(0.5, (d - 0.5).max(0.6)), y0, MAT_STONE);
        }
    }

    // rear wing (L-shaped houses), smaller and one storey lower
    if r.f() < 0.4 {
        let ww = 4.0 + 2.0 * r.below(2) as f32;
        let wd = 3.0 + 2.0 * r.below(2) as f32;
        let wx0 = r.range(0.0, (w - ww).max(0.0));
        let wfloors = floors.saturating_sub(1).max(1);
        let wh = wfloors as f32 * 3.0;
        let (wx1, wz0, wz1) = (wx0 + ww, -wd + 0.02, 0.02);
        // overlap the main block by 2 cm (hides the shared wall seam)
        wall(&mut b, [wx0, wz1], [wx1, wz1], wfloors, wall_mat, None);
        wall(&mut b, [wx1, wz1], [wx1, wz0], wfloors, wall_mat, None);
        wall(&mut b, [wx1, wz0], [wx0, wz0], wfloors, wall_mat, None);
        wall(&mut b, [wx0, wz0], [wx0, wz1], wfloors, wall_mat, None);
        let wrh = (ww.min(wd) * 0.5 * r.range(0.5, 0.75)).max(0.8);
        roof(&mut b, wx0, wx1, wz0, wz1, wh, wrh, roof_mat, wall_mat);
    }

    b.finish(6)
}

// -------------------------------------------------------------- textures ---

/// Procedurally painted tiles for the six material slots, aligned with
/// `mesh.texture_ids`. Wall tiles are 2 m x 3 m (128x192), roofs/stone are
/// 2 m x 2 m (128x128).
pub fn textures() -> Vec<(u32, u32, Vec<u8>)> {
    vec![
        plaster_tile(),
        timber_tile(),
        roof_tile([166, 90, 60], [108, 54, 38], [124, 66, 44], 21),
        roof_tile([98, 106, 122], [64, 70, 84], [80, 86, 100], 31),
        stone_tile(),
        door_tile(),
    ]
}

fn h2(x: i32, y: i32, seed: u32) -> f32 {
    let mut v = (x as u32).wrapping_mul(374_761_393)
        ^ (y as u32).wrapping_mul(668_265_263)
        ^ seed.wrapping_mul(1_274_126_177);
    v ^= v >> 13;
    v = v.wrapping_mul(1_274_126_177);
    v ^= v >> 16;
    (v & 0xffff) as f32 / 65535.0
}

fn put(px: &mut [u8], w: u32, x: u32, y: u32, c: [u8; 3], sh: f32) {
    let o = ((y * w + x) * 4) as usize;
    let f = 1.0 + sh;
    px[o] = (c[0] as f32 * f).clamp(0.0, 255.0) as u8;
    px[o + 1] = (c[1] as f32 * f).clamp(0.0, 255.0) as u8;
    px[o + 2] = (c[2] as f32 * f).clamp(0.0, 255.0) as u8;
    px[o + 3] = 255;
}

/// Flat fill with per-pixel grain + soft 4 px blotches.
fn fill(px: &mut [u8], w: u32, h: u32, c: [u8; 3], seed: u32) {
    for y in 0..h {
        for x in 0..w {
            let n = (h2(x as i32, y as i32, seed) - 0.5) * 0.06
                + (h2(x as i32 / 4, y as i32 / 4, seed + 1) - 0.5) * 0.09;
            put(px, w, x, y, c, n);
        }
    }
}

fn rect(px: &mut [u8], w: u32, x0: u32, y0: u32, x1: u32, y1: u32, c: [u8; 3], seed: u32) {
    for y in y0..y1 {
        for x in x0..x1 {
            let n = (h2(x as i32 / 2, y as i32 / 2, seed) - 0.5) * 0.07;
            put(px, w, x, y, c, n);
        }
    }
}

/// Dark round blotches (weathering).
fn speckle(px: &mut [u8], w: u32, h: u32, seed: u32, n: u32, dark: f32) {
    for i in 0..n {
        let cx = (h2(i as i32, 3, seed) * w as f32) as i32;
        let cy = (h2(i as i32, 7, seed) * h as f32) as i32;
        let r = 4.0 + h2(i as i32, 11, seed) * 10.0;
        let x0 = (cx as f32 - r).max(0.0) as u32;
        let x1 = ((cx as f32 + r) as u32).min(w);
        let y0 = (cy as f32 - r).max(0.0) as u32;
        let y1 = ((cy as f32 + r) as u32).min(h);
        for y in y0..y1 {
            for x in x0..x1 {
                let d = (((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2)).sqrt()
                    / r)
                    .min(1.0);
                let o = ((y * w + x) * 4) as usize;
                let f = 1.0 - dark * (1.0 - d);
                px[o] = (px[o] as f32 * f) as u8;
                px[o + 1] = (px[o + 1] as f32 * f) as u8;
                px[o + 2] = (px[o + 2] as f32 * f) as u8;
            }
        }
    }
}

/// Window (frame + glass with a sky gradient and a diagonal glint) at
/// (x0..x1, y0..y1) of a 128-wide tile.
fn window(px: &mut [u8], w: u32, x0: u32, y0: u32, x1: u32, y1: u32) {
    let fw = 6; // frame width
    rect(px, w, x0, y0, x1, y1, [124, 106, 82], 4);
    let (gx0, gy0, gx1, gy1) = (x0 + fw, y0 + fw, x1 - fw, y1 - fw);
    for y in gy0..gy1 {
        for x in gx0..gx1 {
            let mut f = (1.18 - 0.34 * ((y - gy0) as f32 / (gy1 - gy0).max(1) as f32))
                * (1.0 + (h2(x as i32, y as i32, 6) - 0.5) * 0.08);
            if ((x as i32 - y as i32 + 512) % 72) < 10 {
                f *= 1.55; // glint
            }
            let o = ((y * w + x) * 4) as usize;
            px[o] = (62.0 * f).clamp(0.0, 255.0) as u8;
            px[o + 1] = (76.0 * f).clamp(0.0, 255.0) as u8;
            px[o + 2] = (98.0 * f).clamp(0.0, 255.0) as u8;
            px[o + 3] = 255;
        }
    }
    // muntins + sill + lintel
    rect(px, w, (x0 + x1) / 2 - 2, gy0, (x0 + x1) / 2 + 2, gy1, [124, 106, 82], 5);
    rect(px, w, gx0, (gy0 + gy1) / 2 - 2, gx1, (gy0 + gy1) / 2 + 2, [124, 106, 82], 5);
    rect(px, w, x0 - 4, y1, x1 + 4, y1 + 8, [206, 192, 168], 6);
    rect(px, w, x0, y0 - 8, x1, y0, [104, 88, 70], 7);
}

fn plaster_tile() -> (u32, u32, Vec<u8>) {
    let (w, h) = (128u32, 192u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    fill(&mut px, w, h, [226, 212, 182], 1);
    speckle(&mut px, w, h, 3, 6, 0.10);
    window(&mut px, w, 40, 40, 88, 138);
    (w, h, px)
}

fn timber_tile() -> (u32, u32, Vec<u8>) {
    let (w, h) = (128u32, 192u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    fill(&mut px, w, h, [228, 214, 186], 9);
    speckle(&mut px, w, h, 10, 4, 0.08);
    let beam = [92, 66, 44];
    // frame + corner posts
    rect(&mut px, w, 0, 0, 9, h, beam, 11);
    rect(&mut px, w, w - 9, 0, w, h, beam, 11);
    rect(&mut px, w, 0, 0, w, 9, beam, 11);
    rect(&mut px, w, 0, h - 9, w, h, beam, 11);
    // one diagonal brace (bottom-left to top-right)
    for t in 0..112i32 {
        let x = 9 + t;
        let y = (h as i32 - 9) - (t * (h as i32 - 58) / 112);
        for dy in 0..6i32 {
            for dx in 0..6i32 {
                let xx = x + dx;
                let yy = y.max(0) + dy;
                if (xx as u32) < w && (yy as u32) < h {
                    let n = (h2(xx, yy, 12) - 0.5) * 0.10;
                    put(&mut px, w, xx as u32, yy as u32, beam, n);
                }
            }
        }
    }
    window(&mut px, w, 44, 56, 84, 140);
    (w, h, px)
}

fn roof_tile(base: [u8; 3], seam: [u8; 3], notch: [u8; 3], seed: u32) -> (u32, u32, Vec<u8>) {
    let (w, h) = (128u32, 128u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let row = y / 16;
        for x in 0..w {
            let c = if y % 16 < 3 {
                seam
            } else {
                let sx = (x + if row % 2 == 0 { 0 } else { 8 }) % 16;
                if sx < 2 {
                    notch
                } else {
                    base
                }
            };
            let n = (h2(x as i32, y as i32, seed) - 0.5) * 0.13
                + (h2(x as i32 / 3, y as i32 / 3, seed + 1) - 0.5) * 0.11;
            put(&mut px, w, x, y, c, n);
        }
    }
    speckle(&mut px, w, h, seed + 2, 7, 0.16);
    (w, h, px)
}

fn stone_tile() -> (u32, u32, Vec<u8>) {
    let (w, h) = (128u32, 128u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let row = y / 32;
        for x in 0..w {
            let c = if y % 32 < 3 {
                [112, 106, 98]
            } else {
                let sx = (x + if row % 2 == 0 { 0 } else { 24 }) % 48;
                if sx < 2 {
                    [118, 112, 104]
                } else {
                    [154, 148, 138]
                }
            };
            let n = (h2(x as i32, y as i32, 41) - 0.5) * 0.08
                + (h2(x as i32 / 5, y as i32 / 5, 42) - 0.5) * 0.10;
            put(&mut px, w, x, y, c, n);
        }
    }
    (w, h, px)
}

fn door_tile() -> (u32, u32, Vec<u8>) {
    let (w, h) = (128u32, 192u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    fill(&mut px, w, h, [224, 210, 180], 51);
    window(&mut px, w, 40, 40, 88, 138);
    speckle(&mut px, w, h, 52, 4, 0.08);
    // door in the lower part: jambs, planks, handle, threshold
    rect(&mut px, w, 38, 58, 90, 190, [74, 58, 44], 53);
    rect(&mut px, w, 44, 64, 84, 186, [108, 76, 50], 54);
    for k in 1..3 {
        rect(&mut px, w, 44 + k * 13, 64, 46 + k * 13, 186, [86, 60, 40], 55);
    }
    rect(&mut px, w, 74, 116, 82, 126, [206, 184, 120], 56);
    rect(&mut px, w, 34, 186, 94, 192, [118, 104, 88], 57);
    (w, h, px)
}
