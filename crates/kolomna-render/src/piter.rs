//! Petersburg tenement generator (доходный дом, эклектика XIX в.) — a 3D port
//! of the element set from E:/My/Clicker (Petersburg Generators):
//! basement + piano nobile with rustication, windows with nalidniks / sandriks
//! / arch tops, balconies with cast-iron railings, floor bands, crowning
//! cornice, hipped / gabled iron roof with dormers, brick chimneys and a
//! weather-vane, a grand entrance with a transom, steps, canopy and lanterns,
//! drain pipes and anchor plates. Palette follows the Clicker project.
//!
//! Output: (ModelMesh, textures) — textures carry the per-house palette, so
//! every building gets its own generated tile set.

use crate::model::ModelMesh;
use crate::procgen::{Builder, Rng};

// ---------------------------------------------------------------- material --

pub const M_WALL: u32 = 0; // plain plaster
pub const M_RUST: u32 = 1; // rusticated stone
pub const M_WIN: u32 = 2; // window, plain nalidnik
pub const M_WIN_SANDRIK: u32 = 3; // parade window with sandrik
pub const M_WIN_ARCH: u32 = 4; // arched window with keystone
pub const M_BASE: u32 = 5; // basement stone
pub const M_BASE_WIN: u32 = 6; // basement window with grille
pub const M_DOOR: u32 = 7; // entrance door with transom
pub const M_ROOF: u32 = 8; // iron roofing with seams
pub const M_BRICK: u32 = 9; // brick chimney
pub const M_IRON: u32 = 10; // solid cast iron (pipes, lanterns, anchors)
pub const M_BAND: u32 = 11; // string course / cornice band
pub const M_DORMER: u32 = 12; // dormer front
pub const M_RAILING: u32 = 13; // balcony railing (alpha cut-out)
pub const M_WIN_PLAIN: u32 = 14; // yard-side window: bare frame, no trim
pub const M_BLACK_DOOR: u32 = 15; // service entrance (black door)
pub const M_STABLE: u32 = 16; // coach-house gates seen from the yard
pub const M_STONE: u32 = 17; // pale smooth stone: columns, entablature, pediment
pub const N_MATS: u32 = 18;

/// Clear height of the carriage arch (above the basement + piano nobile line).
pub const ARCH_H: f32 = 4.6;

pub const TEX_W: usize = 128; // 2 m at 64 px/m
pub const TEX_H: usize = 192; // 3 m
pub const TEX_S: usize = 128; // square tiles

// ---------------------------------------------------------------- palette ---

#[derive(Clone, Copy)]
pub struct Palette {
    pub wall: [u8; 3],
    pub wall_light: [u8; 3],
    pub rust_dark: [u8; 3],
    pub basement: [u8; 3],
    pub brick: [u8; 3],
    pub brick_dark: [u8; 3],
    pub roof: [u8; 3],
    pub roof_hl: [u8; 3],
    pub frame: [u8; 3],
    pub glass: [u8; 3],
    pub glass_hl: [u8; 3],
    pub iron: [u8; 3],
    pub gold: [u8; 3],
    pub door: [u8; 3],
}

fn rgb(v: u32) -> [u8; 3] {
    [
        ((v >> 16) & 255) as u8,
        ((v >> 8) & 255) as u8,
        (v & 255) as u8,
    ]
}

pub fn shift(c: [u8; 3], d: i32) -> [u8; 3] {
    [
        (c[0] as i32 + d).clamp(0, 255) as u8,
        (c[1] as i32 + d).clamp(0, 255) as u8,
        (c[2] as i32 + d).clamp(0, 255) as u8,
    ]
}

impl Palette {
    /// Clicker base palette (colors.ts) with a per-seed brightness shift.
    /// The wall hue varies: ochre / rose / grey-green / terracotta / grey-blue.
    fn new(r: &mut Rng) -> Palette {
        let hues: [[u32; 2]; 5] = [
            [0xd9c8a9, 0xe8dcc8], // ochre (Clicker base)
            [0xd8b8a8, 0xe8d0c4], // rose
            [0xc2c8bc, 0xd4d8cc], // grey-green
            [0xc09070, 0xd0a888], // terracotta
            [0xbcc6cc, 0xd0d8dc], // grey-blue
        ];
        let h = hues[r.below(5) as usize];
        let d = r.below(25) as i32 - 12;
        Palette {
            wall: shift(rgb(h[0]), d),
            wall_light: shift(rgb(h[1]), d),
            rust_dark: shift(rgb(0x6e6557), d / 2),
            basement: shift(rgb(0x7a7060), d / 2),
            brick: shift(rgb(0x8b3a2a), d / 2),
            brick_dark: shift(rgb(0x732d1b), d / 2),
            roof: shift(rgb(0x5d7b6a), d / 2),
            roof_hl: shift(rgb(0x6b8b78), d / 2),
            frame: rgb(0x3a3028),
            glass: shift(rgb(0x7ba4b5), d / 3),
            glass_hl: rgb(0x8fb8c8),
            iron: rgb(0x2d2d2d),
            gold: rgb(0xc4a44a),
            door: shift(rgb(0x4a3020), d / 3),
        }
    }
}

// ----------------------------------------------------------- texture utils --

struct Img {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Img {
    fn new(w: usize, h: usize, c: [u8; 3]) -> Self {
        let mut px = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            px.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        Img { w, h, px }
    }
    fn set(&mut self, x: i32, y: i32, c: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let i = ((y as usize) * self.w + x as usize) * 4;
        self.px[i] = c[0];
        self.px[i + 1] = c[1];
        self.px[i + 2] = c[2];
        self.px[i + 3] = 255;
    }
    fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 3]) {
        for y in y0..y1.max(y0) {
            for x in x0..x1.max(x0) {
                self.set(x, y, c);
            }
        }
    }
    fn line_h(&mut self, x0: i32, x1: i32, y: i32, c: [u8; 3]) {
        for x in x0..x1.max(x0) {
            self.set(x, y, c);
        }
    }
    fn line_v(&mut self, x: i32, y0: i32, y1: i32, c: [u8; 3]) {
        for y in y0..y1.max(y0) {
            self.set(x, y, c);
        }
    }
    fn noise(&mut self, r: &mut Rng, amp: i32) {
        for i in 0..self.w * self.h {
            let n = r.below((2 * amp + 1) as u32) as i32 - amp;
            let o = i * 4;
            for k in 0..3 {
                self.px[o + k] = (self.px[o + k] as i32 + n).clamp(0, 255) as u8;
            }
        }
    }
}

/// Vertical row flip. Wall quads carry uv v=0 on their BOTTOM edge and the
/// GPU maps v=0 to the first stored row, so a texture drawn top-down (as
/// every drawing fn here does) renders vertically mirrored — the entrance
/// doors read upside down (caught by the user). `textures()` stores every
/// tile bottom-up through this helper.
fn flip_rows(px: &mut [u8], w: usize, h: usize) {
    let stride = w * 4;
    let mut tmp = vec![0u8; stride];
    for y in 0..h / 2 {
        let a = y * stride;
        let b = (h - 1 - y) * stride;
        tmp.copy_from_slice(&px[a..a + stride]);
        px.copy_within(b..b + stride, a);
        px[b..b + stride].copy_from_slice(&tmp);
    }
}

/// Horizontal rustication (stone courses) on any field colour.
fn rusticate(img: &mut Img, base: [u8; 3], course: i32) {
    let dark = shift(base, -58);
    let light = shift(base, 30);
    let mut y = course;
    let mut off = 0;
    while y < img.h as i32 {
        img.line_h(0, img.w as i32, y, dark);
        img.line_h(0, img.w as i32, y + 1, dark);
        img.line_h(0, img.w as i32, y + 2, light);
        img.line_h(0, img.w as i32, y + 3, light);
        off ^= 24;
        let mut x = off;
        while x < img.w as i32 {
            img.line_v(x, (y - course).max(0), y, dark);
            img.line_v(x + 1, (y - course).max(0), y, dark);
            x += 48;
        }
        y += course;
    }
}

/// Draw one window into the tile (style: 0 plain, 1 sandrik, 2 arch,
/// 3 basement grille, 4 none).
fn draw_window(img: &mut Img, pal: &Palette, base: [u8; 3], style: u32) {
    let w = img.w as i32;
    let nl = shift(pal.wall_light, 22);
    let edge = shift(base, -52);
    let wx0 = 24;
    let wx1 = w - 24; // 80 px = 1.25 m
    let wy1 = img.h as i32 - 40; // sill 40 px above the floor line
    let wy0 = match style {
        0 => 62,
        1 => 66,
        2 => 58,
        3 => img.h as i32 - 96,
        4 => 66,
        _ => 62,
    };
    // nalidnik: bright band with a dark contour so it reads when zoomed out
    // (styles 3 = basement grille and 4 = plain yard window have none)
    if style != 3 && style != 4 {
        img.rect(wx0 - 13, wy0 - 13, wx1 + 13, wy1 + 11, edge);
        img.rect(wx0 - 11, wy0 - 11, wx1 + 11, wy1 + 9, nl);
    }
    // glass, gloss band, diagonal glint
    img.rect(wx0, wy0, wx1, wy1, pal.glass);
    let hl_x = wx0 + 12;
    img.rect(hl_x, wy0, hl_x + 7, wy1, pal.glass_hl);
    for k in 0..22 {
        img.set(wx0 + 34 + k, wy0 + 30 - k, pal.glass_hl);
        img.set(wx0 + 35 + k, wy0 + 30 - k, pal.glass_hl);
    }
    // frame + mullion + transom
    let f = pal.frame;
    img.rect(wx0, wy0, wx1, wy0 + 4, f);
    img.rect(wx0, wy1 - 4, wx1, wy1, f);
    img.rect(wx0, wy0, wx0 + 4, wy1, f);
    img.rect(wx1 - 4, wy0, wx1, wy1, f);
    let mid = (wx0 + wx1) / 2;
    img.rect(mid - 2, wy0, mid + 2, wy1, f);
    img.rect(wx0, wy0 + 34, wx1, wy0 + 38, f);
    // sill: plate + drip shadow
    img.rect(wx0 - 11, wy1, wx1 + 11, wy1 + 6, nl);
    img.rect(wx0 - 11, wy1 + 6, wx1 + 11, wy1 + 9, edge);
    if style == 1 {
        // sandrik: white slab projecting past the nalidnik + shadow + consoles
        img.rect(wx0 - 19, wy0 - 24, wx1 + 19, wy0 - 15, nl);
        img.rect(wx0 - 19, wy0 - 15, wx1 + 19, wy0 - 10, edge);
        img.rect(wx0 - 17, wy0 - 10, wx0 - 12, wy0 - 2, nl);
        img.rect(wx1 + 12, wy0 - 10, wx1 + 17, wy0 - 2, nl);
    }
    if style == 2 {
        // arch: round the top corners with stone wedges + keystone
        for k in 0..12 {
            let cut = ((12 - k) * 3) / 2;
            img.rect(wx0, wy0 + k * 2, wx0 + 12 - cut, wy0 + k * 2 + 3, nl);
            img.rect(wx1 - 12 + cut, wy0 + k * 2, wx1, wy0 + k * 2 + 3, nl);
        }
        img.rect(mid - 10, wy0 - 16, mid + 10, wy0 + 2, edge);
        img.rect(mid - 8, wy0 - 14, mid + 8, wy0 + 2, nl);
        img.rect(mid - 4, wy0 - 10, mid + 4, wy0, edge);
    }
    if style <= 2 {
        // apron panel with a gold rosette under the sill
        let ay0 = wy1 + 14;
        img.rect(mid - 26, ay0, mid + 26, ay0 + 30, nl);
        img.rect(mid - 22, ay0 + 4, mid + 22, ay0 + 26, shift(base, -8));
        img.rect(mid - 3, ay0 + 9, mid + 3, ay0 + 21, pal.gold);
        img.rect(mid - 8, ay0 + 13, mid + 8, ay0 + 17, pal.gold);
    }
    if style == 3 {
        // basement grille bars
        for k in 0..5 {
            let x = wx0 + 8 + k * 12;
            img.rect(x, wy0, x + 3, wy1, pal.iron);
        }
        img.rect(wx0, wy1 - 5, wx1, wy1, pal.iron);
        img.rect(wx0, wy0, wx1, wy0 + 3, pal.iron);
    }
}

/// One facade tile: field colour, optional rustication, optional window.
fn tile_tex(
    pal: &Palette,
    base: [u8; 3],
    win: Option<u32>,
    rust_bg: bool,
    r: &mut Rng,
) -> Vec<u8> {
    let mut img = Img::new(TEX_W, TEX_H, base);
    if rust_bg {
        rusticate(&mut img, base, if base == pal.basement { 44 } else { 30 });
    } else {
        img.noise(r, 4);
    }
    if let Some(style) = win {
        draw_window(&mut img, pal, base, style);
    }
    img.px
}

/// Entrance door: stone portal, two paneled leaves, arched radial transom,
/// keystone, handle pair, lanterns on the reveals.
fn door_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_W, TEX_H, pal.basement);
    rusticate(&mut img, pal.basement, 44);
    img.noise(r, 3);
    let dx0 = 34;
    let dx1 = TEX_W as i32 - 34; // 60 px = 0.95 m leaf pair
    let dy1 = TEX_H as i32 - 10;
    // stone portal
    img.rect(dx0 - 12, 26, dx1 + 12, dy1 + 6, pal.wall_light);
    img.rect(dx0 - 12, 26, dx0 - 6, dy1 + 6, pal.rust_dark);
    // arched transom (semicircle by shrinking rows)
    for k in 0..26 {
        let cut = ((26 - k) as f32 * 0.62) as i32;
        img.rect(dx0 + cut, 26 + k, dx1 - cut, 27 + k, pal.glass);
    }
    // radial glazing
    let cx = (dx0 + dx1) / 2;
    for ang in -4..=4i32 {
        let mut x = cx;
        let mut y = 50;
        for _ in 0..10 {
            img.set(x, y - 1, pal.frame);
            img.set(x, y, pal.frame);
            x += ang;
            y -= 2;
        }
    }
    img.rect(dx0, 48, dx1, 52, pal.frame);
    // keystone
    img.rect(cx - 8, 16, cx + 8, 30, pal.wall_light);
    img.rect(cx - 4, 20, cx + 4, 28, pal.rust_dark);
    // leaves
    img.rect(dx0, 52, dx1, dy1, pal.door);
    img.rect(cx - 2, 52, cx + 2, dy1, pal.frame);
    for lx in [dx0 + 8, cx + 8] {
        img.rect(lx, 60, lx + 18, 88, shift(pal.door, 14));
        img.rect(lx, 96, lx + 18, 124, shift(pal.door, 14));
        img.rect(lx + 2, 62, lx + 16, 86, shift(pal.door, -8));
        img.rect(lx + 2, 98, lx + 16, 122, shift(pal.door, -8));
    }
    img.rect(cx - 10, 96, cx - 5, 104, pal.gold);
    img.rect(cx + 5, 96, cx + 10, 104, pal.gold);
    // lanterns beside the portal
    for lx in [dx0 - 10, dx1 + 4] {
        img.rect(lx, 56, lx + 6, 74, pal.iron);
        img.rect(lx - 2, 64, lx + 8, 76, pal.gold);
    }
    img.px
}

/// Service entrance on the yard side: a plain plank door in a stone reveal,
/// no portal, no transom, a stone lintel above.
fn black_door_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_W, TEX_H, pal.basement);
    rusticate(&mut img, pal.basement, 44);
    img.noise(r, 3);
    let dx0 = 40;
    let dx1 = TEX_W as i32 - 40;
    let dy0 = 60;
    let dy1 = TEX_H as i32 - 12;
    img.rect(dx0 - 6, dy0 - 6, dx1 + 6, dy1 + 2, shift(pal.basement, -40));
    img.rect(dx0, dy0, dx1, dy1, shift(pal.door, -16));
    for x in (dx0 + 8..dx1).step_by(12) {
        img.line_v(x, dy0, dy1, shift(pal.door, -28));
    }
    img.rect(dx1 - 20, dy0 + 42, dx1 - 15, dy0 + 48, pal.iron);
    img.rect(dx0 - 8, dy0 - 14, dx1 + 8, dy0 - 6, shift(pal.basement, 24));
    img.px
}

/// Coach-house gates: broad double doors of dark planks with iron straps and
/// studs, an arched head — seen from inside the yard.
fn stable_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_W, TEX_H, pal.basement);
    rusticate(&mut img, pal.basement, 44);
    img.noise(r, 3);
    let gx0 = 12;
    let gx1 = TEX_W as i32 - 12;
    let gy0 = 30;
    let gy1 = TEX_H as i32 - 8;
    img.rect(gx0 - 6, gy0 - 10, gx1 + 6, gy1 + 2, shift(pal.basement, -44));
    for k in 0..14 {
        let cut = ((14 - k) as f32 * 1.1) as i32;
        img.rect(gx0 + cut, gy0 - 8 + k, gx1 - cut, gy0 - 7 + k, shift(pal.basement, -18));
    }
    let dk = shift(pal.door, -24);
    img.rect(gx0, gy0, gx1, gy1, dk);
    for x in (gx0 + 10..gx1).step_by(14) {
        img.line_v(x, gy0, gy1, shift(dk, -12));
        img.line_v(x + 1, gy0, gy1, shift(dk, 10));
    }
    let mid = (gx0 + gx1) / 2;
    img.line_v(mid - 1, gy0, gy1, shift(dk, -22));
    img.line_v(mid, gy0, gy1, shift(dk, -22));
    for yy in [gy0 + 26, gy1 - 36] {
        img.rect(gx0 + 4, yy, gx1 - 4, yy + 5, pal.iron);
    }
    let mut x = gx0 + 16;
    while x < gx1 - 8 {
        img.rect(x, gy0 + 27, x + 4, gy0 + 31, pal.gold);
        img.rect(x, gy1 - 35, x + 4, gy1 - 31, pal.gold);
        x += 26;
    }
    img.px
}

fn roof_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_S, TEX_S, pal.roof);
    img.noise(r, 5);
    for x in (0..TEX_S as i32).step_by(24) {
        img.line_v(x, 0, TEX_S as i32, shift(pal.roof, -18));
        img.line_v(x + 1, 0, TEX_S as i32, pal.roof_hl);
    }
    img.line_h(0, TEX_S as i32, 4, pal.roof_hl);
    img.px
}

fn brick_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_S, TEX_S, pal.brick);
    img.noise(r, 9);
    let mut y = 10;
    let mut row = 0;
    while y < TEX_S as i32 {
        img.line_h(0, TEX_S as i32, y, pal.brick_dark);
        img.line_h(0, TEX_S as i32, y + 1, shift(pal.brick_dark, 22));
        let off = if row % 2 == 0 { 0 } else { 16 };
        for x in (off..TEX_S as i32).step_by(32) {
            img.line_v(x, y - 10, y, pal.brick_dark);
        }
        y += 10;
        row += 1;
    }
    img.px
}

fn iron_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_S, TEX_S, pal.iron);
    img.noise(r, 3);
    img.px
}

/// Cast-iron balcony railing with an alpha cut-out: rails, balusters with a
/// mid bulge, meander touches under the top rail. Rendered with the model
/// shader's alpha test.
fn railing_tex(pal: &Palette) -> Vec<u8> {
    let h = 64usize;
    let mut img = Img::new(TEX_S, h, [0, 0, 0]);
    for i in 0..TEX_S * h {
        img.px[i * 4 + 3] = 0;
    }
    let c = pal.iron;
    fn put(img: &mut Img, x0: i32, y0: i32, x1: i32, y1: i32, col: [u8; 3]) {
        for y in y0..y1 {
            for x in x0..x1 {
                if x >= 0 && y >= 0 && (x as usize) < img.w && (y as usize) < img.h {
                    let i = ((y as usize) * img.w + x as usize) * 4;
                    img.px[i] = col[0];
                    img.px[i + 1] = col[1];
                    img.px[i + 2] = col[2];
                    img.px[i + 3] = 255;
                }
            }
        }
    }
    put(&mut img, 0, 2, TEX_S as i32, 8, c);
    put(&mut img, 0, 54, TEX_S as i32, 62, c);
    for x in (6..TEX_S as i32).step_by(16) {
        put(&mut img, x, 6, x + 4, 56, c);
        put(&mut img, x - 2, 26, x + 6, 34, shift(c, 22));
    }
    for x in (0..TEX_S as i32).step_by(32) {
        put(&mut img, x, 10, x + 10, 13, c);
        put(&mut img, x + 10, 10, x + 13, 20, c);
    }
    img.px
}

/// Pale ashlar with faint vertical flutes: columns, entablature, pediment.
fn stone_tex(pal: &Palette) -> Vec<u8> {
    let base = shift(pal.wall_light, 24);
    let mut img = Img::new(TEX_S, TEX_S, base);
    let mut x = 0usize;
    while x < TEX_S {
        img.line_v(x as i32, 0, TEX_S as i32, shift(base, -16));
        img.line_v(x as i32 + 1, 0, TEX_S as i32, shift(base, 8));
        img.line_v(x as i32 + 2, 0, TEX_S as i32, base);
        img.line_v(x as i32 + 3, 0, TEX_S as i32, base);
        x += 8;
    }
    let mut y = 0i32;
    while y < TEX_S as i32 {
        img.line_h(0, TEX_S as i32, y, shift(base, -22));
        y += 48;
    }
    img.px
}

fn band_tex(pal: &Palette) -> Vec<u8> {
    let mut img = Img::new(TEX_S, 32, pal.wall_light);
    img.line_h(0, TEX_S as i32, 0, shift(pal.wall_light, -30));
    img.line_h(0, TEX_S as i32, 20, shift(pal.wall_light, -20));
    img.line_h(0, TEX_S as i32, 30, shift(pal.wall_light, -50));
    img.px
}

fn dormer_tex(pal: &Palette, r: &mut Rng) -> Vec<u8> {
    let mut img = Img::new(TEX_S, TEX_S, pal.wall_light);
    img.noise(r, 4);
    // round window
    let cx = TEX_S as i32 / 2;
    for k in 0..22 {
        for y in -22..22i32 {
            if k * k + y * y <= 22 * 22 {
                img.set(cx + k, 46 + y, pal.glass);
                img.set(cx - k, 46 + y, pal.glass);
            }
        }
    }
    img.rect(cx - 3, 26, cx + 3, 66, pal.frame);
    img.rect(cx - 20, 26, cx + 20, 30, pal.frame);
    img.rect(cx - 20, 62, cx + 20, 66, pal.frame);
    img.rect(0, 80, TEX_S as i32, 88, pal.wall_light);
    img.rect(0, 78, TEX_S as i32, 80, shift(pal.wall_light, -30));
    img.px
}

// ---------------------------------------------------------------- geometry --
//
// Quad winding used throughout: p0->p1 is edge u, p0->p2 edge v, normal
// n = u x v must point away from the solid.

struct Manor {
    b: Builder,
    pal: Palette,
    w: f32,
    d: f32,
    heights: Vec<f32>,
    y_band: Vec<f32>,
    tiles: usize,
    win_tiles: Vec<usize>,
    door_tile: usize,
    /// carriage arch: first of the two street-wall tiles it pierces
    arch: Option<usize>,
    /// service (black) entrance on the yard side
    yard_door: Option<usize>,
    /// coach-house gates on the yard side
    stable: Option<usize>,
    /// corner house: the east end is a second parade facade (two streets)
    corner: bool,
    /// Roman-style parade facade: a four-column portico with a pediment and a
    /// tall portal over a wide perron
    roman: bool,
    /// bay windows (эркеры) flanking the entrance
    bays: bool,
    /// x-spans of the built bays — balconies keep off them
    bay_x: Vec<(f32, f32)>,
}

/// Basement plinth that follows the real ground instead of a fixed depth.
/// The caller passes `low(lx, lz)`: for a LOCAL house coordinate, how far
/// below the floor slab (y = 0) the ground lies, in metres (negative, with
/// any safety margin already applied). Every 2 m strip of every wall is
/// drawn down to the lowest ground under that strip, so on a slope the
/// plinth digs into the hill — the house reads as BUILT ON it: nothing is
/// ever cropped by the slope (the house is seated on the high ground) and
/// nothing hovers (the plinth reaches the low ground).
pub fn manor_plinth(
    w: f32,
    d: f32,
    low: &dyn Fn(f32, f32) -> f32,
    arch: Option<(f32, f32)>,
) -> ModelMesh {
    const YMIN: f32 = -14.0; // sane clamp (we are not pouring into a lake)
    let mut b = Builder::new();
    let yl = |l: &dyn Fn(f32, f32) -> f32, x: f32, z: f32| (l(x, z) - 0.45).max(YMIN);
    let in_arch = |x0: f32, x1: f32| {
        if let Some((a0, a1)) = arch {
            x1 > a0 && x0 < a1
        } else {
            false
        }
    };
    // south (+z): strips along x
    let mut x = 0.0f32;
    while x < w - 1e-4 {
        let x1 = (x + 2.0).min(w);
        if in_arch(x, x1) {
            x = x1;
            continue; // the carriage passage goes down to the ground
        }
        let yy = yl(low, x, d).min(yl(low, x1, d));
        b.quad(
            0, 
            [[x, 0.0, d], [x1, 0.0, d], [x1, yy, d], [x, yy, d]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, yy / 2.6], [0.0, yy / 2.6]],
        );
        x = x1;
    }
    // north (−z)
    x = w;
    while x > 1e-4 {
        let x0 = (x - 2.0).max(0.0);
        if in_arch(x0, x) {
            x = x0;
            continue;
        }
        let yy = yl(low, x, 0.0).min(yl(low, x0, 0.0));
        b.quad(
            0,
            [[x, 0.0, 0.0], [x0, 0.0, 0.0], [x0, yy, 0.0], [x, yy, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, yy / 2.6], [0.0, yy / 2.6]],
        );
        x = x0;
    }
    // west (−x): strips along z
    let mut z = 0.0f32;
    while z < d - 1e-4 {
        let z1 = (z + 2.0).min(d);
        let yy = yl(low, 0.0, z).min(yl(low, 0.0, z1));
        b.quad(
            0,
            [[0.0, 0.0, z], [0.0, 0.0, z1], [0.0, yy, z1], [0.0, yy, z]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, yy / 2.6], [0.0, yy / 2.6]],
        );
        z = z1;
    }
    // east (+x)
    z = d;
    while z > 1e-4 {
        let z0 = (z - 2.0).max(0.0);
        let yy = yl(low, w, z).min(yl(low, w, z0));
        b.quad(
            0,
            [[w, 0.0, z], [w, 0.0, z0], [w, yy, z0], [w, yy, z]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, yy / 2.6], [0.0, yy / 2.6]],
        );
        z = z0;
    }
    b.finish(1)
}

/// Map a mesh bilinearly from its local rectangle (0..w, 0..d) onto the quad
/// `c` = [(0,0), (w,0), (0,d), (w,d)] -> four target points. This is how a
/// rectangular generator output becomes an irregular four-sided house that
/// slots into the block plan without gaps.
pub fn deform(mesh: &mut ModelMesh, w: f32, d: f32, c: [[f32; 2]; 4]) {
    for v in mesh.verts.iter_mut() {
        let u = v.pos[0] / w;
        let t = v.pos[2] / d;
        let x = (1.0 - u) * (1.0 - t) * c[0][0]
            + u * (1.0 - t) * c[1][0]
            + (1.0 - u) * t * c[2][0]
            + u * t * c[3][0];
        let z = (1.0 - u) * (1.0 - t) * c[0][1]
            + u * (1.0 - t) * c[1][1]
            + (1.0 - u) * t * c[2][1]
            + u * t * c[3][1];
        v.pos[0] = x;
        v.pos[2] = z;
    }
}

impl Manor {
    /// Free quad with the default uv (facade order: bl, br, tr, tl).
    fn face(&mut self, mat: u32, p: [[f32; 3]; 4]) {
        self.b
            .quad(mat, p, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    }

    /// Roman-style parade entrance: four pale columns on bases carry an
    /// entablature and a triangular pediment; a wide three-step perron leads
    /// to a tall portal. The portal opening itself was carved out of the
    /// street wall by the wall builder for `roman` houses.
    fn roman_portico(&mut self) {
        let d = self.d;
        let xc = 2.0 * self.door_tile as f32 + 1.0;

        // ---- perron: podium (top 0.45) + two steps -------------------------
        let pw = 2.7f32;
        self.face(
            M_STONE,
            [
                [xc - pw, 0.45, d],
                [xc - pw, 0.45, d + 1.7],
                [xc + pw, 0.45, d + 1.7],
                [xc + pw, 0.45, d],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - pw, 0.30, d + 1.7],
                [xc + pw, 0.30, d + 1.7],
                [xc + pw, 0.45, d + 1.7],
                [xc - pw, 0.45, d + 1.7],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - pw + 0.3, 0.30, d + 1.7],
                [xc - pw + 0.3, 0.30, d + 2.15],
                [xc + pw - 0.3, 0.30, d + 2.15],
                [xc + pw - 0.3, 0.30, d + 1.7],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - pw + 0.3, 0.15, d + 2.15],
                [xc + pw - 0.3, 0.15, d + 2.15],
                [xc + pw - 0.3, 0.30, d + 2.15],
                [xc - pw + 0.3, 0.30, d + 2.15],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - pw + 0.6, 0.15, d + 2.15],
                [xc - pw + 0.6, 0.15, d + 2.6],
                [xc + pw - 0.6, 0.15, d + 2.6],
                [xc + pw - 0.6, 0.15, d + 2.15],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - pw + 0.6, 0.0, d + 2.6],
                [xc + pw - 0.6, 0.0, d + 2.6],
                [xc + pw - 0.6, 0.15, d + 2.6],
                [xc - pw + 0.6, 0.15, d + 2.6],
            ],
        );
        // side cheeks close the podium and both steps down to the ground —
        // flat treads alone read as cardboard from the sides (user report)
        for sx in [xc - pw, xc + pw] {
            self.face(
                M_STONE,
                [
                    [sx, -1.2, d],
                    [sx, -1.2, d + 1.7],
                    [sx, 0.45, d + 1.7],
                    [sx, 0.45, d],
                ],
            );
        }
        for sx in [xc - pw + 0.3, xc + pw - 0.3] {
            self.face(
                M_STONE,
                [
                    [sx, -1.2, d + 1.7],
                    [sx, -1.2, d + 2.15],
                    [sx, 0.30, d + 2.15],
                    [sx, 0.30, d + 1.7],
                ],
            );
        }
        for sx in [xc - pw + 0.6, xc + pw - 0.6] {
            self.face(
                M_STONE,
                [
                    [sx, -1.2, d + 2.15],
                    [sx, -1.2, d + 2.6],
                    [sx, 0.15, d + 2.6],
                    [sx, 0.15, d + 2.15],
                ],
            );
        }

        // ---- the tall portal itself (0.45 .. 4.75) --------------------------
        self.face(
            M_DOOR,
            [
                [xc - 1.0, 0.45, d],
                [xc + 1.0, 0.45, d],
                [xc + 1.0, 4.75, d],
                [xc - 1.0, 4.75, d],
            ],
        );

        // ---- four columns ----------------------------------------------------
        for cx in [xc - 2.65, xc - 1.35, xc + 1.35, xc + 2.65] {
            let zc = d + 1.2;
            let rb = 0.33f32; // plinth block
            self.face(
                M_STONE,
                [
                    [cx - rb, 0.45, zc + rb],
                    [cx + rb, 0.45, zc + rb],
                    [cx + rb, 0.75, zc + rb],
                    [cx - rb, 0.75, zc + rb],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx - rb, 0.45, zc - rb],
                    [cx - rb, 0.45, zc + rb],
                    [cx - rb, 0.75, zc + rb],
                    [cx - rb, 0.75, zc - rb],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx + rb, 0.45, zc + rb],
                    [cx + rb, 0.45, zc - rb],
                    [cx + rb, 0.75, zc - rb],
                    [cx + rb, 0.75, zc + rb],
                ],
            );
            // shaft: octagonal prism
            let r = 0.24f32;
            let yt = 6.05f32;
            for k in 0..8 {
                let a0 = (k as f32 + 0.5) * std::f32::consts::PI / 4.0;
                let a1 = ((k + 1) as f32 + 0.5) * std::f32::consts::PI / 4.0;
                let (s0, c0) = a0.sin_cos();
                let (s1, c1) = a1.sin_cos();
                self.face(
                    M_STONE,
                    [
                        [cx + r * c0, 0.75, zc + r * s0],
                        [cx + r * c1, 0.75, zc + r * s1],
                        [cx + r * c1, yt, zc + r * s1],
                        [cx + r * c0, yt, zc + r * s0],
                    ],
                );
            }
            // capital: two stepped blocks
            let r1 = 0.30f32;
            self.face(
                M_STONE,
                [
                    [cx - r1, yt, zc + r1],
                    [cx + r1, yt, zc + r1],
                    [cx + r1, yt + 0.18, zc + r1],
                    [cx - r1, yt + 0.18, zc + r1],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx - r1, yt, zc - r1],
                    [cx - r1, yt, zc + r1],
                    [cx - r1, yt + 0.18, zc + r1],
                    [cx - r1, yt + 0.18, zc - r1],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx + r1, yt, zc + r1],
                    [cx + r1, yt, zc - r1],
                    [cx + r1, yt + 0.18, zc - r1],
                    [cx + r1, yt + 0.18, zc + r1],
                ],
            );
            let r2 = 0.4f32;
            self.face(
                M_STONE,
                [
                    [cx - r2, yt + 0.18, zc + r2],
                    [cx + r2, yt + 0.18, zc + r2],
                    [cx + r2, yt + 0.4, zc + r2],
                    [cx - r2, yt + 0.4, zc + r2],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx - r2, yt + 0.18, zc - r2],
                    [cx - r2, yt + 0.18, zc + r2],
                    [cx - r2, yt + 0.4, zc + r2],
                    [cx - r2, yt + 0.4, zc - r2],
                ],
            );
            self.face(
                M_STONE,
                [
                    [cx + r2, yt + 0.18, zc + r2],
                    [cx + r2, yt + 0.18, zc - r2],
                    [cx + r2, yt + 0.4, zc - r2],
                    [cx + r2, yt + 0.4, zc + r2],
                ],
            );
        }

        // ---- entablature: architrave / frieze / cornice ---------------------
        let ex = 3.15f32;
        let zf = d + 1.85;
        let zb = d + 0.08;
        for (y0, y1, mat) in [
            (6.45f32, 6.75, M_STONE),
            (6.75, 7.0, M_BAND),
            (7.0, 7.35, M_STONE),
        ] {
            self.face(
                mat,
                [
                    [xc - ex, y0, zf],
                    [xc + ex, y0, zf],
                    [xc + ex, y1, zf],
                    [xc - ex, y1, zf],
                ],
            );
            self.face(
                mat,
                [
                    [xc - ex, y0, zb],
                    [xc - ex, y0, zf],
                    [xc - ex, y1, zf],
                    [xc - ex, y1, zb],
                ],
            );
            self.face(
                mat,
                [
                    [xc + ex, y0, zf],
                    [xc + ex, y0, zb],
                    [xc + ex, y1, zb],
                    [xc + ex, y1, zf],
                ],
            );
        }

        // ---- pediment: tympanum + two raking slopes -------------------------
        let py = 7.35f32;
        let ph = 1.65f32;
        let zf2 = d + 1.88;
        let zb2 = d + 0.12;
        self.face(
            M_STONE,
            [
                [xc - ex, py, zf2],
                [xc + ex, py, zf2],
                [xc + ex, py, zf2],
                [xc, py + ph, zf2],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc - ex, py, zf2],
                [xc, py + ph, zf2],
                [xc, py + ph, zb2],
                [xc - ex, py, zb2],
            ],
        );
        self.face(
            M_STONE,
            [
                [xc, py + ph, zf2],
                [xc + ex, py, zf2],
                [xc + ex, py, zb2],
                [xc, py + ph, zb2],
            ],
        );
    }

    fn wall_quad(&mut self, mat: u32, side: u8, t0: f32, t1: f32, y0: f32, y1: f32, x_or_z: f32) {
        let p = match side {
            // south (+z): u=+x
            0 => [
                [t0, y0, x_or_z],
                [t1, y0, x_or_z],
                [t1, y1, x_or_z],
                [t0, y1, x_or_z],
            ],
            // north (-z): u=-x
            1 => [
                [t1, y0, x_or_z],
                [t0, y0, x_or_z],
                [t0, y1, x_or_z],
                [t1, y1, x_or_z],
            ],
            // west (-x): u=+z
            2 => [
                [x_or_z, y0, t0],
                [x_or_z, y0, t1],
                [x_or_z, y1, t1],
                [x_or_z, y1, t0],
            ],
            // east (+x): u=-z
            _ => [
                [x_or_z, y0, t1],
                [x_or_z, y0, t0],
                [x_or_z, y1, t0],
                [x_or_z, y1, t1],
            ],
        };
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        self.b.quad(mat, p, uv);
    }

    /// Like [`wall_quad`] but with an explicit u range so sub-tile strips
    /// (bay margins, piers) show the wall texture at its natural 2 m scale
    /// instead of stretching the whole tile into the strip.
    fn wall_quad_uv(
        &mut self,
        mat: u32,
        side: u8,
        t0: f32,
        t1: f32,
        y0: f32,
        y1: f32,
        x_or_z: f32,
        u0: f32,
        u1: f32,
    ) {
        let p = match side {
            // south (+z): u=+x
            0 => [
                [t0, y0, x_or_z],
                [t1, y0, x_or_z],
                [t1, y1, x_or_z],
                [t0, y1, x_or_z],
            ],
            // north (-z): u=-x
            1 => [
                [t1, y0, x_or_z],
                [t0, y0, x_or_z],
                [t0, y1, x_or_z],
                [t1, y1, x_or_z],
            ],
            // west (-x): u=+z
            2 => [
                [x_or_z, y0, t0],
                [x_or_z, y0, t1],
                [x_or_z, y1, t1],
                [x_or_z, y1, t0],
            ],
            // east (+x): u=-z
            _ => [
                [x_or_z, y0, t1],
                [x_or_z, y0, t0],
                [x_or_z, y1, t0],
                [x_or_z, y1, t1],
            ],
        };
        let uv = [[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]];
        self.b.quad(mat, p, uv);
    }

    /// Bay windows (эркеры) flanking the entrance: the nearest free 4 m span
    /// on each side of the door zone becomes a projection of the upper floors
    /// (0.85 m deep) on rusticated cheeks, with a closed underside on three
    /// corbels and a small cornice cap just under the eaves. The built spans
    /// are recorded in `bay_x` so balconies stay off the bays.
    fn erkers(&mut self, r: &mut Rng) {
        let tiles = self.tiles;
        if tiles < 8 {
            return; // too narrow to flank the entrance with bays
        }
        let dt = self.door_tile;
        let clear = |t: usize| -> bool {
            if t == dt || t >= tiles {
                return false;
            }
            if let Some(a) = self.arch {
                if t == a || t == a + 1 {
                    return false;
                }
            }
            true
        };
        // every wall tile the span [x0, x1) touches must be free
        let fits = |x0: f32, x1: f32| -> bool {
            let t0 = (x0 / 2.0).floor().max(0.0) as usize;
            let t1 = ((x1 - 1e-3) / 2.0).floor() as usize;
            t0 >= 1 && t1 + 1 < tiles && (t0..=t1).all(clear)
        };
        // bays flank the entrance flush against its jamb; the width is
        // picked at random among the spans that fit, so narrow and wide
        // bays mix along a facade (a wider bay takes two windows per
        // floor, a narrow one — a single window)
        let mut bays: Vec<(f32, f32)> = Vec::new();
        let widths = [6.4f32, 5.8, 5.2, 4.6, 4.0, 3.6];
        let x_door = 2.0 * dt as f32;
        // left of the entrance: the right edge sits flush on the door jamb
        let cand: Vec<f32> = widths
            .iter()
            .copied()
            .filter(|&bw| fits(x_door - bw, x_door))
            .collect();
        if !cand.is_empty() {
            // half the time take the widest span that fits (those take two
            // windows), the rest of the time any one — so both narrow and
            // wide bays show up
            let bw = if r.f() < 0.5 {
                cand[0]
            } else {
                cand[r.below(cand.len() as u32) as usize]
            };
            bays.push((x_door - bw, x_door));
        }
        // right of the entrance: the left edge sits flush on the right jamb
        let xr = x_door + 2.0;
        let cand: Vec<f32> = widths
            .iter()
            .copied()
            .filter(|&bw| fits(xr, xr + bw))
            .collect();
        if !cand.is_empty() {
            let bw = if r.f() < 0.5 {
                cand[0]
            } else {
                cand[r.below(cand.len() as u32) as usize]
            };
            bays.push((xr, xr + bw));
        }
        let y0 = self.heights[0] + self.heights[1];
        let y_top: f32 = self.heights.iter().sum::<f32>() - 0.25; // under the eaves
        if y_top <= y0 + 0.8 || bays.is_empty() {
            return; // no room above the entrance floor
        }
        println!(
            "  erkers: {n} bay(s) at {bays:?} (tiles {tiles}, door tile {dt}, w {w} m)",
            n = bays.len(),
            w = self.w
        );
        let heights = self.heights.clone(); // avoid an immutable self borrow
        for &(bx0, bx1) in &bays {
            // the bay depth varies a little from house to house
            let dep = 0.75 + 0.1 * r.below(3) as f32;
            let zf = self.d + dep;
            // front: one window per floor on a narrow bay, two on a wide
            // one; the margins and the pier between the windows show the
            // wall texture at its natural 2 m scale
            let bw = bx1 - bx0;
            let two = bw >= 5.5;
            let lo = bw - if two { 4.0 } else { 2.0 };
            let (m, pier) = if two {
                (lo / 4.0, lo / 2.0)
            } else {
                (lo / 2.0, 0.0)
            };
            let mut y = 0.0f32;
            for (fi, fh) in heights.iter().enumerate() {
                if fi >= 2 {
                    let mwin = if fi == 2 && r.f() < 0.6 {
                        M_WIN_SANDRIK
                    } else {
                        M_WIN
                    };
                    let ye = (y + fh).min(y_top);
                    if ye > y + 0.05 {
                        let mut xx = bx0;
                        self.wall_quad_uv(M_WALL, 0, xx, xx + m, y, ye, zf, 0.0, m / 2.0);
                        xx += m;
                        self.wall_quad_uv(mwin, 0, xx, xx + 2.0, y, ye, zf, 0.0, 1.0);
                        xx += 2.0;
                        if two {
                            self.wall_quad_uv(M_WALL, 0, xx, xx + pier, y, ye, zf, 0.0, pier / 2.0);
                            xx += pier;
                            self.wall_quad_uv(mwin, 0, xx, xx + 2.0, y, ye, zf, 0.0, 1.0);
                            xx += 2.0;
                        }
                        self.wall_quad_uv(M_WALL, 0, xx, bx1, y, ye, zf, 0.0, (bx1 - xx) / 2.0);
                    }
                }
                y += fh;
            }
            // cheeks
            for sx in [bx0, bx1] {
                self.face(
                    M_RUST,
                    [
                        [sx, y0, self.d],
                        [sx, y0, zf],
                        [sx, y_top, zf],
                        [sx, y_top, self.d],
                    ],
                );
            }
            // closed underside + three corbels
            self.face(
                M_RUST,
                [
                    [bx0, y0, zf],
                    [bx1, y0, zf],
                    [bx1, y0, self.d],
                    [bx0, y0, self.d],
                ],
            );
            for cx in [bx0 + 0.35, (bx0 + bx1) * 0.5, bx1 - 0.35] {
                self.face(
                    M_RUST,
                    [
                        [cx - 0.12, y0 - 0.55, self.d + 0.25],
                        [cx + 0.12, y0 - 0.55, self.d + 0.25],
                        [cx + 0.12, y0, zf - 0.1],
                        [cx - 0.12, y0, zf - 0.1],
                    ],
                );
            }
            // cornice cap: top slab + a 0.1 m fascia ring
            for p in [
                [
                    [bx0 - 0.12, y_top + 0.10, self.d - 0.05],
                    [bx1 + 0.12, y_top + 0.10, self.d - 0.05],
                    [bx1 + 0.12, y_top + 0.10, zf + 0.12],
                    [bx0 - 0.12, y_top + 0.10, zf + 0.12],
                ],
                [
                    [bx0 - 0.12, y_top, zf + 0.12],
                    [bx1 + 0.12, y_top, zf + 0.12],
                    [bx1 + 0.12, y_top + 0.10, zf + 0.12],
                    [bx0 - 0.12, y_top + 0.10, zf + 0.12],
                ],
                [
                    [bx0 - 0.12, y_top, self.d - 0.05],
                    [bx0 - 0.12, y_top, zf + 0.12],
                    [bx0 - 0.12, y_top + 0.10, zf + 0.12],
                    [bx0 - 0.12, y_top + 0.10, self.d - 0.05],
                ],
                [
                    [bx1 + 0.12, y_top, zf + 0.12],
                    [bx1 + 0.12, y_top, self.d - 0.05],
                    [bx1 + 0.12, y_top + 0.10, self.d - 0.05],
                    [bx1 + 0.12, y_top + 0.10, zf + 0.12],
                ],
            ] {
                self.face(M_BAND, p);
            }
        }
        self.bay_x = bays;
    }
}

/// Build one tenement house; returns the mesh, its palette textures and the
/// footprint size (w, d) in metres so the caller can place and rotate it.
pub fn generate_manor(
    seed: u64,
    arch: bool,
    size: Option<(f32, f32)>,
    corner: bool,
    roman: bool,
    bays: bool,
) -> (ModelMesh, Vec<(u32, u32, Vec<u8>)>, f32, f32, Option<(f32, f32)>) {
    let mut r = Rng::new(seed ^ 0x0050_4954_4552_0000); // "PITER"
    let pal = Palette::new(&mut r);

    // footprint: fixed when the block planner sums a row exactly, else rolled
    let (w, d) = if let Some((fw, fd)) = size {
        (fw.round(), fd.round())
    } else {
        let t = [6usize, 8, 10, 12, 14][r.below(5) as usize]; // 12..28 m
        (2.0 * t as f32, [10.0f32, 12.0, 14.0][r.below(3) as usize])
    };
    let tiles = (w / 2.0).round() as usize;
    let upper = 1 + r.below(5); // 1..5 upper floors → 3..7 storeys with basement
    let mut heights = vec![2.6f32, 3.9];
    for k in 0..upper {
        heights.push(3.4 - 0.25 * k as f32);
    }
    let win_start = r.below(2) as usize;
    let win_tiles: Vec<usize> = (win_start..tiles)
        .step_by(2)
        // keep one 2 m tile of blank wall between the outermost windows and
        // the building corners — a nalidnik/sandrik glued to the corner reads
        // wrong (the user asked for corner clearance)
        .filter(|&t| t >= 1 && t <= tiles - 2)
        .collect();
    let door_tile = win_tiles[win_tiles.len() / 2];
    // a Roman portico wants clean wall around the portal: no windows on the
    // two tiles flanking the entrance
    let win_tiles: Vec<usize> = if roman {
        win_tiles
            .into_iter()
            .filter(|&t| t + 1 != door_tile && t != door_tile + 1)
            .collect()
    } else {
        win_tiles
    };
    // carriage arch: two street-wall tiles (4 m), close to the facade centre,
    // never overlapping the entrance; the passage runs right through to the yard
    let arch = if arch && tiles >= 6 {
        // the Roman portico owns the wall around the entrance: an arch squeezed
        // behind a column reads as blocked, so keep the opening outside the
        // colonnade footprint (columns at xc ± 2.65, edge 2.91 + 0.25 air)
        let m = if roman { 2.9 } else { 0.0 };
        let xc = 2.0 * door_tile as f32 + 1.0;
        let mut cand: Vec<usize> = (1..tiles - 2)
            // only the Roman house needs the arch clear of the (absent for
            // non-Roman arch houses) entrance tiles
            .filter(|&a| roman || (a != door_tile && a + 1 != door_tile))
            .filter(|&a| {
                let (x0, x1) = (2.0 * a as f32, 2.0 * (a + 2) as f32);
                !(x0 < xc + m && x1 > xc - m)
            })
            .collect();
        cand.sort_by_key(|&a| ((2 * a as i32 + 1) - tiles as i32).abs());
        cand.first().copied()
    } else {
        None
    };
    if roman && arch.is_some() {
        println!("  roman arch at tiles {a}..{b} of {tiles} m/{tw}", a = arch.unwrap(), b = arch.unwrap() + 1, tiles = tiles, tw = 2 * tiles);
    }
    // yard side: a black entrance (always) and sometimes coach-house gates
    let mut yard_door = None;
    let mut stable = None;
    {
        let plain_tiles: Vec<usize> = (0..tiles).filter(|t| !win_tiles.contains(t)).collect();
        if !plain_tiles.is_empty() && r.f() < 0.85 {
            yard_door = Some(plain_tiles[r.below(plain_tiles.len() as u32) as usize]);
        }
        if r.f() < 0.45 {
            let cand: Vec<usize> = win_tiles
                .iter()
                .copied()
                .filter(|&t| Some(t) != yard_door && Some(t) != arch && Some(t) != arch.map(|a| a + 1))
                .collect();
            if !cand.is_empty() {
                stable = Some(cand[r.below(cand.len() as u32) as usize]);
            }
        }
    }

    let mut m = Manor {
        b: Builder::new(),
        pal,
        w,
        d,
        heights,
        y_band: Vec::new(),
        tiles,
        win_tiles,
        door_tile,
        arch,
        yard_door,
        stable,
        corner,
        roman,
        bays,
        bay_x: Vec::new(),
    };
    m.build(&mut r);
    let texs = m.textures();
    let mesh = m.b.finish(N_MATS);
    let arch_range = arch.map(|a| (2.0 * a as f32, 2.0 * (a + 2) as f32));
    (mesh, texs, w, d, arch_range)
}

impl Manor {
    fn build(&mut self, r: &mut Rng) {
        let tiles = self.tiles;
        let w = self.w;
        let d = self.d;
        let n_floors = self.heights.len();
        // a house with a carriage arch carries no parade entrance: no door,
        // no porch, no bays on its street facade (user's rule)
        let no_parade = self.arch.is_some() && !self.roman;
        if self.bays && !self.roman && self.arch.is_none() {
            self.erkers(r);
        }

        // ---- walls, per floor, per 2 m tile ------------------------------
        let mut y = 0.0f32;
        for fi in 0..n_floors {
            let fh = self.heights[fi];
            let (mat_wall, mat_win) = match fi {
                0 => (M_BASE, M_BASE_WIN),
                1 => (
                    M_RUST,
                    if r.f() < 0.55 {
                        M_WIN_SANDRIK
                    } else {
                        M_WIN_ARCH
                    },
                ),
                _ => (M_WALL, M_WIN),
            };
            // south (street): the parade facade — entrance, nalidniks,
            // sandriks; pierced by the carriage arch tiles
            for t in 0..tiles {
                let x0 = 2.0 * t as f32;
                if let Some(a) = self.arch {
                    if t == a || t == a + 1 {
                        if fi == 0 {
                            continue; // the passage: no wall here
                        }
                        if fi == 1 {
                            self.wall_quad(mat_wall, 0, x0, x0 + 2.0, ARCH_H, y + fh, d);
                            continue;
                        }
                    }
                }
                // Roman portal: one tall opening through both lower floors
                // (the door body itself is drawn by the portico builder)
                if self.roman && t == self.door_tile {
                    if fi == 0 {
                        self.wall_quad(M_BASE, 0, x0, x0 + 2.0, 0.0, 0.45, d);
                        continue;
                    }
                    if fi == 1 {
                        self.wall_quad(mat_wall, 0, x0, x0 + 2.0, 4.75, y + fh, d);
                        continue;
                    }
                }
                let mat = if fi == 0 && t == self.door_tile && !no_parade {
                    M_DOOR
                } else if self.win_tiles.contains(&t)
                    && (!(fi == 0 && t == self.door_tile) || no_parade)
                {
                    if fi == 1 && t == self.door_tile && !no_parade {
                        mat_wall // above the entrance keep it plain
                    } else {
                        mat_win
                    }
                } else {
                    mat_wall
                };
                self.wall_quad(mat, 0, x0, x0 + 2.0, y, y + fh, d);
            }
            // north (yard): a plainer back — bare windows, a black entrance,
            // coach-house gates; the arch passes through here as well
            for t in 0..tiles {
                let x0 = 2.0 * t as f32;
                if let Some(a) = self.arch {
                    if t == a || t == a + 1 {
                        if fi == 0 {
                            continue;
                        }
                        if fi == 1 {
                            self.wall_quad(mat_wall, 1, x0, x0 + 2.0, ARCH_H, y + fh, 0.0);
                            continue;
                        }
                    }
                }
                let mat = if self.corner {
                    // a corner house backs onto the neighbouring wing: this
                    // wall is a blind firewall as well — no yard door, no
                    // windows, nothing
                    mat_wall
                } else if fi == 0 && Some(t) == self.stable {
                    M_STABLE
                } else if fi == 0 && Some(t) == self.yard_door {
                    M_BLACK_DOOR
                } else if self.win_tiles.contains(&t) {
                    if fi == 0 {
                        M_BASE_WIN
                    } else {
                        M_WIN_PLAIN
                    }
                } else {
                    mat_wall
                };
                self.wall_quad(mat, 1, x0, x0 + 2.0, y, y + fh, 0.0);
            }
            // ends: adjoining end walls are blank brick firewalls
            // (брандмауэры) — no windows, no trim anywhere on them; only a
            // corner house keeps its second parade facade on the east end,
            // which alone keeps windows and its plaster wall
            let dts = (d / 2.0).floor() as usize;
            for t in 0..dts {
                let z0 = 2.0 * t as f32;
                if self.corner && t >= 1 && t <= dts.saturating_sub(2) && t % 2 == 1 {
                    let mat = if fi == 0 { M_BASE_WIN } else { mat_win };
                    self.wall_quad(mat, 3, z0, z0 + 2.0, y, y + fh, w);
                } else {
                    // blind firewalls are rendered in the house's own
                    // plaster (Petersburg: the side walls matched the
                    // facade, they were not left as raw brick)
                    self.wall_quad(mat_wall, 3, z0, z0 + 2.0, y, y + fh, w);
                }
                self.wall_quad(mat_wall, 2, z0, z0 + 2.0, y, y + fh, 0.0);
            }
            y += fh;
            self.y_band.push(y);
        }
        let y_wall_top = y;

        // ---- carriage arch: the passage through the house ------------------
        if let Some(a) = self.arch {
            let (a0, a1) = (2.0 * a as f32, 2.0 * (a + 2) as f32);
            let ah = ARCH_H;
            // jambs (inside faces of the passage)
            self.b.quad(
                M_BASE,
                [[a0, 0.0, 0.0], [a0, 0.0, d], [a0, ah, d], [a0, ah, 0.0]],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 2.0, ah / 2.6], [0.0, ah / 2.6]],
            );
            self.b.quad(
                M_BASE,
                [[a1, 0.0, d], [a1, 0.0, 0.0], [a1, ah, 0.0], [a1, ah, d]],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 2.0, ah / 2.6], [0.0, ah / 2.6]],
            );
            // ceiling
            self.b.quad(
                M_BASE,
                [[a0, ah, d], [a1, ah, d], [a1, ah, 0.0], [a0, ah, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [1.0, d / 2.0], [0.0, d / 2.0]],
            );
            // stone edging around the opening, street and yard faces
            for zz in [d + 0.06, -0.06] {
                for (x0, x1) in [(a0 - 0.22, a0), (a1, a1 + 0.22)] {
                    self.b.quad(
                        M_BAND,
                        [[x0, 0.0, zz], [x1, 0.0, zz], [x1, ah + 0.5, zz], [x0, ah + 0.5, zz]],
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    );
                }
                self.b.quad(
                    M_BAND,
                    [
                        [a0 - 0.22, ah + 0.25, zz],
                        [a1 + 0.22, ah + 0.25, zz],
                        [a1 + 0.22, ah + 0.5, zz],
                        [a0 - 0.22, ah + 0.5, zz],
                    ],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                );
            }
        }

        // ---- string courses between floors + stepped crowning cornice -----
        // the courses wrap the facades only: adjoining end walls are blank
        // firewalls and carry no trim; only a corner house trims its east end
        let band = |b: &mut Builder, yy: f32, out: f32, w: f32, d: f32, eas: bool, nor: bool| {
            let a = 0.0f32;
            // south
            b.quad(
                M_BAND,
                [
                    [-a, yy - 0.10, d + out],
                    [w + a, yy - 0.10, d + out],
                    [w + a, yy + 0.14, d + out],
                    [-a, yy + 0.14, d + out],
                ],
                [[0.0, 0.0], [w / 2.0, 0.0], [w / 2.0, 1.0], [0.0, 1.0]],
            );
            // north: skipped on corner houses (that wall is a firewall)
            if nor {
                b.quad(
                    M_BAND,
                    [
                        [w + a, yy - 0.10, -out],
                        [-a, yy - 0.10, -out],
                        [-a, yy + 0.14, -out],
                        [w + a, yy + 0.14, -out],
                    ],
                    [[0.0, 0.0], [w / 2.0, 0.0], [w / 2.0, 1.0], [0.0, 1.0]],
                );
            }
            // east: only when this end is a parade facade
            if eas {
                b.quad(
                    M_BAND,
                    [
                        [w + out, yy - 0.10, d + a],
                        [w + out, yy - 0.10, -a],
                        [w + out, yy + 0.14, -a],
                        [w + out, yy + 0.14, d + a],
                    ],
                    [[0.0, 0.0], [d / 2.0, 0.0], [d / 2.0, 1.0], [0.0, 1.0]],
                );
            }
        };
        let eas = self.corner;
        let nor = !self.corner;
        for k in 1..self.y_band.len() {
            band(&mut self.b, self.y_band[k], 0.10, w, d, eas, nor);
        }
        // crowning cornice: two stepped bands overhanging the wall
        band(&mut self.b, y_wall_top + 0.06, 0.24, w, d, eas, nor);
        band(&mut self.b, y_wall_top + 0.34, 0.38, w, d, eas, nor);

        // ---- balconies ----------------------------------------------------
        let balcony_floor = if r.f() < 0.75 { Some(2usize) } else { None };
        if let Some(bf) = balcony_floor {
            if bf < n_floors {
                let y0: f32 = self.heights[..bf].iter().sum();
                for (wi, &wt) in self.win_tiles.iter().enumerate() {
                    if wi % 2 == 1 && self.win_tiles.len() > 2 {
                        continue;
                    }
                    if bf == 1 && wt == self.door_tile {
                        continue;
                    }
                    if let Some(a) = self.arch {
                        if wt == a || wt == a + 1 {
                            continue; // no balcony over the carriage arch
                        }
                    }
                    {
                        let xcx = 2.0 * wt as f32 + 1.0;
                        if self
                            .bay_x
                            .iter()
                            .any(|&(x0, x1)| xcx > x0 - 0.6 && xcx < x1 + 0.6)
                        {
                            continue; // a bay window owns this wall span
                        }
                    }
                    // no balcony over the entrance (its canopy and lanterns
                    // hang right there) and none inside the Roman colonnade
                    // (the portico entablature would cut straight through it)
                    if wt == self.door_tile || wt + 1 == self.door_tile || wt == self.door_tile + 1 {
                        continue;
                    }
                    if self.roman {
                        let px_c = 2.0 * self.door_tile as f32 + 1.0;
                        if ((2.0 * wt as f32 + 1.0) - px_c).abs() < 4.6 {
                            continue;
                        }
                    }
                    let xc = 2.0 * wt as f32 + 1.0;
                    let x0 = xc - 1.15;
                    let x1 = xc + 1.15;
                    let z_out = d + 0.95;
                    // slab top
                    self.b.quad(
                        M_IRON,
                        [
                            [x0, y0 + 0.18, d],
                            [x0, y0 + 0.18, z_out],
                            [x1, y0 + 0.18, z_out],
                            [x1, y0 + 0.18, d],
                        ],
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                    );
                    // slab underside
                    self.b.quad(
                        M_IRON,
                        [
                            [x0, y0 + 0.02, z_out],
                            [x0, y0 + 0.02, d],
                            [x1, y0 + 0.02, d],
                            [x1, y0 + 0.02, z_out],
                        ],
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                    );
                    // slab front fascia
                    self.b.quad(
                        M_IRON,
                        [
                            [x0, y0 + 0.02, z_out],
                            [x1, y0 + 0.02, z_out],
                            [x1, y0 + 0.18, z_out],
                            [x0, y0 + 0.18, z_out],
                        ],
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    );
                    // corbels
                    for cx_ in [x0 + 0.12, x1 - 0.12] {
                        self.b.quad(
                            M_IRON,
                            [
                                [cx_ - 0.06, y0 - 0.55, d],
                                [cx_ + 0.06, y0 - 0.55, d],
                                [cx_ + 0.06, y0 + 0.02, d + 0.75],
                                [cx_ - 0.06, y0 + 0.02, d + 0.75],
                            ],
                            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        );
                    }
                    // railings (alpha texture): front + two sides
                    let rh = 1.05f32;
                    let rail = |b: &mut Builder, p: [[f32; 3]; 4], u: f32| {
                        b.quad(M_RAILING, p, [[0.0, 0.0], [u, 0.0], [u, 1.0], [0.0, 1.0]]);
                    };
                    rail(
                        &mut self.b,
                        [
                            [x0, y0 + 0.18, z_out],
                            [x1, y0 + 0.18, z_out],
                            [x1, y0 + 0.18 + rh, z_out],
                            [x0, y0 + 0.18 + rh, z_out],
                        ],
                        1.4,
                    );
                    rail(
                        &mut self.b,
                        [
                            [x0, y0 + 0.18, d],
                            [x0, y0 + 0.18, z_out],
                            [x0, y0 + 0.18 + rh, z_out],
                            [x0, y0 + 0.18 + rh, d],
                        ],
                        0.6,
                    );
                    rail(
                        &mut self.b,
                        [
                            [x1, y0 + 0.18, z_out],
                            [x1, y0 + 0.18, d],
                            [x1, y0 + 0.18 + rh, d],
                            [x1, y0 + 0.18 + rh, z_out],
                        ],
                        0.6,
                    );
                }
            }
        }

        // ---- porch: steps, canopy on brackets, lanterns -------------------
        if !self.roman && !no_parade {
            let dt = self.door_tile;
            let xc = 2.0 * dt as f32 + 1.0;
            // three SOLID steps: tread, riser and side cheeks per step,
            // sunk into the ground (-1.2) so nothing hollow shows from the
            // sides or below (the user caught flat risers reading as cardboard)
            for s in 0..3 {
                let sw = 1.9 - 0.18 * s as f32;
                let zh0 = d + 0.30 * (3 - s) as f32;
                let zh1 = zh0 + 0.30;
                let yt = 0.14 * (s + 1) as f32 + 0.16;
                let yb = -1.2f32;
                self.b.quad(
                    M_BAND,
                    [
                        [xc - sw, yt, zh1],
                        [xc + sw, yt, zh1],
                        [xc + sw, yt, zh0],
                        [xc - sw, yt, zh0],
                    ],
                    [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
                );
                self.b.quad(
                    M_BAND,
                    [
                        [xc - sw, yb, zh1],
                        [xc + sw, yb, zh1],
                        [xc + sw, yt, zh1],
                        [xc - sw, yt, zh1],
                    ],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                );
                for sx in [xc - sw, xc + sw] {
                    self.b.quad(
                        M_BAND,
                        [
                            [sx, yb, zh0],
                            [sx, yb, zh1],
                            [sx, yt, zh1],
                            [sx, yt, zh0],
                        ],
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    );
                }
            }
            // canopy slab over the entrance + brackets
            let can_y = self.heights[0] + 0.55;
            self.b.quad(
                M_BAND,
                [
                    [xc - 1.6, can_y + 0.14, d],
                    [xc - 1.6, can_y + 0.14, d + 1.25],
                    [xc + 1.6, can_y + 0.14, d + 1.25],
                    [xc + 1.6, can_y + 0.14, d],
                ],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            );
            self.b.quad(
                M_BAND,
                [
                    [xc - 1.6, can_y, d + 1.25],
                    [xc + 1.6, can_y, d + 1.25],
                    [xc + 1.6, can_y + 0.14, d + 1.25],
                    [xc - 1.6, can_y + 0.14, d + 1.25],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            );
            for bx in [xc - 1.3, xc + 1.3] {
                self.b.quad(
                    M_IRON,
                    [
                        [bx - 0.06, can_y - 0.85, d],
                        [bx + 0.06, can_y - 0.85, d],
                        [bx + 0.06, can_y, d + 0.9],
                        [bx - 0.06, can_y, d + 0.9],
                    ],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                );
            }
            // wall lanterns beside the door
            for lx in [xc - 1.35, xc + 1.35] {
                let ly = 2.5f32;
                self.b.quad(
                    M_IRON,
                    [
                        [lx - 0.14, ly - 0.28, d + 0.42],
                        [lx + 0.14, ly - 0.28, d + 0.42],
                        [lx + 0.14, ly + 0.28, d + 0.42],
                        [lx - 0.14, ly + 0.28, d + 0.42],
                    ],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                );
            }
        }
        if self.roman {
            self.roman_portico();
        }

        // ---- roof ---------------------------------------------------------
        let rh = if r.f() < 0.5 { 3.6 } else { 4.2 };
        let y1 = y_wall_top + rh;
        let hipped = r.f() < 0.6;
        let zm = d / 2.0;
        let inset = if hipped { (d * 0.42).min(w * 0.28) } else { 0.0 };
        let roof_uv = |len: f32, dep: f32| {
            [
                [0.0, 0.0],
                [len / 2.0, 0.0],
                [len / 2.0, dep / 2.0],
                [0.0, dep / 2.0],
            ]
        };
        if hipped {
            // south slope
            self.b.quad(
                M_ROOF,
                [
                    [-0.25, y_wall_top, d + 0.35],
                    [w + 0.25, y_wall_top, d + 0.35],
                    [w - inset, y1, zm],
                    [inset, y1, zm],
                ],
                roof_uv(w, d / 2.0),
            );
            // north slope
            self.b.quad(
                M_ROOF,
                [
                    [w + 0.25, y_wall_top, -0.35],
                    [-0.25, y_wall_top, -0.35],
                    [inset, y1, zm],
                    [w - inset, y1, zm],
                ],
                roof_uv(w, d / 2.0),
            );
            // west hip
            self.b.tri(
                M_ROOF,
                [
                    [-0.25, y_wall_top, d + 0.35],
                    [-0.25, y_wall_top, -0.35],
                    [inset, y1, zm],
                ],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, 1.6]],
            );
            // east hip
            self.b.tri(
                M_ROOF,
                [
                    [w + 0.25, y_wall_top, -0.35],
                    [w + 0.25, y_wall_top, d + 0.35],
                    [w - inset, y1, zm],
                ],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, 1.6]],
            );
        } else {
            // gable: two full slopes + gable walls at the ends
            self.b.quad(
                M_ROOF,
                [
                    [-0.25, y_wall_top, d + 0.35],
                    [w + 0.25, y_wall_top, d + 0.35],
                    [w + 0.25, y1, zm],
                    [-0.25, y1, zm],
                ],
                roof_uv(w, d / 2.0),
            );
            self.b.quad(
                M_ROOF,
                [
                    [w + 0.25, y_wall_top, -0.35],
                    [-0.25, y_wall_top, -0.35],
                    [-0.25, y1, zm],
                    [w + 0.25, y1, zm],
                ],
                roof_uv(w, d / 2.0),
            );
            self.b.tri(
                M_WALL,
                [
                    [-0.02, y_wall_top, -0.02],
                    [-0.02, y_wall_top, d + 0.02],
                    [-0.02, y1 - 0.15, zm],
                ],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, 1.0]],
            );
            self.b.tri(
                M_WALL,
                [
                    [w + 0.02, y_wall_top, d + 0.02],
                    [w + 0.02, y_wall_top, -0.02],
                    [w + 0.02, y1 - 0.15, zm],
                ],
                [[0.0, 0.0], [d / 2.0, 0.0], [d / 4.0, 1.0]],
            );
        }

        // ---- chimneys ------------------------------------------------------
        let chimneys = if n_floors <= 4 { 1 } else { 2 };
        let chim_x: Vec<f32> = (0..chimneys)
            .map(|c| {
                if chimneys == 1 {
                    w * 0.5
                } else {
                    w * (0.28 + 0.44 * c as f32)
                }
            })
            .collect();
        for &cx in &chim_x {
            chimney(&mut self.b, cx, zm, y1 - 0.4, 1.7, 0.38);
        }

        // ---- dormers on the front slope -------------------------------------
        {
            let dormers = if tiles >= 10 { 3 } else { 2 };
            for i in 0..dormers {
                let dx = w * (i as f32 + 1.0) / (dormers as f32 + 1.0);
                // elements must not intersect: a dormer never sits on a flue
                if chim_x.iter().any(|&c| (dx - c).abs() < 1.6) {
                    continue;
                }
                let t = 0.45f32;
                let dz = (d + 0.35) - (d + 0.35 - zm) * t;
                let dy = y_wall_top + rh * t;
                dormer(&mut self.b, dx, dz, dy);
            }
        }

        // ---- weather-vane on the ridge end ----------------------------------
        if r.f() < 0.6 {
            let vx = inset;
            let vy0 = y1 - 0.3;
            let vy1 = y1 + 1.6;
            self.b.quad(
                M_IRON,
                [
                    [vx - 0.04, vy0, zm - 0.04],
                    [vx + 0.04, vy0, zm - 0.04],
                    [vx + 0.04, vy1, zm - 0.04],
                    [vx - 0.04, vy1, zm - 0.04],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 4.0], [0.0, 4.0]],
            );
            self.b.quad(
                M_IRON,
                [
                    [vx, vy1 - 0.35, zm - 0.04],
                    [vx, vy1 - 0.35, zm + 0.51],
                    [vx, vy1 + 0.05, zm + 0.26],
                    [vx, vy1 + 0.05, zm - 0.04],
                ],
                [[0.0, 0.0], [1.0, 0.0], [0.7, 1.0], [0.0, 1.0]],
            );
        }

        // ---- drain pipes ----------------------------------------------------
        for px_ in [0.30f32, w - 0.30] {
            let py0 = 0.15;
            let py1 = y_wall_top + 0.12;
            let zc = d + 0.10;
            let hh = 0.10;
            // south face
            self.b.quad(
                M_IRON,
                [
                    [px_ - hh, py0, zc + hh],
                    [px_ + hh, py0, zc + hh],
                    [px_ + hh, py1, zc + hh],
                    [px_ - hh, py1, zc + hh],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 8.0], [0.0, 8.0]],
            );
            // north face
            self.b.quad(
                M_IRON,
                [
                    [px_ + hh, py0, zc - hh],
                    [px_ - hh, py0, zc - hh],
                    [px_ - hh, py1, zc - hh],
                    [px_ + hh, py1, zc - hh],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 8.0], [0.0, 8.0]],
            );
            // east face
            self.b.quad(
                M_IRON,
                [
                    [px_ + hh, py0, zc + hh],
                    [px_ + hh, py0, zc - hh],
                    [px_ + hh, py1, zc - hh],
                    [px_ + hh, py1, zc + hh],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 8.0], [0.0, 8.0]],
            );
            // west face
            self.b.quad(
                M_IRON,
                [
                    [px_ - hh, py0, zc - hh],
                    [px_ - hh, py0, zc + hh],
                    [px_ - hh, py1, zc + hh],
                    [px_ - hh, py1, zc - hh],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 8.0], [0.0, 8.0]],
            );
            // funnel
            self.b.quad(
                M_IRON,
                [
                    [px_ - 0.13, py1, zc + 0.02],
                    [px_ + 0.13, py1, zc + 0.02],
                    [px_ + 0.13, py1 + 0.3, zc + 0.02],
                    [px_ - 0.13, py1 + 0.3, zc + 0.02],
                ],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            );
        }

        // ---- anchor plates on the facade --------------------------------------
        for k in 0..n_floors.saturating_sub(1) {
            let yy = (self.y_band[k] - 1.4).max(1.2);
            for ax in [1.0f32, w - 1.0] {
                self.b.quad(
                    M_IRON,
                    [
                        [ax - 0.14, yy - 0.14, d + 0.03],
                        [ax + 0.14, yy - 0.14, d + 0.03],
                        [ax + 0.14, yy + 0.14, d + 0.03],
                        [ax - 0.14, yy + 0.14, d + 0.03],
                    ],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                );
            }
        }
    }

    /// All 14 house textures in material-slot order.
    fn textures(&self) -> Vec<(u32, u32, Vec<u8>)> {
        let pal = self.pal;
        let mut rr = Rng::new(0x7E57_0000 ^ ((self.tiles as u64) << 8) ^ self.heights.len() as u64);
        let wall = pal.wall;
        let base = pal.basement;
        let mut out = vec![
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, None, false, &mut rr)), // 0 M_WALL
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, None, true, &mut rr)),  // 1 M_RUST
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, Some(0), false, &mut rr)), // 2 M_WIN
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, Some(1), true, &mut rr)), // 3 M_WIN_SANDRIK
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, Some(2), true, &mut rr)), // 4 M_WIN_ARCH
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, base, None, true, &mut rr)), // 5 M_BASE
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, base, Some(3), true, &mut rr)), // 6 M_BASE_WIN
            (TEX_W as u32, TEX_H as u32, door_tex(&pal, &mut rr)),                    // 7 M_DOOR
            (TEX_S as u32, TEX_S as u32, roof_tex(&pal, &mut rr)),                    // 8 M_ROOF
            (TEX_S as u32, TEX_S as u32, brick_tex(&pal, &mut rr)),                   // 9 M_BRICK
            (TEX_S as u32, TEX_S as u32, iron_tex(&pal, &mut rr)),                    // 10 M_IRON
            (TEX_S as u32, 32, band_tex(&pal)),                                       // 11 M_BAND
            (TEX_S as u32, TEX_S as u32, dormer_tex(&pal, &mut rr)),                  // 12 M_DORMER
            (TEX_S as u32, 64, railing_tex(&pal)),                                    // 13 M_RAILING (alpha)
            (TEX_W as u32, TEX_H as u32, tile_tex(&pal, wall, Some(4), false, &mut rr)), // 14 M_WIN_PLAIN
            (TEX_W as u32, TEX_H as u32, black_door_tex(&pal, &mut rr)),              // 15 M_BLACK_DOOR
            (TEX_W as u32, TEX_H as u32, stable_tex(&pal, &mut rr)),                  // 16 M_STABLE
            (TEX_S as u32, TEX_S as u32, stone_tex(&pal)),                            // 17 M_STONE
        ];
        // store every tile bottom-up (see `flip_rows`): without this the whole
        // facade renders vertically mirrored and the doors show upside down
        for t in out.iter_mut() {
            flip_rows(&mut t.2, t.0 as usize, t.1 as usize);
        }
        out
    }
}

/// Brick chimney with a crown cap.
fn chimney(b: &mut Builder, x: f32, z: f32, y0: f32, h: f32, a: f32) {
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let (x0, x1, z0, z1) = (x - a, x + a, z - a, z + a);
    let y1 = y0 + h;
    b.quad(M_BRICK, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], uv);
    b.quad(M_BRICK, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], uv);
    b.quad(M_BRICK, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], uv);
    b.quad(M_BRICK, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], uv);
    b.quad(M_BRICK, [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], uv);
    // crown cap
    let c = a + 0.14;
    let y2 = y1 + 0.22;
    b.quad(
        M_BRICK,
        [
            [x - c, y1, z + c],
            [x + c, y1, z + c],
            [x + c, y2, z + c],
            [x - c, y2, z + c],
        ],
        uv,
    );
    b.quad(
        M_BRICK,
        [
            [x + c, y1, z - c],
            [x - c, y1, z - c],
            [x - c, y2, z - c],
            [x + c, y2, z - c],
        ],
        uv,
    );
    b.quad(
        M_BRICK,
        [
            [x + c, y1, z + c],
            [x + c, y1, z - c],
            [x + c, y2, z - c],
            [x + c, y2, z + c],
        ],
        uv,
    );
    b.quad(
        M_BRICK,
        [
            [x - c, y1, z - c],
            [x - c, y1, z + c],
            [x - c, y2, z + c],
            [x - c, y2, z - c],
        ],
        uv,
    );
    b.quad(
        M_BRICK,
        [
            [x - c, y2, z - c],
            [x - c, y2, z + c],
            [x + c, y2, z + c],
            [x + c, y2, z - c],
        ],
        uv,
    );
}

/// Small dormer: body with a window, a little gabled roof.
fn dormer(b: &mut Builder, x: f32, z: f32, y0: f32) {
    let w = 0.55f32;
    let dep = 0.5f32;
    let h = 1.0f32;
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    // front face with the dormer texture (round window)
    b.quad(
        M_DORMER,
        [
            [x - w, y0, z + dep],
            [x + w, y0, z + dep],
            [x + w, y0 + h, z + dep],
            [x - w, y0 + h, z + dep],
        ],
        uv,
    );
    // sides
    b.quad(
        M_WALL,
        [
            [x - w, y0, z],
            [x - w, y0, z + dep],
            [x - w, y0 + h, z + dep],
            [x - w, y0 + h, z],
        ],
        uv,
    );
    b.quad(
        M_WALL,
        [
            [x + w, y0, z + dep],
            [x + w, y0, z],
            [x + w, y0 + h, z],
            [x + w, y0 + h, z + dep],
        ],
        uv,
    );
    // little gable roof: front slope + two cheeks
    let zc = z + dep / 2.0;
    b.tri(
        M_ROOF,
        [
            [x - w - 0.08, y0 + h, z + dep + 0.08],
            [x + w + 0.08, y0 + h, z + dep + 0.08],
            [x, y0 + h + 0.5, zc],
        ],
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    );
    b.tri(
        M_ROOF,
        [
            [x - w - 0.08, y0 + h, z + dep + 0.08],
            [x, y0 + h + 0.5, zc],
            [x - w - 0.08, y0 + h, z - 0.02],
        ],
        [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
    );
    b.tri(
        M_ROOF,
        [
            [x + w + 0.08, y0 + h, z + dep + 0.08],
            [x + w + 0.08, y0 + h, z - 0.02],
            [x, y0 + h + 0.5, zc],
        ],
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    );
}
