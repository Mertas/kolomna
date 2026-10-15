//! Classify the facade "handwriting" rolls straight from the generated
//! textures: window-trim family (plain / eyebrow / keystone), entrance-door
//! drawing (ampire / classic / glazed) and railing pattern
//! (posts+scrolls / diagonal lattice / spears).
//!
//! Tiles are stored bottom-up (`flip_rows`), so original pixel (x, y)
//! lives at stored row `h - 1 - y`.
use kolomna_render::piter::generate_manor;

fn lu(p: &[u8], w: usize, x: usize, y: usize) -> i32 {
    let i = (y * w + x) * 4;
    (p[i] as i32 * 299 + p[i + 1] as i32 * 587 + p[i + 2] as i32 * 114) / 1000
}

fn median_row(p: &[u8], w: usize, y: usize, x0: usize, x1: usize, step: usize) -> i32 {
    let mut v: Vec<i32> = (x0..x1).step_by(step).map(|x| lu(p, w, x, y)).collect();
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let n = 600u64;
    let (mut f0, mut f6, mut f5) = (0u32, 0u32, 0u32);
    let (mut d_amp, mut d_cls, mut d_glz) = (0u32, 0u32, 0u32);
    let (mut r0, mut r1, mut r2) = (0u32, 0u32, 0u32);
    for seed in 0..n {
        let (_m, tx, _w, _d, _a) =
            generate_manor(seed, false, Some((20.0, 12.0)), false, false, false);

        // ---- tile 2 (M_WIN): trim family of the upper floors ------------
        // Every trimmed style sits inside a bright nalidnik, so the cues
        // live ABOVE the window: style5 raises a keystone block over
        // y36..55, style6 throws the browka hood over y42..58, style0 is
        // bare there.
        let (t2w, t2h, t2) = (&tx[2].0, &tx[2].1, &tx[2].2);
        let (w, h) = (*t2w as usize, *t2h as usize);
        let l = |x: usize, y: usize| lu(t2, w, x, h - 1 - y);
        let med40 = median_row(t2, w, h - 1 - 40, 20, 108, 1);
        let bright40 = (20..108).filter(|&x| l(x, 40) - med40 > 22).count();
        if bright40 >= 12 {
            f5 += 1; // keystone lintel
        } else if l(64, 48) - l(64, 30) > 28 {
            f6 += 1; // eyebrow hood over a bare band
        } else {
            f0 += 1; // plain nalidnik
        }

        // ---- tile 7 (M_DOOR): door drawing ------------------------------
        let (d7w, d7h, t7) = (&tx[7].0, &tx[7].1, &tx[7].2);
        let (w7, h7) = (*d7w as usize, *d7h as usize);
        let l7 = |x: usize, y: usize| lu(t7, w7, x, h7 - 1 - y);
        let glz_at = |x: usize, y: usize| -> u32 {
            let i = ((h7 - 1 - y) * w7 + x) * 4;
            if t7[i + 2] as i32 - t7[i] as i32 > 35 {
                1
            } else {
                0
            }
        };
        // keystone over the arch → ampire; glazed leaf heads (clear of the
        // central mullion bars at x50..53 / x80..83) → glazed pair
        let med19 = median_row(t7, w7, h7 - 1 - 19, 36, 94, 2);
        if l7(64, 19) - med19 > 18 {
            d_amp += 1;
        } else if glz_at(46, 80) + glz_at(76, 80) + glz_at(46, 90) + glz_at(76, 90) >= 3 {
            d_glz += 1;
        } else {
            d_cls += 1;
        }

        // ---- tile 13 (M_RAILING, alpha): railing pattern ----------------
        // Spears break above the top rail at y0; scrolls bulge to 8 px at
        // y26, the thin 2 px lattice stays below 48 there.
        let (r13w, r13h, t13) = (&tx[13].0, &tx[13].1, &tx[13].2);
        let (w13, h13) = (*r13w as usize, *r13h as usize);
        let opaque_row = |y: usize| -> u32 {
            let yy = h13 - 1 - y;
            (0..w13)
                .filter(|&x| t13[(yy * w13 + x) * 4 + 3] > 128)
                .count() as u32
        };
        let c0 = opaque_row(0);
        let c26 = opaque_row(26);
        if c0 >= 20 {
            r2 += 1; // spear-topped posts
        } else if c26 >= 48 {
            r0 += 1; // bar posts with scrolls
        } else {
            r1 += 1; // diagonal lattice
        }
    }
    let t = n as f64;
    println!(
        "trim fam: plain={f0} ({:.0}%) eyebrow={f6} ({:.0}%) keystone={f5} ({:.0}%) | expect 50/30/20",
        100.0 * f0 as f64 / t,
        100.0 * f6 as f64 / t,
        100.0 * f5 as f64 / t
    );
    println!(
        "door: ampire={d_amp} ({:.0}%) classic={d_cls} ({:.0}%) glazed={d_glz} ({:.0}%) | expect ~33/33/33",
        100.0 * d_amp as f64 / t,
        100.0 * d_cls as f64 / t,
        100.0 * d_glz as f64 / t
    );
    println!(
        "railing: posts={r0} ({:.0}%) lattice={r1} ({:.0}%) spears={r2} ({:.0}%) | expect ~33/33/33",
        100.0 * r0 as f64 / t,
        100.0 * r1 as f64 / t,
        100.0 * r2 as f64 / t
    );
}
