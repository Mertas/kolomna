//! Top-down plan dump of the junction meshes: rasterise `build_roads` parts by
//! material so gaps, kerb rings and zebra placement can be read without eyes.
//! Usage: cargo run -q -p kolomna-render --example jdump
use kolomna_render::model::ModelMesh;
use kolomna_render::streets::{build_roads, Params};
use kolomna_render::terrain_gen;
use kolomna_render::{grid_side, smooth_field};

fn smp(f: &[f32], n: usize, cx: f32, cy: f32) -> f32 {
    let x = cx.clamp(0.0, (n - 1) as f32);
    let y = cy.clamp(0.0, (n - 1) as f32);
    let x0 = (x as usize).min(n.saturating_sub(2));
    let y0 = (y as usize).min(n.saturating_sub(2));
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let h00 = f[y0 * n + x0];
    let h10 = f[y0 * n + x0 + 1];
    let h01 = f[(y0 + 1) * n + x0];
    let h11 = f[(y0 + 1) * n + x0 + 1];
    (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fy) + (h01 * (1.0 - fx) + h11 * fx) * fy
}

fn pt_in_tri(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let s = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0]);
    let t = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let u = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]);
    (s >= 0.0 && t >= 0.0 && u >= 0.0) || (s <= 0.0 && t <= 0.0 && u <= 0.0)
}

fn dump(mesh: &ModelMesh, c: [f32; 2], title: &str) {
    let half = 18.0f32;
    let cell = 0.5f32;
    let n = (2.0 * half / cell) as usize; // 72
    let mut g = vec![0u8; n * n];
    // priority: paint > lamp > kerb > walk > asphalt; 0 = no geometry here
    let pri = |m: u32| -> u8 {
        match m {
            3 => 5,
            12 => 4,
            1 => 3,
            2 => 2,
            0 => 1,
            _ => 0,
        }
    };
    let sym = ['.', '#', 'w', 'K', 'L', '+'];
    for part in &mesh.parts {
        let p = pri(part.texture);
        if p == 0 {
            continue;
        }
        let mut t = part.first_index;
        while t + 2 < part.first_index + part.index_count {
            let mut vs = [[0f32; 2]; 3];
            for k in 0usize..3 {
                let v = &mesh.verts[mesh.indices[(t + k as u32) as usize] as usize].pos;
                vs[k] = [v[0] - 1024.0 - c[0], v[2] - 1024.0 - c[1]];
            }
            let (mut i0, mut i1, mut j0, mut j1) = (n, 0usize, n, 0usize);
            for v in &vs {
                let i = ((v[0] + half) / cell) as isize;
                let j = ((v[1] + half) / cell) as isize;
                i0 = i0.min(i.clamp(0, n as isize - 1) as usize);
                i1 = i1.max(i.clamp(0, n as isize - 1) as usize);
                j0 = j0.min(j.clamp(0, n as isize - 1) as usize);
                j1 = j1.max(j.clamp(0, n as isize - 1) as usize);
            }
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let px = -half + (i as f32 + 0.5) * cell;
                    let py = -half + (j as f32 + 0.5) * cell;
                    if pt_in_tri([px, py], vs[0], vs[1], vs[2]) && p > g[j * n + i] {
                        g[j * n + i] = p;
                    }
                }
            }
            t += 3;
        }
    }
    println!("== {title} ==  cell {cell} m, window +-{half} m, centre = junction");
    println!("   # asphalt  w walk  K kerb  + paint  L lamp  * internal hole  . bare");
    let mut holes = 0usize;
    for j in (0..n).rev() {
        let mut row = String::with_capacity(n);
        for i in 0..n {
            let c0 = g[j * n + i];
            let mut ch = if c0 == 0 { '.' } else { sym[c0 as usize] };
            if c0 == 0 {
                let mut cov = 0;
                for dj in -1i32..=1 {
                    for di in -1i32..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ii = i as i32 + di;
                        let jj = j as i32 + dj;
                        if ii >= 0 && jj >= 0 && (ii as usize) < n && (jj as usize) < n {
                            if g[jj as usize * n + ii as usize] > 0 {
                                cov += 1;
                            }
                        }
                    }
                }
                if cov >= 5 {
                    ch = '*';
                    holes += 1;
                }
            }
            row.push(ch);
        }
        println!("{row}");
    }
    println!("internal holes: {holes} (cells {cell} m² ≈ {:.1} m²)", holes as f32 * cell * cell);
}

fn main() {
    let w = terrain_gen::generate(0x5EED_1957);
    let (mn, wl) = (w.mn, w.wl);
    let cn = grid_side(&w.coarse);
    let sf = smooth_field(&w.fine, wl);
    let fnx = grid_side(&sf);
    let kf = (fnx - 1) as f32 / (cn - 1) as f32;
    let h_at = |x: f32, z: f32| smp(&sf, fnx, x * kf, z * kf);
    let anchor = [64.0, 64.0];
    let eu = [1.0, 0.0];
    let nv = [0.0, 1.0];

    // set A: exactly the EDIT_TEST roads (junction (0,0) chain×main,
    // junction (100,100) T-attachment of the stub onto the diagonal)
    let net_full = Params {
        main_t0: -1500.0,
        main_t1: 1500.0,
        cross_half: 0.0,
        cross: vec![],
    };
    let seta = vec![
        vec![[-600.0, -600.0], [600.0, 600.0]],
        vec![[-200.0, 200.0], [0.0, 260.0]],
        vec![[240.0, 30.0], [100.0, 100.0]],
    ];
    let (ma, _t) = build_roads(&h_at, anchor, mn, eu, nv, &seta, &net_full);
    dump(&ma, [0.0, 0.0], "junction (0,0): diagonal chain x main street");
    dump(&ma, [100.0, 100.0], "junction (100,100): T of the stub onto the diagonal");

    // set B: the demo loop crossing itself
    let net_none = Params {
        main_t0: 0.0,
        main_t1: 0.0,
        cross_half: 0.0,
        cross: vec![],
    };
    let setb = vec![vec![
        [-180.0, 40.0],
        [-120.0, 100.0],
        [-180.0, 150.0],
        [-250.0, 95.0],
        [-148.0, 41.0],
    ]];
    let (mb, _t) = build_roads(&h_at, anchor, mn, eu, nv, &setb, &net_none);
    dump(&mb, [-168.3, 51.7], "self-crossing of the loop");
}
