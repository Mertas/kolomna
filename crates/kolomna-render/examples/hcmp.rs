//! Diagnose "terrain poking through the roads": how far the 16 m `world.coarse`
//! field (what the app used to pose roads on) deviates from the 4 m field the
//! terrain mesh actually renders — and count asphalt vertices where the
//! rendered terrain rises through the road slab, for both samplings.
use kolomna_render::streets::{build_roads, Params, M_ASPHALT};
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

fn main() {
    let w = terrain_gen::generate(0x5EED_1957);
    let (mn, wl) = (w.mn, w.wl);
    let cn = grid_side(&w.coarse);
    let sc = smooth_field(&w.coarse, wl); // the OLD scene field (16 m smoothed)
    let sf = smooth_field(&w.fine, wl); // the field the render mesh is built from
    let fnx = grid_side(&sf);
    let kf = (fnx - 1) as f32 / (cn - 1) as f32;

    // ---- 1: how far the two fields differ (metres, dry land only)
    let (mut mx, mut mean, mut cnt, mut worst) = (0f32, 0f64, 0usize, (0usize, 0usize));
    let edge = 10usize; // stay off the smoothed map rim
    for j in edge..cn - edge {
        for i in edge..cn - edge {
            let a = sc[j * cn + i];
            if a < wl + 0.3 {
                continue; // skip the water bed
            }
            let b = smp(&sf, fnx, i as f32 * kf, j as f32 * kf);
            let d = (b - a).abs();
            mean += d as f64;
            cnt += 1;
            if d > mx {
                mx = d;
                worst = (i, j);
            }
        }
    }
    println!(
        "field mismatch (16 m vs 4 m): max {mx:.2} m at cell {worst:?} (world {:.0},{:.0} m), mean {:.2} m over {cnt} land nodes",
        worst.0 as f32 * 16.0,
        worst.1 as f32 * 16.0,
        mean / cnt.max(1) as f64
    );

    // ---- 2: a straight road through the worst spot; count terrain pokes
    let (wxm, wzm) = (worst.0 as f32 * 16.0, worst.1 as f32 * 16.0);
    // plan metres: world = 1024 + u (anchor (64,64) -> 1024 m)
    let chain = vec![
        [wxm - 160.0 - 1024.0, wzm - 1024.0],
        [wxm + 160.0 + 1024.0, wzm - 1024.0],
    ];
    let coarse_h = |x: f32, z: f32| smp(&sc, cn, x, z);
    let fine_h = |x: f32, z: f32| smp(&sf, fnx, x * kf, z * kf);

    let net = Params {
        main_t0: 0.0,
        main_t1: 0.0,
        cross_half: 0.0,
        cross: vec![],
    };
    let variants: [(&str, &dyn Fn(f32, f32) -> f32); 2] =
        [("coarse (old)", &coarse_h), ("fine (new)", &fine_h)];
    for (name, h_at) in variants {
        let (mesh, _t) = build_roads(
            h_at,
            [64.0, 64.0],
            mn,
            [1.0, 0.0],
            [0.0, 1.0],
            &[chain.clone()],
            &net,
        );
        let (mut pokes, mut worstp, mut total) = (0usize, 0f32, 0usize);
        for part in &mesh.parts {
            if part.texture != M_ASPHALT {
                continue;
            }
            for ii in part.first_index..part.first_index + part.index_count {
                let v = &mesh.verts[mesh.indices[ii as usize] as usize];
                let u = v.pos[0] - 1024.0;
                let vv = v.pos[2] - 1024.0;
                let cx = 64.0 + u / 16.0;
                let cz = 64.0 + vv / 16.0;
                let h = smp(&sf, fnx, cx * kf, cz * kf).max(wl);
                let p = (h - mn) - v.pos[1]; // both metres (mesh y = z - mn)
                total += 1;
                if p > 0.01 {
                    pokes += 1;
                }
                if p > worstp {
                    worstp = p;
                }
            }
        }
        println!(
            "{name}: {pokes} asphalt verts under the rendered terrain of {total} (worst +{worstp:.2} m)"
        );
    }
}
