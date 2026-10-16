//! Junction probe: samples the TOP surface material at every plan point of
//! a window (exactly what the renderer would show from straight above) and
//! lists the traffic-light / sign geometry clusters with plan coordinates.
//!
//! usage: cargo run -q -p kolomna-render --example jprobe -- [self|cross] [du dv]
use kolomna_render::streets;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let preset = args.get(1).map(|s| s.as_str()).unwrap_or("self");
    let (du_off, dv_off) = if args.len() > 3 {
        (
            args[2].parse::<f32>().unwrap_or(0.0),
            args[3].parse::<f32>().unwrap_or(0.0),
        )
    } else {
        (0.0, 0.0)
    };
    let anchor = [64.0f32, 64.0];
    let eu = [1.0f32, 0.0];
    let nvec = [0.0f32, 1.0];
    let h_at = |_x: f32, _y: f32| 0.0f32;
    let chains: Vec<Vec<[f32; 2]>> = match preset {
        "cross" => vec![
            vec![[-40.0, 0.0], [40.0, 0.0]],
            vec![[-28.0, -28.0], [28.0, 28.0]],
        ],
        _ => vec![vec![
            [-180.0, 40.0],
            [-120.0, 100.0],
            [-180.0, 150.0],
            [-250.0, 95.0],
            [-148.0, 41.0],
        ]],
    };
    let net = streets::Params::default();
    let (mesh, _texs) = streets::build_roads(&h_at, anchor, 0.0, eu, nvec, &chains, &net);
    let base = [anchor[0] * 16.0, anchor[1] * 16.0];
    // dump mode: jprobe [preset] dump u0 u1 v0 v1 -- list triangles in a plan rect
    if args.get(2).map(|s| s.as_str()) == Some("dump") {
        let (u0, u1, v0, v1) = (
            args[3].parse::<f32>().unwrap(),
            args[4].parse::<f32>().unwrap(),
            args[5].parse::<f32>().unwrap(),
            args[6].parse::<f32>().unwrap(),
        );
        let mut list: Vec<(f32, u32, [[f32; 2]; 3], [f32; 3])> = Vec::new();
        for p in &mesh.parts {
            let f = p.first_index as usize;
            let n = p.index_count as usize;
            for t in (f..f + n).step_by(3) {
                let (ia, ib, ic) = (
                    mesh.indices[t] as usize,
                    mesh.indices[t + 1] as usize,
                    mesh.indices[t + 2] as usize,
                );
                let (pa, pb, pc) = (
                    mesh.verts[ia].pos,
                    mesh.verts[ib].pos,
                    mesh.verts[ic].pos,
                );
                let pus = [pa[0] - base[0], pb[0] - base[0], pc[0] - base[0]];
                let pvs = [pa[2] - base[1], pb[2] - base[1], pc[2] - base[1]];
                let umin = pus.iter().cloned().fold(f32::MAX, f32::min);
                let umax = pus.iter().cloned().fold(f32::MIN, f32::max);
                let vmin = pvs.iter().cloned().fold(f32::MAX, f32::min);
                let vmax = pvs.iter().cloned().fold(f32::MIN, f32::max);
                if umax < u0 || umin > u1 || vmax < v0 || vmin > v1 {
                    continue;
                }
                let ys = [pa[1], pb[1], pc[1]];
                let yavg = (ys[0] + ys[1] + ys[2]) / 3.0;
                list.push((
                    yavg,
                    p.texture,
                    [[pus[0], pvs[0]], [pus[1], pvs[1]], [pus[2], pvs[2]]],
                    ys,
                ));
            }
        }
        list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("# dump u {u0}..{u1} v {v0}..{v1}: {} triangles (top first)", list.len());
        for (y, mat, pts, ys) in list.iter().take(120) {
            println!(
                "  mat {:2} yavg {:7.3} ({:7.2},{:7.2}) ({:7.2},{:7.2}) ({:7.2},{:7.2})  y {:6.3}/{:6.3}/{:6.3}",
                mat, y, pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1],
                ys[0], ys[1], ys[2]
            );
        }
        return;
    }
    // point mode: jprobe [preset] point u v [u v]... -- stack of triangles at a plan point
    if args.get(2).map(|s| s.as_str()) == Some("point") {
        let mut k = 3;
        while k + 1 < args.len() {
            let (pu, pv) = (
                args[k].parse::<f32>().unwrap(),
                args[k + 1].parse::<f32>().unwrap(),
            );
            k += 2;
            println!("# point ({pu:.2},{pv:.2})");
            let mut stack: Vec<(f32, u32, [[f32; 2]; 3], [f32; 3])> = Vec::new();
            for p in &mesh.parts {
                let f = p.first_index as usize;
                let n = p.index_count as usize;
                for t in (f..f + n).step_by(3) {
                    let (ia, ib, ic) = (
                        mesh.indices[t] as usize,
                        mesh.indices[t + 1] as usize,
                        mesh.indices[t + 2] as usize,
                    );
                    let (pa, pb, pc) = (
                        mesh.verts[ia].pos,
                        mesh.verts[ib].pos,
                        mesh.verts[ic].pos,
                    );
                    let ax = pa[0] - base[0];
                    let az = pa[2] - base[1];
                    let bx = pb[0] - base[0];
                    let bz = pb[2] - base[1];
                    let cx = pc[0] - base[0];
                    let cz = pc[2] - base[1];
                    let den = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
                    if den.abs() < 1e-9 {
                        continue;
                    }
                    let w0 = ((bz - cz) * (pu - cx) + (cx - bx) * (pv - cz)) / den;
                    let w1 = ((cz - az) * (pu - cx) + (ax - cx) * (pv - cz)) / den;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                        continue;
                    }
                    let y = w0 * pa[1] + w1 * pb[1] + w2 * pc[1];
                    stack.push((
                        y,
                        p.texture,
                        [[ax, az], [bx, bz], [cx, cz]],
                        [pa[1], pb[1], pc[1]],
                    ));
                }
            }
            stack.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (y, mat, pts, ys) in stack.iter().take(24) {
                println!(
                    "  y {:7.3} mat {:2}  verts ({:7.2},{:7.2})y{:.3} ({:7.2},{:7.2})y{:.3} ({:7.2},{:7.2})y{:.3}",
                    y, mat, pts[0][0], pts[0][1], ys[0], pts[1][0], pts[1][1], ys[1], pts[2][0], pts[2][1], ys[2]
                );
            }
        }
        return;
    }
    let (cu, cv, half) = if preset == "cross" {
        (0.0f32, 0.0f32, 18.0f32)
    } else {
        (-168.3f32, 51.7f32, 16.0f32)
    };
    let (u0, u1) = (cu + du_off - half, cu + du_off + half);
    let (v0, v1) = (cv + dv_off - half, cv + dv_off + half);
    let step = 0.5f32;

    let ch = |m: u32| -> char {
        match m {
            0 => 'a',
            1 => 'k',
            2 => 'w',
            3 => 'P',
            4 => 'I',
            5 => 'g',
            6 => 'b',
            7 => ',',
            8 => '.',
            9 => 'S',
            10 => 'W',
            11 => 'H',
            12 => 'L',
            _ => '?',
        }
    };

    // ---- bin triangles into 2 m plan cells over the window ----------------
    let cell = 2.0f32;
    let gw = ((u1 - u0) / cell).ceil() as usize + 1;
    let gh = ((v1 - v0) / cell).ceil() as usize + 1;
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); gw * gh];
    let tris: Vec<(usize, usize, usize)> = mesh
        .parts
        .iter()
        .flat_map(|p| {
            let f = p.first_index as usize;
            let n = p.index_count as usize;
            (f..f + n).step_by(3).map(move |t| (t, p.texture as usize, 0))
        })
        .map(|(t, _, _)| {
            (
                mesh.indices[t] as usize,
                mesh.indices[t + 1] as usize,
                mesh.indices[t + 2] as usize,
            )
        })
        .collect();
    let mat_of: Vec<u32> = mesh
        .parts
        .iter()
        .flat_map(|p| {
            let n = (p.index_count / 3) as usize;
            std::iter::repeat(p.texture).take(n)
        })
        .collect();
    for (ti, (ia, ib, ic)) in tris.iter().enumerate() {
        let (pa, pb, pc) = (mesh.verts[*ia].pos, mesh.verts[*ib].pos, mesh.verts[*ic].pos);
        let xmin = pa[0].min(pb[0]).min(pc[0]) - base[0];
        let xmax = pa[0].max(pb[0]).max(pc[0]) - base[0];
        let zmin = pa[2].min(pb[2]).min(pc[2]) - base[1];
        let zmax = pa[2].max(pb[2]).max(pc[2]) - base[1];
        let gx0 = (((xmin - u0) / cell).floor().max(0.0)) as usize;
        let gx1 = (((xmax - u0) / cell).ceil().min((gw - 1) as f32)) as usize;
        let gz0 = (((zmin - v0) / cell).floor().max(0.0)) as usize;
        let gz1 = (((zmax - v0) / cell).ceil().min((gh - 1) as f32)) as usize;
        if xmax < u0 || xmin > u1 || zmax < v0 || zmin > v1 {
            continue;
        }
        for gz in gz0..=gz1 {
            for gx in gx0..=gx1 {
                bins[gz * gw + gx].push(ti);
            }
        }
    }

    println!("# preset {preset}: top-surface map u {u0}..{u1} v {v0}..{v1} step {step}");
    println!("# legend: a=asphalt k=kerb w=walk P=paint I=iron g=glass b=bark ,=leaf .=leaf2 S=sign W=wood H=white L=lights  space=no surface");
    let mut v = v0;
    while v <= v1 + 1e-4 {
        let mut row = String::new();
        let mut u = u0;
        while u <= u1 + 1e-4 {
            let gx = (((u - u0) / cell) as usize).min(gw - 1);
            let gz = (((v - v0) / cell) as usize).min(gh - 1);
            let mut best: Option<(f32, u32)> = None;
            for &ti in &bins[gz * gw + gx] {
                let (ia, ib, ic) = tris[ti];
                let (pa, pb, pc) = (mesh.verts[ia].pos, mesh.verts[ib].pos, mesh.verts[ic].pos);
                let ax = pa[0] - base[0];
                let az = pa[2] - base[1];
                let bx = pb[0] - base[0];
                let bz = pb[2] - base[1];
                let cx = pc[0] - base[0];
                let cz = pc[2] - base[1];
                let den = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
                if den.abs() < 1e-9 {
                    continue;
                }
                let w0 = ((bz - cz) * (u - cx) + (cx - bx) * (v - cz)) / den;
                let w1 = ((cz - az) * (u - cx) + (ax - cx) * (v - cz)) / den;
                let w2 = 1.0 - w0 - w1;
                if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                    continue;
                }
                let y = w0 * pa[1] + w1 * pb[1] + w2 * pc[1];
                if best.map(|(by, _)| y > by).unwrap_or(true) {
                    best = Some((y, mat_of[ti]));
                }
            }
            row.push(best.map(|(_, m)| ch(m)).unwrap_or(' '));
            u += step;
        }
        println!("v={v:7.1} {row}");
        v += step;
    }

    // ---- traffic light / furniture inventory ------------------------------
    // cluster iron (4) triangles by centroid proximity
    let mut cents: Vec<([f32; 2], u32, f32, f32)> = Vec::new(); // (plan u,v), mat, y_top, y_bot
    for (ti, (ia, ib, ic)) in tris.iter().enumerate() {
        let m = mat_of[ti];
        if m != 4 && m != 12 && m != 9 {
            continue;
        }
        let (pa, pb, pc) = (mesh.verts[*ia].pos, mesh.verts[*ib].pos, mesh.verts[*ic].pos);
        let uu = (pa[0] + pb[0] + pc[0]) / 3.0 - base[0];
        let vv = (pa[2] + pb[2] + pc[2]) / 3.0 - base[1];
        let ymax = pa[1].max(pb[1]).max(pc[1]);
        let ymin = pa[1].min(pb[1]).min(pc[1]);
        cents.push(([uu, vv], m, ymax, ymin));
    }
    // simple clustering: merge points within 1.5 m of a cluster centroid
    let mut cl: Vec<([f32; 2], usize, u32, f32, f32, f32, f32, f32, f32)> = Vec::new();
    // (centroid u,v), count, mat, ymax, ymin, umin, umax, vmin, vmax
    for (p, m, ymax, ymin) in cents {
        let mut hit = None;
        for (k, c) in cl.iter_mut().enumerate() {
            if vlen2(p, c.0) < 1.5 * 1.5 {
                hit = Some(k);
                break;
            }
        }
        if let Some(k) = hit {
            let c = &mut cl[k];
            let n = c.1 as f32;
            c.0 = [(c.0[0] * n + p[0]) / (n + 1.0), (c.0[1] * n + p[1]) / (n + 1.0)];
            c.1 += 1;
            c.3 = c.3.max(ymax);
            c.4 = c.4.min(ymin);
            c.5 = c.5.min(p[0]);
            c.6 = c.6.max(p[0]);
            c.7 = c.7.min(p[1]);
            c.8 = c.8.max(p[1]);
        } else {
            cl.push((p, 1, m, ymax, ymin, p[0], p[0], p[1], p[1]));
        }
    }
    println!("# clusters (iron/lights/sign furniture), plan coords:");
    for (c, n, m, ymax, ymin, u0b, u1b, v0b, v1b) in &cl {
        println!(
            "  mat {:2} n={:3} centre ({:7.2},{:7.2})  y {:.2}..{:.2}  bbox u {:.2}..{:.2} v {:.2}..{:.2}",
            m, n, c[0], c[1], ymin, ymax, u0b, u1b, v0b, v1b
        );
    }
}

fn vlen2(a: [f32; 2], b: [f32; 2]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1]];
    d[0] * d[0] + d[1] * d[1]
}
