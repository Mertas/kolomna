//! Classify every manor variant by its roof geometry alone:
//! flat deck (parapet) / gable-across (ridge along z, x-top edge at w/2) /
//! hipped (2 quads + 2 hip triangles) / ordinary gable.
use kolomna_render::piter::{generate_manor, M_ROOF};

fn main() {
    let n = 800u64;
    let (mut flat, mut gz, mut hipped, mut gable) = (0u32, 0u32, 0u32, 0u32);
    for seed in 0..n {
        let (mesh, _tx, wa, _da, _ar) =
            generate_manor(seed, false, Some((20.0, 12.0)), false, false, false);
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for p in mesh.parts.iter().filter(|p| p.texture == M_ROOF) {
            let a = p.first_index as usize;
            let b = a + p.index_count as usize;
            for &i in &mesh.indices[a..b] {
                pts.push(mesh.verts[i as usize].pos);
            }
        }
        if pts.is_empty() {
            continue;
        }
        let ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
        let ymax = ys.iter().cloned().fold(f32::MIN, f32::max);
        let ymin = ys.iter().cloned().fold(f32::MAX, f32::min);
        let spread = ymax - ymin;
        if spread < 0.6 {
            flat += 1;
        } else {
            let cut = ymin + 0.75 * spread;
            let top: Vec<&[f32; 3]> = pts.iter().filter(|p| p[1] >= cut).collect();
            let xs: Vec<f32> = top.iter().map(|p| p[0]).collect();
            let xmax = xs.iter().cloned().fold(f32::MIN, f32::max);
            let xmin = xs.iter().cloned().fold(f32::MAX, f32::min);
            let mx: f32 = xs.iter().sum::<f32>() / xs.len() as f32;
            if xmax - xmin < 0.5 && (mx - wa / 2.0).abs() < 0.4 {
                gz += 1;
            } else if xmax - xmin > 0.6 * wa {
                gable += 1;
            } else {
                hipped += 1;
            }
        }
    }
    let t = n as f64;
    println!(
        "census {n} seeds: flat={flat} ({:.0}%) gz={gz} ({:.0}%) hipped={hipped} ({:.0}%) gable={gable} ({:.0}%)",
        100.0 * flat as f64 / t,
        100.0 * gz as f64 / t,
        100.0 * hipped as f64 / t,
        100.0 * gable as f64 / t
    );
    println!("expect: flat~13% gz~17% hipped~42% gable~28%");
}
