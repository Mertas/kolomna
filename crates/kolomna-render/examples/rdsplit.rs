//! Debug: (1) asphalt quad count for a straight chain (dashed output means
//! the same-chain duplicate guard misfires on consecutive stations);
//! (2) what split_duplicate keeps around chain corners and tails.
use kolomna_render::streets::{build_roads, split_duplicate, Params};

fn d(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    let net = Params {
        main_t0: 0.0,
        main_t1: 0.0,
        cross_half: 0.0,
        cross: vec![],
    };
    let h_at = |_x: f32, _y: f32| -> f32 { 0.0 };

    // ---- (1) asphalt slabs for one straight chain
    let chain = vec![[-600.0f32, -600.0], [600.0, 600.0]];
    let (mesh, _texs) = build_roads(&h_at, [64.0, 64.0], 0.0, [1.0, 0.0], [0.0, 1.0], &[chain], &net);
    let mut asph = 0usize;
    for part in &mesh.parts {
        if part.texture == kolomna_render::streets::M_ASPHALT {
            asph += part.index_count as usize / 6;
        }
    }
    let l = (1200.0f64) * 2.0f64.sqrt();
    let pairs = (l / 1.5).ceil() as usize;
    println!("straight chain: {asph} asphalt quads (expected ~{pairs} if continuous)");

    // ---- (2) split_duplicate around corners
    let ch = vec![[-40.0f32, 130.0], [40.0, 150.0], [108.0, 125.0], [95.0, 55.0]];
    for (name, extra_roads) in [("no other roads", vec![]), ("with a covering twin", vec![ch.clone()])] {
        let (kept_n, skipped) = {
            let (kept, sk) =
                split_duplicate(&ch, &extra_roads, &net);
            for (i, run) in kept.iter().enumerate() {
                let mut c1 = f32::MAX;
                let mut c2 = f32::MAX;
                for p in run {
                    c1 = c1.min(d(*p, ch[1]));
                    c2 = c2.min(d(*p, ch[2]));
                }
                println!(
                    "{name}: run{i} {} pts, first {:?}, last {:?}, corner hits {c1:.2}/{c2:.2} m",
                    run.len(),
                    run[0],
                    run[run.len() - 1]
                );
            }
            (kept.len(), sk)
        };
        println!("{name}: {kept_n} run(s), skipped {skipped:.1} m");
    }
}
