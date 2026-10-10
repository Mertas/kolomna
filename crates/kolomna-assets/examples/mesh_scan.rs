//! Scan every 3DMD model in a DBPF archive and print TGI + bbox + names,
//! one line per model (names from material textures and animation groups).
//!
//! usage: cargo run -p kolomna-assets --example mesh3d_scan -- <file.dat> [keyword ...]
//!
//! With keywords, only models whose joined names contain any keyword
//! (case-insensitive) are printed.

use std::env;

use kolomna_assets::mesh3d;
use kolomna_dbpf::Archive;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: mesh3d_scan <file.dat> [keyword ...]");
        std::process::exit(1);
    }
    let archive = Archive::open(&args[0]).expect("open archive");
    let kws: Vec<String> = args[1..].iter().map(|s| s.to_lowercase()).collect();
    let (mut n_ok, mut n_err, mut n_hits) = (0usize, 0usize, 0usize);
    for e in archive.entries_of_type(0x5ad0e817) {
        let raw = match archive.read_subfile(e) {
            Ok(r) => r,
            Err(_) => {
                n_err += 1;
                continue;
            }
        };
        let m = match mesh3d::parse(&raw) {
            Ok(m) => {
                n_ok += 1;
                m
            }
            Err(_) => {
                n_err += 1;
                continue;
            }
        };
        let mut names: Vec<String> = m.animation.groups.iter().map(|g| g.name.clone()).collect();
        for g in &m.material_groups {
            for t in &g.textures {
                names.push(t.name.clone());
            }
        }
        let joined = names.join(" | ");
        if !kws.is_empty() {
            let low = joined.to_lowercase();
            if !kws.iter().any(|k| low.contains(k.as_str())) {
                continue;
            }
        }
        n_hits += 1;
        let (mn, mx) = m.bbox();
        println!(
            "{:08x}/{:08x}/{:08x} bbox=({:.2},{:.2},{:.2})..({:.2},{:.2},{:.2}) {}",
            e.tgi.type_id,
            e.tgi.group_id,
            e.tgi.instance_id,
            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
            joined
        );
    }
    eprintln!("mesh3d_scan: ok={n_ok} err={n_err} hits={n_hits}");
}
