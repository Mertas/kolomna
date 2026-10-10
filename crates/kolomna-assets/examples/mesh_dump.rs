//! Debug dump of one 3DMD model from a DBPF archive.
//!
//! usage: cargo run -p kolomna-assets --example mesh3d_dump -- <file.dat> <type_hex> <group_hex> <instance_hex>

use std::process::ExitCode;

use kolomna_assets::mesh3d;
use kolomna_dbpf::Archive;

fn hx(s: &str) -> Result<u32, String> {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| format!("bad hex {s}: {e}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("usage: mesh3d_dump <file.dat> <type_hex> <group_hex> <instance_hex>");
        return ExitCode::FAILURE;
    }
    let (t, g, i) = match (hx(&args[1]), hx(&args[2]), hx(&args[3])) {
        (Ok(t), Ok(g), Ok(i)) => (t, g, i),
        _ => return ExitCode::FAILURE,
    };
    let archive = match Archive::open(&args[0]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    let e = match archive.find(t, g, i) {
        Some(e) => e,
        None => {
            eprintln!("subfile {t:08x}/{g:08x}/{i:08x} not found");
            return ExitCode::FAILURE;
        }
    };
    let raw = match archive.read_subfile(e) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("read error: {err}");
            return ExitCode::FAILURE;
        }
    };
    println!("{t:08x}/{g:08x}/{i:08x}: {} bytes (decoded)", raw.len());
    match mesh3d::parse(&raw) {
        Ok(m) => {
            println!(
                "version {}x{}  vert_groups {}  index_groups {}  prim_groups {}  materials {}  anim_frames {}  anim_groups {}",
                m.version.0, m.version.1,
                m.vertex_groups.len(), m.index_groups.len(), m.prim_groups.len(),
                m.material_groups.len(), m.animation.num_frames, m.animation.groups.len(),
            );
            for (k, g) in m.vertex_groups.iter().enumerate() {
                let (mn, mx) = m.bbox();
                let uv = g.verts.iter().fold(
                    ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
                    |(mut a, mut b), v| {
                        a[0] = a[0].min(v.uv[0]);
                        a[1] = a[1].min(v.uv[1]);
                        b[0] = b[0].max(v.uv[0]);
                        b[1] = b[1].max(v.uv[1]);
                        (a, b)
                    },
                );
                println!(
                    "  VERT[{k}] flags={:#06x} fmt={:#010x} n={} bbox=({:.2},{:.2},{:.2})..({:.2},{:.2},{:.2}) uv=({:.4},{:.4})..({:.4},{:.4})",
                    g.flags, g.format, g.verts.len(),
                    mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], uv.0[0], uv.0[1], uv.1[0], uv.1[1],
                );
            }
            for (k, g) in m.index_groups.iter().enumerate() {
                let maxv = g.indices.iter().copied().max().unwrap_or(0);
                println!("  INDX[{k}] n={} max_index={maxv}", g.indices.len());
            }
            for (k, g) in m.prim_groups.iter().enumerate() {
                println!("  PRIM[{k}] count={}", g.len());
                for (j, p) in g.iter().enumerate().take(8) {
                    println!(
                        "    [{j}] type={} first={} num_index={}",
                        p.kind, p.first, p.num_index
                    );
                }
            }
            for (k, g) in m.material_groups.iter().enumerate() {
                println!("  MATS[{k}] flags={:#010x} textures={}", g.flags, g.textures.len());
                for t in &g.textures {
                    println!(
                        "    id={:#010x} wrap=({},{}) filters=({},{}) anim=({},{}) name={:?}",
                        t.id, t.wrap_u, t.wrap_v, t.mag_filter, t.min_filter,
                        t.anim_rate, t.anim_mode, t.name
                    );
                }
            }
            for g in &m.animation.groups {
                println!("  ANIM group {:?} frames={}", g.name, g.blocks.len());
            }
            for p in &m.properties {
                println!(
                    "  PROP mesh={} frame={} {} = {}",
                    p.mesh_index, p.frame_index, p.assignment_type, p.assigned_value
                );
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("parse error: {err}");
            ExitCode::FAILURE
        }
    }
}
