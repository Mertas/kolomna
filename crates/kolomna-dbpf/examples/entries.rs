//! Usage: cargo run -p kolomna-dbpf --example entries -- <file.dat> [more.dat ...]
//!
//! Lists every subfile in a DBPF archive: TGI, raw size, decompressed size,
//! compression flag — sorted by decompressed size (largest first). Useful for
//! spotting big structured chunks (terrain, network, zone grids) in city saves.

use std::env;
use std::process::ExitCode;

use kolomna_dbpf::{type_name, Archive};

fn main() -> ExitCode {
    let files: Vec<String> = env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: entries <file.dat> [more.dat ...]");
        return ExitCode::FAILURE;
    }
    let mut failed = false;
    for path in &files {
        if let Err(err) = list(path) {
            eprintln!("!! {path}: {err}");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn list(path: &str) -> std::io::Result<()> {
    let a = Archive::open(path)?;
    println!(
        "== {path} ({} bytes, {} entries, {} compressed)",
        a.file_len(),
        a.entries.len(),
        a.count_compressed()
    );

    let mut rows: Vec<(u32, String)> = Vec::with_capacity(a.entries.len());
    for e in &a.entries {
        let comp = e.is_compressed() || a.compressed.contains_key(&e.tgi);
        let dec = if comp {
            a.compressed.get(&e.tgi).copied()
        } else {
            Some(e.packed_size())
        };
        let name = type_name(e.tgi.type_id).unwrap_or("");
        rows.push((
            dec.unwrap_or(0),
            format!(
                "  {:08x}/{:08x}/{:08x} {:<5} raw {:>9} dec {:>9} {}",
                e.tgi.type_id,
                e.tgi.group_id,
                e.tgi.instance_id,
                name,
                e.packed_size(),
                dec.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
                if comp { "Z" } else { "-" }
            ),
        ));
    }
    rows.sort_by(|x, y| y.0.cmp(&x.0));
    for (_, line) in &rows {
        println!("{line}");
    }
    Ok(())
}
