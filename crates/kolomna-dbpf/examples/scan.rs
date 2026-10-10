//! Usage: cargo run -p kolomna-dbpf --example scan -- <file.dat> [more.dat ...]
//!
//! Prints archive header info, entry/type statistics and compression counts.

use std::env;
use std::process::ExitCode;

use kolomna_dbpf::{Archive, type_name};

fn main() -> ExitCode {
    let files: Vec<String> = env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: scan <file.dat> [more.dat ...]");
        return ExitCode::FAILURE;
    }
    let mut failed = false;
    for path in &files {
        if let Err(err) = scan(path) {
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

fn scan(path: &str) -> std::io::Result<()> {
    let archive = Archive::open(path)?;
    let h = &archive.header;
    println!("== {} ({} bytes)", path, archive.file_len());
    println!(
        "   v{}.{}, index v{}.{}: {} entries @ {} ({} bytes), created {}",
        h.major,
        h.minor,
        h.index_major,
        h.index_minor,
        h.index_count,
        h.index_offset,
        h.index_size,
        h.created
    );

    let mut hist: Vec<(u32, usize)> = archive.types_histogram().into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!(
        "   types: {} distinct, compressed subfiles: {}",
        hist.len(),
        archive.count_compressed()
    );
    for (t, count) in hist.iter().take(10) {
        let name = type_name(*t).unwrap_or("");
        println!("     {t:08x} {name:<5} x{count}");
    }
    Ok(())
}
