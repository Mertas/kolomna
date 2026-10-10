//! Usage: cargo run -p kolomna-assets --example exmp_scan -- <file.dat>
//!
//! Prints `<type>/<group>/<instance> <name>` for every EXMP (6534284a) subfile,
//! where <name> is the longest printable ASCII run found in the first 512 bytes
//! (exemplar property lists embed the exemplar name as plain text).

use std::process::ExitCode;

use kolomna_dbpf::Archive;

const EXMP_TYPE: u32 = 0x6534284a;

fn extract_name(raw: &[u8]) -> Option<String> {
    let window = &raw[..raw.len().min(512)];
    let mut best = String::new();
    let mut cur = String::new();
    let mut flush = |cur: &mut String, best: &mut String| {
        if cur.chars().any(|c| c.is_ascii_alphabetic()) && cur.chars().count() > best.chars().count() {
            *best = std::mem::take(cur);
        }
        cur.clear();
    };
    for &b in window {
        let c = b as char;
        if (0x20..0x7f).contains(&b) && !matches!(c, '(' | ')' | '\'') {
            cur.push(c);
        } else {
            flush(&mut cur, &mut best);
        }
    }
    flush(&mut cur, &mut best);
    let trimmed = best.trim().to_string();
    if trimmed.chars().count() >= 4 {
        Some(trimmed)
    } else {
        None
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: exmp_scan <file.dat>");
        return ExitCode::FAILURE;
    }
    let archive = match Archive::open(&args[0]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    let mut n = 0usize;
    for e in archive.entries_of_type(EXMP_TYPE) {
        if let Ok(raw) = archive.read_subfile(e) {
            let name = extract_name(&raw).unwrap_or_else(|| "?".into());
            println!(
                "{:08x}/{:08x}/{:08x} {}",
                e.tgi.type_id, e.tgi.group_id, e.tgi.instance_id, name
            );
            n += 1;
        }
    }
    eprintln!("{n} EXMP entries printed");
    ExitCode::SUCCESS
}
