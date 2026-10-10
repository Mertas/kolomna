//! Usage: cargo run -p kolomna-dbpf --example extract -- <file.dat> <type_hex> <group_hex> <instance_hex> <out.bin>
//!
//! Extracts a single subfile (decompressing when needed) to a raw binary file.

use std::env;
use std::process::ExitCode;

use kolomna_dbpf::Archive;

fn hx(s: &str) -> Result<u32, String> {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| format!("bad hex {s}: {e}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 5 {
        eprintln!("usage: extract <file.dat> <type_hex> <group_hex> <instance_hex> <out.bin>");
        return ExitCode::FAILURE;
    }
    let (t, g, i) = match (hx(&args[1]), hx(&args[2]), hx(&args[3])) {
        (Ok(t), Ok(g), Ok(i)) => (t, g, i),
        (a, b, c) => {
            for e in [a, b, c] {
                if let Err(err) = e {
                    eprintln!("{err}");
                }
            }
            return ExitCode::FAILURE;
        }
    };
    match run(&args[0], t, g, i, &args[4]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("!! {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str, t: u32, g: u32, i: u32, out: &str) -> Result<(), String> {
    let archive = Archive::open(path).map_err(|e| format!("{path}: {e}"))?;
    let e = archive
        .find(t, g, i)
        .ok_or_else(|| format!("subfile {t:08x}/{g:08x}/{i:08x} not found"))?;
    let bytes = archive
        .read_subfile(e)
        .map_err(|e| format!("read {t:08x}/{g:08x}/{i:08x}: {e}"))?;
    std::fs::write(out, &bytes).map_err(|e| format!("write {out}: {e}"))?;
    println!("{t:08x}/{g:08x}/{i:08x}: {} bytes -> {out}", bytes.len());
    Ok(())
}
