//! Dump FSH textures from a DBPF archive to PNG files.
//!
//! usage: cargo run -p kolomna-assets --example fsh_dump -- <file.dat> <out_dir> [max]

use std::env;
use std::fs;
use std::path::Path;

use kolomna_assets::fsh::Fsh;
use kolomna_dbpf::Archive;

const FSH_TYPE: u32 = 0x7AB50E44;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: fsh_dump <file.dat> <out_dir> [max]");
        std::process::exit(1);
    }
    let max: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let archive = Archive::open(&args[0])?;
    fs::create_dir_all(&args[1])?;

    let mut dumped = 0usize;
    let mut attempts = 0usize;
    let mut errors = 0usize;
    for e in archive.entries_of_type(FSH_TYPE) {
        if dumped >= max || attempts >= max * 5 {
            break;
        }
        attempts += 1;
        let raw = match archive.read_subfile(e) {
            Ok(raw) => raw,
            Err(err) => {
                eprintln!("{}: read error: {err}", e.tgi);
                continue;
            }
        };
        let fsh = match Fsh::parse(&raw) {
            Ok(fsh) => fsh,
            Err(err) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("{}: parse error: {err}", e.tgi);
                }
                continue;
            }
        };
        if fsh.entries.is_empty() {
            continue;
        }
        let first = &fsh.entries[0];
        if first.width < 16 || first.height < 16 {
            continue;
        }
        match fsh.decode_entry(&raw, 0) {
            Ok(img) => {
                let tag: String = first
                    .name
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                    .collect();
                let name = format!("{:03}_{tag}_{}x{}.png", dumped, first.width, first.height);
                let path = Path::new(&args[1]).join(&name);
                write_png(&path, img.width, img.height, &img.rgba)?;
                println!(
                    "{:29} id=0x{:02x} tag={:<4} {:>4}x{:<4} mips={} dir={} -> {name}",
                    e.tgi.to_string(),
                    first.entry_id,
                    first.name,
                    first.width,
                    first.height,
                    first.mip_count,
                    fsh.dir_id,
                );
                dumped += 1;
            }
            Err(err) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("{}: decode error: {err}", e.tgi);
                }
            }
        }
    }
    println!("dumped {dumped} textures, {errors} errors ({attempts} attempts)");
    Ok(())
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let file = fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}
