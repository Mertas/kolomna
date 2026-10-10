//! Dump specific FSH textures (by TGI) from a DBPF archive to PNG files.
//!
//! usage: cargo run -p kolomna-assets --example fsh_pick -- <file.dat> <out_dir> <type_hex> <group_hex> <instance_hex> [<type> <group> <instance> ...]

use std::env;
use std::fs;
use std::path::Path;

use kolomna_assets::fsh::Fsh;
use kolomna_dbpf::Archive;

fn hx(s: &str) -> Result<u32, String> {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| format!("bad hex {s}: {e}"))
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 5 || (args.len() - 2) % 3 != 0 {
        eprintln!("usage: fsh_pick <file.dat> <out_dir> <type> <group> <instance> [...]");
        std::process::exit(1);
    }
    let archive = Archive::open(&args[0])?;
    fs::create_dir_all(&args[1])?;

    let mut dumped = 0usize;
    for (k, chunk) in args[2..].chunks(3).enumerate() {
        let (t, g, i) = match (hx(&chunk[0]), hx(&chunk[1]), hx(&chunk[2])) {
            (Ok(t), Ok(g), Ok(i)) => (t, g, i),
            _ => {
                eprintln!("skip bad triple: {chunk:?}");
                continue;
            }
        };
        let e = match archive.find(t, g, i) {
            Some(e) => e,
            None => {
                eprintln!("{t:08x}/{g:08x}/{i:08x}: not found");
                continue;
            }
        };
        let raw = match archive.read_subfile(e) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("{t:08x}/{g:08x}/{i:08x}: read error {err}");
                continue;
            }
        };
        let fsh = match Fsh::parse(&raw) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("{t:08x}/{g:08x}/{i:08x}: parse error {err}");
                continue;
            }
        };
        let img = match fsh.decode_entry(&raw, 0) {
            Ok(img) => img,
            Err(err) => {
                eprintln!("{t:08x}/{g:08x}/{i:08x}: decode error {err}");
                continue;
            }
        };
        let name = format!(
            "{:02}_{:08x}_{:08x}_{:08x}_{}x{}.png",
            k, t, g, i, img.width, img.height
        );
        let path = Path::new(&args[1]).join(&name);
        write_png(&path, img.width, img.height, &img.rgba)?;
        println!(
            "{t:08x}/{g:08x}/{i:08x}: {}x{} -> {name}",
            img.width, img.height
        );
        dumped += 1;
    }
    println!("dumped {dumped} textures");
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
