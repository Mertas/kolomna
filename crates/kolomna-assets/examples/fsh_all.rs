//! Dump ALL entries of one FSH texture to PNG files.
//!
//! usage: cargo run -p kolomna-assets --example fsh_all -- <file.dat> <out_dir> <type_hex> <group_hex> <instance_hex>

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
    if args.len() != 5 {
        eprintln!("usage: fsh_all <file.dat> <out_dir> <type> <group> <instance>");
        std::process::exit(1);
    }
    let (t, g, i) = (
        hx(&args[2]).expect("type"),
        hx(&args[3]).expect("group"),
        hx(&args[4]).expect("instance"),
    );
    let archive = Archive::open(&args[0])?;
    fs::create_dir_all(&args[1])?;
    let e = archive
        .find(t, g, i)
        .unwrap_or_else(|| panic!("{t:08x}/{g:08x}/{i:08x} not found"));
    let raw = archive.read_subfile(e)?;
    let fsh = Fsh::parse(&raw).unwrap_or_else(|err| panic!("parse: {err}"));
    println!(
        "{t:08x}/{g:08x}/{i:08x}: {} bytes, dir_id {:?}, {} entries",
        raw.len(),
        fsh.dir_id,
        fsh.entries.len()
    );
    for (k, entry) in fsh.entries.iter().enumerate() {
        println!(
            "  [{k}] name={:?} id={:#04x} {}x{} mips={}",
            entry.name, entry.entry_id, entry.width, entry.height, entry.mip_count
        );
        match fsh.decode_entry(&raw, k) {
            Ok(img) => {
                let fname = format!(
                    "{:02}_{}_entry{k}_{}x{}.png",
                    k, entry.name.trim(), img.width, img.height
                );
                write_png(&Path::new(&args[1]).join(&fname), img.width, img.height, &img.rgba)?;
                println!("      -> {fname}");
            }
            Err(err) => println!("      decode error: {err}"),
        }
    }
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
