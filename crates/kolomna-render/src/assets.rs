//! Loaders for archive assets: model meshes and their textures.
//!
//! Model meshes live in `data1.dat` (type 5ad0e817); textures live in
//! `data2..5.dat` as FSH files (type 7ab50e44) in group 1abe787d, keyed by
//! the material texture id from the 3DMD. `fsh_rgba_search` scans every
//! `data*.dat` sitting next to a given archive path.

use std::error::Error;
use std::path::Path;

use kolomna_assets::fsh::Fsh;
use kolomna_assets::mesh3d;
use kolomna_dbpf::Archive;

pub const FSH_TYPE: u32 = 0x7ab5_0e44;
pub const FSH_GROUP: u32 = 0x1abe_787d;

/// Read and parse one 3DMD model by TGI.
pub fn mesh3d_from_dat(dat: &str, t: u32, g: u32, i: u32) -> Result<mesh3d::Mesh3d, Box<dyn Error>> {
    let archive = Archive::open(dat).map_err(|e| format!("{dat}: {e}"))?;
    let e = archive
        .find(t, g, i)
        .ok_or_else(|| format!("mesh3d {t:08x}/{g:08x}/{i:08x} not found in {dat}"))?;
    let raw = archive.read_subfile(e)?;
    match mesh3d::parse(&raw) {
        Ok(m) => Ok(m),
        Err(e) => Err(format!("mesh3d {t:08x}/{g:08x}/{i:08x}: parse error: {e}").into()),
    }
}

/// Decode FSH entry 0 (group 1abe787d) from `dat`, if present.
pub fn fsh_rgba(dat: &str, instance: u32) -> Result<Option<(u32, u32, Vec<u8>)>, Box<dyn Error>> {
    let archive = match Archive::open(dat) {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };
    let Some(e) = archive.find(FSH_TYPE, FSH_GROUP, instance) else {
        return Ok(None);
    };
    let raw = archive.read_subfile(e)?;
    let Ok(f) = Fsh::parse(&raw) else {
        return Ok(None);
    };
    match f.decode_entry(&raw, 0) {
        Ok(img) => Ok(Some((img.width, img.height, img.rgba))),
        Err(_) => Ok(None),
    }
}

/// Find and decode FSH `instance` in any `data*.dat` next to `any_dat`.
pub fn fsh_rgba_search(any_dat: &str, instance: u32) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let dir = Path::new(any_dat)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    for name in [
        "data5.dat",
        "data4.dat",
        "data3.dat",
        "data2.dat",
        "data1.dat",
    ] {
        let p = dir.join(name);
        if !p.exists() {
            continue;
        }
        if let Some(img) = fsh_rgba(&p.to_string_lossy(), instance)? {
            return Ok(img);
        }
    }
    Err(format!("FSH 1abe787d/{instance:08x} not found next to {any_dat}").into())
}
