//! Offscreen isometric render of a real city-save's terrain.
//!
//! Thin CLI over the `kolomna-render` library (rendering internals in `src/lib.rs`).
//!
//! usage: cargo run -p kolomna-render --example city_terrain -- <save> <archive.dat> <out.png>

use std::env;

use kolomna_render::{
    SOURCES, TILE, VIEW_H, VIEW_W, ZSCALE, build_mesh, grid_side, load_atlas, load_terrain,
    render_offscreen, water_level, write_png,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: city_terrain <save> <archive.dat> <out.png>");
        std::process::exit(1);
    }

    let terrain = load_terrain(&args[0])?;
    let n = grid_side(&terrain);
    if n * n != terrain.len() {
        return Err(format!("terrain grid is not square ({} values)", terrain.len()).into());
    }
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    println!("terrain: {n}x{n} vertices, altitude {mn:.1}..{mx:.1} m, water surface {wl:.2} m");

    let atlas = load_atlas(&args[1])?;
    println!("atlas: {} textures -> {}x{} px layers", SOURCES.len(), TILE, TILE);

    let verts = build_mesh(&terrain, mn, mx, wl);
    println!("mesh: {} triangles", verts.len() / 3);

    let zspan = (mx - mn) * ZSCALE;
    let rgba = render_offscreen(VIEW_W, VIEW_H, &atlas, &verts, (n - 1) as f32, zspan, mn, wl)?;
    write_png(&args[2], VIEW_W, VIEW_H, &rgba)?;
    println!("saved {}", args[2]);
    Ok(())
}
