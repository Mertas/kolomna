//! Show all four Z5 orientation variants of the a small house side by side.
//!
//! Buildings ship four rotated copies (instance base N, N+0x10, N+0x20,
//! N+0x30); they swap when the camera rotates. This example puts
//! all four "straight to camera" (rot=0, angle=0) in a row so each facade
//! can be inspected.
//!
//! usage: house4 <city.save> <data2.dat> <out_prefix>

use std::env;
use std::path::Path;

use kolomna_render::assets::{fsh_rgba_search, mesh3d_from_dat};
use kolomna_render::model::{mesh_from_mesh3d, render_scene_with_models, ModelMesh, PlacedModel};
use kolomna_render::road::{build_road_mesh, load_road_tiles};
use kolomna_render::{
    build_mesh, grid_side, load_atlas, load_terrain, water_level, write_png, VIEW_H, VIEW_W, ZSCALE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: house4 <city.save> <data2.dat> <out_prefix>");
        std::process::exit(1);
    }
    let (city, dat, out) = (&args[0], &args[1], &args[2]);

    let terrain = load_terrain(city)?;
    let n = grid_side(&terrain);
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    let atlas = load_atlas(dat)?;
    let verts = build_mesh(&terrain, mn, mx, wl);
    let grid = (n - 1) as f32;
    let zspan = (mx - mn) * ZSCALE;

    let tiles = load_road_tiles(city)?;
    println!("road tiles: {}", tiles.len());
    let road_mesh = build_road_mesh(&tiles, mn, wl);
    let mut road_tex = Vec::new();
    for id in &road_mesh.texture_ids {
        road_tex.push(fsh_rgba_search(dat, *id)?);
    }

    let s1 = sibling(dat, "data1.dat");
    let insts = [0x00ad0400u32, 0x00ad0410, 0x00ad0420, 0x00ad0430];
    let mut meshes: Vec<ModelMesh> = Vec::new();
    let mut texsets: Vec<Vec<(u32, u32, Vec<u8>)>> = Vec::new();
    for i in insts {
        let m = mesh3d_from_dat(&s1, 0x5ad0e817, 0xbadb57f1, i)?;
        let mesh = mesh_from_mesh3d(&m);
        let mut ts = Vec::new();
        for id in &mesh.texture_ids {
            ts.push(fsh_rgba_search(dat, *id)?);
        }
        println!(
            "{i:08x}: {} verts, tex {:08x?}",
            mesh.verts.len(),
            mesh.texture_ids
        );
        meshes.push(mesh);
        texsets.push(ts);
    }

    // street position: same as road_shot — near the midline tile
    let t = &tiles[tiles.len() / 2];
    let cx = (t.verts[0][0] + t.verts[2][0]) * 0.5 / 16.0;
    let cz = (t.verts[0][2] + t.verts[2][2]) * 0.5 / 16.0;
    let alt = (t.verts[0][1] + t.verts[2][1]) * 0.5;
    let hz_mesh = (alt - mn) * ZSCALE;

    // row of 4 houses, one per orientation, each rendered "straight to camera"
    let hy = cz - 2.0;
    let mut models = Vec::new();
    for k in 0..4usize {
        models.push(PlacedModel {
            mesh: &meshes[k],
            textures: &texsets[k],
            pos: [cx + 1.6 * k as f32, cz - 2.0, hz_mesh],
            rot: 0.0,
            spin: 1.0,
        });
    }
    let ccx = cx + 1.7 * 1.5;
    let center = [(ccx - hy) * 0.5, (ccx + hy) * 0.25 - hz_mesh - 0.7];
    let rgba = render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &atlas,
        &verts,
        grid,
        zspan,
        11.0,
        mn,
        wl,
        Some((center, 0.0)),
        &models,
        Some(&road_mesh),
        &road_tex,
    )?;
    let f = format!("{out}_row.png");
    write_png(&f, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {f}");
    Ok(())
}

fn sibling(any_dat: &str, name: &str) -> String {
    Path::new(any_dat)
        .parent()
        .map(|p| p.join(name).to_string_lossy().into_owned())
        .unwrap_or_else(|| name.to_string())
}
