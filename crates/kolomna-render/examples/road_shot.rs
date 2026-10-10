//! Terrain + roads from the city save + one house (a small house) -> PNGs.
//!
//! usage: cargo run -p kolomna-render --example road_shot -- <city.save> <data2.dat> <out_prefix>

use std::env;
use std::path::Path;

use kolomna_render::assets::{fsh_rgba_search, mesh3d_from_dat};
use kolomna_render::model::{mesh_from_mesh3d, render_scene_with_models, PlacedModel};
use kolomna_render::road::{build_road_mesh, load_road_tiles};
use kolomna_render::{
    build_mesh, grid_side, load_atlas, load_terrain, water_level, write_png, VIEW_H, VIEW_W, ZSCALE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: road_shot <city.save> <data2.dat> <out_prefix>");
        std::process::exit(1);
    }
    let (city, dat, out) = (&args[0], &args[1], &args[2]);

    let terrain = load_terrain(city)?;
    let n = grid_side(&terrain);
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    println!("terrain: {n}x{n}, {mn:.1}..{mx:.1} m, water {wl:.2} m");

    let atlas = load_atlas(dat)?;
    let verts = build_mesh(&terrain, mn, mx, wl);
    let grid = (n - 1) as f32;
    let zspan = (mx - mn) * ZSCALE;

    // ---- roads from the save
    let tiles = load_road_tiles(city)?;
    println!("road tiles: {}", tiles.len());
    let road_mesh = build_road_mesh(&tiles, mn, wl);
    let mut road_tex = Vec::new();
    for id in &road_mesh.texture_ids {
        let img = fsh_rgba_search(dat, *id)?;
        println!("road tex {id:08x}: {}x{}", img.0, img.1);
        road_tex.push(img);
    }

    // ---- house: Z5S model of a small house
    let s1 = sibling(dat, "data1.dat");
    let m = mesh3d_from_dat(&s1, 0x5ad0e817, 0xbadb57f1, 0x00ad0400)?;
    let mesh = mesh_from_mesh3d(&m);
    println!(
        "house mesh: {} verts, {} indices, textures {:08x?}",
        mesh.verts.len(),
        mesh.indices.len(),
        mesh.texture_ids
    );
    let mut htex = Vec::new();
    for id in &mesh.texture_ids {
        let img = fsh_rgba_search(dat, *id)?;
        println!("house tex {id:08x}: {}x{}", img.0, img.1);
        htex.push(img);
    }

    // place the house next to the middle road tile
    let t = &tiles[tiles.len() / 2];
    let cx = (t.verts[0][0] + t.verts[2][0]) * 0.5 / 16.0;
    let cz = (t.verts[0][2] + t.verts[2][2]) * 0.5 / 16.0;
    let alt = (t.verts[0][1] + t.verts[2][1]) * 0.5;
    let hx = cx;
    let hz = cz - 1.0;
    let hz_mesh = (alt - mn) * ZSCALE;
    println!("house at cell ({hx:.1}, {hz:.1}), alt {alt:.1} m");

    // wide shot: whole grid
    let house = PlacedModel {
        mesh: &mesh,
        textures: &htex,
        pos: [hx, hz, hz_mesh],
        rot: 0.0,
    };
    let rgba = render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &atlas,
        &verts,
        grid,
        zspan,
        1.0,
        mn,
        wl,
        None,
        &[house],
        Some(&road_mesh),
        &road_tex,
    )?;
    let out_wide = format!("{out}_wide.png");
    write_png(&out_wide, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {out_wide}");

    // close shot: centre on the house
    let center = [(hx - hz) * 0.5, (hx + hz) * 0.25 - hz_mesh - 0.4];
    let house = PlacedModel {
        mesh: &mesh,
        textures: &htex,
        pos: [hx, hz, hz_mesh],
        rot: 0.0,
    };
    let rgba = render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &atlas,
        &verts,
        grid,
        zspan,
        9.0,
        mn,
        wl,
        Some((center, 0.0)),
        &[house],
        Some(&road_mesh),
        &road_tex,
    )?;
    let out_close = format!("{out}_close.png");
    write_png(&out_close, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {out_close}");

    // turntable: render the house from 4 camera angles (check all walls)
    let p = grid * 0.5;
    let d = [hx - p, hz - p];
    for (k, a) in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 4.712389]
        .iter()
        .enumerate()
    {
        let (ca, sa) = (a.cos(), a.sin());
        let rpx = d[0] * ca - d[1] * sa + p;
        let rpy = d[0] * sa + d[1] * ca + p;
        let center = [(rpx - rpy) * 0.5, (rpx + rpy) * 0.25 - hz_mesh - 0.6];
        let house = PlacedModel {
            mesh: &mesh,
            textures: &htex,
            pos: [hx, hz, hz_mesh],
            rot: 0.0,
        };
        let rgba = render_scene_with_models(
            VIEW_W,
            VIEW_H,
            &atlas,
            &verts,
            grid,
            zspan,
            20.0,
            mn,
            wl,
            Some((center, *a)),
            &[house],
            Some(&road_mesh),
            &road_tex,
        )?;
        let f = format!("{out}_turn{k}.png");
        write_png(&f, VIEW_W, VIEW_H, &rgba)?;
        println!("saved {f}");
    }
    Ok(())
}

fn sibling(any_dat: &str, name: &str) -> String {
    Path::new(any_dat)
        .parent()
        .map(|p| p.join(name).to_string_lossy().into_owned())
        .unwrap_or_else(|| name.to_string())
}
