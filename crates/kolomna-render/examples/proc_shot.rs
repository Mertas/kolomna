//! Render a row of procedurally generated houses (Foundation-style).
//!
//! usage: proc_shot <city.save> <data2.dat> <out_prefix> [count] [zoom] [seed0]

use std::env;

use kolomna_render::assets::fsh_rgba_search;
use kolomna_render::model::{render_scene_with_models, PlacedModel};
use kolomna_render::procgen::{generate_house, textures};
use kolomna_render::road::{build_road_mesh, load_road_tiles};
use kolomna_render::{
    build_mesh, grid_side, load_atlas, load_terrain, water_level, write_png, VIEW_H, VIEW_W,
    ZSCALE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: proc_shot <city.save> <data2.dat> <out_prefix> [count] [zoom] [seed0]");
        std::process::exit(1);
    }
    let (city, dat, out) = (&args[0], &args[1], &args[2]);
    let count: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let zoom: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(7.5);
    let seed0: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(11);

    let terrain = load_terrain(city)?;
    let ni = grid_side(&terrain) as usize;
    let mn = terrain.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wl = water_level(&terrain, mn);
    let atlas = load_atlas(dat)?;
    let verts = build_mesh(&terrain, mn, mx, wl);
    let grid = (ni - 1) as f32;
    let zspan = (mx - mn) * ZSCALE;

    let tiles = load_road_tiles(city)?;
    let road_mesh = build_road_mesh(&tiles, mn, wl);
    let mut road_tex = Vec::new();
    for id in &road_mesh.texture_ids {
        road_tex.push(fsh_rgba_search(dat, *id)?);
    }

    // road cells, so the row is not planted on the street
    let mut road_cells: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    for t in &tiles {
        let x = ((t.verts[0][0] + t.verts[2][0]) * 0.5 / 16.0) as i32;
        let z = ((t.verts[0][2] + t.verts[2][2]) * 0.5 / 16.0) as i32;
        road_cells.insert((x, z));
    }

    // find a flat, dry patch of terrain (13x13 cells) away from roads
    let mut best = (f32::MAX, ni / 2, ni / 2);
    for j in (8..ni - 8).step_by(4) {
        for i in (8..ni - 8).step_by(4) {
            if road_cells
                .iter()
                .any(|&(rx, rz)| (rx - i as i32).abs() <= 3 && (rz - j as i32).abs() <= 3)
            {
                continue;
            }
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for jj in j - 6..=j + 6 {
                for ii in i - 6..=i + 6 {
                    let v = terrain[jj * ni + ii];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi - lo < best.0 && terrain[j * ni + i] > wl + 2.5 {
                best = (hi - lo, i, j);
            }
        }
    }
    let cx = best.1 as f32;
    let cz = best.2 as f32;
    println!(
        "patch at cell ({cx:.0},{cz:.0}), {:.2} m height spread",
        best.0
    );
    let height_at = |x: f32, z: f32| -> f32 {
        let i = (x.round() as isize).clamp(0, ni as isize - 1) as usize;
        let j = (z.round() as isize).clamp(0, ni as isize - 1) as usize;
        (terrain[j * ni + i] - mn) * ZSCALE
    };

    let tex = textures();
    let mut meshes = Vec::new();
    for i in 0..count {
        meshes.push(generate_house(seed0 + i as u64));
    }
    let models: Vec<PlacedModel> = meshes
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let px = cx + 1.2 * i as f32 - 4.0;
            PlacedModel {
                mesh: m,
                textures: &tex,
                pos: [px, cz, height_at(px, cz)],
                rot: 0.0,
                spin: 0.0,
            }
        })
        .collect();

    // frame the middle of the row
    let midx = cx + 1.2 * (count as f32 - 1.0) * 0.5 - 4.0;
    let hc = height_at(midx, cz);
    let view = ([(midx - cz) * 0.5, (midx + cz) * 0.25 - hc - 0.6], 0.0);
    let rgba = render_scene_with_models(
        VIEW_W,
        VIEW_H,
        &atlas,
        &verts,
        grid,
        zspan,
        zoom,
        mn,
        wl,
        Some(view),
        &models,
        Some(&road_mesh),
        &road_tex,
    )?;
    let f = format!("{out}.png");
    write_png(&f, VIEW_W, VIEW_H, &rgba)?;
    println!("saved {f}");
    Ok(())
}
