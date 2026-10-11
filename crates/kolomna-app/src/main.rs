//! Kolomna application binary.
//!
//! Modes:
//! - interactive (default): opens a window and renders the city terrain with a
//!   soft isometric camera.
//! - headless: `kolomna-app --shot <out.png> [--zoom N]` renders exactly one
//!   frame offscreen (zoom > 1 for close-up checks) and exits.
//!
//! With no arguments the exe opens a window with a procedural city;
//! it can simply be double-clicked (no external data needed).

mod ui_atlas;
mod win;

use std::process::exit;

const DEFAULT_CITY: &str =
    "C:/Users/<user>/Kolomna/city.save";
const DEFAULT_DAT: &str = "data2.dat";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        exit(1);
    }
}

fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = args.to_vec();
    // optional --zoom <factor> (close-up render for --shot; ignored in window mode)
    let mut zoom = 1.0f32;
    if let Some(i) = args.iter().position(|a| a == "--zoom") {
        if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()) {
            zoom = v;
        }
        if i + 1 < args.len() {
            args.drain(i..i + 2);
        } else {
            args.remove(i);
        }
    }

    if args.first().map(String::as_str) == Some("--help") || args.first().map(String::as_str) == Some("-h") {
        usage();
        return Ok(());
    }
    if let Ok(p) = std::env::var("KOLOMNA_UI_SHOT") {
        win::ui_shot(&p);
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("--shot") {
        let out = args
            .get(1)
            .cloned()
            .unwrap_or_else(|| "out/shot.png".to_string());
        let city = args.get(2).cloned().unwrap_or_else(|| DEFAULT_CITY.to_string());
        let dat = args.get(3).cloned().unwrap_or_else(|| DEFAULT_DAT.to_string());
        return win::shot(&out, &city, &dat, zoom);
    }
    let city = args.first().cloned().unwrap_or_else(|| DEFAULT_CITY.to_string());
    let dat = args.get(1).cloned().unwrap_or_else(|| DEFAULT_DAT.to_string());
    win::run(&city, &dat)
}

fn usage() {
    println!("Kolomna — изометрический просмотр сохранённого города");
    println!();
    println!("  kolomna-app [<city.save> [<data2.dat>]]   окно (по умолчанию — процедурный город)");
    println!("  kolomna-app --shot [out.png] [city] [dat] [--zoom N]   один кадр в PNG (N — приближение)");
    println!();
    println!("Управление:");
    println!("  колесо мыши            мягкий зум");
    println!("  средняя кнопка (drag)  мягкий поворот");
    println!("  Shift+средняя / ЛКМ / ПКМ (drag) — перенос карты");
    println!("  края окна              навигация (карта сама едет, если мышь у края)");
    println!("  WASD / стрелки         перенос;  Q / E — поворот на 90°");
    println!("  R — сброс вида;  Esc — выход");
}
