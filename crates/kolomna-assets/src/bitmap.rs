//! Decoders for the raw bitmap encodings found inside FSH entries.
//!
//! All functions produce RGBA8 (r, g, b, a byte order).
//! Bitmaps are stored little-endian, i.e. 32-bit data is BGRA on disk.
//! 16-bit packed formats are read big-endian, matching community readers.
//! Ported from public community format descriptions.

/// Decode `data` (already sliced to exactly the needed length) into RGBA8.
pub fn decode(code: u8, data: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    Ok(match code & 0x7f {
        0x7B => grayscale8(data),
        0x7D => bgra32(data),
        0x7F => bgr24(data),
        0x7E => packed16(data, unpack1555),
        0x78 => packed16(data, unpack565),
        0x6D => packed16(data, unpack4444),
        0x60 => dxt1(data, width, height),
        0x61 => dxt3(data, width, height),
        other => return Err(format!("unsupported FSH entry id 0x{other:02x}")),
    })
}

/// Raw byte size of the main image for an entry, given its id.
pub fn bytes_for(code: u8, width: usize, height: usize) -> Option<usize> {
    let pixels = width * height;
    Some(match code & 0x7f {
        0x7B => pixels,
        0x7D => pixels * 4,
        0x7F => pixels * 3,
        0x7E | 0x78 | 0x6D => pixels * 2,
        0x60 => pixels / 2,
        0x61 => pixels,
        _ => return None,
    })
}

fn grayscale8(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&[v, v, v, 255]);
    }
    out
}

fn bgra32(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for px in data.chunks_exact(4) {
        out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
    }
    out
}

fn bgr24(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 3 * 4);
    for px in data.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}

fn packed16(data: &[u8], unpack: fn(u16) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 * 4);
    for px in data.chunks_exact(2) {
        let v = ((px[0] as u16) << 8) | px[1] as u16;
        out.extend_from_slice(&unpack(v));
    }
    out
}

fn scale5(v: u16) -> u8 {
    (v as u32 * 255 / 31) as u8
}
fn scale6(v: u16) -> u8 {
    (v as u32 * 255 / 63) as u8
}

fn unpack565(v: u16) -> [u8; 4] {
    [
        scale5((v >> 11) & 31),
        scale6((v >> 5) & 63),
        scale5(v & 31),
        255,
    ]
}

fn unpack1555(v: u16) -> [u8; 4] {
    [
        scale5((v >> 10) & 31),
        scale5((v >> 5) & 31),
        scale5(v & 31),
        if v & 0x8000 != 0 { 255 } else { 0 },
    ]
}

fn unpack4444(v: u16) -> [u8; 4] {
    let s = |x: u16| (x as u32 * 255 / 15) as u8;
    [
        s((v >> 8) & 15),
        s((v >> 4) & 15),
        s(v & 15),
        s((v >> 12) & 15),
    ]
}

fn mix(a: [u8; 4], b: [u8; 4], wa: u32, wb: u32, wsum: u32) -> [u8; 4] {
    let c = |i: usize| ((a[i] as u32 * wa + b[i] as u32 * wb) / wsum) as u8;
    [c(0), c(1), c(2), 255]
}

fn dxt_colors(c0: u16, c1: u16, four_color: bool) -> [[u8; 4]; 4] {
    let a = unpack565(c0);
    let b = unpack565(c1);
    if four_color {
        [a, b, mix(a, b, 2, 1, 3), mix(a, b, 1, 2, 3)]
    } else {
        [a, b, mix(a, b, 1, 1, 2), [0, 0, 0, 0]]
    }
}

fn blit_block(
    out: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    colors: &[[u8; 4]; 4],
    bits: u32,
) {
    for row in 0..4 {
        for col in 0..4 {
            let (px, py) = (x + col, y + row);
            if px >= width || py >= height {
                continue;
            }
            let idx = ((bits >> (2 * (row * 4 + col))) & 3) as usize;
            let di = (py * width + px) * 4;
            out[di..di + 4].copy_from_slice(&colors[idx]);
        }
    }
}

fn dxt1(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 4];
    let mut off = 0usize;
    'blocks: for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            if off + 8 > data.len() {
                break 'blocks;
            }
            let c0 = u16::from_le_bytes([data[off], data[off + 1]]);
            let c1 = u16::from_le_bytes([data[off + 2], data[off + 3]]);
            let bits =
                u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
            let colors = dxt_colors(c0, c1, c0 > c1);
            blit_block(&mut out, width, height, x, y, &colors, bits);
            off += 8;
        }
    }
    out
}

fn dxt3(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 4];
    let mut off = 0usize;
    'blocks: for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            if off + 16 > data.len() {
                break 'blocks;
            }
            // 8 bytes of 4-bit alpha, low nibble first.
            let mut alpha = [0u8; 16];
            for i in 0..8 {
                let byte = data[off + i];
                alpha[i * 2] = (byte & 0x0F) * 17;
                alpha[i * 2 + 1] = ((byte >> 4) & 0x0F) * 17;
            }
            let c0 = u16::from_le_bytes([data[off + 8], data[off + 9]]);
            let c1 = u16::from_le_bytes([data[off + 10], data[off + 11]]);
            let bits = u32::from_le_bytes([
                data[off + 12],
                data[off + 13],
                data[off + 14],
                data[off + 15],
            ]);
            // DXT3 color part always uses the 4-color scheme.
            let colors = dxt_colors(c0, c1, true);
            blit_block(&mut out, width, height, x, y, &colors, bits);
            for row in 0..4 {
                for col in 0..4 {
                    let (px, py) = (x + col, y + row);
                    if px >= width || py >= height {
                        continue;
                    }
                    let di = (py * width + px) * 4;
                    out[di + 3] = alpha[row * 4 + col];
                }
            }
            off += 16;
        }
    }
    out
}
