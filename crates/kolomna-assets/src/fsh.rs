//! FSH ("SHPI") texture container: parsing and decoding to RGBA8.
//!
//! Layout (little-endian):
//! ```text
//! 0x00 "SHPI"
//! 0x04 u32 total file size
//! 0x08 u32 number of entries
//! 0x0C 4-byte directory id
//! directory: per entry — 4-byte name, u32 offset
//! entry header at `offset`:
//!   u8  record id (0x60 DXT1, 0x61 DXT3, 0x7B 8-bit, 0x7D 32-bit, 0x7F 24-bit, ...)
//!   u24 block size
//!   u16 width, u16 height
//!   u16 center x, u16 center y
//!   u16 x position (+flags)
//!   u16 y position (+4-bit mipmap count in the high nibble)
//!   ... image data (and mipmaps) follow at offset + 16
//! ```

use std::io;

use crate::bitmap;

#[derive(Debug, Clone)]
pub struct FshEntry {
    pub name: String,
    pub offset: u32,
    pub entry_id: u8,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub data_offset: u32,
}

#[derive(Debug)]
pub struct Fsh {
    pub file_size: u32,
    pub dir_id: String,
    pub entries: Vec<FshEntry>,
}

#[derive(Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Fsh {
    pub fn parse(bytes: &[u8]) -> io::Result<Fsh> {
        if bytes.len() < 16 || !bytes.starts_with(b"SHPI") {
            return Err(bad("not an FSH file"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let file_size = u32_at(4);
        let num = u32_at(8) as usize;
        let dir_id = String::from_utf8_lossy(&bytes[12..16]).to_string();

        let mut entries = Vec::with_capacity(num);
        let mut o = 16usize;
        for _ in 0..num {
            if o + 8 > bytes.len() {
                break;
            }
            let name = String::from_utf8_lossy(&bytes[o..o + 4]).to_string();
            let offset = u32_at(o + 4);
            entries.push(FshEntry {
                name,
                offset,
                entry_id: 0,
                width: 0,
                height: 0,
                mip_count: 0,
                data_offset: 0,
            });
            o += 8;
        }

        for e in entries.iter_mut() {
            let b = e.offset as usize;
            if b + 16 > bytes.len() {
                continue;
            }
            e.entry_id = bytes[b];
            e.width = u16::from_le_bytes(bytes[b + 4..b + 6].try_into().unwrap()) as u32;
            e.height = u16::from_le_bytes(bytes[b + 6..b + 8].try_into().unwrap()) as u32;
            let y_word = u16::from_le_bytes(bytes[b + 14..b + 16].try_into().unwrap());
            e.mip_count = (y_word >> 12) as u32;
            e.data_offset = e.offset + 16;
        }

        Ok(Fsh {
            file_size,
            dir_id,
            entries,
        })
    }

    /// Decode entry `index` (main image only, no mipmaps) to RGBA8.
    pub fn decode_entry(&self, bytes: &[u8], index: usize) -> Result<Image, String> {
        let e = self
            .entries
            .get(index)
            .ok_or_else(|| "no such entry".to_string())?;
        let needed = bitmap::bytes_for(e.entry_id, e.width as usize, e.height as usize)
            .ok_or_else(|| format!("unknown FSH entry id 0x{:02x}", e.entry_id))?;
        let start = e.data_offset as usize;
        let end = start + needed;
        if end > bytes.len() {
            return Err("entry image data out of bounds".to_string());
        }
        let rgba = bitmap::decode(
            e.entry_id,
            &bytes[start..end],
            e.width as usize,
            e.height as usize,
        )?;
        Ok(Image {
            width: e.width,
            height: e.height,
            rgba,
        })
    }
}
