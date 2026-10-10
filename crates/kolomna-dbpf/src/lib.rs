//! Reader for DBPF archives (DBPF 1.0, 96-byte header).
//!
//! Layout validated against real archive data:
//! - header: magic `DBPF`, major=1, minor=0, created/modified (unix seconds),
//!   index_major=7, index_count, index_offset, index_size (= count * 20).
//! - index entries: 20 bytes — type, group, instance, offset, size
//!   (bit 31 of size = RefPack/QFS-compressed subfile).
//!
//! References (read-only, not vendored):
//! - <public DBPF format documentation>
//! - open-source DBPF archive readers.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const HEADER_SIZE: usize = 0x60;
pub const ENTRY_SIZE: usize = 20;

/// Type / Group / Instance identifier of a DBPF subfile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tgi {
    pub type_id: u32,
    pub group_id: u32,
    pub instance_id: u32,
}

impl fmt::Display for Tgi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:08x}/{:08x}/{:08x}",
            self.type_id, self.group_id, self.instance_id
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub major: u32,
    pub minor: u32,
    pub created: u32,
    pub modified: u32,
    pub index_major: u32,
    pub index_count: u32,
    pub index_offset: u32,
    pub index_size: u32,
    pub index_minor: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub tgi: Tgi,
    pub offset: u32,
    /// Raw size word; bit 31 set = compressed (RefPack/QFS).
    pub size: u32,
}

impl Entry {
    pub fn is_compressed(&self) -> bool {
        self.size & 0x8000_0000 != 0
    }
    pub fn packed_size(&self) -> u32 {
        self.size & 0x7fff_ffff
    }
}

/// A whole DBPF archive held in memory. Data files are ~100–170 MB, fine for now.
pub struct Archive {
    path: PathBuf,
    data: Vec<u8>,
    pub header: Header,
    pub entries: Vec<Entry>,
}

impl Archive {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let data = fs::read(&path)?;
        let header = parse_header(&data)?;
        let start = header.index_offset as usize;
        let count = header.index_count as usize;
        let end = start
            .checked_add(count * ENTRY_SIZE)
            .ok_or_else(|| bad("index size overflows"))?;
        if end > data.len() {
            return Err(bad("index outside file bounds"));
        }
        let mut entries = Vec::with_capacity(count);
        for chunk in data[start..end].chunks_exact(ENTRY_SIZE) {
            let g = |i: usize| u32::from_le_bytes(chunk[i..i + 4].try_into().unwrap());
            entries.push(Entry {
                tgi: Tgi {
                    type_id: g(0),
                    group_id: g(4),
                    instance_id: g(8),
                },
                offset: g(12),
                size: g(16),
            });
        }
        Ok(Archive {
            path,
            data,
            header,
            entries,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_len(&self) -> usize {
        self.data.len()
    }

    pub fn find(&self, type_id: u32, group_id: u32, instance_id: u32) -> Option<&Entry> {
        self.entries.iter().find(|e| {
            e.tgi.type_id == type_id
                && e.tgi.group_id == group_id
                && e.tgi.instance_id == instance_id
        })
    }

    pub fn entries_of_type(&self, type_id: u32) -> impl Iterator<Item = &Entry> {
        self.entries
            .iter()
            .filter(move |e| e.tgi.type_id == type_id)
    }

    /// Raw bytes of a subfile as stored on disk (no decompression).
    pub fn read_raw(&self, e: &Entry) -> io::Result<&[u8]> {
        let start = e.offset as usize;
        let end = start
            .checked_add(e.packed_size() as usize)
            .ok_or_else(|| bad("subfile size overflows"))?;
        self.data
            .get(start..end)
            .ok_or_else(|| bad("subfile outside file bounds"))
    }

    /// Read a subfile, decompressing it with RefPack when flagged.
    pub fn read_subfile(&self, e: &Entry) -> io::Result<Vec<u8>> {
        let raw = self.read_raw(e)?;
        if e.is_compressed() {
            decompress_refpack(raw)
        } else {
            Ok(raw.to_vec())
        }
    }

    pub fn types_histogram(&self) -> BTreeMap<u32, usize> {
        let mut h = BTreeMap::new();
        for e in &self.entries {
            *h.entry(e.tgi.type_id).or_insert(0) += 1;
        }
        h
    }

    pub fn count_compressed(&self) -> usize {
        self.entries.iter().filter(|e| e.is_compressed()).count()
    }
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn parse_header(data: &[u8]) -> io::Result<Header> {
    if data.len() < HEADER_SIZE || !data.starts_with(b"DBPF") {
        return Err(bad("not a DBPF file"));
    }
    let u = |off: usize| u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let h = Header {
        major: u(0x04),
        minor: u(0x08),
        created: u(0x18),
        modified: u(0x1c),
        index_major: u(0x20),
        index_count: u(0x24),
        index_offset: u(0x28),
        index_size: u(0x2c),
        index_minor: u(0x3c),
    };
    if h.major != 1 {
        return Err(bad("unsupported DBPF major version"));
    }
    Ok(h)
}

/// Decompress a RefPack/QFS stream.
///
/// Compressed subfiles carry a 9-byte prelude:
/// `u32 reserved, u16 compression type, u24 big-endian decompressed size`.
///
/// NOTE: ported from the public RefPack description ;
/// not yet exercised on a compressed subfile — first test pending.
pub fn decompress_refpack(raw: &[u8]) -> io::Result<Vec<u8>> {
    if raw.len() < 9 {
        return Err(bad("compressed subfile too short"));
    }
    let expected = ((raw[6] as usize) << 16) | ((raw[7] as usize) << 8) | raw[8] as usize;
    let input = &raw[9..];
    let mut out: Vec<u8> = Vec::with_capacity(expected);
    let mut pos = 0usize;
    while pos < input.len() {
        let cc = input[pos];
        pos += 1;
        let num_plain;
        let mut num_copy = 0usize;
        let mut offset = 0usize;
        if cc >= 0xFC {
            num_plain = (cc & 3) as usize;
        } else if cc >= 0xE0 {
            num_plain = ((cc as usize) - 0xDF) << 2;
        } else if cc >= 0xC0 {
            if pos + 3 > input.len() {
                return Err(bad("truncated refpack stream"));
            }
            let (b1, b2, b3) = (
                input[pos] as usize,
                input[pos + 1] as usize,
                input[pos + 2] as usize,
            );
            pos += 3;
            num_plain = (cc & 3) as usize;
            num_copy = (((cc & 0x0C) as usize) << 6) + 5 + b3;
            offset = (((cc & 0x10) as usize) << 12) + (b1 << 8) + b2;
        } else if cc >= 0x80 {
            if pos + 2 > input.len() {
                return Err(bad("truncated refpack stream"));
            }
            let (b1, b2) = (input[pos] as usize, input[pos + 1] as usize);
            pos += 2;
            num_plain = (b1 & 0xC0) >> 6;
            num_copy = ((cc & 0x3F) as usize) + 4;
            offset = ((b1 & 0x3F) << 8) + b2;
        } else {
            if pos + 1 > input.len() {
                return Err(bad("truncated refpack stream"));
            }
            let b1 = input[pos] as usize;
            pos += 1;
            num_plain = (cc & 3) as usize;
            num_copy = (((cc & 0x1C) >> 2) as usize) + 3;
            offset = (((cc & 0x60) as usize) << 3) + b1;
        }
        if num_plain > 0 {
            if pos + num_plain > input.len() {
                return Err(bad("truncated refpack literals"));
            }
            out.extend_from_slice(&input[pos..pos + num_plain]);
            pos += num_plain;
        }
        if num_copy > 0 {
            if offset + 1 > out.len() {
                return Err(bad("refpack back-reference out of range"));
            }
            let start = out.len() - offset - 1;
            for k in 0..num_copy {
                let b = out[start + k];
                out.push(b);
            }
        }
        if cc >= 0xFC {
            break;
        }
    }
    Ok(out)
}

/// Human name for common TGI type ids.
/// Source: community DBPF type list (see open-source readers `file-types.ts`).
pub fn type_name(t: u32) -> Option<&'static str> {
    Some(match t {
        0x5AD0E817 => "3DMD",
        0x6534284A => "EXMP",
        0x7AB50E44 => "FSH",
        0x856DDBAC => "PNG",
        0x2026960B => "LTEXT",
        0xE86B1EEF => "DIR",
        0x296678F7 => "Path",
        0x29A5D1EC => "ATC",
        0x09ADCD75 => "AVP",
        0xAA5C3144 => "CURSOR",
        0xCA63E2A3 => "LUA",
        0x05342861 => "Cohort",
        0x0A5BCF4B => "RUL",
        0xEA5118B0 => "EffectDir",
        0xA2E3D533 => "KEYCFG",
        0x66778001 => "BMP",
        0x74807101 => "JFIF",
        0x88777601 | 0x88777602 => "XML",
        0x00000000 => "UI/INI",
        _ => return None,
    })
}
