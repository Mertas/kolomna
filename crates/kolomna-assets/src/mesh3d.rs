//! 3DMD model parser.
//!
//! Container: "3DMD" magic, then tagged chunks HEAD/VERT/INDX/PRIM/MATS/ANIM/
//! PROP/REGP. Each chunk: 4-byte tag + u32 size. The stored size is NOT
//! trustworthy (the stored size is ignored in practice), so chunks are parsed
//! structurally; the parser asserts all bytes are consumed in the end.
//!
//! Layouts (cross-checked against public open-source parsers and
//! byte-verified on archived models):
//! - HEAD: u16 major + u16 minor version ("1.5" -> 1, 5)
//! - VERT: u32 group count; per group: u16 flags, u16 numVerts, u32 format;
//!   vertices = numVerts x 20 bytes (pos f32x3 + uv f32x2)
//! - INDX: u32 group count; per group: u16 flags (0), u16 unused (2),
//!   u16 count, count x u16 indices (triangle list)
//! - PRIM: u32 group count; per group: u16 prim count, then prims x
//!   { u32 type, u32 first, u32 numIndex } — first/numIndex slice the
//!   matching INDX group
//! - MATS: u32 group count; per group: u32 flags, u8 alphaFunc, u8 depthFunc,
//!   u8 srcBlend, u8 dstBlend, u16 alphaThreshold, u32 matClass, u8 reserved,
//!   u8 textureCount; textures x { u32 id, u8 wrapU, u8 wrapV,
//!   [u8 mag, u8 min if minor >= 5], u16 animRate, u16 animMode, u8 nameLen,
//!   name } — the texture `id` is the FSH instance inside group 1abe787d
//! - ANIM: u16 numFrames, u16 frameRate, u16 playMode, u32 flags,
//!   f32 displacement, u16 group count; per group: u8 nameLen, u8 flags,
//!   name, numFrames x { u16 vertex, u16 index, u16 prim, u16 material }
//! - PROP: u32 group count; per group: u16 meshIndex, u16 frameIndex,
//!   u8 len+str assignmentType, u8 len+str assignedValue
//! - REGP: u32 group count; per group: u8 nameLen+name, u16 subgroup count;
//!   subgroups x { 3 x f32 translation, 4 x f32 orientation }

/// FSH texture group used by 3DMD material texture ids.
pub const TEXTURE_GROUP: u32 = 0x1abe787d;

#[derive(Clone, Copy, Debug)]
pub struct Vert {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct VertexGroup {
    pub flags: u16,
    pub format: u32,
    pub verts: Vec<Vert>,
}

#[derive(Clone, Debug, Default)]
pub struct IndexGroup {
    pub indices: Vec<u16>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Prim {
    /// 0 = triangle list (observed in all vanilla models)
    pub kind: u32,
    pub first: u32,
    pub num_index: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialTexture {
    pub id: u32,
    pub wrap_u: u8,
    pub wrap_v: u8,
    pub mag_filter: u8,
    pub min_filter: u8,
    pub anim_rate: u16,
    pub anim_mode: u16,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialGroup {
    pub flags: u32,
    pub textures: Vec<MaterialTexture>,
}

#[derive(Clone, Debug, Default)]
pub struct AnimGroup {
    pub name: String,
    pub flags: u8,
    /// per frame: vertex/index/prim/material group indices
    pub blocks: Vec<[u16; 4]>,
}

#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub num_frames: u16,
    pub frame_rate: u16,
    pub play_mode: u16,
    pub flags: u32,
    pub displacement: f32,
    pub groups: Vec<AnimGroup>,
}

#[derive(Clone, Debug, Default)]
pub struct PropGroup {
    pub mesh_index: u16,
    pub frame_index: u16,
    pub assignment_type: String,
    pub assigned_value: String,
}

#[derive(Clone, Debug, Default)]
pub struct RegpSubgroup {
    pub translation: [f32; 3],
    pub orientation: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct RegpGroup {
    pub name: String,
    pub groups: Vec<RegpSubgroup>,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh3d {
    pub version: (u16, u16),
    pub vertex_groups: Vec<VertexGroup>,
    pub index_groups: Vec<IndexGroup>,
    pub prim_groups: Vec<Vec<Prim>>,
    pub material_groups: Vec<MaterialGroup>,
    pub animation: Animation,
    pub properties: Vec<PropGroup>,
    pub regp_groups: Vec<RegpGroup>,
}

struct R<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> R<'a> {
    fn need(&self, n: usize) -> Result<(), String> {
        if self.o + n > self.b.len() {
            Err(format!(
                "mesh3d: unexpected end of data at {:#x} (need {} more bytes)",
                self.o, n
            ))
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, String> {
        self.need(1)?;
        let v = self.b[self.o];
        self.o += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, String> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.b[self.o..self.o + 2].try_into().unwrap());
        self.o += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, String> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.b[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        self.need(n)?;
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    /// Read an u8-length-prefixed string (trailing NULs trimmed).
    fn string8(&mut self) -> Result<String, String> {
        let n = self.u8()? as usize;
        let b = self.bytes(n)?;
        let end = b.iter().rposition(|&c| c != 0).map(|i| i + 1).unwrap_or(0);
        Ok(String::from_utf8_lossy(&b[..end]).into_owned())
    }
    fn remaining(&self) -> usize {
        self.b.len() - self.o
    }
    /// Read a chunk tag + (ignored) size.
    fn tag(&mut self, expect: &str) -> Result<(), String> {
        let b = self.bytes(4)?;
        if b != expect.as_bytes() {
            return Err(format!(
                "mesh3d: expected chunk {expect}, got {:?} at {:#x}",
                String::from_utf8_lossy(b),
                self.o - 4
            ));
        }
        let _size = self.u32()?;
        Ok(())
    }
}

pub fn parse(raw: &[u8]) -> Result<Mesh3d, String> {
    let mut r = R { b: raw, o: 0 };
    r.tag("3DMD")?;

    r.tag("HEAD")?;
    let mut s = Mesh3d {
        version: (r.u16()?, r.u16()?),
        ..Default::default()
    };

    r.tag("VERT")?;
    let vgroups = r.u32()? as usize;
    for _ in 0..vgroups {
        let mut g = VertexGroup {
            flags: r.u16()?,
            ..Default::default()
        };
        let n = r.u16()? as usize;
        g.format = r.u32()?;
        g.verts.reserve(n);
        for _ in 0..n {
            let pos = [r.f32()?, r.f32()?, r.f32()?];
            let uv = [r.f32()?, r.f32()?];
            g.verts.push(Vert { pos, uv });
        }
        s.vertex_groups.push(g);
    }

    r.tag("INDX")?;
    let igroups = r.u32()? as usize;
    for _ in 0..igroups {
        let _flags = r.u16()?;
        let _unused = r.u16()?;
        let n = r.u16()? as usize;
        let mut g = IndexGroup {
            indices: Vec::with_capacity(n),
        };
        for _ in 0..n {
            g.indices.push(r.u16()?);
        }
        s.index_groups.push(g);
    }

    r.tag("PRIM")?;
    let pgroups = r.u32()? as usize;
    for _ in 0..pgroups {
        let n = r.u16()? as usize;
        let mut g = Vec::with_capacity(n);
        for _ in 0..n {
            g.push(Prim {
                kind: r.u32()?,
                first: r.u32()?,
                num_index: r.u32()?,
            });
        }
        s.prim_groups.push(g);
    }

    r.tag("MATS")?;
    let mgroups = r.u32()? as usize;
    let minor = s.version.1;
    for _ in 0..mgroups {
        let mut g = MaterialGroup {
            flags: r.u32()?,
            ..Default::default()
        };
        let _alpha_func = r.u8()?;
        let _depth_func = r.u8()?;
        let _src_blend = r.u8()?;
        let _dst_blend = r.u8()?;
        let _alpha_threshold = r.u16()?;
        let _mat_class = r.u32()?;
        let _reserved = r.u8()?;
        let nt = r.u8()? as usize;
        for _ in 0..nt {
            let mut t = MaterialTexture {
                id: r.u32()?,
                wrap_u: r.u8()?,
                wrap_v: r.u8()?,
                ..Default::default()
            };
            if minor >= 5 {
                t.mag_filter = r.u8()?;
                t.min_filter = r.u8()?;
            }
            t.anim_rate = r.u16()?;
            t.anim_mode = r.u16()?;
            t.name = r.string8()?;
            g.textures.push(t);
        }
        s.material_groups.push(g);
    }

    r.tag("ANIM")?;
    s.animation.num_frames = r.u16()?;
    s.animation.frame_rate = r.u16()?;
    s.animation.play_mode = r.u16()?;
    s.animation.flags = r.u32()?;
    s.animation.displacement = r.f32()?;
    let agroups = r.u16()? as usize;
    for _ in 0..agroups {
        let mut g = AnimGroup::default();
        let name_len = r.u8()? as usize;
        g.flags = r.u8()?;
        let b = r.bytes(name_len)?;
        let end = b.iter().rposition(|&c| c != 0).map(|i| i + 1).unwrap_or(0);
        g.name = String::from_utf8_lossy(&b[..end]).into_owned();
        for _ in 0..s.animation.num_frames {
            g.blocks
                .push([r.u16()?, r.u16()?, r.u16()?, r.u16()?]);
        }
        s.animation.groups.push(g);
    }

    r.tag("PROP")?;
    let nr = r.u32()? as usize;
    for _ in 0..nr {
        let mut g = PropGroup {
            mesh_index: r.u16()?,
            frame_index: r.u16()?,
            ..Default::default()
        };
        g.assignment_type = r.string8()?;
        g.assigned_value = r.string8()?;
        s.properties.push(g);
    }

    r.tag("REGP")?;
    let ng = r.u32()? as usize;
    for _ in 0..ng {
        let mut g = RegpGroup::default();
        g.name = r.string8()?;
        let subs = r.u16()? as usize;
        for _ in 0..subs {
            let translation = [r.f32()?, r.f32()?, r.f32()?];
            let orientation = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
            g.groups.push(RegpSubgroup {
                translation,
                orientation,
            });
        }
        s.regp_groups.push(g);
    }

    if r.remaining() != 0 {
        return Err(format!(
            "mesh3d: {} trailing bytes after REGP at {:#x}",
            r.remaining(),
            r.o
        ));
    }
    Ok(s)
}

impl Mesh3d {
    /// All vertices of group 0 (all vanilla models use a single VERT group).
    pub fn verts(&self) -> &[Vert] {
        self.vertex_groups
            .first()
            .map(|g| g.verts.as_slice())
            .unwrap_or(&[])
    }

    /// Axis-aligned bounds over all vertex groups (model space, metres).
    pub fn bbox(&self) -> ([f32; 3], [f32; 3]) {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for g in &self.vertex_groups {
            for v in &g.verts {
                for k in 0..3 {
                    mn[k] = mn[k].min(v.pos[k]);
                    mx[k] = mx[k].max(v.pos[k]);
                }
            }
        }
        (mn, mx)
    }

    /// FSH instance ids of every material texture (group 1abe787d).
    pub fn texture_ids(&self) -> Vec<u32> {
        self.material_groups
            .iter()
            .flat_map(|g| g.textures.iter().map(|t| t.id))
            .collect()
    }
}
