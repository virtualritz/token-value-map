// AIDEV-NOTE: ultraviolet backend -- type aliases and utility functions.
//
// ZERO-COST: All utility functions have #[inline(always)] for complete inlining.

pub type Vec2Impl = ultraviolet::Vec2;
pub type Vec3Impl = ultraviolet::Vec3;
pub type Mat3Impl = ultraviolet::Mat3;
pub type Mat4Impl = ultraviolet::DMat4;

// AIDEV-NOTE: ultraviolet has no separate Point3 type -- Vec3 serves as both.
pub type Point3Impl = Vec3Impl;

// --- Utility functions ---

/// Create a zero `Vec2`.
#[inline(always)]
pub fn vec2_zeros() -> Vec2Impl {
    Vec2Impl::zero()
}

/// Create a zero `Vec3`.
#[inline(always)]
pub fn vec3_zeros() -> Vec3Impl {
    Vec3Impl::zero()
}

/// Get `Vec2` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec2_as_slice(v: &Vec2Impl) -> &[f32] {
    v.as_slice()
}

/// Get `Vec2` data as a `&[f32; 2]` ref.
#[inline(always)]
pub fn vec2_as_ref(v: &Vec2Impl) -> &[f32; 2] {
    v.as_array()
}

/// Get `Vec3` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec3_as_slice(v: &Vec3Impl) -> &[f32] {
    v.as_slice()
}

/// Get `Vec3` data as a `&[f32; 3]` ref.
#[inline(always)]
pub fn vec3_as_ref(v: &Vec3Impl) -> &[f32; 3] {
    v.as_array()
}

/// Return a normalized copy of a `Vec3`.
#[inline(always)]
pub fn vec3_normalized(v: &Vec3Impl) -> Vec3Impl {
    v.normalized()
}

/// Create a `Mat3` from a row-major slice.
#[inline(always)]
pub fn mat3_from_row_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    // ultraviolet stores column-major, so transpose from row-major input.
    Mat3Impl::new(
        ultraviolet::Vec3::new(data[0], data[3], data[6]),
        ultraviolet::Vec3::new(data[1], data[4], data[7]),
        ultraviolet::Vec3::new(data[2], data[5], data[8]),
    )
}

/// Create a `Mat3` from a column-major slice.
#[inline(always)]
pub fn mat3_from_column_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    Mat3Impl::new(
        ultraviolet::Vec3::new(data[0], data[1], data[2]),
        ultraviolet::Vec3::new(data[3], data[4], data[5]),
        ultraviolet::Vec3::new(data[6], data[7], data[8]),
    )
}

/// Return a zero `Mat3`.
#[inline(always)]
pub fn mat3_zeros() -> Mat3Impl {
    Mat3Impl::default()
}

/// Get `Mat3` data as a `&[f32]` slice (column-major order).
#[inline(always)]
pub fn mat3_as_slice(m: &Mat3Impl) -> &[f32] {
    m.as_slice()
}

/// Iterate over `Mat3` elements (column-major order).
#[inline(always)]
pub fn mat3_iter(m: &Mat3Impl) -> impl Iterator<Item = &f32> {
    m.as_slice().iter()
}

/// Extract a row from a `Mat3` as `[f32; 3]`.
#[inline(always)]
pub fn mat3_row(m: &Mat3Impl, i: usize) -> [f32; 3] {
    // ultraviolet stores column-major: cols[c][component].
    [
        m.cols[0].as_slice()[i],
        m.cols[1].as_slice()[i],
        m.cols[2].as_slice()[i],
    ]
}

/// Create a `DMat4` from a row-major slice.
#[inline(always)]
pub fn mat4_from_row_slice(data: &[f64]) -> Mat4Impl {
    assert_eq!(data.len(), 16, "Matrix4 requires exactly 16 elements");
    Mat4Impl::new(
        ultraviolet::DVec4::new(data[0], data[4], data[8], data[12]),
        ultraviolet::DVec4::new(data[1], data[5], data[9], data[13]),
        ultraviolet::DVec4::new(data[2], data[6], data[10], data[14]),
        ultraviolet::DVec4::new(data[3], data[7], data[11], data[15]),
    )
}

/// Subtract two `DMat4` matrices element-wise.
///
/// AIDEV-NOTE: ultraviolet `DMat4` lacks a `Sub` impl, so we negate and add.
#[inline(always)]
pub fn mat4_sub(a: Mat4Impl, b: Mat4Impl) -> Mat4Impl {
    // DMat4 has Add and Mul<f64> but no Sub. Negate via scalar mul and add.
    a + b * -1.0
}

/// Return a zero `DMat4`.
#[inline(always)]
pub fn mat4_zeros() -> Mat4Impl {
    Mat4Impl::default()
}

/// Iterate over `DMat4` elements (column-major order).
#[inline(always)]
pub fn mat4_iter(m: &Mat4Impl) -> impl Iterator<Item = &f64> {
    m.as_slice().iter()
}

/// Return `DMat4` data as a `&[f64]` slice.
#[inline(always)]
pub fn mat4_as_slice(m: &Mat4Impl) -> &[f64] {
    m.as_slice()
}

/// Return the origin point (zero vector).
#[inline(always)]
pub fn point3_origin() -> Point3Impl {
    Vec3Impl::zero()
}

/// Get point coordinates as a slice.
#[inline(always)]
pub fn point3_as_slice(p: &Point3Impl) -> &[f32] {
    p.as_slice()
}

/// Get point coordinates as a `&[f32; 3]` ref.
#[inline(always)]
pub fn point3_as_ref(p: &Point3Impl) -> &[f32; 3] {
    bytemuck::cast_ref(p)
}

/// Return the identity `Mat3`.
#[inline(always)]
pub fn mat3_identity() -> Mat3Impl {
    Mat3Impl::identity()
}

/// Return the identity `DMat4`.
#[inline(always)]
pub fn mat4_identity() -> Mat4Impl {
    Mat4Impl::identity()
}

/// Get element at `(row, col)` from a `Mat3`.
#[inline(always)]
pub fn mat3(m: &Mat3Impl, row: usize, col: usize) -> f32 {
    m.cols[col].as_slice()[row]
}

/// Get element at `(row, col)` from a `DMat4`.
#[inline(always)]
pub fn mat4(m: &Mat4Impl, row: usize, col: usize) -> f64 {
    m.cols[col].as_slice()[row]
}

/// Subtract two `Mat3` matrices element-wise.
///
/// AIDEV-NOTE: ultraviolet `Mat3` lacks a `Sub` impl, so we do it manually.
#[inline(always)]
pub fn mat3_sub(a: Mat3Impl, b: Mat3Impl) -> Mat3Impl {
    Mat3Impl::new(
        a.cols[0] - b.cols[0],
        a.cols[1] - b.cols[1],
        a.cols[2] - b.cols[2],
    )
}

/// Create a diagonal `Mat3` with `v` on the diagonal.
#[inline(always)]
pub fn mat3_from_diagonal_element(v: f32) -> Mat3Impl {
    Mat3Impl::from_scale(v)
}
