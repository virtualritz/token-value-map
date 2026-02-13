// AIDEV-NOTE: glam backend -- type aliases and utility functions.
//
// ZERO-COST: All utility functions have #[inline(always)] for complete inlining.

pub type Vec2Impl = glam::Vec2;
pub type Vec3Impl = glam::Vec3;
pub type Mat3Impl = glam::Mat3;
pub type Mat4Impl = glam::DMat4; // f64 version.

// AIDEV-NOTE: glam has no separate Point3 type -- Vec3 serves as both.
pub type Point3Impl = Vec3Impl;

// --- Utility functions ---

/// Create a zero `Vec2`.
#[inline(always)]
pub fn vec2_zeros() -> Vec2Impl {
    Vec2Impl::ZERO
}

/// Create a zero `Vec3`.
#[inline(always)]
pub fn vec3_zeros() -> Vec3Impl {
    Vec3Impl::ZERO
}

/// Get `Vec2` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec2_as_slice(v: &Vec2Impl) -> &[f32] {
    v.as_ref()
}

/// Get `Vec2` data as a `&[f32; 2]` ref.
#[inline(always)]
pub fn vec2_as_ref(v: &Vec2Impl) -> &[f32; 2] {
    v.as_ref()
}

/// Get `Vec3` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec3_as_slice(v: &Vec3Impl) -> &[f32] {
    v.as_ref()
}

/// Get `Vec3` data as a `&[f32; 3]` ref.
#[inline(always)]
pub fn vec3_as_ref(v: &Vec3Impl) -> &[f32; 3] {
    v.as_ref()
}

/// Return a normalized copy of a `Vec3`.
#[inline(always)]
pub fn vec3_normalized(v: &Vec3Impl) -> Vec3Impl {
    v.normalize()
}

/// Create a `Mat3` from a row-major slice.
#[inline(always)]
pub fn mat3_from_row_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    // glam stores column-major, so transpose from row-major input.
    Mat3Impl::from_cols_array(&[
        data[0], data[3], data[6], // Column 0.
        data[1], data[4], data[7], // Column 1.
        data[2], data[5], data[8], // Column 2.
    ])
}

/// Create a `Mat3` from a column-major slice.
#[inline(always)]
pub fn mat3_from_column_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    Mat3Impl::from_cols_array(data.try_into().unwrap())
}

/// Return a zero `Mat3`.
#[inline(always)]
pub fn mat3_zeros() -> Mat3Impl {
    Mat3Impl::ZERO
}

/// Get `Mat3` data as a `&[f32]` slice (column-major order).
#[inline(always)]
pub fn mat3_as_slice(m: &Mat3Impl) -> &[f32] {
    m.as_ref()
}

/// Iterate over `Mat3` elements (column-major order).
#[inline(always)]
pub fn mat3_iter(m: &Mat3Impl) -> impl Iterator<Item = &f32> {
    m.as_ref().iter()
}

/// Extract a row from a `Mat3` as `[f32; 3]`.
#[inline(always)]
pub fn mat3_row(m: &Mat3Impl, i: usize) -> [f32; 3] {
    // glam stores column-major: col[c][r].
    let cols: &[f32; 9] = m.as_ref();
    [cols[i], cols[i + 3], cols[i + 6]]
}

/// Create a `DMat4` from a row-major slice.
#[inline(always)]
pub fn mat4_from_row_slice(data: &[f64]) -> Mat4Impl {
    assert_eq!(data.len(), 16, "Matrix4 requires exactly 16 elements");
    Mat4Impl::from_cols_array(&[
        data[0], data[4], data[8], data[12], // Column 0.
        data[1], data[5], data[9], data[13], // Column 1.
        data[2], data[6], data[10], data[14], // Column 2.
        data[3], data[7], data[11], data[15], // Column 3.
    ])
}

/// Subtract two `DMat4` matrices element-wise.
#[inline(always)]
pub fn mat4_sub(a: Mat4Impl, b: Mat4Impl) -> Mat4Impl {
    a - b
}

/// Return a zero `DMat4`.
#[inline(always)]
pub fn mat4_zeros() -> Mat4Impl {
    Mat4Impl::ZERO
}

/// Iterate over `DMat4` elements (column-major order).
#[inline(always)]
pub fn mat4_iter(m: &Mat4Impl) -> impl Iterator<Item = &f64> {
    m.as_ref().iter()
}

/// Return `DMat4` data as a `&[f64]` slice.
#[inline(always)]
pub fn mat4_as_slice(m: &Mat4Impl) -> &[f64] {
    m.as_ref()
}

/// Return the origin point (zero vector).
#[inline(always)]
pub fn point3_origin() -> Point3Impl {
    Vec3Impl::ZERO
}

/// Get point coordinates as a slice.
#[inline(always)]
pub fn point3_as_slice(p: &Point3Impl) -> &[f32] {
    p.as_ref()
}

/// Get point coordinates as a `&[f32; 3]` ref.
#[inline(always)]
pub fn point3_as_ref(p: &Point3Impl) -> &[f32; 3] {
    p.as_ref()
}

/// Return the identity `Mat3`.
#[inline(always)]
pub fn mat3_identity() -> Mat3Impl {
    Mat3Impl::IDENTITY
}

/// Return the identity `DMat4`.
#[inline(always)]
pub fn mat4_identity() -> Mat4Impl {
    Mat4Impl::IDENTITY
}

/// Get element at `(row, col)` from a `Mat3`.
#[inline(always)]
pub fn mat3(m: &Mat3Impl, row: usize, col: usize) -> f32 {
    // glam stores column-major.
    m.col(col)[row]
}

/// Get element at `(row, col)` from a `DMat4`.
#[inline(always)]
pub fn mat4(m: &Mat4Impl, row: usize, col: usize) -> f64 {
    // glam stores column-major.
    m.col(col)[row]
}

/// Subtract two `Mat3` matrices element-wise.
#[inline(always)]
pub fn mat3_sub(a: Mat3Impl, b: Mat3Impl) -> Mat3Impl {
    a - b
}

/// Create a diagonal `Mat3` with `v` on the diagonal.
#[inline(always)]
pub fn mat3_from_diagonal_element(v: f32) -> Mat3Impl {
    Mat3Impl::from_diagonal(glam::Vec3::splat(v))
}
