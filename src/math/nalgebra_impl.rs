// AIDEV-NOTE: nalgebra backend -- type aliases and utility functions.
//
// ZERO-COST: All utility functions have #[inline(always)] for complete inlining.

pub type Vec2Impl = nalgebra::Vector2<f32>;
pub type Vec3Impl = nalgebra::Vector3<f32>;
pub type Mat3Impl = nalgebra::Matrix3<f32>;
pub type Mat4Impl = nalgebra::Matrix4<f64>;
pub type Point3Impl = nalgebra::Point3<f32>;

// --- Utility functions ---

/// Create a zero `Vector2`.
#[inline(always)]
pub fn vec2_zeros() -> Vec2Impl {
    Vec2Impl::zeros()
}

/// Create a zero `Vector3`.
#[inline(always)]
pub fn vec3_zeros() -> Vec3Impl {
    Vec3Impl::zeros()
}

/// Get `Vector2` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec2_as_slice(v: &Vec2Impl) -> &[f32] {
    v.as_slice()
}

/// Get `Vector2` data as a `&[f32; 2]` ref.
#[inline(always)]
pub fn vec2_as_ref(v: &Vec2Impl) -> &[f32; 2] {
    v.as_ref()
}

/// Get `Vector3` data as a `&[f32]` slice.
#[inline(always)]
pub fn vec3_as_slice(v: &Vec3Impl) -> &[f32] {
    v.as_slice()
}

/// Get `Vector3` data as a `&[f32; 3]` ref.
#[inline(always)]
pub fn vec3_as_ref(v: &Vec3Impl) -> &[f32; 3] {
    v.as_ref()
}

/// Return a normalized copy of a `Vector3`.
#[inline(always)]
pub fn vec3_normalized(v: &Vec3Impl) -> Vec3Impl {
    v.normalize()
}

/// Create a `Matrix3` from a row-major slice.
#[inline(always)]
pub fn mat3_from_row_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    Mat3Impl::from_row_slice(data)
}

/// Create a `Matrix3` from a column-major slice.
#[inline(always)]
pub fn mat3_from_column_slice(data: &[f32]) -> Mat3Impl {
    assert_eq!(data.len(), 9, "Matrix3 requires exactly 9 elements");
    Mat3Impl::from_column_slice(data)
}

/// Return a zero `Matrix3`.
#[inline(always)]
pub fn mat3_zeros() -> Mat3Impl {
    Mat3Impl::zeros()
}

/// Get `Matrix3` data as a `&[f32]` slice (column-major order).
#[inline(always)]
pub fn mat3_as_slice(m: &Mat3Impl) -> &[f32] {
    m.as_slice()
}

/// Iterate over `Matrix3` elements (column-major order).
#[inline(always)]
pub fn mat3_iter(m: &Mat3Impl) -> impl Iterator<Item = &f32> {
    m.iter()
}

/// Extract a row from a `Matrix3` as `[f32; 3]`.
#[inline(always)]
pub fn mat3_row(m: &Mat3Impl, i: usize) -> [f32; 3] {
    let row = m.row(i);
    [row[0], row[1], row[2]]
}

/// Create a `Matrix4` from a row-major slice.
#[inline(always)]
pub fn mat4_from_row_slice(data: &[f64]) -> Mat4Impl {
    assert_eq!(data.len(), 16, "Matrix4 requires exactly 16 elements");
    Mat4Impl::from_row_slice(data)
}

/// Subtract two `Matrix4` matrices element-wise.
#[inline(always)]
pub fn mat4_sub(a: Mat4Impl, b: Mat4Impl) -> Mat4Impl {
    a - b
}

/// Return a zero `Matrix4`.
#[inline(always)]
pub fn mat4_zeros() -> Mat4Impl {
    Mat4Impl::zeros()
}

/// Iterate over `Matrix4` elements (column-major order).
#[inline(always)]
pub fn mat4_iter(m: &Mat4Impl) -> impl Iterator<Item = &f64> {
    m.iter()
}

/// Return `Matrix4` data as a `&[f64]` slice.
#[inline(always)]
pub fn mat4_as_slice(m: &Mat4Impl) -> &[f64] {
    m.as_slice()
}

/// Return the origin point.
#[inline(always)]
pub fn point3_origin() -> Point3Impl {
    Point3Impl::origin()
}

/// Get point coordinates as a slice.
#[inline(always)]
pub fn point3_as_slice(p: &Point3Impl) -> &[f32] {
    p.coords.as_slice()
}

/// Get point coordinates as a `&[f32; 3]` ref.
#[inline(always)]
pub fn point3_as_ref(p: &Point3Impl) -> &[f32; 3] {
    p.coords.as_ref()
}

/// Return the identity `Matrix3`.
#[inline(always)]
pub fn mat3_identity() -> Mat3Impl {
    Mat3Impl::identity()
}

/// Return the identity `Matrix4`.
#[inline(always)]
pub fn mat4_identity() -> Mat4Impl {
    Mat4Impl::identity()
}

/// Get element at `(row, col)` from a `Matrix3`.
#[inline(always)]
pub fn mat3(m: &Mat3Impl, row: usize, col: usize) -> f32 {
    m[(row, col)]
}

/// Get element at `(row, col)` from a `Matrix4`.
#[inline(always)]
pub fn mat4(m: &Mat4Impl, row: usize, col: usize) -> f64 {
    m[(row, col)]
}

/// Subtract two `Matrix3` matrices element-wise.
#[inline(always)]
pub fn mat3_sub(a: Mat3Impl, b: Mat3Impl) -> Mat3Impl {
    a - b
}

/// Create a diagonal `Matrix3` with `v` on the diagonal.
#[inline(always)]
pub fn mat3_from_diagonal_element(v: f32) -> Mat3Impl {
    Mat3Impl::from_diagonal_element(v)
}
