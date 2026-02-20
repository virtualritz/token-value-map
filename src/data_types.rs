use crate::math::*;
use crate::{DataTypeOps, Error, Result, *};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
};

// Macro to implement DataTypeOps for each variant
macro_rules! impl_data_ops {
    ($type:ty, $name:expr, $data_type:expr) => {
        impl DataTypeOps for $type {
            fn type_name(&self) -> &'static str {
                $name
            }

            fn data_type(&self) -> DataType {
                $data_type
            }
        }
    };
}

// AIDEV-NOTE: Macro to reduce duplication for nalgebra-based types.
// Implements Add, Sub, Mul<f32>, and Mul<f64> for wrapper types.
macro_rules! impl_nalgebra_arithmetic {
    // For f32-based types
    ($type:ty) => {
        impl Add for $type {
            type Output = $type;

            fn add(self, other: $type) -> $type {
                Self(self.0 + other.0)
            }
        }

        impl Sub for $type {
            type Output = $type;

            fn sub(self, other: $type) -> $type {
                Self(self.0 - other.0)
            }
        }

        impl Mul<f32> for $type {
            type Output = $type;

            fn mul(self, scalar: f32) -> $type {
                Self(self.0 * scalar)
            }
        }

        impl Mul<f64> for $type {
            type Output = $type;

            fn mul(self, scalar: f64) -> $type {
                Self(self.0 * scalar as f32)
            }
        }
    };

    // For f64-based types
    ($type:ty, f64) => {
        impl Add for $type {
            type Output = $type;

            fn add(self, other: $type) -> $type {
                Self(self.0 + other.0)
            }
        }

        impl Sub for $type {
            type Output = $type;

            fn sub(self, other: $type) -> $type {
                Self(self.0 - other.0)
            }
        }

        impl Mul<f32> for $type {
            type Output = $type;

            fn mul(self, scalar: f32) -> $type {
                Self(self.0 * scalar as f64)
            }
        }

        impl Mul<f64> for $type {
            type Output = $type;

            fn mul(self, scalar: f64) -> $type {
                Self(self.0 * scalar)
            }
        }
    };
}

/// A boolean value wrapper.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Boolean(pub bool);

impl From<Data> for Boolean {
    fn from(data: Data) -> Self {
        match data {
            Data::Boolean(b) => b,
            Data::Real(r) => Boolean(r.0 != 0.0),
            Data::Integer(i) => Boolean(i.0 != 0),
            Data::String(s) => Boolean(s.0.parse::<bool>().unwrap_or(false)),
            _ => panic!("Cannot convert {data:?} to Boolean"),
        }
    }
}

/// A 64-bit signed integer wrapper.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Integer(pub i64);

impl From<Data> for Integer {
    fn from(data: Data) -> Self {
        match data {
            Data::Boolean(b) => Integer(if b.0 { 1 } else { 0 }),
            Data::Real(r) => Integer(r.0 as i64),
            Data::Integer(i) => i,
            Data::String(s) => Integer(s.0.parse::<i64>().unwrap_or(0)),
            _ => panic!("Cannot convert {data:?} to Integer"),
        }
    }
}

/// A 64-bit floating-point number wrapper.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Real(pub f64);

impl Eq for Real {}

impl From<Data> for Real {
    fn from(data: Data) -> Self {
        match data {
            Data::Boolean(b) => Real(if b.0 { 1.0 } else { 0.0 }),
            Data::Real(r) => r,
            Data::Integer(i) => Real(i.0 as f64),
            Data::String(s) => Real(s.0.parse::<f64>().unwrap_or(0.0)),
            _ => panic!("Cannot convert {data:?} to Real"),
        }
    }
}

impl From<f64> for Real {
    fn from(value: f64) -> Self {
        Real(value)
    }
}

impl From<f32> for Real {
    fn from(value: f32) -> Self {
        Real(value as f64)
    }
}

/// A UTF-8 string wrapper.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct String(pub std::string::String);

impl From<Data> for String {
    fn from(data: Data) -> Self {
        match data {
            Data::Boolean(b) => String(b.0.to_string()),
            Data::Real(r) => String(r.0.to_string()),
            Data::Integer(i) => String(i.0.to_string()),
            Data::String(s) => s,
            Data::Color(c) => String(format!("{c:?}")),
            #[cfg(feature = "vector2")]
            Data::Vector2(v) => String(format!("{v:?}")),
            #[cfg(feature = "vector3")]
            Data::Vector3(v) => String(format!("{v:?}")),
            #[cfg(feature = "matrix3")]
            Data::Matrix3(m) => String(format!("{m:?}")),
            #[cfg(feature = "normal3")]
            Data::Normal3(n) => String(format!("{n:?}")),
            #[cfg(feature = "point3")]
            Data::Point3(p) => String(format!("{p:?}")),
            #[cfg(feature = "matrix4")]
            Data::Matrix4(m) => String(format!("{m:?}")),
            Data::BooleanVec(v) => String(format!("{v:?}")),
            Data::RealVec(v) => String(format!("{v:?}")),
            Data::IntegerVec(v) => String(format!("{v:?}")),
            Data::StringVec(v) => String(format!("{v:?}")),
            Data::ColorVec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(v) => String(format!("{v:?}")),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(v) => String(format!("{v:?}")),
            #[cfg(feature = "curves")]
            Data::RealCurve(c) => String(format!("{c:?}")),
            #[cfg(feature = "curves")]
            Data::ColorCurve(c) => String(format!("{c:?}")),
        }
    }
}

/// A 4-component RGBA color value.
///
/// AIDEV-NOTE: Color uses `[f32; 4]` directly -- it's a domain concept (RGBA),
/// not a math-library type. Both glam and nalgebra lack a canonical "Color".
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
#[repr(transparent)]
pub struct Color(pub [f32; 4]);

impl Eq for Color {}

impl From<[f32; 4]> for Color {
    #[inline(always)]
    fn from(array: [f32; 4]) -> Self {
        Color(array)
    }
}

impl From<Color> for [f32; 4] {
    #[inline(always)]
    fn from(color: Color) -> Self {
        color.0
    }
}

impl Color {
    /// Create a new color from RGBA components.
    #[inline(always)]
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Color([r, g, b, a])
    }

    /// Get RGBA components as array.
    #[inline(always)]
    pub fn to_array(&self) -> [f32; 4] {
        self.0
    }
}

impl From<Data> for Color {
    fn from(data: Data) -> Self {
        match data {
            Data::Boolean(b) => Color::from([b.0.into(), b.0.into(), b.0.into(), 1.0]),
            Data::Real(r) => Color::from([r.0 as _, r.0 as _, r.0 as _, 1.0]),
            Data::Integer(i) => Color::from([i.0 as _, i.0 as _, i.0 as _, 1.0]),
            Data::String(s) => Color::from([s.0.parse::<f32>().unwrap_or(0.0); 4]),
            Data::Color(c) => c,
            #[cfg(feature = "vector2")]
            Data::Vector2(v) => Color::from([v.0.x, v.0.y, 0.0, 1.0]),
            #[cfg(feature = "vector3")]
            Data::Vector3(v) => Color::from([v.0.x, v.0.y, v.0.z, 1.0]),
            Data::BooleanVec(v) => Color::from([v.0[0].into(), v.0[1].into(), v.0[2].into(), 1.0]),
            Data::RealVec(v) => Color::from([v.0[0] as _, v.0[1] as _, v.0[2] as _, 1.0]),
            Data::IntegerVec(v) => Color::from([v.0[0] as _, v.0[1] as _, v.0[2] as _, 1.0]),
            Data::StringVec(v) => Color::from([v.0[0].parse::<f32>().unwrap_or(0.0); 4]),
            Data::ColorVec(v) => Color::from([v.0[0][0], v.0[0][1], v.0[0][2], v.0[0][3]]),
            _ => panic!("Cannot convert {data:?} to Color"),
        }
    }
}

/// A 2D vector.
#[cfg(feature = "vector2")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Vector2(pub Vec2Impl);

#[cfg(feature = "vector2")]
impl Eq for Vector2 {}

/// A 3D vector.
#[cfg(feature = "vector3")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Vector3(pub Vec3Impl);

#[cfg(feature = "vector3")]
impl Eq for Vector3 {}

/// A 3×3 transformation matrix.
#[cfg(feature = "matrix3")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Matrix3(pub Mat3Impl);

#[cfg(feature = "matrix3")]
impl Eq for Matrix3 {}

#[cfg(feature = "matrix3")]
impl From<Vec<f32>> for Matrix3 {
    fn from(vec: Vec<f32>) -> Self {
        assert_eq!(vec.len(), 9, "Matrix3 requires exactly 9 elements");
        Matrix3(mat3_from_row_slice(&vec))
    }
}

#[cfg(feature = "matrix3")]
impl From<Vec<f64>> for Matrix3 {
    fn from(vec: Vec<f64>) -> Self {
        assert_eq!(vec.len(), 9, "Matrix3 requires exactly 9 elements");
        let vec_f32: Vec<f32> = vec.into_iter().map(|v| v as f32).collect();
        Matrix3(mat3_from_row_slice(&vec_f32))
    }
}

#[cfg(feature = "matrix3")]
impl From<[f32; 9]> for Matrix3 {
    fn from(arr: [f32; 9]) -> Self {
        Matrix3(mat3_from_row_slice(&arr))
    }
}

/// A 3D normal vector.
///
/// AIDEV-NOTE: Inner type is `Vec3Impl` for both backends. Normalization
/// is the caller's responsibility -- typically done at construction time.
#[cfg(feature = "normal3")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
#[repr(transparent)]
pub struct Normal3(pub Vec3Impl);

#[cfg(feature = "normal3")]
impl Eq for Normal3 {}

/// A 3D point.
#[cfg(feature = "point3")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Point3(pub Point3Impl);

#[cfg(feature = "point3")]
impl Eq for Point3 {}

/// A 4×4 transformation matrix.
#[cfg(feature = "matrix4")]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Matrix4(pub Mat4Impl);

#[cfg(feature = "matrix4")]
impl Eq for Matrix4 {}

/// A vector of integer values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct IntegerVec(pub Vec<i64>);

impl IntegerVec {
    pub fn new(vec: Vec<i64>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "IntegerVec",
            });
        }
        Ok(IntegerVec(vec))
    }
}

impl From<Vec<i64>> for IntegerVec {
    fn from(vec: Vec<i64>) -> Self {
        IntegerVec(vec)
    }
}

impl From<Vec<i32>> for IntegerVec {
    fn from(vec: Vec<i32>) -> Self {
        IntegerVec(vec.into_iter().map(|v| v as i64).collect())
    }
}

/// A vector of real values.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct RealVec(pub Vec<f64>);

impl RealVec {
    pub fn new(vec: Vec<f64>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "RealVec",
            });
        }
        Ok(RealVec(vec))
    }
}

impl From<Vec<f64>> for RealVec {
    fn from(vec: Vec<f64>) -> Self {
        RealVec(vec)
    }
}

impl From<Vec<f32>> for RealVec {
    fn from(vec: Vec<f32>) -> Self {
        RealVec(vec.into_iter().map(|v| v as f64).collect())
    }
}

impl Eq for RealVec {}

/// A vector of boolean values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct BooleanVec(pub Vec<bool>);

impl BooleanVec {
    pub fn new(vec: Vec<bool>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "BooleanVec",
            });
        }
        Ok(BooleanVec(vec))
    }
}

/// A vector of string values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct StringVec(pub Vec<std::string::String>);

impl StringVec {
    pub fn new(vec: Vec<std::string::String>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "StringVec",
            });
        }
        Ok(StringVec(vec))
    }
}

/// A vector of color values.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct ColorVec(pub Vec<[f32; 4]>);

impl ColorVec {
    pub fn new(vec: Vec<[f32; 4]>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "ColorVec",
            });
        }
        Ok(ColorVec(vec))
    }
}

impl Eq for ColorVec {}

/// A real-valued curve mapping Position → Real.
///
/// Used for falloff curves and other 1D parameter mappings.
#[cfg(feature = "curves")]
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct RealCurve(pub KeyDataMap<Position, Real>);

#[cfg(feature = "curves")]
impl Eq for RealCurve {}

#[cfg(feature = "curves")]
impl RealCurve {
    /// Create a linear ramp from 0 to 1.
    pub fn linear() -> Self {
        RealCurve(KeyDataMap::from(std::collections::BTreeMap::from([
            (Position(0.0), Real(0.0)),
            (Position(1.0), Real(1.0)),
        ])))
    }

    /// Create a constant curve.
    pub fn constant(v: f64) -> Self {
        RealCurve(KeyDataMap::from(std::collections::BTreeMap::from([(
            Position(0.0),
            Real(v),
        )])))
    }

    /// Evaluate the curve at a normalized position.
    pub fn evaluate(&self, position: f32) -> f64 {
        self.0.interpolate(Position(position)).0
    }
}

/// A color-valued curve mapping Position → Color.
///
/// Used for multi-stop color gradients.
#[cfg(feature = "curves")]
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct ColorCurve(pub KeyDataMap<Position, Color>);

#[cfg(feature = "curves")]
impl Eq for ColorCurve {}

#[cfg(feature = "curves")]
impl ColorCurve {
    /// Create a black-to-white gradient.
    pub fn black_to_white() -> Self {
        ColorCurve(KeyDataMap::from(std::collections::BTreeMap::from([
            (Position(0.0), Color([0.0, 0.0, 0.0, 1.0])),
            (Position(1.0), Color([1.0, 1.0, 1.0, 1.0])),
        ])))
    }

    /// Evaluate the curve at a normalized position.
    pub fn evaluate(&self, position: f32) -> [f32; 4] {
        self.0.interpolate(Position(position)).0
    }
}

/// A vector of 2D vectors.
#[cfg(all(feature = "vector2", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Vector2Vec(pub Vec<Vec2Impl>);

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Vector2Vec {
    pub fn new(vec: Vec<Vec2Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Vector2Vec",
            });
        }
        Ok(Vector2Vec(vec))
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Eq for Vector2Vec {}

/// A vector of 3D vectors.
#[cfg(all(feature = "vector3", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Vector3Vec(pub Vec<Vec3Impl>);

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Vector3Vec {
    pub fn new(vec: Vec<Vec3Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Vector3Vec",
            });
        }
        Ok(Vector3Vec(vec))
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Eq for Vector3Vec {}

/// A vector of transformation matrices.
#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Matrix3Vec(pub Vec<Mat3Impl>);

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Matrix3Vec {
    pub fn new(vec: Vec<Mat3Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Matrix3Vec",
            });
        }
        Ok(Matrix3Vec(vec))
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Eq for Matrix3Vec {}

/// A vector of 3D normals.
#[cfg(all(feature = "normal3", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Normal3Vec(pub Vec<Vec3Impl>);

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Normal3Vec {
    pub fn new(vec: Vec<Vec3Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Normal3Vec",
            });
        }
        Ok(Normal3Vec(vec))
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Eq for Normal3Vec {}

/// A vector of 3D points.
#[cfg(all(feature = "point3", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Point3Vec(pub Vec<Point3Impl>);

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Point3Vec {
    pub fn new(vec: Vec<Point3Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Point3Vec",
            });
        }
        Ok(Point3Vec(vec))
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Eq for Point3Vec {}

/// A vector of 4×4 transformation matrices.
#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Matrix4Vec(pub Vec<Mat4Impl>);

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Matrix4Vec {
    pub fn new(vec: Vec<Mat4Impl>) -> Result<Self> {
        if vec.is_empty() {
            return Err(Error::EmptyVec {
                type_name: "Matrix4Vec",
            });
        }
        Ok(Matrix4Vec(vec))
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Eq for Matrix4Vec {}

// Arithmetic operations for new types
#[cfg(feature = "normal3")]
impl_nalgebra_arithmetic!(Normal3);

#[cfg(feature = "point3")]
impl Add for Point3 {
    type Output = Point3;

    fn add(self, other: Point3) -> Point3 {
        Point3(Point3Impl::new(
            self.0.x + other.0.x,
            self.0.y + other.0.y,
            self.0.z + other.0.z,
        ))
    }
}

#[cfg(feature = "point3")]
impl Sub for Point3 {
    type Output = Point3;

    fn sub(self, other: Point3) -> Point3 {
        Point3(Point3Impl::new(
            self.0.x - other.0.x,
            self.0.y - other.0.y,
            self.0.z - other.0.z,
        ))
    }
}

#[cfg(feature = "point3")]
impl Mul<f32> for Point3 {
    type Output = Point3;

    fn mul(self, scalar: f32) -> Point3 {
        Point3(Point3Impl::new(
            self.0.x * scalar,
            self.0.y * scalar,
            self.0.z * scalar,
        ))
    }
}

#[cfg(feature = "point3")]
impl Mul<f64> for Point3 {
    type Output = Point3;

    fn mul(self, scalar: f64) -> Point3 {
        let s = scalar as f32;
        Point3(Point3Impl::new(self.0.x * s, self.0.y * s, self.0.z * s))
    }
}

// Matrix4 arithmetic operations.
// AIDEV-NOTE: Manual impls using math utility functions because ultraviolet's
// DMat4 lacks Sub.
#[cfg(feature = "matrix4")]
impl Add for Matrix4 {
    type Output = Matrix4;
    fn add(self, other: Matrix4) -> Matrix4 {
        Matrix4(self.0 + other.0)
    }
}

#[cfg(feature = "matrix4")]
impl Sub for Matrix4 {
    type Output = Matrix4;
    fn sub(self, other: Matrix4) -> Matrix4 {
        Matrix4(mat4_sub(self.0, other.0))
    }
}

#[cfg(feature = "matrix4")]
impl Mul<f32> for Matrix4 {
    type Output = Matrix4;
    fn mul(self, scalar: f32) -> Matrix4 {
        Matrix4(self.0 * scalar as f64)
    }
}

#[cfg(feature = "matrix4")]
impl Mul<f64> for Matrix4 {
    type Output = Matrix4;
    fn mul(self, scalar: f64) -> Matrix4 {
        Matrix4(self.0 * scalar)
    }
}

// Arithmetic trait implementations for interpolation
impl Add for Real {
    type Output = Real;

    fn add(self, other: Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl Sub for Real {
    type Output = Real;

    fn sub(self, other: Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl Mul<f32> for Real {
    type Output = Real;

    fn mul(self, scalar: f32) -> Real {
        Real(self.0 * scalar as f64)
    }
}

impl Mul<f64> for Real {
    type Output = Real;

    fn mul(self, scalar: f64) -> Real {
        Real(self.0 * scalar)
    }
}

impl Add for Integer {
    type Output = Integer;

    fn add(self, other: Integer) -> Integer {
        Integer(self.0 + other.0)
    }
}

impl Sub for Integer {
    type Output = Integer;

    fn sub(self, other: Integer) -> Integer {
        Integer(self.0 - other.0)
    }
}

impl Mul<f32> for Integer {
    type Output = Integer;

    fn mul(self, scalar: f32) -> Integer {
        Integer((self.0 as f64 * scalar as f64) as i64)
    }
}

impl Mul<f64> for Integer {
    type Output = Integer;

    fn mul(self, scalar: f64) -> Integer {
        Integer((self.0 as f64 * scalar) as i64)
    }
}

// Boolean arithmetic operations (treat as 0.0 and 1.0)
impl Add for Boolean {
    type Output = Boolean;

    fn add(self, other: Boolean) -> Boolean {
        Boolean(self.0 || other.0)
    }
}

impl Sub for Boolean {
    type Output = Boolean;

    fn sub(self, other: Boolean) -> Boolean {
        Boolean(self.0 && !other.0)
    }
}

impl Mul<f32> for Boolean {
    type Output = Boolean;

    fn mul(self, _scalar: f32) -> Boolean {
        self
    }
}

impl Mul<f64> for Boolean {
    type Output = Boolean;

    fn mul(self, _scalar: f64) -> Boolean {
        self
    }
}

// String arithmetic operations (concatenation for add, identity for others)
impl Add for String {
    type Output = String;

    fn add(self, _other: String) -> String {
        self
    }
}

impl Sub for String {
    type Output = String;

    fn sub(self, _other: String) -> String {
        self
    }
}

impl Mul<f32> for String {
    type Output = String;

    fn mul(self, _scalar: f32) -> String {
        self
    }
}

impl Mul<f64> for String {
    type Output = String;

    fn mul(self, _scalar: f64) -> String {
        self
    }
}

// BooleanVec arithmetic operations
impl Add for BooleanVec {
    type Output = BooleanVec;

    fn add(self, other: BooleanVec) -> BooleanVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        BooleanVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a || b)
                .collect(),
        )
    }
}

impl Sub for BooleanVec {
    type Output = BooleanVec;

    fn sub(self, other: BooleanVec) -> BooleanVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        BooleanVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a && !b)
                .collect(),
        )
    }
}

impl Mul<f32> for BooleanVec {
    type Output = BooleanVec;

    fn mul(self, _scalar: f32) -> BooleanVec {
        self
    }
}

impl Mul<f64> for BooleanVec {
    type Output = BooleanVec;

    fn mul(self, _scalar: f64) -> BooleanVec {
        self
    }
}

// StringVec arithmetic operations
impl Add for StringVec {
    type Output = StringVec;

    fn add(self, other: StringVec) -> StringVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        StringVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| format!("{a}{b}"))
                .collect(),
        )
    }
}

impl Sub for StringVec {
    type Output = StringVec;

    fn sub(self, _other: StringVec) -> StringVec {
        self
    }
}

impl Mul<f32> for StringVec {
    type Output = StringVec;

    fn mul(self, _scalar: f32) -> StringVec {
        self
    }
}

impl Mul<f64> for StringVec {
    type Output = StringVec;

    fn mul(self, _scalar: f64) -> StringVec {
        self
    }
}

// Color arithmetic operations on [f32; 4].
impl Add for Color {
    type Output = Color;

    #[inline(always)]
    fn add(self, other: Color) -> Color {
        Color([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
            self.0[3] + other.0[3],
        ])
    }
}

impl Sub for Color {
    type Output = Color;

    #[inline(always)]
    fn sub(self, other: Color) -> Color {
        Color([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
            self.0[3] - other.0[3],
        ])
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    #[inline(always)]
    fn mul(self, s: f32) -> Color {
        Color([self.0[0] * s, self.0[1] * s, self.0[2] * s, self.0[3] * s])
    }
}

impl Div<f32> for Color {
    type Output = Color;

    #[inline(always)]
    fn div(self, s: f32) -> Color {
        Color([self.0[0] / s, self.0[1] / s, self.0[2] / s, self.0[3] / s])
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    #[inline(always)]
    fn mul(self, scalar: f64) -> Color {
        self * (scalar as f32)
    }
}

impl Div<f64> for Color {
    type Output = Color;

    #[inline(always)]
    fn div(self, scalar: f64) -> Color {
        self / (scalar as f32)
    }
}

// Vector2 arithmetic operations
#[cfg(feature = "vector2")]
impl_nalgebra_arithmetic!(Vector2);

// Vector3 arithmetic operations
#[cfg(feature = "vector3")]
impl_nalgebra_arithmetic!(Vector3);

// Matrix3 arithmetic operations.
// AIDEV-NOTE: Manual impls using math utility functions because ultraviolet's
// Mat3 lacks Sub and Div<f32>.
#[cfg(feature = "matrix3")]
impl Add for Matrix3 {
    type Output = Matrix3;
    fn add(self, other: Matrix3) -> Matrix3 {
        Matrix3(self.0 + other.0)
    }
}

#[cfg(feature = "matrix3")]
impl Sub for Matrix3 {
    type Output = Matrix3;
    fn sub(self, other: Matrix3) -> Matrix3 {
        Matrix3(mat3_sub(self.0, other.0))
    }
}

#[cfg(feature = "matrix3")]
impl Mul<f32> for Matrix3 {
    type Output = Matrix3;
    fn mul(self, scalar: f32) -> Matrix3 {
        Matrix3(self.0 * scalar)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<f64> for Matrix3 {
    type Output = Matrix3;
    fn mul(self, scalar: f64) -> Matrix3 {
        Matrix3(self.0 * scalar as f32)
    }
}

// Matrix3 multiplication (matrix * matrix)
#[cfg(feature = "matrix3")]
impl Mul for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: Matrix3) -> Matrix3 {
        Matrix3(self.0 * other.0)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<&Matrix3> for Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &Matrix3) -> Matrix3 {
        Matrix3(self.0 * other.0)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<Matrix3> for &Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: Matrix3) -> Matrix3 {
        Matrix3(self.0 * other.0)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<&Matrix3> for &Matrix3 {
    type Output = Matrix3;

    fn mul(self, other: &Matrix3) -> Matrix3 {
        Matrix3(self.0 * other.0)
    }
}

// Vector types arithmetic operations
impl Add for RealVec {
    type Output = RealVec;

    fn add(self, other: RealVec) -> RealVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        RealVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl Sub for RealVec {
    type Output = RealVec;

    fn sub(self, other: RealVec) -> RealVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        RealVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

impl Mul<f32> for RealVec {
    type Output = RealVec;

    fn mul(self, scalar: f32) -> RealVec {
        RealVec(self.0.into_iter().map(|x| x * scalar as f64).collect())
    }
}

impl Mul<f64> for RealVec {
    type Output = RealVec;

    fn mul(self, scalar: f64) -> RealVec {
        RealVec(self.0.into_iter().map(|x| x * scalar).collect())
    }
}

impl Add for IntegerVec {
    type Output = IntegerVec;

    fn add(self, other: IntegerVec) -> IntegerVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        IntegerVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl Sub for IntegerVec {
    type Output = IntegerVec;

    fn sub(self, other: IntegerVec) -> IntegerVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        IntegerVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

impl Mul<f32> for IntegerVec {
    type Output = IntegerVec;

    fn mul(self, scalar: f32) -> IntegerVec {
        IntegerVec(
            self.0
                .into_iter()
                .map(|x| (x as f64 * scalar as f64) as i64)
                .collect(),
        )
    }
}

impl Mul<f64> for IntegerVec {
    type Output = IntegerVec;

    fn mul(self, scalar: f64) -> IntegerVec {
        IntegerVec(
            self.0
                .into_iter()
                .map(|x| (x as f64 * scalar) as i64)
                .collect(),
        )
    }
}

impl Add for ColorVec {
    type Output = ColorVec;

    fn add(self, other: ColorVec) -> ColorVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        ColorVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
                .collect(),
        )
    }
}

impl Sub for ColorVec {
    type Output = ColorVec;

    fn sub(self, other: ColorVec) -> ColorVec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        ColorVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
                .collect(),
        )
    }
}

impl Mul<f32> for ColorVec {
    type Output = ColorVec;

    fn mul(self, scalar: f32) -> ColorVec {
        ColorVec(
            self.0
                .into_iter()
                .map(|color| {
                    [
                        color[0] * scalar,
                        color[1] * scalar,
                        color[2] * scalar,
                        color[3] * scalar,
                    ]
                })
                .collect(),
        )
    }
}

impl Mul<f64> for ColorVec {
    type Output = ColorVec;

    fn mul(self, scalar: f64) -> ColorVec {
        let scalar = scalar as f32;
        ColorVec(
            self.0
                .into_iter()
                .map(|color| {
                    [
                        color[0] * scalar,
                        color[1] * scalar,
                        color[2] * scalar,
                        color[3] * scalar,
                    ]
                })
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Add for Vector2Vec {
    type Output = Vector2Vec;

    fn add(self, other: Vector2Vec) -> Vector2Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        Vector2Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Sub for Vector2Vec {
    type Output = Vector2Vec;

    fn sub(self, other: Vector2Vec) -> Vector2Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        Vector2Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Mul<f32> for Vector2Vec {
    type Output = Vector2Vec;

    fn mul(self, scalar: f32) -> Vector2Vec {
        Vector2Vec(self.0.into_iter().map(|vec| vec * scalar).collect())
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Mul<f64> for Vector2Vec {
    type Output = Vector2Vec;

    fn mul(self, scalar: f64) -> Vector2Vec {
        let scalar = scalar as f32;
        Vector2Vec(self.0.into_iter().map(|vec| vec * scalar).collect())
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Add for Vector3Vec {
    type Output = Vector3Vec;

    fn add(self, other: Vector3Vec) -> Vector3Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        Vector3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Sub for Vector3Vec {
    type Output = Vector3Vec;

    fn sub(self, other: Vector3Vec) -> Vector3Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        Vector3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Mul<f32> for Vector3Vec {
    type Output = Vector3Vec;

    fn mul(self, scalar: f32) -> Vector3Vec {
        Vector3Vec(self.0.into_iter().map(|vec| vec * scalar).collect())
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Mul<f64> for Vector3Vec {
    type Output = Vector3Vec;

    fn mul(self, scalar: f64) -> Vector3Vec {
        let scalar = scalar as f32;
        Vector3Vec(self.0.into_iter().map(|vec| vec * scalar).collect())
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Add for Matrix3Vec {
    type Output = Matrix3Vec;

    fn add(mut self, other: Matrix3Vec) -> Matrix3Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for addition");
        }
        self.0.iter_mut().zip(other.0).for_each(|(a, b)| *a += b);
        self
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Sub for Matrix3Vec {
    type Output = Matrix3Vec;

    fn sub(mut self, other: Matrix3Vec) -> Matrix3Vec {
        if self.0.len() != other.0.len() {
            panic!("Vector lengths must match for subtraction");
        }
        self.0
            .iter_mut()
            .zip(other.0)
            .for_each(|(a, b)| *a = mat3_sub(*a, b));
        self
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Mul<f32> for Matrix3Vec {
    type Output = Matrix3Vec;

    #[allow(clippy::assign_op_pattern)]
    fn mul(mut self, scalar: f32) -> Matrix3Vec {
        // AIDEV-NOTE: no MulAssign on ultraviolet Mat3.
        self.0.iter_mut().for_each(|mat| *mat = *mat * scalar);
        self
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Mul<f64> for Matrix3Vec {
    type Output = Matrix3Vec;

    #[allow(clippy::assign_op_pattern)]
    fn mul(mut self, scalar: f64) -> Matrix3Vec {
        let scalar = scalar as f32;
        self.0.iter_mut().for_each(|mat| *mat = *mat * scalar);
        self
    }
}

// Normal3Vec arithmetic operations
#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Add for Normal3Vec {
    type Output = Normal3Vec;

    fn add(self, other: Normal3Vec) -> Self::Output {
        Normal3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Sub for Normal3Vec {
    type Output = Normal3Vec;

    fn sub(self, other: Normal3Vec) -> Self::Output {
        Normal3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Mul<f32> for Normal3Vec {
    type Output = Normal3Vec;

    fn mul(self, scalar: f32) -> Self::Output {
        Normal3Vec(self.0.into_iter().map(|v| v * scalar).collect())
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Mul<f64> for Normal3Vec {
    type Output = Normal3Vec;

    fn mul(self, scalar: f64) -> Self::Output {
        Normal3Vec(self.0.into_iter().map(|v| v * scalar as f32).collect())
    }
}

// Point3Vec arithmetic operations.
#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Add for Point3Vec {
    type Output = Point3Vec;

    fn add(self, other: Point3Vec) -> Self::Output {
        Point3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| Point3Impl::new(a.x + b.x, a.y + b.y, a.z + b.z))
                .collect(),
        )
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Sub for Point3Vec {
    type Output = Point3Vec;

    fn sub(self, other: Point3Vec) -> Self::Output {
        Point3Vec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| Point3Impl::new(a.x - b.x, a.y - b.y, a.z - b.z))
                .collect(),
        )
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Mul<f32> for Point3Vec {
    type Output = Point3Vec;

    fn mul(self, s: f32) -> Self::Output {
        Point3Vec(
            self.0
                .into_iter()
                .map(|p| Point3Impl::new(p.x * s, p.y * s, p.z * s))
                .collect(),
        )
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Mul<f64> for Point3Vec {
    type Output = Point3Vec;

    fn mul(self, scalar: f64) -> Self::Output {
        let s = scalar as f32;
        Point3Vec(
            self.0
                .into_iter()
                .map(|p| Point3Impl::new(p.x * s, p.y * s, p.z * s))
                .collect(),
        )
    }
}

// Matrix4Vec arithmetic operations
#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Add for Matrix4Vec {
    type Output = Matrix4Vec;

    fn add(mut self, other: Matrix4Vec) -> Self::Output {
        self.0.iter_mut().zip(other.0).for_each(|(a, b)| *a += b);
        self
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Sub for Matrix4Vec {
    type Output = Matrix4Vec;

    fn sub(mut self, other: Matrix4Vec) -> Self::Output {
        self.0
            .iter_mut()
            .zip(other.0)
            .for_each(|(a, b)| *a = mat4_sub(*a, b));
        self
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Mul<f32> for Matrix4Vec {
    type Output = Matrix4Vec;

    #[allow(clippy::assign_op_pattern)]
    fn mul(mut self, scalar: f32) -> Self::Output {
        // AIDEV-NOTE: no MulAssign on ultraviolet DMat4.
        let scalar = scalar as f64;
        self.0.iter_mut().for_each(|v| *v = *v * scalar);
        self
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Mul<f64> for Matrix4Vec {
    type Output = Matrix4Vec;

    #[allow(clippy::assign_op_pattern)]
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.0.iter_mut().for_each(|v| *v = *v * scalar);
        self
    }
}

// Division implementations for f32
#[cfg(feature = "normal3")]
impl Div<f32> for Normal3 {
    type Output = Normal3;

    fn div(self, scalar: f32) -> Normal3 {
        Normal3(self.0 / scalar)
    }
}

#[cfg(feature = "point3")]
impl Div<f32> for Point3 {
    type Output = Point3;

    fn div(self, s: f32) -> Point3 {
        Point3(Point3Impl::new(self.0.x / s, self.0.y / s, self.0.z / s))
    }
}

#[cfg(feature = "matrix4")]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for Matrix4 {
    type Output = Matrix4;

    fn div(self, scalar: f32) -> Matrix4 {
        // AIDEV-NOTE: ultraviolet DMat4 lacks Div<f64>.
        #[cfg(not(feature = "ultraviolet"))]
        {
            Matrix4(self.0 / scalar as f64)
        }
        #[cfg(feature = "ultraviolet")]
        {
            self * (1.0 / scalar as f64)
        }
    }
}

impl Div<f32> for Real {
    type Output = Real;

    fn div(self, scalar: f32) -> Real {
        Real(self.0 / scalar as f64)
    }
}

impl Div<f32> for Integer {
    type Output = Integer;

    fn div(self, scalar: f32) -> Integer {
        Integer((self.0 as f64 / scalar as f64) as i64)
    }
}

impl Div<f32> for Boolean {
    type Output = Boolean;

    fn div(self, _scalar: f32) -> Boolean {
        self
    }
}

impl Div<f32> for String {
    type Output = String;

    fn div(self, _scalar: f32) -> String {
        self
    }
}

impl Div<f32> for BooleanVec {
    type Output = BooleanVec;

    fn div(self, _scalar: f32) -> BooleanVec {
        self
    }
}

impl Div<f32> for StringVec {
    type Output = StringVec;

    fn div(self, _scalar: f32) -> StringVec {
        self
    }
}

#[cfg(feature = "vector2")]
impl Div<f32> for Vector2 {
    type Output = Vector2;

    fn div(self, scalar: f32) -> Vector2 {
        Vector2(self.0 / scalar)
    }
}

#[cfg(feature = "vector3")]
impl Div<f32> for Vector3 {
    type Output = Vector3;

    fn div(self, scalar: f32) -> Vector3 {
        Vector3(self.0 / scalar)
    }
}

#[cfg(feature = "matrix3")]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for Matrix3 {
    type Output = Matrix3;

    fn div(self, scalar: f32) -> Matrix3 {
        // AIDEV-NOTE: ultraviolet Mat3 lacks Div<f32>.
        #[cfg(not(feature = "ultraviolet"))]
        {
            Matrix3(self.0 / scalar)
        }
        #[cfg(feature = "ultraviolet")]
        {
            self * (1.0 / scalar)
        }
    }
}

impl Div<f32> for RealVec {
    type Output = RealVec;

    fn div(self, scalar: f32) -> RealVec {
        RealVec(self.0.into_iter().map(|x| x / scalar as f64).collect())
    }
}

impl Div<f32> for IntegerVec {
    type Output = IntegerVec;

    fn div(self, scalar: f32) -> IntegerVec {
        IntegerVec(
            self.0
                .into_iter()
                .map(|x| (x as f64 / scalar as f64) as i64)
                .collect(),
        )
    }
}

impl Div<f32> for ColorVec {
    type Output = ColorVec;

    fn div(self, scalar: f32) -> ColorVec {
        ColorVec(
            self.0
                .into_iter()
                .map(|color| {
                    [
                        color[0] / scalar,
                        color[1] / scalar,
                        color[2] / scalar,
                        color[3] / scalar,
                    ]
                })
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Div<f32> for Vector2Vec {
    type Output = Vector2Vec;

    fn div(self, scalar: f32) -> Vector2Vec {
        Vector2Vec(self.0.into_iter().map(|vec| vec / scalar).collect())
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Div<f32> for Vector3Vec {
    type Output = Vector3Vec;

    fn div(self, scalar: f32) -> Vector3Vec {
        Vector3Vec(self.0.into_iter().map(|vec| vec / scalar).collect())
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for Matrix3Vec {
    type Output = Matrix3Vec;

    fn div(mut self, scalar: f32) -> Matrix3Vec {
        #[cfg(not(feature = "ultraviolet"))]
        self.0.iter_mut().for_each(|mat| *mat /= scalar);
        #[cfg(feature = "ultraviolet")]
        {
            let recip = 1.0 / scalar;
            self.0.iter_mut().for_each(|mat| *mat = *mat * recip);
        }
        self
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Div<f32> for Normal3Vec {
    type Output = Normal3Vec;

    fn div(self, scalar: f32) -> Self::Output {
        Normal3Vec(self.0.into_iter().map(|v| v / scalar).collect())
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Div<f32> for Point3Vec {
    type Output = Point3Vec;

    fn div(self, s: f32) -> Self::Output {
        Point3Vec(
            self.0
                .into_iter()
                .map(|p| Point3Impl::new(p.x / s, p.y / s, p.z / s))
                .collect(),
        )
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for Matrix4Vec {
    type Output = Matrix4Vec;

    fn div(mut self, scalar: f32) -> Self::Output {
        let scalar = scalar as f64;
        #[cfg(not(feature = "ultraviolet"))]
        self.0.iter_mut().for_each(|v| *v /= scalar);
        #[cfg(feature = "ultraviolet")]
        {
            let recip = 1.0 / scalar;
            self.0.iter_mut().for_each(|v| *v = *v * recip);
        }
        self
    }
}

// Division implementations for f64
#[cfg(feature = "normal3")]
impl Div<f64> for Normal3 {
    type Output = Normal3;

    fn div(self, scalar: f64) -> Normal3 {
        Normal3(self.0 / scalar as f32)
    }
}

#[cfg(feature = "point3")]
impl Div<f64> for Point3 {
    type Output = Point3;

    fn div(self, scalar: f64) -> Point3 {
        let s = scalar as f32;
        Point3(Point3Impl::new(self.0.x / s, self.0.y / s, self.0.z / s))
    }
}

#[cfg(feature = "matrix4")]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f64> for Matrix4 {
    type Output = Matrix4;

    fn div(self, scalar: f64) -> Matrix4 {
        #[cfg(not(feature = "ultraviolet"))]
        {
            Matrix4(self.0 / scalar)
        }
        #[cfg(feature = "ultraviolet")]
        {
            self * (1.0 / scalar)
        }
    }
}

impl Div<f64> for Real {
    type Output = Real;

    fn div(self, scalar: f64) -> Real {
        Real(self.0 / scalar)
    }
}

impl Div<f64> for Integer {
    type Output = Integer;

    fn div(self, scalar: f64) -> Integer {
        Integer((self.0 as f64 / scalar) as i64)
    }
}

impl Div<f64> for Boolean {
    type Output = Boolean;

    fn div(self, _scalar: f64) -> Boolean {
        self
    }
}

impl Div<f64> for String {
    type Output = String;

    fn div(self, _scalar: f64) -> String {
        self
    }
}

impl Div<f64> for BooleanVec {
    type Output = BooleanVec;

    fn div(self, _scalar: f64) -> BooleanVec {
        self
    }
}

impl Div<f64> for StringVec {
    type Output = StringVec;

    fn div(self, _scalar: f64) -> StringVec {
        self
    }
}

#[cfg(feature = "vector2")]
impl Div<f64> for Vector2 {
    type Output = Vector2;

    fn div(self, scalar: f64) -> Vector2 {
        Vector2(self.0 / scalar as f32)
    }
}

#[cfg(feature = "vector3")]
impl Div<f64> for Vector3 {
    type Output = Vector3;

    fn div(self, scalar: f64) -> Vector3 {
        Vector3(self.0 / scalar as f32)
    }
}

#[cfg(feature = "matrix3")]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f64> for Matrix3 {
    type Output = Matrix3;

    fn div(self, scalar: f64) -> Matrix3 {
        #[cfg(not(feature = "ultraviolet"))]
        {
            Matrix3(self.0 / scalar as f32)
        }
        #[cfg(feature = "ultraviolet")]
        {
            self * (1.0 / scalar as f32)
        }
    }
}

impl Div<f64> for RealVec {
    type Output = RealVec;

    fn div(self, scalar: f64) -> RealVec {
        RealVec(self.0.into_iter().map(|x| x / scalar).collect())
    }
}

impl Div<f64> for IntegerVec {
    type Output = IntegerVec;

    fn div(self, scalar: f64) -> IntegerVec {
        IntegerVec(
            self.0
                .into_iter()
                .map(|x| (x as f64 / scalar) as i64)
                .collect(),
        )
    }
}

impl Div<f64> for ColorVec {
    type Output = ColorVec;

    fn div(self, scalar: f64) -> ColorVec {
        let scalar = scalar as f32;
        ColorVec(
            self.0
                .into_iter()
                .map(|color| {
                    [
                        color[0] / scalar,
                        color[1] / scalar,
                        color[2] / scalar,
                        color[3] / scalar,
                    ]
                })
                .collect(),
        )
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Div<f64> for Vector2Vec {
    type Output = Vector2Vec;

    fn div(self, scalar: f64) -> Vector2Vec {
        let scalar = scalar as f32;
        Vector2Vec(self.0.into_iter().map(|vec| vec / scalar).collect())
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Div<f64> for Vector3Vec {
    type Output = Vector3Vec;

    fn div(self, scalar: f64) -> Vector3Vec {
        let scalar = scalar as f32;
        Vector3Vec(self.0.into_iter().map(|vec| vec / scalar).collect())
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f64> for Matrix3Vec {
    type Output = Matrix3Vec;

    fn div(mut self, scalar: f64) -> Matrix3Vec {
        let scalar = scalar as f32;
        #[cfg(not(feature = "ultraviolet"))]
        self.0.iter_mut().for_each(|mat| *mat /= scalar);
        #[cfg(feature = "ultraviolet")]
        {
            let recip = 1.0 / scalar;
            self.0.iter_mut().for_each(|mat| *mat = *mat * recip);
        }
        self
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Div<f64> for Normal3Vec {
    type Output = Normal3Vec;

    fn div(self, scalar: f64) -> Self::Output {
        Normal3Vec(self.0.into_iter().map(|v| v / scalar as f32).collect())
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Div<f64> for Point3Vec {
    type Output = Point3Vec;

    fn div(self, scalar: f64) -> Self::Output {
        let s = scalar as f32;
        Point3Vec(
            self.0
                .into_iter()
                .map(|p| Point3Impl::new(p.x / s, p.y / s, p.z / s))
                .collect(),
        )
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f64> for Matrix4Vec {
    type Output = Matrix4Vec;

    fn div(mut self, scalar: f64) -> Self::Output {
        #[cfg(not(feature = "ultraviolet"))]
        self.0.iter_mut().for_each(|v| *v /= scalar);
        #[cfg(feature = "ultraviolet")]
        {
            let recip = 1.0 / scalar;
            self.0.iter_mut().for_each(|v| *v = *v * recip);
        }
        self
    }
}

// Hash implementations for floating point types
// Using bit representation for deterministic hashing

impl Hash for Real {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize -0.0 to 0.0 for consistent hashing
        // Check if the value is zero (either -0.0 or 0.0) and normalize to 0.0
        let normalized = if self.0 == 0.0 { 0.0_f64 } else { self.0 };
        normalized.to_bits().hash(state);
    }
}

impl Hash for Color {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &component in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing.
            let normalized = if component == 0.0 { 0.0_f32 } else { component };
            normalized.to_bits().hash(state);
        }
    }
}

#[cfg(feature = "vector2")]
impl Hash for Vector2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize -0.0 to 0.0 for consistent hashing
        let x = if self.0.x == 0.0 { 0.0_f32 } else { self.0.x };
        let y = if self.0.y == 0.0 { 0.0_f32 } else { self.0.y };
        x.to_bits().hash(state);
        y.to_bits().hash(state);
    }
}

#[cfg(feature = "vector3")]
impl Hash for Vector3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize -0.0 to 0.0 for consistent hashing
        let x = if self.0.x == 0.0 { 0.0_f32 } else { self.0.x };
        let y = if self.0.y == 0.0 { 0.0_f32 } else { self.0.y };
        let z = if self.0.z == 0.0 { 0.0_f32 } else { self.0.z };
        x.to_bits().hash(state);
        y.to_bits().hash(state);
        z.to_bits().hash(state);
    }
}

#[cfg(feature = "matrix3")]
impl Hash for Matrix3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &element in mat3_iter(&self.0) {
            // Normalize -0.0 to 0.0 for consistent hashing.
            let normalized = if element == 0.0 { 0.0_f32 } else { element };
            normalized.to_bits().hash(state);
        }
    }
}

impl Hash for RealVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for &element in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing
            let normalized = if element == 0.0 { 0.0_f64 } else { element };
            normalized.to_bits().hash(state);
        }
    }
}

impl Hash for ColorVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for color in &self.0 {
            for &component in color {
                // Normalize -0.0 to 0.0 for consistent hashing
                let normalized = if component == 0.0 { 0.0_f32 } else { component };
                normalized.to_bits().hash(state);
            }
        }
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl Hash for Vector2Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for vector in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing
            let x = if vector.x == 0.0 { 0.0_f32 } else { vector.x };
            let y = if vector.y == 0.0 { 0.0_f32 } else { vector.y };
            x.to_bits().hash(state);
            y.to_bits().hash(state);
        }
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl Hash for Vector3Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for vector in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing
            let x = if vector.x == 0.0 { 0.0_f32 } else { vector.x };
            let y = if vector.y == 0.0 { 0.0_f32 } else { vector.y };
            let z = if vector.z == 0.0 { 0.0_f32 } else { vector.z };
            x.to_bits().hash(state);
            y.to_bits().hash(state);
            z.to_bits().hash(state);
        }
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl Hash for Matrix3Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for matrix in &self.0 {
            for &element in mat3_iter(matrix) {
                // Normalize -0.0 to 0.0 for consistent hashing.
                let normalized = if element == 0.0 { 0.0_f32 } else { element };
                normalized.to_bits().hash(state);
            }
        }
    }
}

// Hash implementations for new types
#[cfg(feature = "normal3")]
impl Hash for Normal3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize -0.0 to 0.0 for consistent hashing
        let x = if self.0.x == 0.0 { 0.0_f32 } else { self.0.x };
        let y = if self.0.y == 0.0 { 0.0_f32 } else { self.0.y };
        let z = if self.0.z == 0.0 { 0.0_f32 } else { self.0.z };
        x.to_bits().hash(state);
        y.to_bits().hash(state);
        z.to_bits().hash(state);
    }
}

#[cfg(feature = "point3")]
impl Hash for Point3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize -0.0 to 0.0 for consistent hashing
        let x = if self.0.x == 0.0 { 0.0_f32 } else { self.0.x };
        let y = if self.0.y == 0.0 { 0.0_f32 } else { self.0.y };
        let z = if self.0.z == 0.0 { 0.0_f32 } else { self.0.z };
        x.to_bits().hash(state);
        y.to_bits().hash(state);
        z.to_bits().hash(state);
    }
}

#[cfg(feature = "matrix4")]
impl Hash for Matrix4 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &element in mat4_iter(&self.0) {
            // Normalize -0.0 to 0.0 for consistent hashing.
            let normalized = if element == 0.0 { 0.0_f64 } else { element };
            normalized.to_bits().hash(state);
        }
    }
}

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl Hash for Normal3Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for vector in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing
            let x = if vector.x == 0.0 { 0.0_f32 } else { vector.x };
            let y = if vector.y == 0.0 { 0.0_f32 } else { vector.y };
            let z = if vector.z == 0.0 { 0.0_f32 } else { vector.z };
            x.to_bits().hash(state);
            y.to_bits().hash(state);
            z.to_bits().hash(state);
        }
    }
}

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl Hash for Point3Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for point in &self.0 {
            // Normalize -0.0 to 0.0 for consistent hashing
            let x = if point.x == 0.0 { 0.0_f32 } else { point.x };
            let y = if point.y == 0.0 { 0.0_f32 } else { point.y };
            let z = if point.z == 0.0 { 0.0_f32 } else { point.z };
            x.to_bits().hash(state);
            y.to_bits().hash(state);
            z.to_bits().hash(state);
        }
    }
}

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl Hash for Matrix4Vec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for matrix in &self.0 {
            for &element in mat4_iter(matrix) {
                // Normalize -0.0 to 0.0 for consistent hashing.
                let normalized = if element == 0.0 { 0.0_f64 } else { element };
                normalized.to_bits().hash(state);
            }
        }
    }
}

// Implement DataTypeOps for all types
impl_data_ops!(Integer, "integer", DataType::Integer);
impl_data_ops!(Real, "real", DataType::Real);
impl_data_ops!(Boolean, "boolean", DataType::Boolean);
impl_data_ops!(String, "string", DataType::String);
impl_data_ops!(Color, "color", DataType::Color);
#[cfg(feature = "vector2")]
impl_data_ops!(Vector2, "vec2", DataType::Vector2);
#[cfg(feature = "vector3")]
impl_data_ops!(Vector3, "vec3", DataType::Vector3);
#[cfg(feature = "matrix3")]
impl_data_ops!(Matrix3, "mat3", DataType::Matrix3);

impl_data_ops!(IntegerVec, "integer_vec", DataType::IntegerVec);
impl_data_ops!(RealVec, "real_vec", DataType::RealVec);
impl_data_ops!(BooleanVec, "boolean_vec", DataType::BooleanVec);
impl_data_ops!(StringVec, "string_vec", DataType::StringVec);
impl_data_ops!(ColorVec, "color_vec", DataType::ColorVec);
#[cfg(feature = "curves")]
impl_data_ops!(RealCurve, "real_curve", DataType::RealCurve);
#[cfg(feature = "curves")]
impl_data_ops!(ColorCurve, "color_curve", DataType::ColorCurve);

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl_data_ops!(Vector2Vec, "vec2_vec", DataType::Vector2Vec);
#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl_data_ops!(Vector3Vec, "vec3_vec", DataType::Vector3Vec);
#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl_data_ops!(Matrix3Vec, "mat3_vec", DataType::Matrix3Vec);

// New data type implementations
#[cfg(feature = "normal3")]
impl_data_ops!(Normal3, "normal3", DataType::Normal3);
#[cfg(feature = "point3")]
impl_data_ops!(Point3, "point3", DataType::Point3);
#[cfg(feature = "matrix4")]
impl_data_ops!(Matrix4, "matrix4", DataType::Matrix4);

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl_data_ops!(Normal3Vec, "normal3_vec", DataType::Normal3Vec);
#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl_data_ops!(Point3Vec, "point3_vec", DataType::Point3Vec);
#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl_data_ops!(Matrix4Vec, "matrix4_vec", DataType::Matrix4Vec);

// Macro to implement TryFrom<Value> and TryFrom<&Value> for data types using
// try_convert
macro_rules! impl_try_from_value {
    ($type:ty, $data_type:expr, $variant:ident) => {
        impl TryFrom<Value> for $type {
            type Error = Error;

            fn try_from(value: Value) -> std::result::Result<Self, Self::Error> {
                match value {
                    Value::Uniform(data) => {
                        let converted = data.try_convert($data_type)?;
                        match converted {
                            Data::$variant(v) => Ok(v),
                            _ => unreachable!(
                                "try_convert should return {} type",
                                stringify!($variant)
                            ),
                        }
                    }
                    Value::Animated(_) => Err(Error::AnimatedExtraction {
                        type_name: stringify!($type),
                    }),
                }
            }
        }

        impl TryFrom<&Value> for $type {
            type Error = Error;

            fn try_from(value: &Value) -> std::result::Result<Self, Self::Error> {
                match value {
                    Value::Uniform(data) => {
                        let converted = data.try_convert($data_type)?;
                        match converted {
                            Data::$variant(v) => Ok(v),
                            _ => unreachable!(
                                "try_convert should return {} type",
                                stringify!($variant)
                            ),
                        }
                    }
                    Value::Animated(_) => Err(Error::AnimatedExtraction {
                        type_name: stringify!($type),
                    }),
                }
            }
        }
    };
}

// Apply the macro to all data types
impl_try_from_value!(Boolean, DataType::Boolean, Boolean);
impl_try_from_value!(Integer, DataType::Integer, Integer);
impl_try_from_value!(Real, DataType::Real, Real);
impl_try_from_value!(String, DataType::String, String);
impl_try_from_value!(Color, DataType::Color, Color);
#[cfg(feature = "vector2")]
impl_try_from_value!(Vector2, DataType::Vector2, Vector2);
#[cfg(feature = "vector3")]
impl_try_from_value!(Vector3, DataType::Vector3, Vector3);
#[cfg(feature = "matrix3")]
impl_try_from_value!(Matrix3, DataType::Matrix3, Matrix3);
impl_try_from_value!(BooleanVec, DataType::BooleanVec, BooleanVec);
impl_try_from_value!(IntegerVec, DataType::IntegerVec, IntegerVec);
impl_try_from_value!(RealVec, DataType::RealVec, RealVec);
impl_try_from_value!(StringVec, DataType::StringVec, StringVec);
impl_try_from_value!(ColorVec, DataType::ColorVec, ColorVec);
#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl_try_from_value!(Vector2Vec, DataType::Vector2Vec, Vector2Vec);
#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl_try_from_value!(Vector3Vec, DataType::Vector3Vec, Vector3Vec);
#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl_try_from_value!(Matrix3Vec, DataType::Matrix3Vec, Matrix3Vec);

// New type TryFrom implementations
#[cfg(feature = "normal3")]
impl_try_from_value!(Normal3, DataType::Normal3, Normal3);
#[cfg(feature = "point3")]
impl_try_from_value!(Point3, DataType::Point3, Point3);
#[cfg(feature = "matrix4")]
impl_try_from_value!(Matrix4, DataType::Matrix4, Matrix4);

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl_try_from_value!(Normal3Vec, DataType::Normal3Vec, Normal3Vec);
#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl_try_from_value!(Point3Vec, DataType::Point3Vec, Point3Vec);
#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl_try_from_value!(Matrix4Vec, DataType::Matrix4Vec, Matrix4Vec);
#[cfg(feature = "curves")]
impl_try_from_value!(RealCurve, DataType::RealCurve, RealCurve);
#[cfg(feature = "curves")]
impl_try_from_value!(ColorCurve, DataType::ColorCurve, ColorCurve);
