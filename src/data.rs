use crate::{
    DataTypeOps,
    macros::{impl_data_arithmetic, impl_data_type_ops, impl_try_from_vec},
    *,
};
#[cfg(feature = "matrix3")]
use bytemuck::cast;
use bytemuck::cast_slice;
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
};
use strum::IntoDiscriminant;

/// A variant `enum` containing all supported data types.
///
/// [`Data`] can hold scalar values ([`Boolean`], [`Integer`], [`Real`],
/// [`String`]), vector types ([`Vector2`], [`Vector3`], [`Color`],
/// [`Matrix3`]), and collections of these types ([`BooleanVec`],
/// [`IntegerVec`], etc.).
#[derive(Debug, Clone, PartialEq, strum::AsRefStr, strum::EnumDiscriminants)]
#[strum_discriminants(name(DataType))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Data {
    /// A boolean value.
    Boolean(Boolean),
    /// A 64-bit signed integer.
    Integer(Integer),
    /// A 64-bit floating-point number.
    Real(Real),
    /// A UTF-8 string.
    String(String),
    /// A 4-component RGBA color.
    Color(Color),
    /// A 2D vector.
    #[cfg(feature = "vector2")]
    Vector2(Vector2),
    /// A 3D vector.
    #[cfg(feature = "vector3")]
    Vector3(Vector3),
    /// A 3×3 transformation matrix.
    #[cfg(feature = "matrix3")]
    Matrix3(Matrix3),
    /// A 3D normal vector.
    #[cfg(feature = "normal3")]
    Normal3(Normal3),
    /// A 3D point.
    #[cfg(feature = "point3")]
    Point3(Point3),
    /// A 4×4 transformation matrix.
    #[cfg(feature = "matrix4")]
    Matrix4(Matrix4),
    /// A vector of boolean values.
    BooleanVec(BooleanVec),
    /// A vector of integer values.
    IntegerVec(IntegerVec),
    /// A vector of real values.
    RealVec(RealVec),
    /// A vector of color values.
    ColorVec(ColorVec),
    /// A vector of string values.
    StringVec(StringVec),
    /// A vector of 2D vectors.
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    Vector2Vec(Vector2Vec),
    /// A vector of 3D vectors.
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    Vector3Vec(Vector3Vec),
    /// A vector of matrices.
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    Matrix3Vec(Matrix3Vec),
    /// A vector of 3D normals.
    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
    Normal3Vec(Normal3Vec),
    /// A vector of 3D points.
    #[cfg(all(feature = "point3", feature = "vec_variants"))]
    Point3Vec(Point3Vec),
    /// A vector of 4×4 matrices.
    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
    Matrix4Vec(Matrix4Vec),
}

impl_data_type_ops!(Data);

impl Data {
    /// Get the length of a vector value, or `1` for scalar values
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.try_len().unwrap_or(1)
    }

    /// Get the length of a vector value, or None for scalar values
    pub fn try_len(&self) -> Option<usize> {
        match self {
            Data::BooleanVec(v) => Some(v.0.len()),
            Data::IntegerVec(v) => Some(v.0.len()),
            Data::RealVec(v) => Some(v.0.len()),
            Data::StringVec(v) => Some(v.0.len()),
            Data::ColorVec(v) => Some(v.0.len()),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(v) => Some(v.0.len()),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(v) => Some(v.0.len()),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(v) => Some(v.0.len()),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(v) => Some(v.0.len()),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(v) => Some(v.0.len()),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(v) => Some(v.0.len()),
            _ => None,
        }
    }

    /// If this is a vector value.
    pub fn is_vec(&self) -> bool {
        self.try_len().is_some()
    }

    #[named]
    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Data::Boolean(value) => Ok(value.0),
            Data::Real(value) => Ok(value.0 != 0.0),
            Data::Integer(value) => Ok(value.0 != 0),
            Data::String(value) => Ok(value.0.parse::<bool>().unwrap_or(false)),
            _ => Err(anyhow!(
                "{}: called on '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Data::Boolean(value) => {
                if value.0 {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            Data::Real(value) => Ok(value.0 as _),
            Data::Integer(value) => Ok(value.0 as _),
            _ => Err(anyhow!(
                "to_f32() called on incompatible data type: {:?}",
                self.data_type()
            )),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Data::Boolean(value) => {
                if value.0 {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            }
            Data::Real(value) => Ok(value.0),
            Data::Integer(value) => Ok(value.0 as _),
            _ => Err(anyhow!(
                "to_f64() called on incompatible data type: {:?}",
                self.data_type()
            )),
        }
    }

    #[named]
    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Data::Boolean(value) => Ok(if value.0 { 1 } else { 0 }),
            Data::Real(value) => Ok((value.0 + 0.5) as i32),
            // FIXME: this may wrap around.
            Data::Integer(value) => value
                .0
                .try_into()
                .map_err(|e: std::num::TryFromIntError| anyhow!("Integer conversion error: {}", e)),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Data::Boolean(value) => {
                if value.0 {
                    Ok(1)
                } else {
                    Ok(0)
                }
            }
            Data::Real(value) => Ok((value.0 + 0.5) as _),
            Data::Integer(value) => Ok(value.0),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_slice_f64(&self) -> Result<&[f64]> {
        match self {
            Data::RealVec(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "matrix4")]
            Data::Matrix4(value) => Ok(value.0.as_slice()),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(value) => {
                // Convert Vec<nalgebra::Matrix4<f64>> to &[f64]
                Ok(bytemuck::cast_slice(&value.0))
            }
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_slice_f32(&self) -> Result<&[f32]> {
        match self {
            Data::Color(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "vector2")]
            Data::Vector2(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "vector3")]
            Data::Vector3(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "matrix3")]
            Data::Matrix3(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "normal3")]
            Data::Normal3(value) => Ok(value.0.as_slice()),
            #[cfg(feature = "point3")]
            Data::Point3(value) => Ok(value.0.coords.as_slice()),
            Data::ColorVec(value) => Ok(cast_slice(value.0.as_slice())),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(value) => {
                // Convert Vec<nalgebra::Vector2<f32>> to &[f32]
                Ok(bytemuck::cast_slice(&value.0))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(value) => {
                // Convert Vec<nalgebra::Vector3<f32>> to &[f32]
                Ok(bytemuck::cast_slice(&value.0))
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(value) => {
                // Convert Vec<nalgebra::Matrix3<f32>> to &[f32]
                Ok(bytemuck::cast_slice(&value.0))
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(value) => {
                // Convert Vec<nalgebra::Point3<f32>> to &[f32]
                Ok(bytemuck::cast_slice(&value.0))
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(value) => {
                // Convert Vec<nalgebra::Point3<f32>> to &[f32]
                Ok(bytemuck::cast_slice(&value.0))
            }
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_slice_i64(&self) -> Result<&[i64]> {
        match self {
            Data::IntegerVec(value) => Ok(value.0.as_slice()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_vector2_ref(&self) -> Result<&[f32; 2]> {
        match self {
            #[cfg(feature = "vector2")]
            Data::Vector2(value) => Ok(value.0.as_ref()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_vector3_ref(&self) -> Result<&[f32; 3]> {
        match self {
            #[cfg(feature = "vector3")]
            Data::Vector3(value) => Ok(value.0.as_ref()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_matrix3_ref(&self) -> Result<&[f32; 9]> {
        match self {
            #[cfg(feature = "matrix3")]
            Data::Matrix3(value) => {
                // nalgebra Matrix3 stores data in column-major order, cast
                // directly from Matrix3
                Ok(bytemuck::cast_ref(&value.0))
            }
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_color_ref(&self) -> Result<&[f32; 4]> {
        match self {
            Data::Color(value) => Ok(&value.0),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    #[cfg(feature = "normal3")]
    pub fn as_normal3_ref(&self) -> Result<&[f32; 3]> {
        match self {
            Data::Normal3(value) => Ok(value.0.as_ref()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    #[cfg(feature = "point3")]
    pub fn as_point3_ref(&self) -> Result<&[f32; 3]> {
        match self {
            Data::Point3(value) => Ok(value.0.coords.as_ref()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    #[cfg(feature = "matrix4")]
    pub fn as_matrix4_ref(&self) -> Result<&[f64; 16]> {
        match self {
            Data::Matrix4(value) => {
                // nalgebra Matrix4 stores data in column-major order, cast
                // directly from Matrix4
                Ok(bytemuck::cast_ref(&value.0))
            }
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Data::String(value) => Ok(value.0.as_str()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }

    #[named]
    pub fn as_slice_string(&self) -> Result<&[std::string::String]> {
        match self {
            Data::StringVec(value) => Ok(value.0.as_slice()),
            _ => Err(anyhow!(
                "{}: called on incompatible data type '{:?}'",
                function_name!(),
                self.discriminant()
            )),
        }
    }
}

// Macro to implement From trait for primitive types
macro_rules! impl_from_primitive {
    ($from:ty, $variant:ident, $wrapper:ident) => {
        impl From<$from> for Data {
            fn from(v: $from) -> Self {
                Data::$variant($wrapper(v as _))
            }
        }
    };
}

// Implement From for all primitive types
impl_from_primitive!(i64, Integer, Integer);
impl_from_primitive!(i32, Integer, Integer);
impl_from_primitive!(i16, Integer, Integer);
impl_from_primitive!(i8, Integer, Integer);
impl_from_primitive!(u32, Integer, Integer);
impl_from_primitive!(u16, Integer, Integer);
impl_from_primitive!(u8, Integer, Integer);

impl_from_primitive!(f64, Real, Real);
impl_from_primitive!(f32, Real, Real);

impl_from_primitive!(bool, Boolean, Boolean);

impl From<std::string::String> for Data {
    fn from(v: std::string::String) -> Self {
        Data::String(String(v))
    }
}

impl From<&str> for Data {
    fn from(v: &str) -> Self {
        Data::String(String(v.into()))
    }
}

// From implementations for arrays
#[cfg(feature = "vector2")]
impl From<[f32; 2]> for Data {
    fn from(v: [f32; 2]) -> Self {
        Data::Vector2(Vector2(v.into()))
    }
}

#[cfg(feature = "vector3")]
impl From<[f32; 3]> for Data {
    fn from(v: [f32; 3]) -> Self {
        Data::Vector3(Vector3(v.into()))
    }
}

#[cfg(feature = "matrix3")]
impl From<[[f32; 3]; 3]> for Data {
    fn from(v: [[f32; 3]; 3]) -> Self {
        let arr: [f32; 9] = cast(v);
        Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(&arr)))
    }
}

#[cfg(feature = "matrix3")]
impl From<[f32; 9]> for Data {
    fn from(v: [f32; 9]) -> Self {
        Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(&v)))
    }
}

impl From<[f32; 4]> for Data {
    fn from(v: [f32; 4]) -> Self {
        Data::Color(Color(v))
    }
}

// From implementations for nalgebra types
#[cfg(feature = "vector2")]
impl From<nalgebra::Vector2<f32>> for Data {
    fn from(v: nalgebra::Vector2<f32>) -> Self {
        Data::Vector2(Vector2(v))
    }
}

#[cfg(feature = "vector3")]
impl From<nalgebra::Vector3<f32>> for Data {
    fn from(v: nalgebra::Vector3<f32>) -> Self {
        Data::Vector3(Vector3(v))
    }
}

#[cfg(feature = "matrix3")]
impl From<nalgebra::Matrix3<f32>> for Data {
    fn from(v: nalgebra::Matrix3<f32>) -> Self {
        Data::Matrix3(Matrix3(v))
    }
}

// From implementations for Vec types
impl TryFrom<Vec<i64>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<i64>) -> Result<Self> {
        Ok(Data::IntegerVec(IntegerVec::new(v)?))
    }
}

impl TryFrom<Vec<f64>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<f64>) -> Result<Self> {
        Ok(Data::RealVec(RealVec::new(v)?))
    }
}

impl TryFrom<Vec<bool>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<bool>) -> Result<Self> {
        Ok(Data::BooleanVec(BooleanVec::new(v)?))
    }
}

impl TryFrom<Vec<&str>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<&str>) -> Result<Self> {
        let string_vec: Vec<std::string::String> = v.into_iter().map(|s| s.to_string()).collect();
        Ok(Data::StringVec(StringVec::new(string_vec)?))
    }
}

impl TryFrom<Vec<[f32; 4]>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<[f32; 4]>) -> Result<Self> {
        Ok(Data::ColorVec(ColorVec::new(v)?))
    }
}

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl TryFrom<Vec<nalgebra::Vector2<f32>>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<nalgebra::Vector2<f32>>) -> Result<Self> {
        Ok(Data::Vector2Vec(Vector2Vec::new(v)?))
    }
}

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl TryFrom<Vec<nalgebra::Vector3<f32>>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<nalgebra::Vector3<f32>>) -> Result<Self> {
        Ok(Data::Vector3Vec(Vector3Vec::new(v)?))
    }
}

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl TryFrom<Vec<nalgebra::Matrix3<f32>>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<nalgebra::Matrix3<f32>>) -> Result<Self> {
        Ok(Data::Matrix3Vec(Matrix3Vec::new(v)?))
    }
}

impl From<Vec<u32>> for Data {
    fn from(v: Vec<u32>) -> Self {
        let int_vec: Vec<i64> = v.into_iter().map(|x| x as i64).collect();
        Data::IntegerVec(IntegerVec(int_vec))
    }
}

impl From<Vec<f32>> for Data {
    fn from(v: Vec<f32>) -> Self {
        let real_vec: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
        Data::RealVec(RealVec(real_vec))
    }
}

impl TryFrom<Vec<std::string::String>> for Data {
    type Error = anyhow::Error;

    fn try_from(v: Vec<std::string::String>) -> Result<Self> {
        Ok(Data::StringVec(StringVec::new(v)?))
    }
}

macro_rules! impl_try_from_value {
    ($target:ty, $variant:ident) => {
        impl TryFrom<Data> for $target {
            type Error = anyhow::Error;

            fn try_from(value: Data) -> Result<Self, Self::Error> {
                match value {
                    Data::$variant(v) => Ok(v.0),
                    _ => Err(anyhow!(
                        "Could not convert {} to {}",
                        stringify!($variant),
                        stringify!($target)
                    )),
                }
            }
        }

        impl TryFrom<&Data> for $target {
            type Error = anyhow::Error;

            fn try_from(value: &Data) -> Result<Self, Self::Error> {
                match value {
                    Data::$variant(v) => Ok(v.0.clone()),
                    _ => Err(anyhow!(
                        "Could not convert &{} to {}",
                        stringify!($variant),
                        stringify!($target)
                    )),
                }
            }
        }
    };
}

// Implement `TryFrom` for all types
impl_try_from_value!(bool, Boolean);
impl_try_from_value!(i64, Integer);
impl_try_from_value!(f64, Real);
impl_try_from_value!(std::string::String, String);
impl_try_from_value!([f32; 4], Color);
#[cfg(feature = "vector2")]
impl_try_from_value!(nalgebra::Vector2<f32>, Vector2);
#[cfg(feature = "vector3")]
impl_try_from_value!(nalgebra::Vector3<f32>, Vector3);
#[cfg(feature = "matrix3")]
impl_try_from_value!(nalgebra::Matrix3<f32>, Matrix3);

// TryFrom implementations for Vec types using macro
impl_try_from_vec!(
    bool, BooleanVec, "bool";
    i64, IntegerVec, "i64";
    f64, RealVec, "f64";
    std::string::String, StringVec, "String";
    [f32; 4], ColorVec, "[f32; 4]";
);

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl_try_from_vec!(
    nalgebra::Vector2<f32>, Vector2Vec, "Vector2<f32>";
);

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl_try_from_vec!(
    nalgebra::Vector3<f32>, Vector3Vec, "Vector3<f32>";
);

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl_try_from_vec!(
    nalgebra::Matrix3<f32>, Matrix3Vec, "Matrix3<f32>";
);

// Custom Hash implementation
impl Hash for Data {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Data::Boolean(Boolean(b)) => b.hash(state),
            Data::Integer(Integer(i)) => i.hash(state),
            Data::Real(Real(f)) => f.to_bits().hash(state),
            Data::String(String(s)) => s.hash(state),
            Data::Color(Color(c)) => {
                c.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "vector2")]
            Data::Vector2(Vector2(v)) => {
                v.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "vector3")]
            Data::Vector3(Vector3(v)) => {
                v.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "matrix3")]
            Data::Matrix3(Matrix3(m)) => {
                m.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "normal3")]
            Data::Normal3(Normal3(v)) => {
                v.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "point3")]
            Data::Point3(Point3(p)) => {
                p.iter().for_each(|v| v.to_bits().hash(state));
            }
            #[cfg(feature = "matrix4")]
            Data::Matrix4(Matrix4(m)) => {
                m.iter().for_each(|v| v.to_bits().hash(state));
            }
            Data::BooleanVec(BooleanVec(v)) => v.hash(state),
            Data::IntegerVec(IntegerVec(v)) => v.hash(state),
            Data::RealVec(RealVec(v)) => {
                v.len().hash(state);
                v.iter().for_each(|v| v.to_bits().hash(state));
            }
            Data::StringVec(StringVec(v)) => v.hash(state),
            Data::ColorVec(ColorVec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|c| c.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(Vector2Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|v| v.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(Vector3Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|v| v.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(Matrix3Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|m| m.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(Normal3Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|v| v.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(Point3Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|p| p.iter().for_each(|v| v.to_bits().hash(state)));
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(Matrix4Vec(v)) => {
                v.len().hash(state);
                v.iter()
                    .for_each(|m| m.iter().for_each(|v| v.to_bits().hash(state)));
            }
        }
    }
}

impl Data {
    /// Ensure a vector value has at least the specified length by padding with
    /// defaults
    pub fn pad_to_length(&mut self, target_len: usize) {
        match self {
            Data::BooleanVec(BooleanVec(v)) => v.resize(target_len, false),
            Data::IntegerVec(IntegerVec(v)) => v.resize(target_len, 0),
            Data::RealVec(RealVec(v)) => v.resize(target_len, 0.0),
            Data::StringVec(StringVec(v)) => v.resize(target_len, std::string::String::new()),
            Data::ColorVec(ColorVec(v)) => v.resize(target_len, [0.0, 0.0, 0.0, 1.0]),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(Vector2Vec(v)) => v.resize(target_len, nalgebra::Vector2::zeros()),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(Vector3Vec(v)) => v.resize(target_len, nalgebra::Vector3::zeros()),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(Matrix3Vec(v)) => v.resize(target_len, nalgebra::Matrix3::zeros()),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(Normal3Vec(v)) => v.resize(target_len, nalgebra::Vector3::zeros()),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(Point3Vec(v)) => v.resize(target_len, nalgebra::Point3::origin()),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(Matrix4Vec(v)) => v.resize(target_len, nalgebra::Matrix4::zeros()),
            _ => {} // Non-vector types are ignored
        }
    }

    /// Try to convert this value to another type
    pub fn try_convert(&self, to: DataType) -> Result<Data> {
        match (self, to) {
            // Same type - just clone
            (v, target) if v.data_type() == target => Ok(v.clone()),

            // To Integer conversions
            (Data::Real(Real(f)), DataType::Integer) => Ok(Data::Integer(Integer(*f as i64))),
            (Data::Boolean(Boolean(b)), DataType::Integer) => {
                Ok(Data::Integer(Integer(if *b { 1 } else { 0 })))
            }
            (Data::String(String(s)), DataType::Integer) => s
                .parse::<i64>()
                .map(|i| Data::Integer(Integer(i)))
                .map_err(|_| anyhow!("Cannot parse '{}' as Integer", s)),

            // To Real conversions
            (Data::Integer(Integer(i)), DataType::Real) => Ok(Data::Real(Real(*i as f64))),
            (Data::Boolean(Boolean(b)), DataType::Real) => {
                Ok(Data::Real(Real(if *b { 1.0 } else { 0.0 })))
            }
            (Data::String(String(s)), DataType::Real) => s
                .parse::<f64>()
                .map(|f| Data::Real(Real(f)))
                .map_err(|_| anyhow!("Cannot parse '{}' as Real", s)),

            // To Boolean conversions
            (Data::Integer(Integer(i)), DataType::Boolean) => Ok(Data::Boolean(Boolean(*i != 0))),
            (Data::Real(Real(f)), DataType::Boolean) => Ok(Data::Boolean(Boolean(*f != 0.0))),
            (Data::String(String(s)), DataType::Boolean) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" | "on" => Ok(Data::Boolean(Boolean(true))),
                "false" | "no" | "0" | "off" | "" => Ok(Data::Boolean(Boolean(false))),
                _ => Err(anyhow!("Cannot parse '{}' as Boolean", s)),
            },

            // To String conversions
            (Data::Integer(Integer(i)), DataType::String) => {
                Ok(Data::String(String(i.to_string())))
            }
            (Data::Real(Real(f)), DataType::String) => Ok(Data::String(String(f.to_string()))),
            (Data::Boolean(Boolean(b)), DataType::String) => {
                Ok(Data::String(String(b.to_string())))
            }
            #[cfg(feature = "vector2")]
            (Data::Vector2(Vector2(v)), DataType::String) => {
                Ok(Data::String(String(format!("[{}, {}]", v[0], v[1]))))
            }
            #[cfg(feature = "vector3")]
            (Data::Vector3(Vector3(v)), DataType::String) => Ok(Data::String(String(format!(
                "[{}, {}, {}]",
                v[0], v[1], v[2]
            )))),
            (Data::Color(Color(c)), DataType::String) => Ok(Data::String(String(format!(
                "[{}, {}, {}, {}]",
                c[0], c[1], c[2], c[3]
            )))),
            #[cfg(feature = "matrix3")]
            (Data::Matrix3(Matrix3(m)), DataType::String) => {
                Ok(Data::String(String(format!("{m:?}"))))
            }
            #[cfg(feature = "normal3")]
            (Data::Normal3(Normal3(v)), DataType::String) => Ok(Data::String(String(format!(
                "[{}, {}, {}]",
                v[0], v[1], v[2]
            )))),
            #[cfg(feature = "point3")]
            (Data::Point3(Point3(p)), DataType::String) => {
                Ok(Data::String(String(format!("[{}, {}, {}]", p.x, p.y, p.z))))
            }
            #[cfg(feature = "matrix4")]
            (Data::Matrix4(Matrix4(m)), DataType::String) => {
                Ok(Data::String(String(format!("{m:?}"))))
            }

            // To Vec2 conversions
            #[cfg(feature = "vector2")]
            (Data::Integer(Integer(i)), DataType::Vector2) => {
                let v = *i as f32;
                Ok(Data::Vector2(Vector2(nalgebra::Vector2::new(v, v))))
            }
            #[cfg(feature = "vector2")]
            (Data::Real(Real(f)), DataType::Vector2) => {
                let v = *f as f32;
                Ok(Data::Vector2(Vector2(nalgebra::Vector2::new(v, v))))
            }
            #[cfg(feature = "vector2")]
            (Data::RealVec(RealVec(vec)), DataType::Vector2) if vec.len() >= 2 => {
                let v: Vec<f32> = vec.iter().take(2).map(|&x| x as f32).collect();
                Ok(Data::Vector2(Vector2(nalgebra::Vector2::from_vec(v))))
            }
            #[cfg(feature = "vector2")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::Vector2) if vec.len() >= 2 => {
                let v: Vec<f32> = vec.iter().take(2).map(|&x| x as f32).collect();
                Ok(Data::Vector2(Vector2(nalgebra::Vector2::from_vec(v))))
            }
            #[cfg(feature = "vector2")]
            (Data::String(String(s)), DataType::Vector2) => {
                parse_to_array::<f32, 2>(s).map(|v| Data::Vector2(Vector2(v.into())))
            }

            // To Vec3 conversions
            #[cfg(feature = "vector3")]
            (Data::Integer(Integer(i)), DataType::Vector3) => {
                let v = *i as f32;
                Ok(Data::Vector3(Vector3(nalgebra::Vector3::new(v, v, v))))
            }
            #[cfg(feature = "vector3")]
            (Data::Real(Real(f)), DataType::Vector3) => {
                let v = *f as f32;
                Ok(Data::Vector3(Vector3(nalgebra::Vector3::new(v, v, v))))
            }
            #[cfg(all(feature = "vector2", feature = "vector3"))]
            (Data::Vector2(Vector2(v)), DataType::Vector3) => Ok(Data::Vector3(Vector3(
                nalgebra::Vector3::new(v.x, v.y, 0.0),
            ))),
            #[cfg(feature = "vector3")]
            (Data::RealVec(RealVec(vec)), DataType::Vector3) if vec.len() >= 3 => {
                let v: Vec<f32> = vec.iter().take(3).map(|&x| x as f32).collect();
                Ok(Data::Vector3(Vector3(nalgebra::Vector3::from_vec(v))))
            }
            #[cfg(feature = "vector3")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::Vector3) if vec.len() >= 3 => {
                let v: Vec<f32> = vec.iter().take(3).map(|&x| x as f32).collect();
                Ok(Data::Vector3(Vector3(nalgebra::Vector3::from_vec(v))))
            }
            #[cfg(feature = "vector3")]
            (Data::ColorVec(ColorVec(vec)), DataType::Vector3) if !vec.is_empty() => {
                let c = &vec[0];
                Ok(Data::Vector3(Vector3(nalgebra::Vector3::new(
                    c[0], c[1], c[2],
                ))))
            }
            #[cfg(feature = "vector3")]
            (Data::String(String(s)), DataType::Vector3) => {
                parse_to_array::<f32, 3>(s).map(|v| Data::Vector3(Vector3(v.into())))
            }

            // To Color conversions
            (Data::Real(Real(f)), DataType::Color) => {
                let f = *f as f32;
                Ok(Data::Color(Color([f, f, f, 1.0])))
            }
            (Data::RealVec(RealVec(vec)), DataType::Color) if vec.len() >= 3 => {
                let mut color = [0.0f32; 4];
                vec.iter()
                    .take(4)
                    .enumerate()
                    .for_each(|(i, &v)| color[i] = v as f32);
                if vec.len() < 4 {
                    color[3] = 1.0;
                }
                Ok(Data::Color(Color(color)))
            }
            (Data::IntegerVec(IntegerVec(vec)), DataType::Color) if vec.len() >= 3 => {
                let mut color = [0.0f32; 4];
                vec.iter()
                    .take(4)
                    .enumerate()
                    .for_each(|(i, &v)| color[i] = v as f32);
                if vec.len() < 4 {
                    color[3] = 1.0;
                }
                Ok(Data::Color(Color(color)))
            }
            #[cfg(feature = "vector3")]
            (Data::Vector3(Vector3(v)), DataType::Color) => {
                Ok(Data::Color(Color([v.x, v.y, v.z, 1.0])))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::Vector3Vec(Vector3Vec(vec)), DataType::Color) if !vec.is_empty() => {
                Ok(Data::Color(Color([vec[0].x, vec[0].y, vec[0].z, 1.0])))
            }
            #[cfg(feature = "vector2")]
            (Data::Vector2(Vector2(v)), DataType::Color) => {
                Ok(Data::Color(Color([v.x, v.y, 0.0, 1.0])))
            }
            #[cfg(feature = "point3")]
            (Data::Point3(Point3(p)), DataType::Color) => {
                Ok(Data::Color(Color([p.x, p.y, p.z, 1.0])))
            }
            #[cfg(feature = "normal3")]
            (Data::Normal3(Normal3(v)), DataType::Color) => {
                Ok(Data::Color(Color([v.x, v.y, v.z, 1.0])))
            }
            (Data::String(String(s)), DataType::Color) => parse_color_from_string(s)
                .map(|c| Data::Color(Color(c)))
                .ok_or_else(|| anyhow!("Cannot parse '{}' as Color", s)),

            // To Mat3 conversions
            #[cfg(feature = "matrix3")]
            (Data::Integer(Integer(i)), DataType::Matrix3) => {
                let v = *i as f32;
                Ok(Data::Matrix3(Matrix3(nalgebra::Matrix3::new(
                    v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 1.0,
                ))))
            }
            #[cfg(feature = "matrix3")]
            (Data::Real(Real(f)), DataType::Matrix3) => {
                let v = *f as f32;
                Ok(Data::Matrix3(Matrix3(nalgebra::Matrix3::new(
                    v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 1.0,
                ))))
            }
            #[cfg(feature = "matrix3")]
            (Data::RealVec(RealVec(vec)), DataType::Matrix3) if vec.len() >= 9 => {
                // AIDEV-NOTE: Using iterator for efficient conversion.
                let m: Vec<f32> = vec.iter().take(9).map(|&x| x as f32).collect();
                Ok(Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(
                    &m,
                ))))
            }
            #[cfg(feature = "matrix3")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::Matrix3) if vec.len() >= 9 => {
                let m: Vec<f32> = vec.iter().take(9).map(|&x| x as f32).collect();
                Ok(Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(
                    &m,
                ))))
            }
            #[cfg(all(feature = "vector3", feature = "matrix3"))]
            (Data::Vector3(Vector3(v)), DataType::Matrix3) => Ok(Data::Matrix3(Matrix3(
                nalgebra::Matrix3::new(v.x, 0.0, 0.0, 0.0, v.y, 0.0, 0.0, 0.0, v.z),
            ))),
            #[cfg(all(feature = "vector3", feature = "matrix3", feature = "vec_variants"))]
            (Data::Vector3Vec(Vector3Vec(vec)), DataType::Matrix3) if vec.len() >= 3 => {
                // Use 3 Vector3s as columns of the matrix.
                let cols: Vec<f32> = vec.iter().take(3).flat_map(|v| [v.x, v.y, v.z]).collect();
                Ok(Data::Matrix3(Matrix3(
                    nalgebra::Matrix3::from_column_slice(&cols),
                )))
            }
            #[cfg(feature = "matrix3")]
            (Data::ColorVec(ColorVec(vec)), DataType::Matrix3) if vec.len() >= 3 => {
                // Use RGB components of 3 colors as rows.
                let rows: Vec<f32> = vec
                    .iter()
                    .take(3)
                    .flat_map(|c| c[0..3].iter().copied())
                    .collect();
                Ok(Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(
                    &rows,
                ))))
            }
            #[cfg(feature = "matrix3")]
            (Data::String(String(s)), DataType::Matrix3) => {
                // Try to parse as a single value first for diagonal matrix
                if let Ok(single_val) = s.trim().parse::<f32>() {
                    Ok(Data::Matrix3(Matrix3(
                        nalgebra::Matrix3::from_diagonal_element(single_val),
                    )))
                } else {
                    // Parse as 9 separate values
                    parse_to_array::<f32, 9>(s)
                        .map(|m| Data::Matrix3(Matrix3(nalgebra::Matrix3::from_row_slice(&m))))
                }
            }

            // To Normal3 conversions
            #[cfg(feature = "normal3")]
            (Data::Integer(Integer(i)), DataType::Normal3) => {
                let v = *i as f32;
                Ok(Data::Normal3(Normal3(nalgebra::Vector3::new(v, v, v))))
            }
            #[cfg(feature = "normal3")]
            (Data::Real(Real(f)), DataType::Normal3) => {
                let v = *f as f32;
                Ok(Data::Normal3(Normal3(nalgebra::Vector3::new(v, v, v))))
            }
            #[cfg(all(feature = "vector3", feature = "normal3"))]
            (Data::Vector3(Vector3(v)), DataType::Normal3) => {
                Ok(Data::Normal3(Normal3(v.normalize())))
            }
            #[cfg(feature = "normal3")]
            (Data::String(String(s)), DataType::Normal3) => parse_to_array::<f32, 3>(s)
                .map(|v| Data::Normal3(Normal3(nalgebra::Vector3::from(v).normalize()))),

            // To Point3 conversions
            #[cfg(feature = "point3")]
            (Data::Integer(Integer(i)), DataType::Point3) => {
                let v = *i as f32;
                Ok(Data::Point3(Point3(nalgebra::Point3::new(v, v, v))))
            }
            #[cfg(feature = "point3")]
            (Data::Real(Real(f)), DataType::Point3) => {
                let v = *f as f32;
                Ok(Data::Point3(Point3(nalgebra::Point3::new(v, v, v))))
            }
            #[cfg(all(feature = "vector3", feature = "point3"))]
            (Data::Vector3(Vector3(v)), DataType::Point3) => {
                Ok(Data::Point3(Point3(nalgebra::Point3::new(v.x, v.y, v.z))))
            }
            #[cfg(feature = "point3")]
            (Data::String(String(s)), DataType::Point3) => {
                parse_to_array::<f32, 3>(s).map(|v| Data::Point3(Point3(nalgebra::Point3::from(v))))
            }

            // To Matrix4 conversions
            #[cfg(feature = "matrix4")]
            (Data::Integer(Integer(i)), DataType::Matrix4) => {
                let v = *i as f64;
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::new(
                    v, 0.0, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0, 1.0,
                ))))
            }
            #[cfg(feature = "matrix4")]
            (Data::Real(Real(f)), DataType::Matrix4) => {
                let v = *f;
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::new(
                    v, 0.0, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0, 1.0,
                ))))
            }
            #[cfg(feature = "matrix4")]
            (Data::RealVec(RealVec(vec)), DataType::Matrix4) if vec.len() >= 16 => {
                // AIDEV-NOTE: Direct copy when types match, using bytemuck for exact size.
                if vec.len() == 16 {
                    let arr: &[f64; 16] = vec
                        .as_slice()
                        .try_into()
                        .map_err(|_| anyhow!("Failed to convert slice to array"))?;
                    Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::from_row_slice(
                        arr,
                    ))))
                } else {
                    let m: Vec<f64> = vec.iter().take(16).copied().collect();
                    Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::from_row_slice(
                        &m,
                    ))))
                }
            }
            #[cfg(feature = "matrix4")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::Matrix4) if vec.len() >= 16 => {
                let m: Vec<f64> = vec.iter().take(16).map(|&x| x as f64).collect();
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::from_row_slice(
                    &m,
                ))))
            }
            #[cfg(all(feature = "matrix3", feature = "matrix4"))]
            (Data::Matrix3(Matrix3(m)), DataType::Matrix4) => {
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::new(
                    m.m11 as f64,
                    m.m12 as f64,
                    m.m13 as f64,
                    0.0,
                    m.m21 as f64,
                    m.m22 as f64,
                    m.m23 as f64,
                    0.0,
                    m.m31 as f64,
                    m.m32 as f64,
                    m.m33 as f64,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ))))
            }
            #[cfg(all(feature = "matrix3", feature = "matrix4", feature = "vec_variants"))]
            (Data::Matrix3Vec(Matrix3Vec(vec)), DataType::Matrix4) if !vec.is_empty() => {
                let m = &vec[0];
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::new(
                    m.m11 as f64,
                    m.m12 as f64,
                    m.m13 as f64,
                    0.0,
                    m.m21 as f64,
                    m.m22 as f64,
                    m.m23 as f64,
                    0.0,
                    m.m31 as f64,
                    m.m32 as f64,
                    m.m33 as f64,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ))))
            }
            #[cfg(feature = "matrix4")]
            (Data::ColorVec(ColorVec(vec)), DataType::Matrix4) if vec.len() >= 4 => {
                // Use RGBA components of 4 colors as rows.
                let rows: Vec<f64> = vec
                    .iter()
                    .take(4)
                    .flat_map(|c| c.iter().map(|&x| x as f64))
                    .collect();
                Ok(Data::Matrix4(Matrix4(nalgebra::Matrix4::from_row_slice(
                    &rows,
                ))))
            }
            #[cfg(feature = "matrix4")]
            (Data::String(String(s)), DataType::Matrix4) => {
                // Try to parse as a single value first for diagonal matrix
                if let Ok(single_val) = s.trim().parse::<f64>() {
                    Ok(Data::Matrix4(Matrix4(
                        nalgebra::Matrix4::from_diagonal_element(single_val),
                    )))
                } else {
                    // Parse as 16 separate values
                    parse_to_array::<f64, 16>(s)
                        .map(|m| Data::Matrix4(Matrix4(nalgebra::Matrix4::from_row_slice(&m))))
                }
            }

            // Vec conversions from scalars and other types
            // To RealVec conversions
            (Data::Integer(Integer(i)), DataType::RealVec) => {
                Ok(Data::RealVec(RealVec(vec![*i as f64])))
            }
            (Data::Real(Real(f)), DataType::RealVec) => Ok(Data::RealVec(RealVec(vec![*f]))),
            #[cfg(feature = "vector2")]
            (Data::Vector2(Vector2(v)), DataType::RealVec) => Ok(Data::RealVec(RealVec(
                v.iter().map(|&x| x as f64).collect(),
            ))),
            #[cfg(feature = "vector3")]
            (Data::Vector3(Vector3(v)), DataType::RealVec) => Ok(Data::RealVec(RealVec(
                v.iter().map(|&x| x as f64).collect(),
            ))),
            (Data::Color(Color(c)), DataType::RealVec) => Ok(Data::RealVec(RealVec(
                c.iter().map(|&x| x as f64).collect(),
            ))),
            #[cfg(feature = "matrix3")]
            (Data::Matrix3(Matrix3(m)), DataType::RealVec) => Ok(Data::RealVec(RealVec(
                m.iter().map(|&x| x as f64).collect(),
            ))),
            #[cfg(feature = "matrix4")]
            (Data::Matrix4(Matrix4(m)), DataType::RealVec) => {
                Ok(Data::RealVec(RealVec(m.iter().copied().collect())))
            }
            #[cfg(feature = "normal3")]
            (Data::Normal3(Normal3(v)), DataType::RealVec) => Ok(Data::RealVec(RealVec(
                v.iter().map(|&x| x as f64).collect(),
            ))),
            #[cfg(feature = "point3")]
            (Data::Point3(Point3(p)), DataType::RealVec) => Ok(Data::RealVec(RealVec(vec![
                p.x as f64, p.y as f64, p.z as f64,
            ]))),

            // To IntegerVec conversions
            (Data::Boolean(Boolean(b)), DataType::IntegerVec) => {
                Ok(Data::IntegerVec(IntegerVec(vec![if *b { 1 } else { 0 }])))
            }
            (Data::Integer(Integer(i)), DataType::IntegerVec) => {
                Ok(Data::IntegerVec(IntegerVec(vec![*i])))
            }
            (Data::Real(Real(f)), DataType::IntegerVec) => {
                Ok(Data::IntegerVec(IntegerVec(vec![*f as i64])))
            }
            #[cfg(feature = "vector2")]
            (Data::Vector2(Vector2(v)), DataType::IntegerVec) => Ok(Data::IntegerVec(IntegerVec(
                v.iter().map(|&x| x as i64).collect(),
            ))),
            #[cfg(feature = "vector3")]
            (Data::Vector3(Vector3(v)), DataType::IntegerVec) => Ok(Data::IntegerVec(IntegerVec(
                v.iter().map(|&x| x as i64).collect(),
            ))),
            (Data::Color(Color(c)), DataType::IntegerVec) => Ok(Data::IntegerVec(IntegerVec(
                c.iter().map(|&x| (x * 255.0) as i64).collect(),
            ))),
            #[cfg(feature = "matrix3")]
            (Data::Matrix3(Matrix3(m)), DataType::IntegerVec) => Ok(Data::IntegerVec(IntegerVec(
                m.iter().map(|&x| x as i64).collect(),
            ))),
            #[cfg(feature = "matrix4")]
            (Data::Matrix4(Matrix4(m)), DataType::IntegerVec) => Ok(Data::IntegerVec(IntegerVec(
                m.iter().map(|&x| x as i64).collect(),
            ))),

            // To ColorVec conversions
            (Data::Color(Color(c)), DataType::ColorVec) => Ok(Data::ColorVec(ColorVec(vec![*c]))),
            #[cfg(feature = "vector3")]
            (Data::Vector3(Vector3(v)), DataType::ColorVec) => {
                Ok(Data::ColorVec(ColorVec(vec![[v.x, v.y, v.z, 1.0]])))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::Vector3Vec(Vector3Vec(vec)), DataType::ColorVec) => {
                let colors = vec.iter().map(|v| [v.x, v.y, v.z, 1.0]).collect();
                Ok(Data::ColorVec(ColorVec(colors)))
            }
            #[cfg(feature = "matrix3")]
            (Data::Matrix3(Matrix3(m)), DataType::ColorVec) => {
                // Convert each row to a color
                let colors = (0..3)
                    .map(|i| {
                        let row = m.row(i);
                        [row[0], row[1], row[2], 1.0]
                    })
                    .collect();
                Ok(Data::ColorVec(ColorVec(colors)))
            }

            // To Vector2Vec conversions
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            (Data::Vector2(Vector2(v)), DataType::Vector2Vec) => {
                Ok(Data::Vector2Vec(Vector2Vec(vec![*v])))
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            (Data::RealVec(RealVec(vec)), DataType::Vector2Vec)
                if vec.len() >= 2 && vec.len() % 2 == 0 =>
            {
                let vectors = vec
                    .chunks_exact(2)
                    .map(|chunk| nalgebra::Vector2::new(chunk[0] as f32, chunk[1] as f32))
                    .collect();
                Ok(Data::Vector2Vec(Vector2Vec(vectors)))
            }

            // To Vector3Vec conversions
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::Vector3(Vector3(v)), DataType::Vector3Vec) => {
                Ok(Data::Vector3Vec(Vector3Vec(vec![*v])))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::RealVec(RealVec(vec)), DataType::Vector3Vec)
                if vec.len() >= 3 && vec.len() % 3 == 0 =>
            {
                let vectors = vec
                    .chunks_exact(3)
                    .map(|chunk| {
                        nalgebra::Vector3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32)
                    })
                    .collect();
                Ok(Data::Vector3Vec(Vector3Vec(vectors)))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::IntegerVec(IntegerVec(vec)), DataType::Vector3Vec)
                if vec.len() >= 3 && vec.len() % 3 == 0 =>
            {
                let vectors = vec
                    .chunks_exact(3)
                    .map(|chunk| {
                        nalgebra::Vector3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32)
                    })
                    .collect();
                Ok(Data::Vector3Vec(Vector3Vec(vectors)))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (Data::ColorVec(ColorVec(vec)), DataType::Vector3Vec) => {
                let vectors = vec
                    .iter()
                    .map(|c| nalgebra::Vector3::new(c[0], c[1], c[2]))
                    .collect();
                Ok(Data::Vector3Vec(Vector3Vec(vectors)))
            }

            // To Matrix3Vec conversions
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            (Data::Matrix3(Matrix3(m)), DataType::Matrix3Vec) => {
                Ok(Data::Matrix3Vec(Matrix3Vec(vec![*m])))
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            (Data::RealVec(RealVec(vec)), DataType::Matrix3Vec)
                if vec.len() >= 9 && vec.len() % 9 == 0 =>
            {
                let matrices = vec
                    .chunks_exact(9)
                    .map(|chunk| {
                        let m: Vec<f32> = chunk.iter().map(|&x| x as f32).collect();
                        nalgebra::Matrix3::from_row_slice(&m)
                    })
                    .collect();
                Ok(Data::Matrix3Vec(Matrix3Vec(matrices)))
            }

            // To Matrix4Vec conversions
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            (Data::Matrix4(Matrix4(m)), DataType::Matrix4Vec) => {
                Ok(Data::Matrix4Vec(Matrix4Vec(vec![*m])))
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            (Data::RealVec(RealVec(vec)), DataType::Matrix4Vec)
                if vec.len() >= 16 && vec.len() % 16 == 0 =>
            {
                let matrices = vec
                    .chunks_exact(16)
                    .map(nalgebra::Matrix4::from_row_slice)
                    .collect();
                Ok(Data::Matrix4Vec(Matrix4Vec(matrices)))
            }

            // Vec to Vec conversions
            #[cfg(feature = "vec_variants")]
            (Data::RealVec(RealVec(vec)), DataType::IntegerVec) => {
                // AIDEV-NOTE: Converting RealVec to IntegerVec by rounding each element.
                Ok(Data::IntegerVec(IntegerVec(
                    vec.iter().map(|&f| f.round() as i64).collect(),
                )))
            }
            #[cfg(feature = "vec_variants")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::RealVec) => {
                // AIDEV-NOTE: Converting IntegerVec to RealVec by casting each element.
                Ok(Data::RealVec(RealVec(
                    vec.iter().map(|&i| i as f64).collect(),
                )))
            }
            #[cfg(feature = "vec_variants")]
            (Data::BooleanVec(BooleanVec(vec)), DataType::IntegerVec) => {
                // AIDEV-NOTE: Converting BooleanVec to IntegerVec (true -> 1, false -> 0).
                Ok(Data::IntegerVec(IntegerVec(
                    vec.iter().map(|&b| if b { 1 } else { 0 }).collect(),
                )))
            }
            #[cfg(feature = "vec_variants")]
            (Data::IntegerVec(IntegerVec(vec)), DataType::BooleanVec) => {
                // AIDEV-NOTE: Converting IntegerVec to BooleanVec (0 -> false, non-0 -> true).
                Ok(Data::BooleanVec(BooleanVec(
                    vec.iter().map(|&i| i != 0).collect(),
                )))
            }
            #[cfg(feature = "vec_variants")]
            (Data::BooleanVec(BooleanVec(vec)), DataType::RealVec) => {
                // AIDEV-NOTE: Converting BooleanVec to RealVec (true -> 1.0, false -> 0.0).
                Ok(Data::RealVec(RealVec(
                    vec.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
                )))
            }
            #[cfg(feature = "vec_variants")]
            (Data::RealVec(RealVec(vec)), DataType::BooleanVec) => {
                // AIDEV-NOTE: Converting RealVec to BooleanVec (0.0 -> false, non-0.0 -> true).
                Ok(Data::BooleanVec(BooleanVec(
                    vec.iter().map(|&f| f != 0.0).collect(),
                )))
            }

            // Unsupported conversions
            _ => Err(anyhow!("Cannot convert {:?} to {:?}", self.data_type(), to)),
        }
    }
}

fn parse_color_from_string(s: &str) -> Option<[f32; 4]> {
    let s = s.trim();

    // Try hex format first (#RRGGBB or #RRGGBBAA)
    if s.starts_with('#') {
        let hex = s.trim_start_matches('#');
        if (hex.len() == 6 || hex.len() == 8)
            && let Ok(val) = u32::from_str_radix(hex, 16)
        {
            let r = ((val >> 16) & 0xFF) as f32 / 255.0;
            let g = ((val >> 8) & 0xFF) as f32 / 255.0;
            let b = (val & 0xFF) as f32 / 255.0;
            let a = if hex.len() == 8 {
                ((val >> 24) & 0xFF) as f32 / 255.0
            } else {
                1.0
            };
            Some([r, g, b, a])
        } else {
            None
        }
    } else {
        // Try array format
        let s = s.trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = s
            .split([',', ' '])
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();

        match parts.len() {
            4 => {
                if let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<f32>(),
                ) {
                    Some([r, g, b, a])
                } else {
                    None
                }
            }
            3 => {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                ) {
                    Some([r, g, b, 1.0])
                } else {
                    None
                }
            }
            1 => {
                // Single value - grayscale
                if let Ok(v) = parts[0].parse::<f32>() {
                    Some([v, v, v, 1.0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Parse a string into an array of `N` elements.
///
/// Missing elements are filled with `T::default()`.
///
/// Returns an error if the string can't be parsed to an array of N elements.
fn parse_to_array<T, const N: usize>(input: &str) -> Result<[T; N]>
where
    T: FromStr + Default + Debug,
    <T as FromStr>::Err: Display,
{
    // Strip brackets first
    let cleaned_input = input.trim().trim_start_matches('[').trim_end_matches(']');

    let mut result = cleaned_input
        .split(&[',', ' '][..])
        .map(|s| s.trim())
        .filter(|&s| !s.is_empty())
        .take(N)
        .map(|s| {
            s.parse::<T>().map_err(|e| {
                anyhow!(
                    "Can't parse string `{s}` in `{input}` to `{}`: {e}",
                    std::any::type_name::<T>()
                )
            })
        })
        .collect::<Result<SmallVec<[T; N]>, _>>()?;

    if result.len() < N {
        result.extend((0..N - result.len()).map(|_| T::default()));
    }

    // This cannot Err
    Ok(result.into_inner().unwrap())
}

// Arithmetic operations for Data enum - generated by macros
impl_data_arithmetic!(binary Add, add, "add");
impl_data_arithmetic!(binary Sub, sub, "subtract");
impl_data_arithmetic!(scalar f32);
impl_data_arithmetic!(scalar f64);
impl_data_arithmetic!(div f32);
impl_data_arithmetic!(div f64);

// Manual Eq implementation for Data
// This is safe because we handle floating point comparison deterministically
impl Eq for Data {}
