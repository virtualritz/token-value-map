use crate::{
    macros::{impl_animated_data_insert, impl_data_type_ops, impl_sample_for_animated_data},
    *,
};
use anyhow::Result;
use core::num::NonZeroU16;
use enum_dispatch::enum_dispatch;

/// Time-indexed data with interpolation support.
///
/// [`AnimatedData`] `enum` stores a collection of time-value pairs for a
/// specific data type and provides interpolation between keyframes. Each
/// variant contains a [`TimeDataMap`] for the corresponding data type.
#[enum_dispatch(AnimatedDataOps)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnimatedData {
    /// Animated boolean values.
    Boolean(TimeDataMap<Boolean>),
    /// Animated integer values.
    Integer(TimeDataMap<Integer>),
    /// Animated real values.
    Real(TimeDataMap<Real>),
    /// Animated string values.
    String(TimeDataMap<String>),
    /// Animated color values.
    Color(TimeDataMap<Color>),
    /// Animated 2D vectors.
    #[cfg(feature = "vector2")]
    Vector2(TimeDataMap<Vector2>),
    /// Animated 3D vectors.
    #[cfg(feature = "vector3")]
    Vector3(TimeDataMap<Vector3>),
    /// Animated transformation matrices.
    #[cfg(feature = "matrix3")]
    Matrix3(TimeDataMap<Matrix3>),
    /// Animated 3D normal vectors.
    #[cfg(feature = "normal3")]
    Normal3(TimeDataMap<Normal3>),
    /// Animated 3D points.
    #[cfg(feature = "point3")]
    Point3(TimeDataMap<Point3>),
    /// Animated 4x4 transformation matrices.
    #[cfg(feature = "matrix4")]
    Matrix4(TimeDataMap<Matrix4>),
    /// Animated boolean vectors.
    BooleanVec(TimeDataMap<BooleanVec>),
    /// Animated integer vectors.
    IntegerVec(TimeDataMap<IntegerVec>),
    /// Animated real vectors.
    RealVec(TimeDataMap<RealVec>),
    /// Animated color vectors.
    ColorVec(TimeDataMap<ColorVec>),
    /// Animated string vectors.
    StringVec(TimeDataMap<StringVec>),
    /// Animated 2D vector arrays.
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    Vector2Vec(TimeDataMap<Vector2Vec>),
    /// Animated 3D vector arrays.
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    Vector3Vec(TimeDataMap<Vector3Vec>),
    /// Animated matrix arrays.
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    Matrix3Vec(TimeDataMap<Matrix3Vec>),
    /// Animated 3D normal arrays.
    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
    Normal3Vec(TimeDataMap<Normal3Vec>),
    /// Animated 3D point arrays.
    #[cfg(all(feature = "point3", feature = "vec_variants"))]
    Point3Vec(TimeDataMap<Point3Vec>),
    /// Animated 4x4 matrix arrays.
    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
    Matrix4Vec(TimeDataMap<Matrix4Vec>),
}

/// Common operations `trait` for animated data types.
#[enum_dispatch]
pub trait AnimatedDataOps {
    /// Returns the number of time samples.
    fn len(&self) -> usize;
    /// Returns `true` if there are no time samples.
    fn is_empty(&self) -> bool;
    /// Returns `true` if there is more than one time sample.
    fn is_animated(&self) -> bool;
}

impl<T> AnimatedDataOps for TimeDataMap<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn is_animated(&self) -> bool {
        self.0.len() > 1
    }
}

impl_data_type_ops!(AnimatedData);

impl From<(Time, Value)> for AnimatedData {
    fn from((time, value): (Time, Value)) -> Self {
        match value {
            Value::Uniform(data) => AnimatedData::from((time, data)),
            Value::Animated(animated_data) => {
                // If the value is already animated, we need to insert this time
                // sample For simplicity, we'll just return the
                // animated data as-is In a more sophisticated
                // implementation, we might want to merge or replace samples
                animated_data
            }
        }
    }
}

impl From<(Time, Data)> for AnimatedData {
    fn from((time, data): (Time, Data)) -> Self {
        match data {
            Data::Boolean(v) => AnimatedData::Boolean(TimeDataMap::from((time, v))),
            Data::Integer(v) => AnimatedData::Integer(TimeDataMap::from((time, v))),
            Data::Real(v) => AnimatedData::Real(TimeDataMap::from((time, v))),
            Data::String(v) => AnimatedData::String(TimeDataMap::from((time, v))),
            Data::Color(v) => AnimatedData::Color(TimeDataMap::from((time, v))),
            #[cfg(feature = "vector2")]
            Data::Vector2(v) => AnimatedData::Vector2(TimeDataMap::from((time, v))),
            #[cfg(feature = "vector3")]
            Data::Vector3(v) => AnimatedData::Vector3(TimeDataMap::from((time, v))),
            #[cfg(feature = "matrix3")]
            Data::Matrix3(v) => AnimatedData::Matrix3(TimeDataMap::from((time, v))),
            #[cfg(feature = "normal3")]
            Data::Normal3(v) => AnimatedData::Normal3(TimeDataMap::from((time, v))),
            #[cfg(feature = "point3")]
            Data::Point3(v) => AnimatedData::Point3(TimeDataMap::from((time, v))),
            #[cfg(feature = "matrix4")]
            Data::Matrix4(v) => AnimatedData::Matrix4(TimeDataMap::from((time, v))),
            Data::BooleanVec(v) => AnimatedData::BooleanVec(TimeDataMap::from((time, v))),
            Data::IntegerVec(v) => AnimatedData::IntegerVec(TimeDataMap::from((time, v))),
            Data::RealVec(v) => AnimatedData::RealVec(TimeDataMap::from((time, v))),
            Data::ColorVec(v) => AnimatedData::ColorVec(TimeDataMap::from((time, v))),
            Data::StringVec(v) => AnimatedData::StringVec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            Data::Vector2Vec(v) => AnimatedData::Vector2Vec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            Data::Vector3Vec(v) => AnimatedData::Vector3Vec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            Data::Matrix3Vec(v) => AnimatedData::Matrix3Vec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            Data::Normal3Vec(v) => AnimatedData::Normal3Vec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            Data::Point3Vec(v) => AnimatedData::Point3Vec(TimeDataMap::from((time, v))),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            Data::Matrix4Vec(v) => AnimatedData::Matrix4Vec(TimeDataMap::from((time, v))),
        }
    }
}

impl_animated_data_insert!(
    insert_boolean, Boolean, Boolean;
    insert_integer, Integer, Integer;
    insert_real, Real, Real;
    insert_string, String, String;
    insert_color, Color, Color;
);

#[cfg(feature = "vector2")]
impl_animated_data_insert!(
    insert_vector2, Vector2, Vector2;
);

#[cfg(feature = "vector3")]
impl_animated_data_insert!(
    insert_vector3, Vector3, Vector3;
);

#[cfg(feature = "matrix3")]
impl_animated_data_insert!(
    insert_matrix3, Matrix3, Matrix3;
);

impl AnimatedData {
    /// Generic insert method that takes `Data` and matches the type to the
    /// `AnimatedData` variant.
    #[named]
    pub fn try_insert(&mut self, time: Time, value: Data) -> Result<()> {
        match (self, value) {
            (AnimatedData::Boolean(map), Data::Boolean(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::Integer(map), Data::Integer(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::Real(map), Data::Real(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::String(map), Data::String(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::Color(map), Data::Color(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "vector2")]
            (AnimatedData::Vector2(map), Data::Vector2(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "vector3")]
            (AnimatedData::Vector3(map), Data::Vector3(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "matrix3")]
            (AnimatedData::Matrix3(map), Data::Matrix3(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "normal3")]
            (AnimatedData::Normal3(map), Data::Normal3(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "point3")]
            (AnimatedData::Point3(map), Data::Point3(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "matrix4")]
            (AnimatedData::Matrix4(map), Data::Matrix4(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::BooleanVec(map), Data::BooleanVec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::IntegerVec(map), Data::IntegerVec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::RealVec(map), Data::RealVec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::ColorVec(map), Data::ColorVec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (AnimatedData::StringVec(map), Data::StringVec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            (AnimatedData::Vector2Vec(map), Data::Vector2Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            (AnimatedData::Vector3Vec(map), Data::Vector3Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            (AnimatedData::Matrix3Vec(map), Data::Matrix3Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            (AnimatedData::Normal3Vec(map), Data::Normal3Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            (AnimatedData::Point3Vec(map), Data::Point3Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            (AnimatedData::Matrix4Vec(map), Data::Matrix4Vec(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (s, v) => Err(anyhow!(
                "{}: type mismatch: {:?} variant does not match {:?} type",
                function_name!(),
                s.data_type(),
                v.data_type()
            )),
        }
    }

    pub fn sample_at(&self, time: Time) -> Option<Data> {
        match self {
            AnimatedData::Boolean(map) => map.get(&time).map(|v| Data::Boolean(v.clone())),
            AnimatedData::Integer(map) => map.get(&time).map(|v| Data::Integer(v.clone())),
            AnimatedData::Real(map) => map.get(&time).map(|v| Data::Real(v.clone())),
            AnimatedData::String(map) => map.get(&time).map(|v| Data::String(v.clone())),
            AnimatedData::Color(map) => map.get(&time).map(|v| Data::Color(v.clone())),
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => map.get(&time).map(|v| Data::Vector2(v.clone())),
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => map.get(&time).map(|v| Data::Vector3(v.clone())),
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => map.get(&time).map(|v| Data::Matrix3(v.clone())),
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => map.get(&time).map(|v| Data::Normal3(v.clone())),
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => map.get(&time).map(|v| Data::Point3(v.clone())),
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => map.get(&time).map(|v| Data::Matrix4(v.clone())),
            AnimatedData::BooleanVec(map) => map.get(&time).map(|v| Data::BooleanVec(v.clone())),
            AnimatedData::IntegerVec(map) => map.get(&time).map(|v| Data::IntegerVec(v.clone())),
            AnimatedData::RealVec(map) => map.get(&time).map(|v| Data::RealVec(v.clone())),
            AnimatedData::ColorVec(map) => map.get(&time).map(|v| Data::ColorVec(v.clone())),
            AnimatedData::StringVec(map) => map.get(&time).map(|v| Data::StringVec(v.clone())),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => map.get(&time).map(|v| Data::Vector2Vec(v.clone())),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => map.get(&time).map(|v| Data::Vector3Vec(v.clone())),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => map.get(&time).map(|v| Data::Matrix3Vec(v.clone())),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => map.get(&time).map(|v| Data::Normal3Vec(v.clone())),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => map.get(&time).map(|v| Data::Point3Vec(v.clone())),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => map.get(&time).map(|v| Data::Matrix4Vec(v.clone())),
        }
    }

    pub fn interpolate(&self, time: Time) -> Data {
        match self {
            AnimatedData::Boolean(map) => Data::Boolean(map.closest_sample(time).clone()),
            AnimatedData::Integer(map) => {
                if map.is_animated() {
                    Data::Integer(map.interpolate(time))
                } else {
                    Data::Integer(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::Real(map) => {
                if map.is_animated() {
                    Data::Real(map.interpolate(time))
                } else {
                    Data::Real(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::String(map) => Data::String(map.closest_sample(time).clone()),
            AnimatedData::Color(map) => {
                if map.is_animated() {
                    Data::Color(map.interpolate(time))
                } else {
                    Data::Color(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => {
                if map.is_animated() {
                    Data::Vector2(map.interpolate(time))
                } else {
                    Data::Vector2(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => {
                if map.is_animated() {
                    Data::Vector3(map.interpolate(time))
                } else {
                    Data::Vector3(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => {
                if map.is_animated() {
                    Data::Matrix3(map.interpolate(time))
                } else {
                    Data::Matrix3(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => {
                if map.is_animated() {
                    Data::Normal3(map.interpolate(time))
                } else {
                    Data::Normal3(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => {
                if map.is_animated() {
                    Data::Point3(map.interpolate(time))
                } else {
                    Data::Point3(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => {
                if map.is_animated() {
                    Data::Matrix4(map.interpolate(time))
                } else {
                    Data::Matrix4(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::BooleanVec(map) => Data::BooleanVec(map.closest_sample(time).clone()),
            AnimatedData::IntegerVec(map) => {
                if map.is_animated() {
                    Data::IntegerVec(map.interpolate(time))
                } else {
                    Data::IntegerVec(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::RealVec(map) => {
                if map.is_animated() {
                    Data::RealVec(map.interpolate(time))
                } else {
                    Data::RealVec(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::ColorVec(map) => {
                if map.is_animated() {
                    Data::ColorVec(map.interpolate(time))
                } else {
                    Data::ColorVec(map.0.values().next().unwrap().clone())
                }
            }
            AnimatedData::StringVec(map) => Data::StringVec(map.closest_sample(time).clone()),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => {
                if map.is_animated() {
                    Data::Vector2Vec(map.interpolate(time))
                } else {
                    Data::Vector2Vec(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => {
                if map.is_animated() {
                    Data::Vector3Vec(map.interpolate(time))
                } else {
                    Data::Vector3Vec(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => {
                if map.is_animated() {
                    Data::Matrix3Vec(map.interpolate(time))
                } else {
                    Data::Matrix3Vec(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => {
                if map.is_animated() {
                    Data::Normal3Vec(map.interpolate(time))
                } else {
                    Data::Normal3Vec(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => {
                if map.is_animated() {
                    Data::Point3Vec(map.interpolate(time))
                } else {
                    Data::Point3Vec(map.0.values().next().unwrap().clone())
                }
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => {
                if map.is_animated() {
                    Data::Matrix4Vec(map.interpolate(time))
                } else {
                    Data::Matrix4Vec(map.0.values().next().unwrap().clone())
                }
            }
        }
    }
}

impl Hash for AnimatedData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            AnimatedData::Boolean(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            AnimatedData::Integer(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            AnimatedData::Real(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.to_bits().hash(state);
                }
            }
            AnimatedData::String(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            AnimatedData::Color(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.coords.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            AnimatedData::BooleanVec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            AnimatedData::IntegerVec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            AnimatedData::RealVec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
            }
            AnimatedData::ColorVec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|c| {
                        c.iter().for_each(|v| v.to_bits().hash(state));
                    });
                }
            }
            AnimatedData::StringVec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.hash(state);
                }
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        v.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        v.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        m.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        v.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|p| {
                        p.coords.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => {
                map.0.len().hash(state);
                for (time, value) in &map.0 {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        m.iter().for_each(|f| f.to_bits().hash(state));
                    });
                }
            }
        }
    }
}

// Sample trait implementations for AnimatedData
impl_sample_for_animated_data!(
    Real, Real;
    Integer, Integer;
    Color, Color;
);

impl_sample_for_animated_data!(
    Vector2, Vector2, "vector2";
    Vector3, Vector3, "vector3";
    Matrix3, Matrix3, "matrix3";
    Normal3, Normal3, "normal3";
    Point3, Point3, "point3";
    Matrix4, Matrix4, "matrix4";
);
