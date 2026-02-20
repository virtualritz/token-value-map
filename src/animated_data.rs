use crate::{
    macros::{impl_animated_data_insert, impl_data_type_ops, impl_sample_for_animated_data},
    time_data_map::TimeDataMapControl,
    *,
};

use crate::Result;
use core::num::NonZeroU16;
use enum_dispatch::enum_dispatch;
use smallvec::SmallVec;
use std::hash::Hasher;

/// Time-indexed data with interpolation support.
///
/// [`AnimatedData`] `enum` stores a collection of time-value pairs for a
/// specific data type and provides interpolation between keyframes. Each
/// variant contains a [`TimeDataMap`] for the corresponding data type.
#[enum_dispatch(AnimatedDataOps)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "facet", facet(opaque))]
#[cfg_attr(feature = "facet", repr(u8))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
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
    /// Animated 4×4 transformation matrices.
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
    /// Animated 4×4 matrix arrays.
    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
    Matrix4Vec(TimeDataMap<Matrix4Vec>),
    /// Animated real curve.
    #[cfg(feature = "curves")]
    RealCurve(TimeDataMap<RealCurve>),
    /// Animated color curve.
    #[cfg(feature = "curves")]
    ColorCurve(TimeDataMap<ColorCurve>),
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
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn is_animated(&self) -> bool {
        self.values.len() > 1
    }
}

impl_data_type_ops!(AnimatedData);

// Interpolation-specific methods (when feature enabled)
#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
impl AnimatedData {
    /// Get bezier handles at a given time.
    ///
    /// Returns handles for scalar types (Real, Integer). Other types return None.
    pub fn bezier_handles(&self, time: &Time) -> Option<egui_keyframe::BezierHandles> {
        match self {
            AnimatedData::Real(map) => map.interpolation(time).map(crate::key_to_bezier_handles),
            AnimatedData::Integer(map) => map.interpolation(time).map(crate::key_to_bezier_handles),
            // Vector/matrix types don't support scalar bezier handles
            _ => None,
        }
    }

    /// Set bezier handles at a given time.
    ///
    /// Works for scalar types (Real, Integer). Other types return an error.
    pub fn set_bezier_handles(
        &mut self,
        time: &Time,
        handles: egui_keyframe::BezierHandles,
    ) -> Result<()> {
        match self {
            AnimatedData::Real(map) => {
                map.set_interpolation_at(time, crate::bezier_handles_to_key(&handles))
            }
            AnimatedData::Integer(map) => {
                map.set_interpolation_at(time, crate::bezier_handles_to_key(&handles))
            }
            _ => Err(Error::BezierNotSupported {
                got: self.data_type(),
            }),
        }
    }

    /// Set the interpolation type at a given time.
    ///
    /// Works for scalar types (Real, Integer). Other types return an error.
    pub fn set_keyframe_type(
        &mut self,
        time: &Time,
        keyframe_type: egui_keyframe::KeyframeType,
    ) -> Result<()> {
        use crate::interpolation::{Interpolation, Key};

        match keyframe_type {
            egui_keyframe::KeyframeType::Hold => match self {
                AnimatedData::Real(map) => map.set_interpolation_at(
                    time,
                    Key {
                        interpolation_in: Interpolation::Hold,
                        interpolation_out: Interpolation::Hold,
                    },
                ),
                AnimatedData::Integer(map) => map.set_interpolation_at(
                    time,
                    Key {
                        interpolation_in: Interpolation::Hold,
                        interpolation_out: Interpolation::Hold,
                    },
                ),
                _ => Err(Error::BezierNotSupported {
                    got: self.data_type(),
                }),
            },
            egui_keyframe::KeyframeType::Linear => match self {
                AnimatedData::Real(map) => map.set_interpolation_at(
                    time,
                    Key {
                        interpolation_in: Interpolation::Linear,
                        interpolation_out: Interpolation::Linear,
                    },
                ),
                AnimatedData::Integer(map) => map.set_interpolation_at(
                    time,
                    Key {
                        interpolation_in: Interpolation::Linear,
                        interpolation_out: Interpolation::Linear,
                    },
                ),
                _ => Err(Error::BezierNotSupported {
                    got: self.data_type(),
                }),
            },
            egui_keyframe::KeyframeType::Bezier => {
                // Default bezier handles (smooth curve)
                self.set_bezier_handles(time, egui_keyframe::BezierHandles::default())
            }
        }
    }
}

impl AnimatedData {
    /// Get all time samples from this animated data.
    pub fn times(&self) -> SmallVec<[Time; 10]> {
        match self {
            AnimatedData::Boolean(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::Integer(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::Real(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::String(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::Color(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::BooleanVec(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::IntegerVec(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::RealVec(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::ColorVec(map) => map.iter().map(|(t, _)| *t).collect(),
            AnimatedData::StringVec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "curves")]
            AnimatedData::RealCurve(map) => map.iter().map(|(t, _)| *t).collect(),
            #[cfg(feature = "curves")]
            AnimatedData::ColorCurve(map) => map.iter().map(|(t, _)| *t).collect(),
        }
    }
}

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
            #[cfg(feature = "curves")]
            Data::RealCurve(v) => AnimatedData::RealCurve(TimeDataMap::from((time, v))),
            #[cfg(feature = "curves")]
            Data::ColorCurve(v) => AnimatedData::ColorCurve(TimeDataMap::from((time, v))),
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

#[cfg(feature = "normal3")]
impl_animated_data_insert!(
    insert_normal3, Normal3, Normal3;
);

#[cfg(feature = "point3")]
impl_animated_data_insert!(
    insert_point3, Point3, Point3;
);

#[cfg(feature = "matrix4")]
impl_animated_data_insert!(
    insert_matrix4, Matrix4, Matrix4;
);

impl AnimatedData {
    /// Generic insert method that takes `Data` and matches the type to the
    /// `AnimatedData` variant.
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
            #[cfg(feature = "curves")]
            (AnimatedData::RealCurve(map), Data::RealCurve(v)) => {
                map.insert(time, v);
                Ok(())
            }
            #[cfg(feature = "curves")]
            (AnimatedData::ColorCurve(map), Data::ColorCurve(v)) => {
                map.insert(time, v);
                Ok(())
            }
            (s, v) => Err(Error::SampleTypeMismatch {
                expected: s.data_type(),
                got: v.data_type(),
            }),
        }
    }

    /// Remove a sample at the given time.
    ///
    /// Returns the removed value as `Data` if it existed.
    pub fn remove_at(&mut self, time: &Time) -> Option<Data> {
        match self {
            AnimatedData::Boolean(map) => map.remove(time).map(Data::Boolean),
            AnimatedData::Integer(map) => map.remove(time).map(Data::Integer),
            AnimatedData::Real(map) => map.remove(time).map(Data::Real),
            AnimatedData::String(map) => map.remove(time).map(Data::String),
            AnimatedData::Color(map) => map.remove(time).map(Data::Color),
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => map.remove(time).map(Data::Vector2),
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => map.remove(time).map(Data::Vector3),
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => map.remove(time).map(Data::Matrix3),
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => map.remove(time).map(Data::Normal3),
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => map.remove(time).map(Data::Point3),
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => map.remove(time).map(Data::Matrix4),
            AnimatedData::BooleanVec(map) => map.remove(time).map(Data::BooleanVec),
            AnimatedData::IntegerVec(map) => map.remove(time).map(Data::IntegerVec),
            AnimatedData::RealVec(map) => map.remove(time).map(Data::RealVec),
            AnimatedData::ColorVec(map) => map.remove(time).map(Data::ColorVec),
            AnimatedData::StringVec(map) => map.remove(time).map(Data::StringVec),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => map.remove(time).map(Data::Vector2Vec),
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => map.remove(time).map(Data::Vector3Vec),
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => map.remove(time).map(Data::Matrix3Vec),
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => map.remove(time).map(Data::Normal3Vec),
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => map.remove(time).map(Data::Point3Vec),
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => map.remove(time).map(Data::Matrix4Vec),
            #[cfg(feature = "curves")]
            AnimatedData::RealCurve(map) => map.remove(time).map(Data::RealCurve),
            #[cfg(feature = "curves")]
            AnimatedData::ColorCurve(map) => map.remove(time).map(Data::ColorCurve),
        }
    }

    pub fn sample_at(&self, time: Time) -> Option<Data> {
        match self {
            AnimatedData::Boolean(map) => map.get(&time).map(|v| Data::Boolean(v.clone())),
            AnimatedData::Integer(map) => map.get(&time).map(|v| Data::Integer(v.clone())),
            AnimatedData::Real(map) => map.get(&time).map(|v| Data::Real(v.clone())),
            AnimatedData::String(map) => map.get(&time).map(|v| Data::String(v.clone())),
            AnimatedData::Color(map) => map.get(&time).map(|v| Data::Color(*v)),
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
            #[cfg(feature = "curves")]
            AnimatedData::RealCurve(map) => map.get(&time).map(|v| Data::RealCurve(v.clone())),
            #[cfg(feature = "curves")]
            AnimatedData::ColorCurve(map) => map.get(&time).map(|v| Data::ColorCurve(v.clone())),
        }
    }

    pub fn interpolate(&self, time: Time) -> Data {
        match self {
            AnimatedData::Boolean(map) => Data::Boolean(map.closest_sample(time).clone()),
            AnimatedData::Integer(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Integer(map.interpolate(time))
                } else {
                    Data::Integer(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::Real(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Real(map.interpolate(time))
                } else {
                    Data::Real(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::String(map) => Data::String(map.closest_sample(time).clone()),
            AnimatedData::Color(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Color(map.interpolate(time))
                } else {
                    Data::Color(*map.iter().next().unwrap().1)
                }
            }
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Vector2(map.interpolate(time))
                } else {
                    Data::Vector2(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Vector3(map.interpolate(time))
                } else {
                    Data::Vector3(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Matrix3(map.interpolate(time))
                } else {
                    Data::Matrix3(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Normal3(map.interpolate(time))
                } else {
                    Data::Normal3(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Point3(map.interpolate(time))
                } else {
                    Data::Point3(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Matrix4(map.interpolate(time))
                } else {
                    Data::Matrix4(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::BooleanVec(map) => Data::BooleanVec(map.closest_sample(time).clone()),
            AnimatedData::IntegerVec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::IntegerVec(map.interpolate(time))
                } else {
                    Data::IntegerVec(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::RealVec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::RealVec(map.interpolate(time))
                } else {
                    Data::RealVec(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::ColorVec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::ColorVec(map.interpolate(time))
                } else {
                    Data::ColorVec(map.iter().next().unwrap().1.clone())
                }
            }
            AnimatedData::StringVec(map) => Data::StringVec(map.closest_sample(time).clone()),
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Vector2Vec(map.interpolate(time))
                } else {
                    Data::Vector2Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Vector3Vec(map.interpolate(time))
                } else {
                    Data::Vector3Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Matrix3Vec(map.interpolate(time))
                } else {
                    Data::Matrix3Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Normal3Vec(map.interpolate(time))
                } else {
                    Data::Normal3Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Point3Vec(map.interpolate(time))
                } else {
                    Data::Point3Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => {
                if TimeDataMapControl::is_animated(map) {
                    Data::Matrix4Vec(map.interpolate(time))
                } else {
                    Data::Matrix4Vec(map.iter().next().unwrap().1.clone())
                }
            }
            #[cfg(feature = "curves")]
            AnimatedData::RealCurve(map) => Data::RealCurve(map.closest_sample(time).clone()),
            #[cfg(feature = "curves")]
            AnimatedData::ColorCurve(map) => Data::ColorCurve(map.closest_sample(time).clone()),
        }
    }
}

impl Hash for AnimatedData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            AnimatedData::Boolean(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::Integer(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::Real(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.to_bits().hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.to_bits().hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::String(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::Color(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "vector2")]
            AnimatedData::Vector2(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::vec2_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::vec2_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "vector3")]
            AnimatedData::Vector3(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::vec3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::vec3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "matrix3")]
            AnimatedData::Matrix3(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::mat3_iter(&value.0).for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::mat3_iter(&value.0).for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "normal3")]
            AnimatedData::Normal3(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::vec3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::vec3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "point3")]
            AnimatedData::Point3(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::point3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::point3_as_slice(&value.0)
                        .iter()
                        .for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            #[cfg(feature = "matrix4")]
            AnimatedData::Matrix4(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    crate::math::mat4_iter(&value.0).for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    crate::math::mat4_iter(&value.0).for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            AnimatedData::BooleanVec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::IntegerVec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            AnimatedData::RealVec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| v.to_bits().hash(state));
                    spec.hash(state);
                }
            }
            AnimatedData::ColorVec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|c| {
                        c.iter().for_each(|v| v.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|c| {
                        c.iter().for_each(|v| v.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            AnimatedData::StringVec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.hash(state);
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            AnimatedData::Vector2Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec2_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec2_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            AnimatedData::Vector3Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec3_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec3_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            AnimatedData::Matrix3Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        crate::math::mat3_iter(m).for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        crate::math::mat3_iter(m).for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            AnimatedData::Normal3Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec3_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|v| {
                        crate::math::vec3_as_slice(v)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            AnimatedData::Point3Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|p| {
                        crate::math::point3_as_slice(p)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|p| {
                        crate::math::point3_as_slice(p)
                            .iter()
                            .for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            AnimatedData::Matrix4Vec(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        crate::math::mat4_iter(m).for_each(|f| f.to_bits().hash(state));
                    });
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.0.len().hash(state);
                    value.0.iter().for_each(|m| {
                        crate::math::mat4_iter(m).for_each(|f| f.to_bits().hash(state));
                    });
                    spec.hash(state);
                }
            }
            #[cfg(feature = "curves")]
            AnimatedData::RealCurve(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.hash(state);
                    spec.hash(state);
                }
            }
            #[cfg(feature = "curves")]
            AnimatedData::ColorCurve(map) => {
                map.len().hash(state);
                #[cfg(not(feature = "interpolation"))]
                for (time, value) in &map.values {
                    time.hash(state);
                    value.hash(state);
                }
                #[cfg(feature = "interpolation")]
                for (time, (value, spec)) in &map.values {
                    time.hash(state);
                    value.hash(state);
                    spec.hash(state);
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

impl AnimatedData {
    /// Hash the animated data with shutter context for better cache coherency.
    ///
    /// Samples the animation at standardized points within the shutter range
    /// and hashes the interpolated values. If all samples are identical,
    /// only one value is hashed for efficiency.
    pub fn hash_with_shutter<H: Hasher>(&self, state: &mut H, shutter: &Shutter) {
        use smallvec::SmallVec;

        // Sample at 5 standardized points within the shutter.
        const SAMPLE_POSITIONS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

        // Collect interpolated samples using SmallVec to avoid heap allocation.
        let samples: SmallVec<[Data; 5]> = SAMPLE_POSITIONS
            .iter()
            .map(|&pos| {
                let time = shutter.evaluate(pos);
                self.interpolate(time)
            })
            .collect();

        // Check if all samples are identical.
        let all_same = samples.windows(2).all(|w| w[0] == w[1]);

        // Hash the data type discriminant.
        std::mem::discriminant(self).hash(state);

        if all_same {
            // If all samples are the same, just hash one.
            1usize.hash(state); // Indicate single sample.
            samples[0].hash(state);
        } else {
            // Hash all samples.
            samples.len().hash(state); // Indicate multiple samples.
            for sample in &samples {
                sample.hash(state);
            }
        }
    }
}

// Implement AnimatedDataSystem trait for the built-in AnimatedData type.
impl crate::traits::AnimatedDataSystem for AnimatedData {
    type Data = Data;

    fn keyframe_count(&self) -> usize {
        AnimatedDataOps::len(self)
    }

    fn is_keyframes_empty(&self) -> bool {
        AnimatedDataOps::is_empty(self)
    }

    fn has_animation(&self) -> bool {
        AnimatedDataOps::is_animated(self)
    }

    fn times(&self) -> SmallVec<[Time; 10]> {
        AnimatedData::times(self)
    }

    fn interpolate(&self, time: Time) -> Data {
        AnimatedData::interpolate(self, time)
    }

    fn sample_at(&self, time: Time) -> Option<Data> {
        AnimatedData::sample_at(self, time)
    }

    fn try_insert(&mut self, time: Time, value: Data) -> Result<()> {
        AnimatedData::try_insert(self, time, value)
    }

    fn remove_at(&mut self, time: &Time) -> Option<Data> {
        AnimatedData::remove_at(self, time)
    }

    fn discriminant(&self) -> DataType {
        DataTypeOps::data_type(self)
    }

    fn from_single(time: Time, value: Data) -> Self {
        AnimatedData::from((time, value))
    }

    fn variant_name(&self) -> &'static str {
        DataTypeOps::type_name(self)
    }
}
