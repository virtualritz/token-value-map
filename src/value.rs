use crate::*;
use core::num::NonZeroU16;
use std::hash::{Hash, Hasher};

/// Type alias for bracket sampling return type.
type BracketSample = (Option<(Time, Data)>, Option<(Time, Data)>);

/// A value that can be either uniform or animated over time.
///
/// A [`Value`] contains either a single [`Data`] value that remains constant
/// (uniform) or [`AnimatedData`] that changes over time with interpolation.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "facet", facet(opaque))]
#[cfg_attr(feature = "facet", repr(u8))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub enum Value {
    /// A constant value that does not change over time.
    Uniform(Data),
    /// A value that changes over time with keyframe interpolation.
    Animated(AnimatedData),
}

impl Value {
    /// Create a uniform value that does not change over time.
    pub fn uniform<V: Into<Data>>(value: V) -> Self {
        Value::Uniform(value.into())
    }

    /// Create an animated value from time-value pairs.
    ///
    /// All samples must have the same data type. Vector samples are padded
    /// to match the length of the longest vector in the set.
    pub fn animated<I, V>(samples: I) -> Result<Self>
    where
        I: IntoIterator<Item = (Time, V)>,
        V: Into<Data>,
    {
        let mut samples_vec: Vec<(Time, Data)> =
            samples.into_iter().map(|(t, v)| (t, v.into())).collect();

        if samples_vec.is_empty() {
            return Err(Error::EmptySamples);
        }

        // Get the data type from the first sample
        let data_type = samples_vec[0].1.data_type();

        // Check all samples have the same type and handle length consistency
        let mut expected_len: Option<usize> = None;
        for (time, value) in &mut samples_vec {
            if value.data_type() != data_type {
                return Err(Error::AnimatedTypeMismatch {
                    expected: data_type,
                    got: value.data_type(),
                    time: *time,
                });
            }

            // Check vector length consistency
            if let Some(vec_len) = value.try_len() {
                match expected_len {
                    None => expected_len = Some(vec_len),
                    Some(expected) => {
                        if vec_len > expected {
                            return Err(Error::VectorLengthExceeded {
                                actual: vec_len,
                                expected,
                                time: *time,
                            });
                        } else if vec_len < expected {
                            // Pad to expected length
                            value.pad_to_length(expected);
                        }
                    }
                }
            }
        }

        // Create the appropriate AnimatedData variant by extracting the
        // specific data type

        let animated_data = match data_type {
            DataType::Boolean => {
                let typed_samples: Vec<(Time, Boolean)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Boolean(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Boolean(TimeDataMap::from_iter(typed_samples))
            }
            DataType::Integer => {
                let typed_samples: Vec<(Time, Integer)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Integer(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Integer(TimeDataMap::from_iter(typed_samples))
            }
            DataType::Real => {
                let typed_samples: Vec<(Time, Real)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Real(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Real(TimeDataMap::from_iter(typed_samples))
            }
            DataType::String => {
                let typed_samples: Vec<(Time, String)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::String(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::String(TimeDataMap::from_iter(typed_samples))
            }
            DataType::Color => {
                let typed_samples: Vec<(Time, Color)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Color(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Color(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "vector2")]
            DataType::Vector2 => {
                let typed_samples: Vec<(Time, Vector2)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Vector2(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Vector2(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "vector3")]
            DataType::Vector3 => {
                let typed_samples: Vec<(Time, Vector3)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Vector3(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Vector3(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "matrix3")]
            DataType::Matrix3 => {
                let typed_samples: Vec<(Time, Matrix3)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Matrix3(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Matrix3(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "normal3")]
            DataType::Normal3 => {
                let typed_samples: Vec<(Time, Normal3)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Normal3(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Normal3(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "point3")]
            DataType::Point3 => {
                let typed_samples: Vec<(Time, Point3)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Point3(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Point3(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(feature = "matrix4")]
            DataType::Matrix4 => {
                let typed_samples: Vec<(Time, Matrix4)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Matrix4(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Matrix4(TimeDataMap::from_iter(typed_samples))
            }
            DataType::BooleanVec => {
                let typed_samples: Vec<(Time, BooleanVec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::BooleanVec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::BooleanVec(TimeDataMap::from_iter(typed_samples))
            }
            DataType::IntegerVec => {
                let typed_samples: Vec<(Time, IntegerVec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::IntegerVec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::IntegerVec(TimeDataMap::from_iter(typed_samples))
            }
            DataType::RealVec => {
                let typed_samples: Vec<(Time, RealVec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::RealVec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::RealVec(TimeDataMap::from_iter(typed_samples))
            }
            DataType::ColorVec => {
                let typed_samples: Vec<(Time, ColorVec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::ColorVec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::ColorVec(TimeDataMap::from_iter(typed_samples))
            }
            DataType::StringVec => {
                let typed_samples: Vec<(Time, StringVec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::StringVec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::StringVec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "vector2", feature = "vec_variants"))]
            DataType::Vector2Vec => {
                let typed_samples: Vec<(Time, Vector2Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Vector2Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Vector2Vec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "vector3", feature = "vec_variants"))]
            DataType::Vector3Vec => {
                let typed_samples: Vec<(Time, Vector3Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Vector3Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Vector3Vec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
            DataType::Matrix3Vec => {
                let typed_samples: Vec<(Time, Matrix3Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Matrix3Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Matrix3Vec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "normal3", feature = "vec_variants"))]
            DataType::Normal3Vec => {
                let typed_samples: Vec<(Time, Normal3Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Normal3Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Normal3Vec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "point3", feature = "vec_variants"))]
            DataType::Point3Vec => {
                let typed_samples: Vec<(Time, Point3Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Point3Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Point3Vec(TimeDataMap::from_iter(typed_samples))
            }
            #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
            DataType::Matrix4Vec => {
                let typed_samples: Vec<(Time, Matrix4Vec)> = samples_vec
                    .into_iter()
                    .map(|(t, data)| match data {
                        Data::Matrix4Vec(v) => (t, v),
                        _ => unreachable!("Type validation should have caught this"),
                    })
                    .collect();
                AnimatedData::Matrix4Vec(TimeDataMap::from_iter(typed_samples))
            }
        };

        Ok(Value::Animated(animated_data))
    }

    /// Add a sample at a specific time, checking length constraints
    pub fn add_sample<V: Into<Data>>(&mut self, time: Time, val: V) -> Result<()> {
        let value = val.into();

        match self {
            Value::Uniform(_uniform_value) => {
                // Switch to animated and drop/ignore the existing uniform
                // content Create a new animated value with only
                // the new sample
                *self = Value::animated(vec![(time, value)])?;
                Ok(())
            }
            Value::Animated(samples) => {
                let data_type = samples.data_type();
                if value.data_type() != data_type {
                    return Err(Error::SampleTypeMismatch {
                        expected: data_type,
                        got: value.data_type(),
                    });
                }

                // Insert the value using the generic insert method
                samples.try_insert(time, value)
            }
        }
    }

    /// Remove a sample at a specific time.
    ///
    /// Returns the removed value if it existed. For uniform values, this is a
    /// no-op and returns `None`. If the last sample is removed from an
    /// animated value, the value remains animated but empty.
    pub fn remove_sample(&mut self, time: &Time) -> Option<Data> {
        match self {
            Value::Uniform(_) => None,
            Value::Animated(samples) => samples.remove_at(time),
        }
    }

    /// Sample value at exact time without interpolation.
    ///
    /// Returns the exact value if it exists at the given time, or `None` if
    /// no sample exists at that time for animated values.
    pub fn sample_at(&self, time: Time) -> Option<Data> {
        match self {
            Value::Uniform(v) => Some(v.clone()),
            Value::Animated(samples) => samples.sample_at(time),
        }
    }

    /// Get the value at or before the given time
    pub fn sample_at_or_before(&self, time: Time) -> Option<Data> {
        match self {
            Value::Uniform(v) => Some(v.clone()),
            Value::Animated(_samples) => {
                // For now, use interpolation at the exact time
                // TODO: Implement proper at-or-before sampling in AnimatedData
                Some(self.interpolate(time))
            }
        }
    }

    /// Get the value at or after the given time
    pub fn sample_at_or_after(&self, time: Time) -> Option<Data> {
        match self {
            Value::Uniform(v) => Some(v.clone()),
            Value::Animated(_samples) => {
                // For now, use interpolation at the exact time
                // TODO: Implement proper at-or-after sampling in AnimatedData
                Some(self.interpolate(time))
            }
        }
    }

    /// Interpolate value at the given time.
    ///
    /// For uniform values, returns the constant value. For animated values,
    /// interpolates between surrounding keyframes using appropriate
    /// interpolation methods (linear, quadratic, or hermite).
    pub fn interpolate(&self, time: Time) -> Data {
        match self {
            Value::Uniform(v) => v.clone(),
            Value::Animated(samples) => samples.interpolate(time),
        }
    }

    /// Get surrounding samples for interpolation.
    pub fn sample_surrounding<const N: usize>(&self, time: Time) -> SmallVec<[(Time, Data); N]> {
        let mut result = SmallVec::<[(Time, Data); N]>::new_const();
        match self {
            Value::Uniform(v) => result.push((time, v.clone())),
            Value::Animated(_samples) => {
                // TODO: Implement proper surrounding sample collection for
                // AnimatedData For now, just return the
                // interpolated value at the given time
                let value = self.interpolate(time);
                result.push((time, value));
            }
        }
        result
    }

    /// Get the two samples surrounding a time for linear interpolation
    pub fn sample_bracket(&self, time: Time) -> BracketSample {
        match self {
            Value::Uniform(v) => (Some((time, v.clone())), None),
            Value::Animated(_samples) => {
                // TODO: Implement proper bracketing for AnimatedData
                // For now, just return the interpolated value at the given time
                let value = self.interpolate(time);
                (Some((time, value)), None)
            }
        }
    }

    /// Check if the value is animated.
    pub fn is_animated(&self) -> bool {
        match self {
            Value::Uniform(_) => false,
            Value::Animated(samples) => samples.is_animated(),
        }
    }

    /// Get the number of time samples.
    pub fn sample_count(&self) -> usize {
        match self {
            Value::Uniform(_) => 1,
            Value::Animated(samples) => samples.len(),
        }
    }

    /// Get all time samples.
    pub fn times(&self) -> SmallVec<[Time; 10]> {
        match self {
            Value::Uniform(_) => SmallVec::<[Time; 10]>::new_const(),
            Value::Animated(samples) => samples.times(),
        }
    }

    /// Get bezier handles at a given time.
    ///
    /// Returns None for uniform values or non-scalar types.
    #[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
    pub fn bezier_handles(&self, time: &Time) -> Option<egui_keyframe::BezierHandles> {
        match self {
            Value::Uniform(_) => None,
            Value::Animated(samples) => samples.bezier_handles(time),
        }
    }

    /// Set bezier handles at a given time.
    ///
    /// Returns an error for uniform values or non-scalar types.
    #[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
    pub fn set_bezier_handles(
        &mut self,
        time: &Time,
        handles: egui_keyframe::BezierHandles,
    ) -> Result<()> {
        match self {
            Value::Uniform(_) => Err(Error::InterpolationOnUniform),
            Value::Animated(samples) => samples.set_bezier_handles(time, handles),
        }
    }

    /// Set the interpolation type at a given time.
    ///
    /// Returns an error for uniform values or non-scalar types.
    #[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
    pub fn set_keyframe_type(
        &mut self,
        time: &Time,
        keyframe_type: egui_keyframe::KeyframeType,
    ) -> Result<()> {
        match self {
            Value::Uniform(_) => Err(Error::InterpolationOnUniform),
            Value::Animated(samples) => samples.set_keyframe_type(time, keyframe_type),
        }
    }

    /// Merge this value with another using a combiner function.
    ///
    /// For uniform values, applies the combiner once.
    /// For animated values, samples both at the union of all keyframe times
    /// and applies the combiner at each time.
    ///
    /// # Example
    /// ```ignore
    /// // Multiply two matrices
    /// let result = matrix1.merge_with(&matrix2, |a, b| {
    ///     match (a, b) {
    ///         (Data::Matrix3(m1), Data::Matrix3(m2)) => {
    ///             Data::Matrix3(Matrix3(m1.0 * m2.0))
    ///         }
    ///         _ => a, // fallback
    ///     }
    /// })?;
    /// ```
    pub fn merge_with<F>(&self, other: &Value, combiner: F) -> Result<Value>
    where
        F: Fn(&Data, &Data) -> Data,
    {
        match (self, other) {
            // Both uniform: simple case
            (Value::Uniform(a), Value::Uniform(b)) => Ok(Value::Uniform(combiner(a, b))),

            // One or both animated: need to sample at union of times
            _ => {
                // Collect all unique times from both values
                let mut all_times = std::collections::BTreeSet::new();

                // Add times from self
                for t in self.times() {
                    all_times.insert(t);
                }

                // Add times from other
                for t in other.times() {
                    all_times.insert(t);
                }

                // If no times found (both were uniform with no times), sample at default
                if all_times.is_empty() {
                    let a = self.interpolate(Time::default());
                    let b = other.interpolate(Time::default());
                    return Ok(Value::Uniform(combiner(&a, &b)));
                }

                // Sample both values at all times and combine
                let mut combined_samples = Vec::new();
                for time in all_times {
                    let a = self.interpolate(time);
                    let b = other.interpolate(time);
                    let combined = combiner(&a, &b);
                    combined_samples.push((time, combined));
                }

                // If only one sample, return as uniform
                if combined_samples.len() == 1 {
                    Ok(Value::Uniform(combined_samples[0].1.clone()))
                } else {
                    // Create animated value from combined samples
                    Value::animated(combined_samples)
                }
            }
        }
    }
}

// From implementations for Value
impl<V: Into<Data>> From<V> for Value {
    fn from(value: V) -> Self {
        Value::uniform(value)
    }
}

// Sample trait implementations for Value using macro
#[cfg(feature = "vector2")]
impl_sample_for_value!(Vector2, Vector2);
#[cfg(feature = "vector3")]
impl_sample_for_value!(Vector3, Vector3);
impl_sample_for_value!(Color, Color);
#[cfg(feature = "matrix3")]
impl_sample_for_value!(Matrix3, Matrix3);
#[cfg(feature = "normal3")]
impl_sample_for_value!(Normal3, Normal3);
#[cfg(feature = "point3")]
impl_sample_for_value!(Point3, Point3);
#[cfg(feature = "matrix4")]
impl_sample_for_value!(Matrix4, Matrix4);

// Special implementations for Real and Integer that handle type conversion
impl Sample<Real> for Value {
    fn sample(&self, shutter: &Shutter, samples: NonZeroU16) -> Result<Vec<(Real, SampleWeight)>> {
        match self {
            Value::Uniform(data) => {
                let value = Real(data.to_f32()? as f64);
                Ok(vec![(value, 1.0)])
            }
            Value::Animated(animated_data) => animated_data.sample(shutter, samples),
        }
    }
}

impl Sample<Integer> for Value {
    fn sample(
        &self,
        shutter: &Shutter,
        samples: NonZeroU16,
    ) -> Result<Vec<(Integer, SampleWeight)>> {
        match self {
            Value::Uniform(data) => {
                let value = Integer(data.to_i64()?);
                Ok(vec![(value, 1.0)])
            }
            Value::Animated(animated_data) => animated_data.sample(shutter, samples),
        }
    }
}

// Manual Eq implementation for Value
// This is safe because we handle floating point comparison deterministically
impl Eq for Value {}

impl Value {
    /// Hash the value with shutter context for animation-aware caching.
    ///
    /// For animated values, this samples at standardized points within the shutter
    /// range and hashes the interpolated values rather than raw keyframes.
    /// This provides better cache coherency for animations with different absolute
    /// times but identical interpolated values.
    pub fn hash_with_shutter<H: Hasher>(&self, state: &mut H, shutter: &Shutter) {
        match self {
            Value::Uniform(data) => {
                // For uniform values, just use regular hashing.
                data.hash(state);
            }
            Value::Animated(animated) => {
                // For animated values, sample at standardized points.
                animated.hash_with_shutter(state, shutter);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "matrix3")]
    #[test]
    fn test_matrix_merge_uniform() {
        // Create two uniform matrices
        let m1 = crate::math::mat3_from_row_slice(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]); // Scale by 2
        let m2 = crate::math::mat3_from_row_slice(&[1.0, 0.0, 10.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0]); // Translate by (10, 20)

        let v1 = Value::uniform(m1);
        let v2 = Value::uniform(m2);

        // Merge them with multiplication
        let result = v1
            .merge_with(&v2, |a, b| match (a, b) {
                (Data::Matrix3(ma), Data::Matrix3(mb)) => Data::Matrix3(ma.clone() * mb.clone()),
                _ => a.clone(),
            })
            .unwrap();

        // Check result is uniform
        if let Value::Uniform(Data::Matrix3(result_matrix)) = result {
            let expected = m1 * m2;
            assert_eq!(result_matrix.0, expected);
        } else {
            panic!("Expected uniform result");
        }
    }

    #[cfg(feature = "matrix3")]
    #[test]
    fn test_matrix_merge_animated() {
        use frame_tick::Tick;

        // Create first animated matrix (rotation)
        let m1_t0 =
            crate::math::mat3_from_row_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]); // Identity
        let m1_t10 =
            crate::math::mat3_from_row_slice(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]); // 90 degree rotation

        let v1 = Value::animated([
            (Tick::from_secs(0.0), m1_t0),
            (Tick::from_secs(10.0), m1_t10),
        ])
        .unwrap();

        // Create second animated matrix (scale)
        let m2_t5 =
            crate::math::mat3_from_row_slice(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
        let m2_t15 =
            crate::math::mat3_from_row_slice(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);

        let v2 = Value::animated([
            (Tick::from_secs(5.0), m2_t5),
            (Tick::from_secs(15.0), m2_t15),
        ])
        .unwrap();

        // Merge them
        let result = v1
            .merge_with(&v2, |a, b| match (a, b) {
                (Data::Matrix3(ma), Data::Matrix3(mb)) => Data::Matrix3(ma.clone() * mb.clone()),
                _ => a.clone(),
            })
            .unwrap();

        // Check that result is animated with samples at t=0, 5, 10, 15
        if let Value::Animated(animated) = result {
            let times = animated.times();
            assert_eq!(times.len(), 4);
            assert!(times.contains(&Tick::from_secs(0.0)));
            assert!(times.contains(&Tick::from_secs(5.0)));
            assert!(times.contains(&Tick::from_secs(10.0)));
            assert!(times.contains(&Tick::from_secs(15.0)));
        } else {
            panic!("Expected animated result");
        }
    }
}
