use crate::*;
use core::num::NonZeroU16;

/// Type alias for bracket sampling return type.
type BracketSample = (Option<(Time, Data)>, Option<(Time, Data)>);

/// A value that can be either uniform or animated over time.
///
/// A [`Value`] contains either a single [`Data`] value that remains constant
/// (uniform) or [`AnimatedData`] that changes over time with interpolation.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
            return Err(anyhow!("Cannot create animated value with no samples"));
        }

        // Get the data type from the first sample
        let data_type = samples_vec[0].1.data_type();

        // Check all samples have the same type and handle length consistency
        let mut expected_len: Option<usize> = None;
        for (time, value) in &mut samples_vec {
            if value.data_type() != data_type {
                return Err(anyhow!(
                    "All animated samples must have the same type. Expected {:?}, found {:?} at time {}",
                    data_type,
                    value.data_type(),
                    time
                ));
            }

            // Check vector length consistency
            if let Some(vec_len) = value.try_len() {
                match expected_len {
                    None => expected_len = Some(vec_len),
                    Some(expected) => {
                        if vec_len > expected {
                            return Err(anyhow!(
                                "Vector length {} exceeds expected length {} at time {}",
                                vec_len,
                                expected,
                                time
                            ));
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
                    return Err(anyhow!(
                        "Type mismatch: cannot add {:?} to animated {:?}",
                        value.data_type(),
                        data_type
                    ));
                }

                // Insert the value using the generic insert method
                samples.try_insert(time, value)
            }
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
            Value::Animated(_samples) => {
                // TODO: Implement proper time collection for AnimatedData
                SmallVec::<[Time; 10]>::new_const()
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
