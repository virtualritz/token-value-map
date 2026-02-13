//! Generic value type that works with any [`DataSystem`].

use crate::{Result, Shutter, Time, traits::*};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A value that can be either uniform or animated over time.
///
/// This is the generic version of [`Value`](crate::Value) that works with any
/// [`DataSystem`]. Use this when you have a custom data type system.
///
/// For the built-in types, use [`Value`](crate::Value) which is an alias for
/// `GenericValue<Data>`.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D: Serialize, D::Animated: Serialize",
        deserialize = "D: Deserialize<'de>, D::Animated: Deserialize<'de>"
    ))
)]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub enum GenericValue<D: DataSystem> {
    /// A constant value that does not change over time.
    Uniform(D),
    /// A value that changes over time with keyframe interpolation.
    Animated(D::Animated),
}

impl<D: DataSystem> GenericValue<D> {
    /// Creates a uniform value that does not change over time.
    pub fn uniform(value: D) -> Self {
        GenericValue::Uniform(value)
    }

    /// Creates an animated value from time-value pairs.
    ///
    /// All samples must have the same data type. Vector samples are padded
    /// to match the length of the longest vector in the set.
    pub fn animated<I>(samples: I) -> Result<Self>
    where
        I: IntoIterator<Item = (Time, D)>,
    {
        let mut samples_vec: Vec<(Time, D)> = samples.into_iter().collect();

        if samples_vec.is_empty() {
            return Err(crate::Error::EmptySamples);
        }

        // Get the data type from the first sample.
        let data_type = samples_vec[0].1.discriminant();
        let expected_name = samples_vec[0].1.variant_name();

        // Check all samples have the same type and handle length consistency.
        let mut expected_len: Option<usize> = None;
        for (time, value) in &mut samples_vec {
            if value.discriminant() != data_type {
                return Err(crate::Error::GenericTypeMismatch {
                    expected: expected_name,
                    got: value.variant_name(),
                });
            }

            // Check vector length consistency.
            if let Some(vec_len) = value.try_len() {
                match expected_len {
                    None => expected_len = Some(vec_len),
                    Some(expected) => {
                        if vec_len > expected {
                            return Err(crate::Error::VectorLengthExceeded {
                                actual: vec_len,
                                expected,
                                time: *time,
                            });
                        } else if vec_len < expected {
                            value.pad_to_length(expected);
                        }
                    }
                }
            }
        }

        // Create animated data from the first sample, then insert the rest.
        let (first_time, first_value) = samples_vec.remove(0);
        let mut animated = D::Animated::from_single(first_time, first_value);

        for (time, value) in samples_vec {
            animated.try_insert(time, value)?;
        }

        Ok(GenericValue::Animated(animated))
    }

    /// Adds a sample at a specific time.
    ///
    /// If the value is uniform, it becomes animated with the new sample.
    pub fn add_sample(&mut self, time: Time, value: D) -> Result<()> {
        match self {
            GenericValue::Uniform(_) => {
                *self = GenericValue::animated(vec![(time, value)])?;
                Ok(())
            }
            GenericValue::Animated(samples) => {
                if value.discriminant() != samples.discriminant() {
                    return Err(crate::Error::GenericTypeMismatch {
                        expected: samples.variant_name(),
                        got: value.variant_name(),
                    });
                }
                samples.try_insert(time, value)
            }
        }
    }

    /// Removes a sample at a specific time.
    ///
    /// Returns the removed value if it existed. For uniform values, this is a
    /// no-op and returns `None`.
    pub fn remove_sample(&mut self, time: &Time) -> Option<D> {
        match self {
            GenericValue::Uniform(_) => None,
            GenericValue::Animated(samples) => samples.remove_at(time),
        }
    }

    /// Samples the value at an exact time without interpolation.
    ///
    /// Returns the exact value if it exists at the given time, or `None` if
    /// no sample exists at that time for animated values.
    pub fn sample_at(&self, time: Time) -> Option<D> {
        match self {
            GenericValue::Uniform(v) => Some(v.clone()),
            GenericValue::Animated(samples) => samples.sample_at(time),
        }
    }

    /// Interpolates the value at the given time.
    ///
    /// For uniform values, returns the constant value. For animated values,
    /// interpolates between surrounding keyframes.
    pub fn interpolate(&self, time: Time) -> D {
        match self {
            GenericValue::Uniform(v) => v.clone(),
            GenericValue::Animated(samples) => samples.interpolate(time),
        }
    }

    /// Returns `true` if the value is animated (has multiple keyframes).
    pub fn is_animated(&self) -> bool {
        match self {
            GenericValue::Uniform(_) => false,
            GenericValue::Animated(samples) => samples.has_animation(),
        }
    }

    /// Returns the number of time samples.
    pub fn sample_count(&self) -> usize {
        match self {
            GenericValue::Uniform(_) => 1,
            GenericValue::Animated(samples) => samples.keyframe_count(),
        }
    }

    /// Returns all keyframe times.
    pub fn times(&self) -> SmallVec<[Time; 10]> {
        match self {
            GenericValue::Uniform(_) => SmallVec::new_const(),
            GenericValue::Animated(samples) => samples.times(),
        }
    }

    /// Returns the data type discriminant for this value.
    pub fn discriminant(&self) -> D::DataType {
        match self {
            GenericValue::Uniform(data) => data.discriminant(),
            GenericValue::Animated(animated) => animated.discriminant(),
        }
    }

    /// Returns a human-readable type name for this value.
    pub fn variant_name(&self) -> &'static str {
        match self {
            GenericValue::Uniform(data) => data.variant_name(),
            GenericValue::Animated(animated) => animated.variant_name(),
        }
    }

    /// Merges this value with another using a combiner function.
    ///
    /// For uniform values, applies the combiner once. For animated values,
    /// samples both at the union of all keyframe times and applies the
    /// combiner at each time.
    pub fn merge_with<F>(&self, other: &GenericValue<D>, combiner: F) -> Result<GenericValue<D>>
    where
        F: Fn(&D, &D) -> D,
    {
        match (self, other) {
            (GenericValue::Uniform(a), GenericValue::Uniform(b)) => {
                Ok(GenericValue::Uniform(combiner(a, b)))
            }
            _ => {
                let mut all_times = std::collections::BTreeSet::new();
                for t in self.times() {
                    all_times.insert(t);
                }
                for t in other.times() {
                    all_times.insert(t);
                }

                if all_times.is_empty() {
                    let a = self.interpolate(Time::default());
                    let b = other.interpolate(Time::default());
                    return Ok(GenericValue::Uniform(combiner(&a, &b)));
                }

                let combined_samples: Vec<(Time, D)> = all_times
                    .into_iter()
                    .map(|time| {
                        let a = self.interpolate(time);
                        let b = other.interpolate(time);
                        (time, combiner(&a, &b))
                    })
                    .collect();

                if combined_samples.len() == 1 {
                    Ok(GenericValue::Uniform(
                        combined_samples.into_iter().next().unwrap().1,
                    ))
                } else {
                    GenericValue::animated(combined_samples)
                }
            }
        }
    }

    /// Hashes the value with shutter context for animation-aware caching.
    ///
    /// For animated values, samples at standardized points within the shutter
    /// range and hashes the interpolated values.
    pub fn hash_with_shutter<H: Hasher>(&self, state: &mut H, shutter: &Shutter) {
        match self {
            GenericValue::Uniform(data) => {
                data.hash(state);
            }
            GenericValue::Animated(animated) => {
                // Sample at 5 standardized points within the shutter.
                const SAMPLE_POSITIONS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

                let samples: SmallVec<[D; 5]> = SAMPLE_POSITIONS
                    .iter()
                    .map(|&pos| {
                        let time = shutter.evaluate(pos);
                        animated.interpolate(time)
                    })
                    .collect();

                let all_same = samples.windows(2).all(|w| w[0] == w[1]);

                std::mem::discriminant(self).hash(state);

                if all_same {
                    1usize.hash(state);
                    samples[0].hash(state);
                } else {
                    samples.len().hash(state);
                    for sample in &samples {
                        sample.hash(state);
                    }
                }
            }
        }
    }
}

impl<D: DataSystem> From<D> for GenericValue<D> {
    fn from(value: D) -> Self {
        GenericValue::uniform(value)
    }
}

impl<D: DataSystem> Eq for GenericValue<D> {}
