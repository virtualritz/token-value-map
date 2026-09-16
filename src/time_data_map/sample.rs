use super::*;
use core::num::NonZeroU16;
#[cfg(feature = "builtin-types")]
use rayon::prelude::*;

/// `trait` for generating motion blur samples with shutter timing.
///
/// The [`Sample`] `trait` generates multiple samples across a [`Shutter`]
/// interval for motion blur rendering.
///
/// # Why samples carry no weights
///
/// A motion-blurred result is the exposure-weighted average of the value over
/// the shutter interval. There are two ways to estimate it with `n` samples:
///
/// - Spread the sample times evenly and weight each sample by how far the
///   shutter is open at its time, then divide by the sum of the weights.
/// - Place the sample times by the exposure itself -- more of them where the
///   shutter is fully open, fewer on its ramps -- so that each sample stands
///   for the same share of the exposure, and take a plain average.
///
/// Both converge to the same result, but the second spends no samples where
/// the shutter is nearly closed, so it is less noisy for the same `n`. [`Sample`]
/// uses the second: times come from [`Shutter::sample_times`], and callers
/// combine the returned values with a plain average (sum, then divide by the
/// number of samples). Weighting them again by the shutter would count the
/// exposure twice.
pub trait Sample<T> {
    /// Generate samples across the shutter interval.
    ///
    /// Returns up to `samples` values, placed by the shutter's exposure; see
    /// the [trait documentation](Sample) for how to combine them. A value that
    /// does not change over time returns a single sample.
    fn sample(&self, shutter: &Shutter, samples: NonZeroU16) -> Result<Vec<T>>;
}

#[cfg(feature = "builtin-types")]
macro_rules! impl_sample {
    ($data_type:ty) => {
        impl Sample<$data_type> for TimeDataMap<$data_type> {
            fn sample(&self, shutter: &Shutter, samples: NonZeroU16) -> Result<Vec<$data_type>> {
                Ok((0..u16::from(samples))
                    .into_par_iter()
                    .map(|index| self.interpolate(shutter.sample_time(index, samples)))
                    .collect())
            }
        }
    };
}

#[cfg(feature = "builtin-types")]
impl_sample!(Real);
#[cfg(feature = "builtin-types")]
impl_sample!(Integer);
#[cfg(feature = "builtin-types")]
impl_sample!(Color);

#[cfg(all(feature = "builtin-types", feature = "vector2"))]
impl_sample!(Vector2);
#[cfg(all(feature = "builtin-types", feature = "vector3"))]
impl_sample!(Vector3);
#[cfg(all(feature = "builtin-types", feature = "normal3"))]
impl_sample!(Normal3);
#[cfg(all(feature = "builtin-types", feature = "point3"))]
impl_sample!(Point3);
#[cfg(all(feature = "builtin-types", feature = "matrix4"))]
impl_sample!(Matrix4);

// AIDEV-NOTE: Matrix3 sampling uses analytical 2×2 SVD decomposition for proper
// rotation/stretch interpolation on all backends. Rotation is interpolated via
// shortest-path angle slerp; translation and stretch are interpolated linearly.
#[cfg(all(feature = "builtin-types", feature = "matrix3"))]
impl Sample<Matrix3> for TimeDataMap<Matrix3> {
    fn sample(&self, shutter: &Shutter, samples: NonZeroU16) -> Result<Vec<Matrix3>> {
        // Split all matrices into their component parts via analytical 2×2 SVD.
        let mut translations = BTreeMap::new();
        let mut rotations = BTreeMap::new();
        let mut stretches = BTreeMap::new();

        #[cfg(not(feature = "interpolation"))]
        for (time, matrix) in self.values.as_btree_map().iter() {
            let crate::Matrix3(ref inner) = *matrix;
            let (translate, rotate, stretch) = decompose_matrix(inner);
            translations.insert(*time, translate);
            rotations.insert(*time, rotate);
            stretches.insert(*time, stretch);
        }
        #[cfg(feature = "interpolation")]
        for (time, (matrix, _spec)) in self.values.as_btree_map().iter() {
            let crate::Matrix3(ref inner) = *matrix;
            let (translate, rotate, stretch) = decompose_matrix(inner);
            translations.insert(*time, translate);
            rotations.insert(*time, rotate);
            stretches.insert(*time, stretch);
        }

        // Interpolate the samples and recompose the matrices.
        Ok((0..u16::from(samples))
            .into_par_iter()
            .map(|index| {
                let time = shutter.sample_time(index, samples);
                crate::Matrix3(recompose_matrix(
                    interpolate(&translations, time),
                    interpolate_rotation(&rotations, time),
                    interpolate(&stretches, time),
                ))
            })
            .collect())
    }
}
