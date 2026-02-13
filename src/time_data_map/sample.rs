use super::*;
use core::num::NonZeroU16;
#[cfg(feature = "builtin-types")]
use rayon::prelude::*;

/// Weight value for motion blur sampling.
pub type SampleWeight = f32;

/// `trait` for generating motion blur samples with shutter timing.
///
/// The [`Sample`] `trait` generates multiple samples across a [`Shutter`]
/// interval for motion blur rendering. Each sample includes the interpolated
/// value and a weight based on the shutter opening at that time.
pub trait Sample<T> {
    /// Generate samples across the shutter interval.
    ///
    /// Returns a vector of value-weight pairs for the specified number of
    /// samples distributed across the shutter's time range.
    fn sample(&self, shutter: &Shutter, samples: NonZeroU16) -> Result<Vec<(T, SampleWeight)>>;
}

#[cfg(feature = "builtin-types")]
macro_rules! impl_sample {
    ($data_type:ty) => {
        impl Sample<$data_type> for TimeDataMap<$data_type> {
            fn sample(
                &self,
                shutter: &Shutter,
                samples: NonZeroU16,
            ) -> Result<Vec<($data_type, SampleWeight)>> {
                Ok((0..samples.into())
                    .into_par_iter()
                    .map(|t| {
                        let time = shutter.evaluate(t as f32 / u16::from(samples) as f32);
                        (self.interpolate(time), shutter.opening(time))
                    })
                    .collect::<Vec<_>>())
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
    fn sample(
        &self,
        shutter: &Shutter,
        samples: NonZeroU16,
    ) -> Result<Vec<(Matrix3, SampleWeight)>> {
        // Split all matrices into their component parts via analytical 2×2 SVD.
        let mut translations = BTreeMap::new();
        let mut rotations = BTreeMap::new();
        let mut stretches = BTreeMap::new();

        #[cfg(not(feature = "interpolation"))]
        for (time, matrix) in self.values.iter() {
            let crate::Matrix3(inner) = matrix;
            let (translate, rotate, stretch) = decompose_matrix(inner);
            translations.insert(*time, translate);
            rotations.insert(*time, rotate);
            stretches.insert(*time, stretch);
        }
        #[cfg(feature = "interpolation")]
        for (time, (matrix, _spec)) in self.values.iter() {
            let crate::Matrix3(inner) = matrix;
            let (translate, rotate, stretch) = decompose_matrix(inner);
            translations.insert(*time, translate);
            rotations.insert(*time, rotate);
            stretches.insert(*time, stretch);
        }

        // Interpolate the samples and recompose the matrices.
        Ok((0..samples.into())
            .into_par_iter()
            .map(|t| {
                let time = shutter.evaluate(t as f32 / u16::from(samples) as f32);
                (
                    crate::Matrix3(recompose_matrix(
                        interpolate(&translations, time),
                        interpolate_rotation(&rotations, time),
                        interpolate(&stretches, time),
                    )),
                    shutter.opening(time),
                )
            })
            .collect::<Vec<_>>())
    }
}
