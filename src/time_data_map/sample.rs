use super::*;
use core::num::NonZeroU16;
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
                        (interpolate(&self.0, time), shutter.opening(time))
                    })
                    .collect::<Vec<_>>())
            }
        }
    };
}

impl_sample!(Real);
impl_sample!(Integer);
impl_sample!(Color);

#[cfg(feature = "vector2")]
impl_sample!(Vector2);
#[cfg(feature = "vector3")]
impl_sample!(Vector3);
#[cfg(feature = "normal3")]
impl_sample!(Normal3);
#[cfg(feature = "point3")]
impl_sample!(Point3);
#[cfg(feature = "matrix4")]
impl_sample!(Matrix4);

#[cfg(feature = "matrix3")]
impl Sample<Matrix3> for TimeDataMap<Matrix3> {
    fn sample(
        &self,
        shutter: &Shutter,
        samples: NonZeroU16,
    ) -> Result<Vec<(Matrix3, SampleWeight)>> {
        // Split all matrices into their component parts via singular value
        // decomposition.
        let mut translations = BTreeMap::new();
        let mut rotations = BTreeMap::new();
        let mut stretches = BTreeMap::new();

        for (time, matrix) in self.0.iter() {
            let (translate, rotate, stretch) = decompose_matrix(&matrix.0);
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
                    Matrix3(recompose_matrix(
                        interpolate(&translations, time),
                        interpolate_spherical_linear(&rotations, time),
                        interpolate(&stretches, time),
                    )),
                    shutter.opening(time),
                )
            })
            .collect::<Vec<_>>())
    }
}
