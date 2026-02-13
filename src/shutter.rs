use crate::*;
use std::ops::Range;

/// Shutter timing for motion blur sampling.
///
/// A [`Shutter`] `struct` defines a time range and opening pattern for
/// generating motion blur samples. The opening function determines the weight
/// of samples at different times within the shutter interval.
#[derive(Clone, Debug, PartialEq, Hash, Default)]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Shutter {
    /// The overall time range for sampling.
    pub range: Range<Time>,
    /// The time range during which the shutter is opening.
    pub opening: Range<Time>,
}

impl Shutter {
    /// Evaluate the shutter at `time`.
    ///
    /// Returns a value between 0.0 and 1.0 that represents the shutter's
    /// opening at `time`.
    #[inline]
    pub fn opening(&self, pos: Time) -> f32 {
        if pos < self.opening.start {
            f32::from(pos) / f32::from(self.opening.start)
        } else {
            1.0f32 - f32::from(pos - self.opening.end) / f32::from(self.opening.end)
        }
    }

    #[inline]
    pub fn evaluate(&self, pos: f32) -> Time {
        self.range.start.lerp(self.range.end, pos as _)
    }

    /// Returns the center of the shutter.
    #[inline]
    pub fn center(&self) -> Time {
        (self.range.start + self.range.end) * 0.5
    }
}

// Manual Eq implementation for Shutter
// This is safe because we handle floating point comparison deterministically
impl Eq for Shutter {}
