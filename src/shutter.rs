use crate::*;
use std::ops::Range;

/// Shutter timing for motion blur sampling.
///
/// A [`Shutter`] `struct` describes a trapezoidal exposure over time: the
/// shutter opens linearly from `range.start` to `opening.start`, stays fully
/// open until `opening.end`, then closes linearly until `range.end`. A box
/// shutter has `opening == range`.
///
/// [`Sample`] places its samples by this exposure (see
/// [`time_at_exposure`](Self::time_at_exposure)), so the samples it returns
/// carry equal weight.
#[derive(Clone, Debug, PartialEq, Hash, Default)]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct Shutter {
    /// The overall time range for sampling.
    pub range: Range<Time>,
    /// The time range during which the shutter is fully open.
    pub opening: Range<Time>,
}

impl Shutter {
    /// Evaluate the shutter at `pos`.
    ///
    /// Returns how far the shutter is open at `pos`, from 0.0 (closed) to 1.0
    /// (fully open). Outside `range` the shutter is closed.
    #[inline]
    pub fn opening(&self, pos: Time) -> f32 {
        let (start, open_start, open_end, end) = self.segments();
        let pos = pos.to_secs();

        let opening = if pos < start || pos > end {
            0.0
        } else if pos < open_start {
            (pos - start) / (open_start - start)
        } else if pos <= open_end {
            1.0
        } else {
            (end - pos) / (end - open_end)
        };
        opening as f32
    }

    /// The time at which `fraction` of the shutter's total exposure has
    /// elapsed.
    ///
    /// This is the inverse of the exposure's cumulative distribution:
    /// `fraction` 0.0 maps to `range.start`, 1.0 to `range.end`, and equal
    /// steps in `fraction` map to times spaced by how open the shutter is --
    /// densely where it is fully open, sparsely on the ramps.
    #[inline]
    pub fn time_at_exposure(&self, fraction: f32) -> Time {
        let (start, open_start, open_end, end) = self.segments();
        let ramp_up = open_start - start;
        let plateau = open_end - open_start;
        let ramp_down = end - open_end;
        let exposure = 0.5 * ramp_up + plateau + 0.5 * ramp_down;

        if exposure <= 0.0 {
            return self.range.start;
        }

        let area = f64::from(fraction.clamp(0.0, 1.0)) * exposure;
        let secs = if area < 0.5 * ramp_up {
            // Opening ramp: exposure so far is x^2 / (2 ramp_up).
            start + (2.0 * ramp_up * area).sqrt()
        } else if area <= 0.5 * ramp_up + plateau {
            open_start + (area - 0.5 * ramp_up)
        } else {
            // Closing ramp: exposure past `open_end` is y - y^2 / (2 ramp_down).
            let rest = area - 0.5 * ramp_up - plateau;
            open_end + ramp_down
                - (ramp_down * ramp_down - 2.0 * ramp_down * rest)
                    .max(0.0)
                    .sqrt()
        };
        Time::from_secs(secs.clamp(start, end))
    }

    /// The time of sample `index` out of `samples`: the middle of the
    /// `index`-th of `samples` equal slices of the shutter's exposure.
    #[inline]
    pub fn sample_time(&self, index: u16, samples: core::num::NonZeroU16) -> Time {
        self.time_at_exposure((f32::from(index) + 0.5) / f32::from(u16::from(samples)))
    }

    /// `samples` times placed by the shutter's exposure, one at the middle of
    /// each of `samples` equal slices of it.
    ///
    /// Every time stands for the same share of the exposure, so values sampled
    /// at these times are combined with a plain average.
    pub fn sample_times(&self, samples: core::num::NonZeroU16) -> impl Iterator<Item = Time> + '_ {
        (0..u16::from(samples)).map(move |index| self.sample_time(index, samples))
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

    /// `range` and `opening` in seconds, with `opening` clamped into `range`.
    fn segments(&self) -> (f64, f64, f64, f64) {
        let start = self.range.start.to_secs();
        let end = self.range.end.to_secs().max(start);
        let open_start = self.opening.start.to_secs().clamp(start, end);
        let open_end = self.opening.end.to_secs().clamp(open_start, end);
        (start, open_start, open_end, end)
    }
}

// Manual Eq implementation for Shutter
// This is safe because we handle floating point comparison deterministically
impl Eq for Shutter {}
