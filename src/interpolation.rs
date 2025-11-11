//! Interpolation types for animation keyframes.
//!
//! This module provides minimal interpolation primitives for time-based animation.
//! Higher-level animation semantics (smooth curves, angle-based tangents, etc.)
//! should be handled by animation systems like Dopamine.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::Time;

/// A keyframe's interpolation specification.
///
/// Describes how values should be interpolated when entering and leaving this keyframe.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Key<T> {
    /// How to interpolate when coming into this keyframe.
    pub interpolation_in: Interpolation<T>,
    /// How to interpolate when leaving this keyframe.
    pub interpolation_out: Interpolation<T>,
}

impl<T> Default for Key<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            interpolation_in: Interpolation::Linear,
            interpolation_out: Interpolation::Linear,
        }
    }
}

/// Bezier tangent handle specification.
///
/// Describes how to specify a tangent at a keyframe for Bezier interpolation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BezierHandle<T> {
    /// Tangent specified as an angle in radians.
    Angle(f32),
    /// Tangent specified as slope per second.
    SlopePerSecond(T),
    /// Tangent specified as slope per frame.
    SlopePerFrame(T),
    /// Tangent specified as delta time and delta value.
    Delta { time: Time, value: T },
}

/// Interpolation mode between keyframes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum Interpolation<T> {
    /// Hold value until next keyframe (step function).
    Hold,
    /// Linear interpolation between keyframes.
    #[default]
    Linear,
    /// Automatic smooth interpolation (Catmull-Rom style tangent).
    Smooth,
    /// Bezier curve with explicit tangent handle.
    Bezier(BezierHandle<T>),
}

// Manual Hash implementation for Key<T>.
impl<T: Hash> Hash for Key<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.interpolation_in.hash(state);
        self.interpolation_out.hash(state);
    }
}

// Manual Hash implementation for BezierHandle<T>.
impl<T: Hash> Hash for BezierHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            BezierHandle::Angle(angle) => angle.to_bits().hash(state),
            BezierHandle::SlopePerSecond(slope) => slope.hash(state),
            BezierHandle::SlopePerFrame(slope) => slope.hash(state),
            BezierHandle::Delta { time, value } => {
                time.hash(state);
                value.hash(state);
            }
        }
    }
}

// Manual Hash implementation for Interpolation<T>.
impl<T: Hash> Hash for Interpolation<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Interpolation::Hold => {}
            Interpolation::Linear => {}
            Interpolation::Smooth => {}
            Interpolation::Bezier(handle) => handle.hash(state),
        }
    }
}

// AIDEV-NOTE: Helper functions for bezier calculations.
// These are used when implementing bezier interpolation in TimeDataMap with BezierHandle variants.

#[cfg(feature = "interpolation")]
pub(crate) mod bezier_helpers {

    /// Clamp handle lengths to prevent overlap on time axis.
    ///
    /// When handles are too long, they can overlap on the time axis, making the knot
    /// vector non-monotonic which breaks uniform-cubic-splines. Each handle can use
    /// at most slightly under half the interval.
    pub fn clamp_handle_lengths(
        k1_time: f32,
        k2_time: f32,
        handle1_length: f32,
        handle2_length: f32,
    ) -> (f32, f32) {
        let dt = k2_time - k1_time;

        // Ensure handles don't overlap in time.
        // Each handle can use at most 49.5% of the interval to prevent overlap.
        let max_handle = dt * 0.495;

        let clamped_h1 = handle1_length.min(max_handle);
        let clamped_h2 = handle2_length.min(max_handle);

        // Additional check: ensure p1.time < p2.time.
        let p1_time = k1_time + clamped_h1;
        let p2_time = k2_time - clamped_h2;

        if p1_time >= p2_time {
            // Scale both down proportionally.
            let scale = (dt * 0.98) / (clamped_h1 + clamped_h2);
            (clamped_h1 * scale, clamped_h2 * scale)
        } else {
            (clamped_h1, clamped_h2)
        }
    }

    /// Calculate bezier control points from speed values.
    ///
    /// Given two keyframes with speed values, calculates the control points
    /// for a cubic bezier curve between them, ensuring handles don't overlap.
    #[allow(dead_code)]
    pub fn control_points_from_speed<T>(
        t1: f32,
        v1: &T,
        speed1: &T,
        t2: f32,
        v2: &T,
        speed2: &T,
    ) -> ((f32, T), (f32, T))
    where
        T: Clone
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<f32, Output = T>,
    {
        let dt = t2 - t1;

        // Base handle length is 1/3 of the interval.
        let base_handle = dt / 3.0;

        // Clamp handles to prevent overlap.
        let (h1, h2) = clamp_handle_lengths(t1, t2, base_handle, base_handle);

        // Control point 1: Move in direction of outgoing speed from keyframe 1.
        // P1 = P0 + speed1 * h1.
        let p1 = (t1 + h1, v1.clone() + speed1.clone() * h1);

        // Control point 2: Move backwards from keyframe 2 in direction of incoming speed.
        // P2 = P3 - speed2 * h2.
        let p2 = (t2 - h2, v2.clone() - speed2.clone() * h2);

        (p1, p2)
    }

    /// Calculate bezier control points from slope values.
    ///
    /// Similar to [`control_points_from_speed`], but accepts arbitrary slopes
    /// instead of specifically "speed" values. This is used when working with
    /// [`BezierHandle`] variants that specify tangents in different ways.
    pub fn control_points_from_slopes<T>(
        t1: f32,
        v1: &T,
        slope1: &T,
        t2: f32,
        v2: &T,
        slope2: &T,
    ) -> ((f32, T), (f32, T))
    where
        T: Clone
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<f32, Output = T>,
    {
        let dt = t2 - t1;
        let base_handle = dt / 3.0;
        let (h1, h2) = clamp_handle_lengths(t1, t2, base_handle, base_handle);

        let p1 = (t1 + h1, v1.clone() + slope1.clone() * h1);
        let p2 = (t2 - h2, v2.clone() - slope2.clone() * h2);

        (p1, p2)
    }

    /// Evaluate a cubic bezier curve using component-wise interpolation.
    ///
    /// Uses the standard cubic Bezier formula. This works for vector types
    /// where the interpolation is applied component-wise.
    pub fn evaluate_bezier_component_wise<T>(
        t: f32,
        p0: (f32, &T),
        p1: (f32, &T),
        p2: (f32, &T),
        p3: (f32, &T),
    ) -> T
    where
        T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
    {
        // Normalize t to [0, 1] range based on time coordinates.
        let t_norm = ((t - p0.0) / (p3.0 - p0.0)).clamp(0.0, 1.0);

        // Cubic Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3.
        let one_minus_t = 1.0 - t_norm;
        let one_minus_t2 = one_minus_t * one_minus_t;
        let one_minus_t3 = one_minus_t2 * one_minus_t;
        let t2 = t_norm * t_norm;
        let t3 = t2 * t_norm;

        p0.1.clone() * one_minus_t3
            + p1.1.clone() * (3.0 * one_minus_t2 * t_norm)
            + p2.1.clone() * (3.0 * one_minus_t * t2)
            + p3.1.clone() * t3
    }
}
