//! Interpolation types for animation keyframes.
//!
//! This module provides minimal interpolation primitives for time-based animation.
//! Higher-level animation semantics (smooth curves, angle-based tangents, etc.)
//! should be handled by animation systems like Dopamine.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

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
    /// Bezier curve defined by speed (derivative) at keyframe.
    /// The speed value has the same type as the animated value.
    Speed(T),
}

// Manual Hash implementation for Key<T>.
impl<T: Hash> Hash for Key<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.interpolation_in.hash(state);
        self.interpolation_out.hash(state);
    }
}

// Manual Hash implementation for Interpolation<T>.
impl<T: Hash> Hash for Interpolation<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Interpolation::Hold => {}
            Interpolation::Linear => {}
            Interpolation::Speed(speed) => speed.hash(state),
        }
    }
}

// AIDEV-NOTE: Helper functions for bezier calculations.
// These will be used when implementing bezier interpolation in TimeDataMap.

#[cfg(feature = "interpolation")]
pub(crate) mod bezier_helpers {

    /// Calculate bezier control points from speed values.
    ///
    /// Given two keyframes with speed values, calculates the control points
    /// for a cubic bezier curve between them.
    pub fn control_points_from_speed<T>(
        t1: f32,
        v1: &T,
        speed1: &T,
        t2: f32,
        v2: &T,
        speed2: &T,
    ) -> ((f32, T), (f32, T))
    where
        T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
    {
        // Control point 1: Move in direction of outgoing speed from keyframe 1.
        let dt = (t2 - t1) / 3.0;
        let p1 = (t1 + dt, v1.clone() + speed1.clone() * dt);

        // Control point 2: Move in opposite direction of incoming speed to keyframe 2.
        let p2 = (t2 - dt, v2.clone() + speed2.clone() * (-dt));

        (p1, p2)
    }

    /// Evaluate a cubic bezier at time t.
    ///
    /// Uses De Casteljau's algorithm for robust evaluation.
    pub fn evaluate_bezier<T>(t: f32, p0: (f32, T), p1: (f32, T), p2: (f32, T), p3: (f32, T)) -> T
    where
        T: Clone + std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
    {
        // Normalize t to [0, 1] range.
        let t = (t - p0.0) / (p3.0 - p0.0);
        let t = t.clamp(0.0, 1.0);

        // De Casteljau's algorithm.
        let q0 = p0.1.clone() * (1.0 - t) + p1.1.clone() * t;
        let q1 = p1.1.clone() * (1.0 - t) + p2.1.clone() * t;
        let q2 = p2.1.clone() * (1.0 - t) + p3.1.clone() * t;

        let r0 = q0.clone() * (1.0 - t) + q1.clone() * t;
        let r1 = q1 * (1.0 - t) + q2 * t;

        r0 * (1.0 - t) + r1 * t
    }
}
