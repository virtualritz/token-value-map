//! Interpolation types for animation keyframes.
//!
//! This module provides minimal interpolation primitives for time-based animation.
//! Higher-level animation semantics (smooth curves, angle-based tangents, etc.)
//! should be handled by animation systems like Dopamine.

#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::Time;

/// A keyframe's interpolation specification.
///
/// Describes how values should be interpolated when entering and leaving this keyframe.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "facet", derive(facet::Facet))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
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
#[cfg_attr(feature = "facet", derive(facet::Facet))]
#[cfg_attr(feature = "facet", repr(u8))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
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
#[cfg_attr(feature = "facet", derive(facet::Facet))]
#[cfg_attr(feature = "facet", repr(u8))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
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

/// Trait for types that can be used as bezier handle values.
///
/// This allows converting between the type's value and f64 for UI handle editing.
/// Scalar types (Real, Integer) implement this; vector types do not.
#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
pub trait HandleValue: Sized {
    /// Convert to f64 for handle value.
    fn to_handle_f64(&self) -> f64;
    /// Create from f64 handle value.
    fn from_handle_f64(v: f64) -> Self;
}

#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
impl HandleValue for crate::Real {
    fn to_handle_f64(&self) -> f64 {
        self.0
    }
    fn from_handle_f64(v: f64) -> Self {
        crate::Real(v)
    }
}

#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
impl HandleValue for crate::Integer {
    fn to_handle_f64(&self) -> f64 {
        self.0 as f64
    }
    fn from_handle_f64(v: f64) -> Self {
        crate::Integer(v.round() as i64)
    }
}

#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
impl HandleValue for crate::Color {
    fn to_handle_f64(&self) -> f64 {
        // Use luminance as scalar proxy for handle editing.
        self.0[0] as f64 * 0.299 + self.0[1] as f64 * 0.587 + self.0[2] as f64 * 0.114
    }
    fn from_handle_f64(v: f64) -> Self {
        let v = v as f32;
        crate::Color([v, v, v, 1.0])
    }
}

/// Convert a [`Key<T>`](Key) to `egui_keyframe` `BezierHandles`.
#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
pub fn key_to_bezier_handles<T: HandleValue>(key: &Key<T>) -> egui_keyframe::BezierHandles {
    let (left_x, left_y) = match &key.interpolation_in {
        Interpolation::Bezier(BezierHandle::Delta { time, value }) => {
            (time.to_secs() as f32, value.to_handle_f64() as f32)
        }
        _ => (0.0, 0.0),
    };
    let (right_x, right_y) = match &key.interpolation_out {
        Interpolation::Bezier(BezierHandle::Delta { time, value }) => {
            (time.to_secs() as f32, value.to_handle_f64() as f32)
        }
        _ => (0.0, 0.0),
    };
    egui_keyframe::BezierHandles {
        left_x,
        left_y,
        right_x,
        right_y,
    }
}

/// Convert `egui_keyframe` `BezierHandles` to a [`Key<T>`](Key).
#[cfg(all(feature = "interpolation", feature = "egui-keyframe"))]
pub fn bezier_handles_to_key<T: HandleValue>(handles: &egui_keyframe::BezierHandles) -> Key<T> {
    Key {
        interpolation_in: Interpolation::Bezier(BezierHandle::Delta {
            time: Time::from_secs(handles.left_x as f64),
            value: T::from_handle_f64(handles.left_y as f64),
        }),
        interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
            time: Time::from_secs(handles.right_x as f64),
            value: T::from_handle_f64(handles.right_y as f64),
        }),
    }
}

// AIDEV-NOTE: Helper functions for bezier calculations.
// These are used when implementing bezier interpolation in TimeDataMap with BezierHandle variants.

#[cfg(feature = "interpolation")]
pub(crate) mod bezier_helpers {

    /// Clamp handle lengths to prevent overlap on time axis.
    ///
    /// When handles are too long, they can overlap on the time axis, making the
    /// time curve non-monotonic so it no longer has a unique inverse (see
    /// [`evaluate_bezier_component_wise`]). Each handle can use at most slightly
    /// under half the interval.
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

    /// Calculate bezier control points from slopes and explicit handle lengths.
    ///
    /// Each handle reaches `length` along the time axis, clamped by
    /// [`clamp_handle_lengths`]. A handle's length is what makes a curve ease: a
    /// long outgoing handle holds the value near the key before moving on. A
    /// third of the interval keeps the curve's timing linear.
    pub fn control_points_from_slopes_with_handle_lengths<T>(
        (t1, v1, slope1, length1): (f32, &T, &T, f32),
        (t2, v2, slope2, length2): (f32, &T, &T, f32),
    ) -> ((f32, T), (f32, T))
    where
        T: Clone
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<f32, Output = T>,
    {
        let (h1, h2) = clamp_handle_lengths(t1, t2, length1.abs(), length2.abs());

        let p1 = (t1 + h1, v1.clone() + slope1.clone() * h1);
        let p2 = (t2 - h2, v2.clone() - slope2.clone() * h2);

        (p1, p2)
    }

    /// Evaluate a cubic bezier curve using component-wise interpolation.
    ///
    /// The curve is two-dimensional: each control point is `(time, value)`. The
    /// time curve is solved for the bezier parameter at `t` first
    /// (`uniform_cubic_splines::spline_inverse_segment`), then the value curve is
    /// evaluated at that parameter. Mapping `t` linearly onto the parameter is
    /// only correct when the handles sit at exactly a third of the interval;
    /// any other handle length would change the curve's shape but not its
    /// timing. Works for vector types, applied component-wise.
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
        use uniform_cubic_splines::{basis::Bezier, spline_inverse_segment};

        let t = t.clamp(p0.0, p3.0);
        // `clamp_handle_lengths` keeps the time control points monotonic, so the
        // inverse exists; the linear mapping is only a guard for degenerate
        // input.
        let t_norm = spline_inverse_segment::<Bezier, f32>(t, &[p0.0, p1.0, p2.0, p3.0])
            .unwrap_or_else(|| ((t - p0.0) / (p3.0 - p0.0)).clamp(0.0, 1.0));

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
