use crate::*;
use enum_dispatch::enum_dispatch;

use std::{
    collections::BTreeMap,
    iter::FromIterator,
    ops::{Add, Mul, Sub},
};

mod sample;
pub use sample::*;

/// A mapping from time to data values with interpolation support.
///
/// Stores time-value pairs in a [`BTreeMap`] for efficient time-based
/// queries and supports various interpolation methods.
///
/// When the `interpolation` feature is enabled, each value can have an
/// associated interpolation specification for animation curves.
///
/// Without `interpolation` feature: `BTreeMap<Time, T>`.
///
/// With `interpolation` feature an entry supports a [`Key`]s:
/// `BTreeMap<Time, (T, Option<Key<T>>)>`.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeDataMap<T> {
    /// The time-value pairs with optional interpolation keys.
    #[cfg(not(feature = "interpolation"))]
    pub values: BTreeMap<Time, T>,
    #[cfg(feature = "interpolation")]
    pub values: BTreeMap<Time, (T, Option<crate::Key<T>>)>,
}

// Manual Eq implementation.
impl<T: Eq> Eq for TimeDataMap<T> {}

// AsRef implementation for backward compatibility.
#[cfg(not(feature = "interpolation"))]
impl<T> AsRef<BTreeMap<Time, T>> for TimeDataMap<T> {
    fn as_ref(&self) -> &BTreeMap<Time, T> {
        &self.values
    }
}

// AIDEV-NOTE: These methods provide backward compatibility for code that
// previously accessed the BTreeMap directly via .0 field.
// With interpolation feature, these return a view without interpolation specs.
impl<T> TimeDataMap<T> {
    /// Get an iterator over time-value pairs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&Time, &T)> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.iter()
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.iter().map(|(t, (v, _))| (t, v))
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

// Constructor for backward compatibility.
impl<T> From<BTreeMap<Time, T>> for TimeDataMap<T> {
    fn from(values: BTreeMap<Time, T>) -> Self {
        #[cfg(not(feature = "interpolation"))]
        {
            Self { values }
        }
        #[cfg(feature = "interpolation")]
        {
            Self {
                values: values.into_iter().map(|(k, v)| (k, (v, None))).collect(),
            }
        }
    }
}

/// Control operations `trait` for time data maps.
#[enum_dispatch]
pub trait TimeDataMapControl<T> {
    /// Returns the number of time samples.
    fn len(&self) -> usize;
    /// Returns `true` if there are no time samples.
    fn is_empty(&self) -> bool;
    /// Returns `true` if there is more than one time sample.
    fn is_animated(&self) -> bool;
}

impl<T> TimeDataMapControl<T> for TimeDataMap<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn is_animated(&self) -> bool {
        1 < self.values.len()
    }
}

impl<T> crate::DataTypeOps for TimeDataMap<T>
where
    T: crate::DataTypeOps,
{
    fn data_type(&self) -> DataType {
        // Use the first element to determine the type, or return a default if empty.
        #[cfg(not(feature = "interpolation"))]
        {
            self.values
                .values()
                .next()
                .map(|v| v.data_type())
                .unwrap_or(DataType::Real)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values
                .values()
                .next()
                .map(|(v, _)| v.data_type())
                .unwrap_or(DataType::Real)
        }
    }

    fn type_name(&self) -> &'static str {
        // Use the first element to determine the type name.
        #[cfg(not(feature = "interpolation"))]
        {
            self.values
                .values()
                .next()
                .map(|v| v.type_name())
                .unwrap_or("unknown")
        }
        #[cfg(feature = "interpolation")]
        {
            self.values
                .values()
                .next()
                .map(|(v, _)| v.type_name())
                .unwrap_or("unknown")
        }
    }
}

macro_rules! impl_from_at_time {
    ($($t:ty),+) => {
        $(
            impl From<(Time, $t)> for TimeDataMap<$t> {
                fn from((time, value): (Time, $t)) -> Self {
                    TimeDataMap::from(BTreeMap::from([(time, value)]))
                }
            }
        )+
    };
}

impl_from_at_time!(
    Boolean, Real, Integer, String, Color, BooleanVec, RealVec, IntegerVec, StringVec, ColorVec,
    Data
);

#[cfg(feature = "vector2")]
impl_from_at_time!(Vector2);
#[cfg(feature = "vector3")]
impl_from_at_time!(Vector3);
#[cfg(feature = "matrix3")]
impl_from_at_time!(Matrix3);
#[cfg(feature = "normal3")]
impl_from_at_time!(Normal3);
#[cfg(feature = "point3")]
impl_from_at_time!(Point3);
#[cfg(feature = "matrix4")]
impl_from_at_time!(Matrix4);

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl_from_at_time!(Vector2Vec);
#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl_from_at_time!(Vector3Vec);
#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl_from_at_time!(Matrix3Vec);
#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl_from_at_time!(Normal3Vec);
#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl_from_at_time!(Point3Vec);
#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl_from_at_time!(Matrix4Vec);

impl<T> TimeDataMap<T> {
    /// Insert a value at the given time.
    pub fn insert(&mut self, time: Time, value: T) {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.insert(time, value);
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.insert(time, (value, None));
        }
    }

    /// Get a value at the exact time.
    pub fn get(&self, time: &Time) -> Option<&T> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.get(time)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.get(time).map(|(v, _)| v)
        }
    }
}

impl<T> TimeDataMap<T>
where
    T: Clone + Add<Output = T> + Mul<f32, Output = T> + Sub<Output = T>,
{
    pub fn interpolate(&self, time: Time) -> T {
        #[cfg(not(feature = "interpolation"))]
        {
            interpolate(&self.values, time)
        }
        #[cfg(feature = "interpolation")]
        {
            interpolate_with_specs(&self.values, time)
        }
    }
}

// Interpolation-specific methods.
#[cfg(feature = "interpolation")]
impl<T> TimeDataMap<T> {
    /// Insert a value with interpolation specification.
    ///
    /// Sets both the value at the given time and how it should interpolate
    /// to neighboring keyframes.
    pub fn insert_with_interpolation(&mut self, time: Time, value: T, spec: crate::Key<T>) {
        self.values.insert(time, (value, Some(spec)));
    }

    /// Get interpolation spec at a given time.
    ///
    /// Returns `None` if no interpolation metadata exists or no spec at this time.
    pub fn get_interpolation(&self, time: &Time) -> Option<&crate::Key<T>> {
        self.values.get(time)?.1.as_ref()
    }

    /// Clear all interpolation metadata.
    ///
    /// After calling this, all interpolation reverts to automatic mode.
    pub fn clear_interpolation(&mut self) {
        for (_, interp) in self.values.values_mut() {
            *interp = None;
        }
    }

    /// Check if using custom interpolation.
    ///
    /// Returns `true` if any interpolation metadata is present.
    pub fn has_interpolation(&self) -> bool {
        self.values.values().any(|(_, interp)| interp.is_some())
    }
}

impl<T, V> FromIterator<(Time, V)> for TimeDataMap<T>
where
    V: Into<T>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Time, V)>,
    {
        Self::from(BTreeMap::from_iter(
            iter.into_iter().map(|(t, v)| (t, v.into())),
        ))
    }
}

impl<T> TimeDataMap<T> {
    pub fn closest_sample(&self, time: Time) -> &T {
        #[cfg(not(feature = "interpolation"))]
        {
            let mut range = self.values.range(time..);
            let greater_or_equal = range.next();

            let mut range = self.values.range(..time);
            let less_than = range.next_back();

            match (less_than, greater_or_equal) {
                (Some((lower_k, lower_v)), Some((upper_k, upper_v))) => {
                    if (time.as_ref() - lower_k.as_ref()).abs()
                        <= (upper_k.as_ref() - time.as_ref()).abs()
                    {
                        lower_v
                    } else {
                        upper_v
                    }
                }
                (Some(entry), None) | (None, Some(entry)) => entry.1,
                (None, None) => {
                    unreachable!("TimeDataMap can never be empty")
                }
            }
        }
        #[cfg(feature = "interpolation")]
        {
            let mut range = self.values.range(time..);
            let greater_or_equal = range.next();

            let mut range = self.values.range(..time);
            let less_than = range.next_back();

            match (less_than, greater_or_equal) {
                (Some((lower_k, (lower_v, _))), Some((upper_k, (upper_v, _)))) => {
                    if (time.as_ref() - lower_k.as_ref()).abs()
                        <= (upper_k.as_ref() - time.as_ref()).abs()
                    {
                        lower_v
                    } else {
                        upper_v
                    }
                }
                (Some((_, (v, _))), None) | (None, Some((_, (v, _)))) => v,
                (None, None) => {
                    unreachable!("TimeDataMap can never be empty")
                }
            }
        }
    }

    /// Sample value at exact time (no interpolation).
    pub fn sample_at(&self, time: Time) -> Option<&T> {
        self.get(&time)
    }

    /// Get the value at or before the given time.
    pub fn sample_at_or_before(&self, time: Time) -> Option<&T> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.range(..=time).next_back().map(|(_, v)| v)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.range(..=time).next_back().map(|(_, (v, _))| v)
        }
    }

    /// Get the value at or after the given time.
    pub fn sample_at_or_after(&self, time: Time) -> Option<&T> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.range(time..).next().map(|(_, v)| v)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.range(time..).next().map(|(_, (v, _))| v)
        }
    }

    /// Get surrounding samples for interpolation.
    ///
    /// Returns up to `N` samples centered around the given time for
    /// use in interpolation algorithms.
    pub fn sample_surrounding<const N: usize>(&self, time: Time) -> SmallVec<[(Time, &T); N]> {
        // Get samples before the time.
        let before_count = N / 2;

        #[cfg(not(feature = "interpolation"))]
        {
            let mut result = self
                .values
                .range(..time)
                .rev()
                .take(before_count)
                .map(|(t, v)| (*t, v))
                .collect::<SmallVec<[_; N]>>();
            result.reverse();

            // Get samples at or after the time.
            let after_count = N - result.len();
            result.extend(
                self.values
                    .range(time..)
                    .take(after_count)
                    .map(|(t, v)| (*t, v)),
            );
            result
        }

        #[cfg(feature = "interpolation")]
        {
            let mut result = self
                .values
                .range(..time)
                .rev()
                .take(before_count)
                .map(|(t, (v, _))| (*t, v))
                .collect::<SmallVec<[_; N]>>();
            result.reverse();

            // Get samples at or after the time.
            let after_count = N - result.len();
            result.extend(
                self.values
                    .range(time..)
                    .take(after_count)
                    .map(|(t, (v, _))| (*t, v)),
            );
            result
        }
    }
}

#[cfg(feature = "matrix3")]
#[inline(always)]
pub(crate) fn interpolate_spherical_linear(
    map: &BTreeMap<Time, nalgebra::Rotation2<f32>>,
    time: Time,
) -> nalgebra::Rotation2<f32> {
    if map.len() == 1 {
        return *map.values().next().unwrap();
    }

    let first = map.iter().next().unwrap();
    let last = map.iter().next_back().unwrap();

    if time <= *first.0 {
        return *first.1;
    }

    if time >= *last.0 {
        return *last.1;
    }

    let lower = map.range(..=time).next_back().unwrap();
    let upper = map.range(time..).next().unwrap();

    let (t0, r0): (f32, _) = (lower.0.into(), lower.1);
    let (t1, r1): (f32, _) = (upper.0.into(), upper.1);

    // Normalize `t`.
    let t: f32 = (f32::from(time) - t0) / (t1 - t0);

    r0.slerp(r1, t)
}

// AIDEV-NOTE: Helper functions for converting BezierHandle variants to slopes and evaluating mixed interpolation modes.

#[cfg(feature = "interpolation")]
fn bezier_handle_to_slope<V>(
    handle: &crate::BezierHandle<V>,
    t1: f32,
    t2: f32,
    _v1: &V,
    _v2: &V,
) -> Option<V>
where
    V: Clone + Add<Output = V> + Sub<Output = V> + Mul<f32, Output = V>,
{
    match handle {
        crate::BezierHandle::SlopePerSecond(s) => Some(s.clone()),
        crate::BezierHandle::SlopePerFrame(s) => {
            let frames = (t2 - t1).max(f32::EPSILON);
            Some(s.clone() * (frames / (t2 - t1)))
        }
        crate::BezierHandle::Delta { time, value } => {
            let time_f32: f32 = (*time).into();
            if time_f32.abs() <= f32::EPSILON {
                None
            } else {
                Some(value.clone() * (1.0 / time_f32))
            }
        }
        crate::BezierHandle::Angle(_) => None, // angle needs higher-level context; fall back later
    }
}

#[cfg(feature = "interpolation")]
#[allow(clippy::too_many_arguments)]
fn smooth_tangent<V>(
    t1: f32,
    v1: &V,
    t2: f32,
    v2: &V,
    map: &BTreeMap<Time, (V, Option<crate::Key<V>>)>,
    _time: Time,
    anchor: Time,
    incoming: bool,
) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    use smallvec::SmallVec;

    // Minimal Catmull-style tangent: reuse nearby keys via existing helpers.
    // We reuse interpolate() on a virtual window to let the hermite logic derive slopes.
    let mut window = SmallVec::<[(Time, V); 2]>::new();

    if incoming {
        if let Some(prev) = map.range(..anchor).next_back() {
            window.push((*prev.0, prev.1.0.clone()));
        }
        window.push((anchor, v1.clone()));
    } else {
        window.push((anchor, v1.clone()));
        if let Some(next) = map.range(anchor..).nth(1) {
            window.push((*next.0, next.1.0.clone()));
        }
    }

    if window.len() == 2 {
        let (t0, p0) = &window[0];
        let (t1, p1) = &window[1];
        let t0_f: f32 = (*t0).into();
        let t1_f: f32 = (*t1).into();
        let dt = (t1_f - t0_f).max(f32::EPSILON);
        (p1.clone() - p0.clone()) * (1.0 / dt)
    } else {
        // Fall back to local linear slope.
        (v2.clone() - v1.clone()) * (1.0 / (t2 - t1).max(f32::EPSILON))
    }
}

#[cfg(feature = "interpolation")]
fn evaluate_mixed_bezier<V>(
    time: Time,
    t1: Time,
    v1: &V,
    slope_out: &V,
    t2: Time,
    v2: &V,
    slope_in: &V,
) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    use crate::interpolation::bezier_helpers::*;

    let (p1, p2) = control_points_from_slopes(t1.into(), v1, slope_out, t2.into(), v2, slope_in);

    evaluate_bezier_component_wise(
        time.into(),
        (t1.into(), v1),
        (p1.0, &p1.1),
        (p2.0, &p2.1),
        (t2.into(), v2),
    )
}

/// Interpolate with respect to interpolation specifications.
///
/// This function checks for custom interpolation specs and uses them if available,
/// otherwise falls back to automatic interpolation.
#[cfg(feature = "interpolation")]
#[inline(always)]
pub(crate) fn interpolate_with_specs<V>(
    map: &BTreeMap<Time, (V, Option<crate::Key<V>>)>,
    time: Time,
) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    if map.len() == 1 {
        return map.values().next().unwrap().0.clone();
    }

    let first = map.iter().next().unwrap();
    let last = map.iter().next_back().unwrap();

    if time <= *first.0 {
        return first.1.0.clone();
    }

    if time >= *last.0 {
        return last.1.0.clone();
    }

    // Find surrounding keyframes.
    let lower = map.range(..time).next_back().unwrap();
    let upper = map.range(time..).next().unwrap();

    // If we're exactly on a keyframe, return its value.
    if lower.0 == upper.0 {
        return lower.1.0.clone();
    }

    let (t1, (v1, spec1)) = lower;
    let (t2, (v2, spec2)) = upper;

    // Check if we have interpolation specs.
    let interp_out = spec1.as_ref().map(|s| &s.interpolation_out);
    let interp_in = spec2.as_ref().map(|s| &s.interpolation_in);

    // AIDEV-NOTE: Determine interpolation mode based on specs.
    // Hold takes precedence over everything else, then explicit Bezier handles,
    // then Smooth (Catmull-Rom style), then Linear, then fallback to automatic.
    match (interp_out, interp_in) {
        // Step/hold beats everything else.
        (Some(crate::Interpolation::Hold), _) | (_, Some(crate::Interpolation::Hold)) => v1.clone(),

        // Both sides supply explicit handles -- run a cubic Bezier with those tangents.
        (
            Some(crate::Interpolation::Bezier(out_handle)),
            Some(crate::Interpolation::Bezier(in_handle)),
        ) => {
            use crate::interpolation::bezier_helpers::*;

            if let (Some(slope_out), Some(slope_in)) = (
                bezier_handle_to_slope(out_handle, (*t1).into(), (*t2).into(), v1, v2),
                bezier_handle_to_slope(in_handle, (*t1).into(), (*t2).into(), v1, v2),
            ) {
                let (p1, p2) = control_points_from_slopes(
                    (*t1).into(),
                    v1,
                    &slope_out,
                    (*t2).into(),
                    v2,
                    &slope_in,
                );

                evaluate_bezier_component_wise(
                    time.into(),
                    ((*t1).into(), v1),
                    (p1.0, &p1.1),
                    (p2.0, &p2.1),
                    ((*t2).into(), v2),
                )
            } else {
                // Fall back to linear if we can't convert handles to slopes.
                linear_interp(*t1, *t2, v1, v2, time)
            }
        }

        // One side explicit, the other "smooth": derive a Catmull-style tangent for the smooth side.
        (Some(crate::Interpolation::Bezier(out_handle)), Some(crate::Interpolation::Smooth)) => {
            if let Some(slope_out) =
                bezier_handle_to_slope(out_handle, (*t1).into(), (*t2).into(), v1, v2)
            {
                let slope_in =
                    smooth_tangent((*t1).into(), v1, (*t2).into(), v2, map, time, *t1, false);

                evaluate_mixed_bezier(time, *t1, v1, &slope_out, *t2, v2, &slope_in)
            } else {
                linear_interp(*t1, *t2, v1, v2, time)
            }
        }
        (Some(crate::Interpolation::Smooth), Some(crate::Interpolation::Bezier(in_handle))) => {
            if let Some(slope_in) =
                bezier_handle_to_slope(in_handle, (*t1).into(), (*t2).into(), v1, v2)
            {
                let slope_out =
                    smooth_tangent((*t1).into(), v1, (*t2).into(), v2, map, time, *t1, true);

                evaluate_mixed_bezier(time, *t1, v1, &slope_out, *t2, v2, &slope_in)
            } else {
                linear_interp(*t1, *t2, v1, v2, time)
            }
        }

        // Symmetric "smooth" -> fall back to the existing automatic interpolation.
        (Some(crate::Interpolation::Smooth), Some(crate::Interpolation::Smooth)) | (None, None) => {
            let values_only: BTreeMap<Time, V> =
                map.iter().map(|(k, (v, _))| (*k, v.clone())).collect();
            interpolate(&values_only, time)
        }

        // Linear/linear, linear vs smooth, or any unsupported combination -> straight line.
        _ => linear_interp(*t1, *t2, v1, v2, time),
    }
}

#[inline(always)]
pub(crate) fn interpolate<V>(map: &BTreeMap<Time, V>, time: Time) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    if map.len() == 1 {
        return map.values().next().unwrap().clone();
    }

    let first = map.iter().next().unwrap();
    let last = map.iter().next_back().unwrap();

    if time <= *first.0 {
        return first.1.clone();
    }

    if time >= *last.0 {
        return last.1.clone();
    }

    let lower = map.range(..time).next_back().unwrap();
    let upper = map.range(time..).next().unwrap();

    // This is our window for interpolation that holds two to four values.
    //
    // The interpolation algorithm is chosen based on how many values are in
    // the window. See the `match` block below.
    let mut window = SmallVec::<[(Time, &V); 4]>::new();

    // Extend with up to two values before `lower`.
    window.extend(map.range(..*lower.0).rev().take(2).map(|(k, v)| (*k, v)));

    // Add `lower` and `upper` (if distinct).
    window.push((*lower.0, lower.1));

    if lower.0 != upper.0 {
        window.push((*upper.0, upper.1));
    }

    // Extend with up to one value after `upper`.
    window.extend(map.range(*upper.0..).skip(1).take(1).map(|(k, v)| (*k, v)));

    // Ensure chronological order.
    window.reverse();

    match window.len() {
        4 => {
            let (t0, p0) = window[0];
            let (t1, p1) = window[1];
            let (t2, p2) = window[2];
            let (t3, p3) = window[3];
            hermite_interp(HermiteParams {
                t0,
                t1,
                t2,
                t3,
                p0,
                p1,
                p2,
                p3,
                t: time,
            })
        }
        3 => {
            let (t0, p0) = window[0];
            let (t1, p1) = window[1];
            let (t2, p2) = window[2];
            quadratic_interp(t0, t1, t2, p0, p1, p2, time)
        }
        2 => {
            let (x0, y0) = window[0];
            let (x1, y1) = window[1];
            linear_interp(x0, x1, y0, y1, time)
        }
        1 => {
            // Single keyframe - return its value.
            window[0].1.clone()
        }
        0 => {
            // This shouldn't happen given the checks above, but be defensive.
            panic!("Interpolation window is empty - this is a bug in token-value-map")
        }
        _ => {
            // Window has more than 4 elements - shouldn't happen but use first 4.
            let (t0, p0) = window[0];
            let (t1, p1) = window[1];
            let (t2, p2) = window[2];
            let (t3, p3) = window[3];
            hermite_interp(HermiteParams {
                t0,
                t1,
                t2,
                t3,
                p0,
                p1,
                p2,
                p3,
                t: time,
            })
        }
    }
}

#[inline(always)]
fn linear_interp<V, T>(x0: T, x1: T, y0: &V, y1: &V, x: T) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
    T: Into<f32>,
{
    let x0 = x0.into();
    let x1 = x1.into();
    let x = x.into();
    let alpha = (x - x0) / (x1 - x0);
    y0.clone() + (y1.clone() - y0.clone()) * alpha
}

#[inline(always)]
fn quadratic_interp<V, T>(x0: T, x1: T, x2: T, y0: &V, y1: &V, y2: &V, x: T) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
    T: Into<f32>,
{
    let x0 = x0.into();
    let x1 = x1.into();
    let x2 = x2.into();
    let x = x.into();
    let a = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2));
    let b = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2));
    let c = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1));

    y0.clone() * a + y1.clone() * b + y2.clone() * c
}

struct HermiteParams<V, T> {
    t0: T,
    t1: T,
    t2: T,
    t3: T,
    p0: V,
    p1: V,
    p2: V,
    p3: V,
    t: T,
}

#[inline(always)]
fn hermite_interp<V, T>(params: HermiteParams<&V, T>) -> V
where
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
    T: Into<f32>,
{
    let t0 = params.t0.into();
    let t1 = params.t1.into();
    let t2 = params.t2.into();
    let t3 = params.t3.into();
    let t = params.t.into();

    // Account for knot positions
    let tau = if (t2 - t1).abs() < f32::EPSILON {
        0.0 // Handle degenerate case where t1 == t2
    } else {
        (t - t1) / (t2 - t1)
    };

    // Calculate tension/bias parameters based on the spacing between knots.
    // This makes the spline respond to actual time intervals rather than
    // assuming uniform spacing.
    let tension1 = if (t1 - t0).abs() < f32::EPSILON {
        1.0
    } else {
        (t2 - t1) / (t1 - t0)
    };
    let tension2 = if (t3 - t2).abs() < f32::EPSILON {
        1.0
    } else {
        (t2 - t1) / (t3 - t2)
    };

    // Tangent vectors that respect the actual knot spacing
    let m1 = (params.p2.clone() - params.p0.clone()) * (0.5 * tension1);
    let m2 = (params.p3.clone() - params.p1.clone()) * (0.5 * tension2);

    // Hermite basis functions
    let tau2 = tau * tau;
    let tau3 = tau2 * tau;

    let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
    let h10 = tau3 - 2.0 * tau2 + tau;
    let h01 = -2.0 * tau3 + 3.0 * tau2;
    let h11 = tau3 - tau2;

    // Hermite interpolation with proper tangent vectors
    params.p1.clone() * h00 + m1 * h10 + params.p2.clone() * h01 + m2 * h11
}

#[cfg(feature = "matrix3")]
#[inline(always)]
fn decompose_matrix(
    matrix: &nalgebra::Matrix3<f32>,
) -> (
    nalgebra::Vector2<f32>,
    nalgebra::Rotation2<f32>,
    nalgebra::Matrix3<f32>,
) {
    // Extract translation (assuming the matrix is in homogeneous coordinates).
    let translation = nalgebra::Vector2::new(matrix[(0, 2)], matrix[(1, 2)]);

    // Extract the linear part (upper-left 2x2 matrix).
    let linear_part = matrix.fixed_view::<2, 2>(0, 0).into_owned();

    // Perform Singular Value Decomposition (SVD) to separate rotation and
    // stretch.
    let svd = nalgebra::SVD::new(linear_part, true, true);
    let rotation = nalgebra::Rotation2::from_matrix(&svd.u.unwrap().into_owned());

    // Construct the stretch matrix from singular values.
    let singular_values = svd.singular_values;
    let stretch = nalgebra::Matrix3::new(
        singular_values[0],
        0.0,
        0.0,
        0.0,
        singular_values[1],
        0.0,
        0.0,
        0.0,
        1.0, // Homogeneous coordinate
    );

    (translation, rotation, stretch)
}

#[cfg(feature = "matrix3")]
#[inline(always)]
fn recompose_matrix(
    translation: nalgebra::Vector2<f32>,
    rotation: nalgebra::Rotation2<f32>,
    stretch: nalgebra::Matrix3<f32>,
) -> nalgebra::Matrix3<f32> {
    let rotation = rotation.to_homogeneous();
    let mut combined_linear = rotation * stretch;

    combined_linear[(0, 2)] = translation.x;
    combined_linear[(1, 2)] = translation.y;

    combined_linear
}
