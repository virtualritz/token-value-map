use crate::*;
use enum_dispatch::enum_dispatch;

use std::{
    collections::BTreeMap,
    iter::FromIterator,
    ops::{Add, Mul, Sub},
};

mod sample;
pub use sample::*;

/// A generic key-value data map with interpolation support.
///
/// Stores key-value pairs in a [`BTreeMap`] for efficient ordered
/// queries and supports various interpolation methods.
///
/// When the `interpolation` feature is enabled, each value can have an
/// associated interpolation specification for animation curves.
///
/// Use the type alias [`TimeDataMap<V>`] for time-keyed maps (the common case),
/// or use `KeyDataMap<Position, V>` for curve-domain maps.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "K: Serialize + Ord, V: Serialize",
        deserialize = "K: Deserialize<'de> + Ord, V: Deserialize<'de>",
    ))
)]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "facet", facet(opaque))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct KeyDataMap<K, V> {
    /// The key-value pairs with optional interpolation keys.
    #[cfg(not(feature = "interpolation"))]
    pub values: BTreeMap<K, V>,
    #[cfg(feature = "interpolation")]
    pub values: BTreeMap<K, (V, Option<crate::Key<V>>)>,
}

/// A time-keyed data map. Alias for `KeyDataMap<Time, V>`.
pub type TimeDataMap<V> = KeyDataMap<Time, V>;

// Manual Eq implementation.
impl<K: Eq, V: Eq> Eq for KeyDataMap<K, V> {}

// AsRef implementation for backward compatibility.
#[cfg(not(feature = "interpolation"))]
impl<K, V> AsRef<BTreeMap<K, V>> for KeyDataMap<K, V> {
    fn as_ref(&self) -> &BTreeMap<K, V> {
        &self.values
    }
}

impl<K: Ord, V> KeyDataMap<K, V> {
    /// Get an iterator over key-value pairs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.iter()
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.iter().map(|(k, (v, _))| (k, v))
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

    /// Remove a sample at the given key.
    ///
    /// Returns the removed value if it existed.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.remove(key)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.remove(key).map(|(v, _)| v)
        }
    }
}

// Constructor from BTreeMap.
impl<K: Ord, V> From<BTreeMap<K, V>> for KeyDataMap<K, V> {
    fn from(values: BTreeMap<K, V>) -> Self {
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

impl<V> TimeDataMapControl<V> for KeyDataMap<Time, V> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn is_animated(&self) -> bool {
        self.values.len() > 1
    }
}

#[cfg(feature = "builtin-types")]
impl<K, V> crate::DataTypeOps for KeyDataMap<K, V>
where
    V: crate::DataTypeOps,
{
    fn data_type(&self) -> crate::DataType {
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

// AIDEV-NOTE: These From impls are only available with builtin-types feature.
#[cfg(feature = "builtin-types")]
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

#[cfg(feature = "builtin-types")]
impl_from_at_time!(
    Boolean, Real, Integer, String, Color, BooleanVec, RealVec, IntegerVec, StringVec, ColorVec,
    Data
);

#[cfg(all(feature = "builtin-types", feature = "curves"))]
impl_from_at_time!(RealCurve, ColorCurve);

#[cfg(all(feature = "builtin-types", feature = "vector2"))]
impl_from_at_time!(Vector2);
#[cfg(all(feature = "builtin-types", feature = "vector3"))]
impl_from_at_time!(Vector3);
#[cfg(all(feature = "builtin-types", feature = "matrix3"))]
impl_from_at_time!(Matrix3);
#[cfg(all(feature = "builtin-types", feature = "normal3"))]
impl_from_at_time!(Normal3);
#[cfg(all(feature = "builtin-types", feature = "point3"))]
impl_from_at_time!(Point3);
#[cfg(all(feature = "builtin-types", feature = "matrix4"))]
impl_from_at_time!(Matrix4);

#[cfg(all(
    feature = "builtin-types",
    feature = "vector2",
    feature = "vec_variants"
))]
impl_from_at_time!(Vector2Vec);
#[cfg(all(
    feature = "builtin-types",
    feature = "vector3",
    feature = "vec_variants"
))]
impl_from_at_time!(Vector3Vec);
#[cfg(all(
    feature = "builtin-types",
    feature = "matrix3",
    feature = "vec_variants"
))]
impl_from_at_time!(Matrix3Vec);
#[cfg(all(
    feature = "builtin-types",
    feature = "normal3",
    feature = "vec_variants"
))]
impl_from_at_time!(Normal3Vec);
#[cfg(all(
    feature = "builtin-types",
    feature = "point3",
    feature = "vec_variants"
))]
impl_from_at_time!(Point3Vec);
#[cfg(all(
    feature = "builtin-types",
    feature = "matrix4",
    feature = "vec_variants"
))]
impl_from_at_time!(Matrix4Vec);

impl<K: Ord, V> KeyDataMap<K, V> {
    /// Insert a value at the given key.
    pub fn insert(&mut self, key: K, value: V) {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.insert(key, value);
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.insert(key, (value, None));
        }
    }

    /// Get a value at the exact key.
    pub fn get(&self, key: &K) -> Option<&V> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.get(key)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.get(key).map(|(v, _)| v)
        }
    }
}

impl<K, V> KeyDataMap<K, V>
where
    K: Ord + Copy + Into<f32>,
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    pub fn interpolate(&self, key: K) -> V {
        #[cfg(not(feature = "interpolation"))]
        {
            interpolate(&self.values, key)
        }
        #[cfg(feature = "interpolation")]
        {
            interpolate_with_specs(&self.values, key)
        }
    }
}

// Interpolation-specific methods.
#[cfg(feature = "interpolation")]
impl<K: Ord, V> KeyDataMap<K, V> {
    /// Insert a value with interpolation specification.
    ///
    /// Sets both the value at the given key and how it should interpolate
    /// to neighboring keyframes.
    pub fn insert_with_interpolation(&mut self, key: K, value: V, spec: crate::Key<V>) {
        self.values.insert(key, (value, Some(spec)));
    }

    /// Get interpolation spec at a given key.
    ///
    /// Returns `None` if no interpolation metadata exists or no spec at this key.
    pub fn interpolation(&self, key: &K) -> Option<&crate::Key<V>> {
        self.values.get(key)?.1.as_ref()
    }

    /// Set or update the interpolation spec at a given key.
    ///
    /// Returns `Ok(())` if the key existed and was updated, or an error if no
    /// sample exists at that key.
    pub fn set_interpolation_at(&mut self, key: &K, spec: crate::Key<V>) -> crate::Result<()> {
        if let Some(entry) = self.values.get_mut(key) {
            entry.1 = Some(spec);
            Ok(())
        } else {
            Err(crate::Error::KeyNotFound)
        }
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

impl<K: Ord, V, U> FromIterator<(K, U)> for KeyDataMap<K, V>
where
    U: Into<V>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, U)>,
    {
        Self::from(BTreeMap::from_iter(
            iter.into_iter().map(|(k, v)| (k, v.into())),
        ))
    }
}

impl<K: Ord + Copy + Into<f32>, V> KeyDataMap<K, V> {
    pub fn closest_sample(&self, key: K) -> &V {
        let k_f32: f32 = key.into();
        #[cfg(not(feature = "interpolation"))]
        {
            let mut range = self.values.range(key..);
            let greater_or_equal = range.next();

            let mut range = self.values.range(..key);
            let less_than = range.next_back();

            match (less_than, greater_or_equal) {
                (Some((lower_k, lower_v)), Some((upper_k, upper_v))) => {
                    let lo: f32 = (*lower_k).into();
                    let hi: f32 = (*upper_k).into();
                    if (k_f32 - lo).abs() <= (hi - k_f32).abs() {
                        lower_v
                    } else {
                        upper_v
                    }
                }
                (Some(entry), None) | (None, Some(entry)) => entry.1,
                (None, None) => {
                    unreachable!("KeyDataMap can never be empty")
                }
            }
        }
        #[cfg(feature = "interpolation")]
        {
            let mut range = self.values.range(key..);
            let greater_or_equal = range.next();

            let mut range = self.values.range(..key);
            let less_than = range.next_back();

            match (less_than, greater_or_equal) {
                (Some((lower_k, (lower_v, _))), Some((upper_k, (upper_v, _)))) => {
                    let lo: f32 = (*lower_k).into();
                    let hi: f32 = (*upper_k).into();
                    if (k_f32 - lo).abs() <= (hi - k_f32).abs() {
                        lower_v
                    } else {
                        upper_v
                    }
                }
                (Some((_, (v, _))), None) | (None, Some((_, (v, _)))) => v,
                (None, None) => {
                    unreachable!("KeyDataMap can never be empty")
                }
            }
        }
    }

    /// Sample value at exact key (no interpolation).
    pub fn sample_at(&self, key: K) -> Option<&V> {
        self.get(&key)
    }

    /// Get the value at or before the given key.
    pub fn sample_at_or_before(&self, key: K) -> Option<&V> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.range(..=key).next_back().map(|(_, v)| v)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.range(..=key).next_back().map(|(_, (v, _))| v)
        }
    }

    /// Get the value at or after the given key.
    pub fn sample_at_or_after(&self, key: K) -> Option<&V> {
        #[cfg(not(feature = "interpolation"))]
        {
            self.values.range(key..).next().map(|(_, v)| v)
        }
        #[cfg(feature = "interpolation")]
        {
            self.values.range(key..).next().map(|(_, (v, _))| v)
        }
    }

    /// Get surrounding samples for interpolation.
    ///
    /// Returns up to `N` samples centered around the given key for
    /// use in interpolation algorithms.
    pub fn sample_surrounding<const N: usize>(&self, key: K) -> SmallVec<[(K, &V); N]> {
        // Get samples before the key.
        let before_count = N / 2;

        #[cfg(not(feature = "interpolation"))]
        {
            let mut result = self
                .values
                .range(..key)
                .rev()
                .take(before_count)
                .map(|(k, v)| (*k, v))
                .collect::<SmallVec<[_; N]>>();
            result.reverse();

            // Get samples at or after the key.
            let after_count = N - result.len();
            result.extend(
                self.values
                    .range(key..)
                    .take(after_count)
                    .map(|(k, v)| (*k, v)),
            );
            result
        }

        #[cfg(feature = "interpolation")]
        {
            let mut result = self
                .values
                .range(..key)
                .rev()
                .take(before_count)
                .map(|(k, (v, _))| (*k, v))
                .collect::<SmallVec<[_; N]>>();
            result.reverse();

            // Get samples at or after the key.
            let after_count = N - result.len();
            result.extend(
                self.values
                    .range(key..)
                    .take(after_count)
                    .map(|(k, (v, _))| (*k, v)),
            );
            result
        }
    }
}

// AIDEV-NOTE: Internal types for backend-agnostic Matrix3 SVD decomposition.
// These implement the arithmetic traits needed by the generic `interpolate()`
// function, avoiding any dependency on a specific math backend for SVD compute.
// The analytical 2×2 SVD replaces the previous nalgebra-dependent implementation.

/// Internal 2D translation extracted from a 3×3 homogeneous matrix.
#[cfg(feature = "matrix3")]
#[derive(Clone)]
struct Trans2(f32, f32);

#[cfg(feature = "matrix3")]
impl Add for Trans2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Trans2(self.0 + rhs.0, self.1 + rhs.1)
    }
}

#[cfg(feature = "matrix3")]
impl Sub for Trans2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Trans2(self.0 - rhs.0, self.1 - rhs.1)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<f32> for Trans2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Trans2(self.0 * s, self.1 * s)
    }
}

/// Internal diagonal stretch (singular values) for decomposed 3×3 matrices.
#[cfg(feature = "matrix3")]
#[derive(Clone)]
struct DiagStretch(f32, f32);

#[cfg(feature = "matrix3")]
impl Add for DiagStretch {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        DiagStretch(self.0 + rhs.0, self.1 + rhs.1)
    }
}

#[cfg(feature = "matrix3")]
impl Sub for DiagStretch {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        DiagStretch(self.0 - rhs.0, self.1 - rhs.1)
    }
}

#[cfg(feature = "matrix3")]
impl Mul<f32> for DiagStretch {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        DiagStretch(self.0 * s, self.1 * s)
    }
}

/// Interpolate 2D rotation angles via shortest-path slerp across a [`BTreeMap`].
#[cfg(feature = "matrix3")]
#[inline(always)]
fn interpolate_rotation(map: &BTreeMap<Time, f32>, time: Time) -> f32 {
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

    let t0 = f32::from(*lower.0);
    let t1 = f32::from(*upper.0);
    let a0 = *lower.1;
    let a1 = *upper.1;

    // Normalize `t` to [0, 1].
    let t: f32 = (f32::from(time) - t0) / (t1 - t0);

    // Shortest-path angle interpolation.
    use std::f32::consts::{PI, TAU};
    let mut diff = a1 - a0;
    if diff > PI {
        diff -= TAU;
    } else if diff < -PI {
        diff += TAU;
    }
    a0 + diff * t
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
fn smooth_tangent<K, V>(
    t1: f32,
    v1: &V,
    t2: f32,
    v2: &V,
    map: &BTreeMap<K, (V, Option<crate::Key<V>>)>,
    _key: K,
    anchor: K,
    incoming: bool,
) -> V
where
    K: Ord + Copy + Into<f32>,
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    use smallvec::SmallVec;

    // Minimal Catmull-style tangent: reuse nearby keys via existing helpers.
    let mut window = SmallVec::<[(K, V); 2]>::new();

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
        let (k0, p0) = &window[0];
        let (k1, p1) = &window[1];
        let k0_f: f32 = (*k0).into();
        let k1_f: f32 = (*k1).into();
        let dt = (k1_f - k0_f).max(f32::EPSILON);
        (p1.clone() - p0.clone()) * (1.0 / dt)
    } else {
        // Fall back to local linear slope.
        (v2.clone() - v1.clone()) * (1.0 / (t2 - t1).max(f32::EPSILON))
    }
}

#[cfg(feature = "interpolation")]
fn evaluate_mixed_bezier<K, V>(
    key: K,
    k1: K,
    v1: &V,
    slope_out: &V,
    k2: K,
    v2: &V,
    slope_in: &V,
) -> V
where
    K: Copy + Into<f32>,
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    use crate::interpolation::bezier_helpers::*;

    let (p1, p2) = control_points_from_slopes(k1.into(), v1, slope_out, k2.into(), v2, slope_in);

    evaluate_bezier_component_wise(
        key.into(),
        (k1.into(), v1),
        (p1.0, &p1.1),
        (p2.0, &p2.1),
        (k2.into(), v2),
    )
}

/// Interpolate with respect to interpolation specifications.
///
/// This function checks for custom interpolation specs and uses them if available,
/// otherwise falls back to automatic interpolation.
#[cfg(feature = "interpolation")]
#[inline(always)]
pub(crate) fn interpolate_with_specs<K, V>(
    map: &BTreeMap<K, (V, Option<crate::Key<V>>)>,
    key: K,
) -> V
where
    K: Ord + Copy + Into<f32>,
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    if map.len() == 1 {
        return map.values().next().unwrap().0.clone();
    }

    let first = map.iter().next().unwrap();
    let last = map.iter().next_back().unwrap();

    if key <= *first.0 {
        return first.1.0.clone();
    }

    if key >= *last.0 {
        return last.1.0.clone();
    }

    // Find surrounding keyframes.
    let lower = map.range(..key).next_back().unwrap();
    let upper = map.range(key..).next().unwrap();

    // If we're exactly on a keyframe, return its value.
    if lower.0 == upper.0 {
        return lower.1.0.clone();
    }

    let (k1, (v1, spec1)) = lower;
    let (k2, (v2, spec2)) = upper;

    // Check if we have interpolation specs.
    let interp_out = spec1.as_ref().map(|s| &s.interpolation_out);
    let interp_in = spec2.as_ref().map(|s| &s.interpolation_in);

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
                bezier_handle_to_slope(out_handle, (*k1).into(), (*k2).into(), v1, v2),
                bezier_handle_to_slope(in_handle, (*k1).into(), (*k2).into(), v1, v2),
            ) {
                let (p1, p2) = control_points_from_slopes(
                    (*k1).into(),
                    v1,
                    &slope_out,
                    (*k2).into(),
                    v2,
                    &slope_in,
                );

                evaluate_bezier_component_wise(
                    key.into(),
                    ((*k1).into(), v1),
                    (p1.0, &p1.1),
                    (p2.0, &p2.1),
                    ((*k2).into(), v2),
                )
            } else {
                // Fall back to linear if we can't convert handles to slopes.
                linear_interp(*k1, *k2, v1, v2, key)
            }
        }

        // One side explicit, the other "smooth": derive a Catmull-style tangent for the smooth side.
        (Some(crate::Interpolation::Bezier(out_handle)), Some(crate::Interpolation::Smooth)) => {
            if let Some(slope_out) =
                bezier_handle_to_slope(out_handle, (*k1).into(), (*k2).into(), v1, v2)
            {
                let slope_in =
                    smooth_tangent((*k1).into(), v1, (*k2).into(), v2, map, key, *k1, false);

                evaluate_mixed_bezier(key, *k1, v1, &slope_out, *k2, v2, &slope_in)
            } else {
                linear_interp(*k1, *k2, v1, v2, key)
            }
        }
        (Some(crate::Interpolation::Smooth), Some(crate::Interpolation::Bezier(in_handle))) => {
            if let Some(slope_in) =
                bezier_handle_to_slope(in_handle, (*k1).into(), (*k2).into(), v1, v2)
            {
                let slope_out =
                    smooth_tangent((*k1).into(), v1, (*k2).into(), v2, map, key, *k1, true);

                evaluate_mixed_bezier(key, *k1, v1, &slope_out, *k2, v2, &slope_in)
            } else {
                linear_interp(*k1, *k2, v1, v2, key)
            }
        }

        // Symmetric "smooth" -> fall back to the existing automatic interpolation.
        (Some(crate::Interpolation::Smooth), Some(crate::Interpolation::Smooth)) | (None, None) => {
            let values_only: BTreeMap<K, V> =
                map.iter().map(|(k, (v, _))| (*k, v.clone())).collect();
            interpolate(&values_only, key)
        }

        // Linear/linear, linear vs smooth, or any unsupported combination -> straight line.
        _ => linear_interp(*k1, *k2, v1, v2, key),
    }
}

#[inline(always)]
pub(crate) fn interpolate<K, V>(map: &BTreeMap<K, V>, key: K) -> V
where
    K: Ord + Copy + Into<f32>,
    V: Clone + Add<Output = V> + Mul<f32, Output = V> + Sub<Output = V>,
{
    if map.len() == 1 {
        return map.values().next().unwrap().clone();
    }

    let first = map.iter().next().unwrap();
    let last = map.iter().next_back().unwrap();

    if key <= *first.0 {
        return first.1.clone();
    }

    if key >= *last.0 {
        return last.1.clone();
    }

    let lower = map.range(..key).next_back().unwrap();
    let upper = map.range(key..).next().unwrap();

    // This is our window for interpolation that holds two to four values.
    //
    // The interpolation algorithm is chosen based on how many values are in
    // the window. See the `match` block below.
    let mut window = SmallVec::<[(K, &V); 4]>::new();

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
                t: key,
            })
        }
        3 => {
            let (t0, p0) = window[0];
            let (t1, p1) = window[1];
            let (t2, p2) = window[2];
            quadratic_interp(t0, t1, t2, p0, p1, p2, key)
        }
        2 => {
            let (x0, y0) = window[0];
            let (x1, y1) = window[1];
            linear_interp(x0, x1, y0, y1, key)
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
                t: key,
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

/// Analytical 2×2 SVD decomposition of a 3×3 homogeneous matrix.
///
/// Extracts translation, rotation angle (radians), and diagonal stretch
/// (singular values) without depending on any specific math backend.
#[cfg(feature = "matrix3")]
#[inline(always)]
fn decompose_matrix(matrix: &crate::math::Mat3Impl) -> (Trans2, f32, DiagStretch) {
    use crate::math::mat3;

    // Extract translation from the last column.
    let tx = mat3(matrix, 0, 2);
    let ty = mat3(matrix, 1, 2);

    // Upper-left 2×2 linear block.
    let a = mat3(matrix, 0, 0);
    let b = mat3(matrix, 0, 1);
    let c = mat3(matrix, 1, 0);
    let d = mat3(matrix, 1, 1);

    // Analytical 2×2 SVD: M = Rot(θ) × diag(σ₁, σ₂) × Rot(-φ).
    let e = (a + d) * 0.5;
    let f = (a - d) * 0.5;
    let g = (c + b) * 0.5;
    let h = (c - b) * 0.5;

    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();

    let s1 = q + r;
    let s2 = q - r;

    let theta1 = g.atan2(f);
    let theta2 = h.atan2(e);

    // U rotation angle.
    let rotation_angle = (theta2 + theta1) * 0.5;

    (Trans2(tx, ty), rotation_angle, DiagStretch(s1, s2))
}

/// Recompose a 3×3 homogeneous matrix from its decomposed parts.
///
/// Constructs `Rot(angle) × diag(sx, sy, 1)` with translation in the last column.
#[cfg(feature = "matrix3")]
#[inline(always)]
fn recompose_matrix(
    translation: Trans2,
    rotation_angle: f32,
    stretch: DiagStretch,
) -> crate::math::Mat3Impl {
    let cos = rotation_angle.cos();
    let sin = rotation_angle.sin();

    // Rot(θ) × diag(sx, sy, 1) in column-major layout.
    crate::math::mat3_from_column_slice(&[
        stretch.0 * cos,
        stretch.0 * sin,
        0.0,
        -stretch.1 * sin,
        stretch.1 * cos,
        0.0,
        translation.0,
        translation.1,
        1.0,
    ])
}
