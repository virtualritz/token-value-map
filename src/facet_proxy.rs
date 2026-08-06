//! Plain-Rust proxy types that give the crate's non-introspectable containers a
//! real `facet` representation.
//!
//! Three kinds of type in this crate cannot be introspected by `facet` on their
//! own:
//!
//! - the math wrapper types ([`Vector3`](crate::Vector3),
//!   [`Matrix4`](crate::Matrix4), ...), whose payload is a backend type from
//!   `glam`/`nalgebra`/`ultraviolet` and therefore not `Facet`;
//! - [`KeyDataMap`](crate::KeyDataMap), whose storage is `mitsein`'s
//!   `BTreeMap1`, likewise not `Facet`.
//!
//! Each such type carries `#[facet(opaque)]` -- which is what lets the derive
//! skip the payload -- together with `#[facet(proxy = ...)]`, which points at a
//! plain-Rust stand-in that *is* introspectable. `facet` resolves a container
//! proxy ahead of the opaque marker in both directions, so the opaque marker is
//! inert for serialization and only serves to keep the derive off the payload.
//!
//! The conversions in this module are **total in both directions**. The
//! serialize direction cannot fail because every inhabited value has a proxy.
//! The deserialize direction cannot fail because the proxy shape makes an
//! invalid proxy unrepresentable: a keyframe map's proxy carries its first
//! keyframe as a separate, mandatory field, so the non-emptiness `BTreeMap1`
//! demands is enforced by the wire shape rather than by a runtime check. This
//! matters beyond tidiness -- `facet-reflect` 0.46.5 discards the error string a
//! failing `convert_in` returns (`partial/partial_api/misc.rs`,
//! `complete_proxy_frame`), so a runtime rejection here would reach the user as
//! "value was not initialized" instead of its actual reason. A missing
//! mandatory field, by contrast, is reported by name.

#[cfg(feature = "interpolation")]
use crate::Key;
use crate::*;
use facet::Facet;
use mitsein::btree_map1::BTreeMap1;
use std::collections::BTreeMap;

/// The interpolation key of one keyframe, in a form TOML can carry.
///
/// [`KeyDataMap`] stores `Option<Key<V>>` per keyframe. `Option::None` has no
/// TOML spelling and `facet`'s `skip_serializing_if` cannot be used on a
/// generic container (the derive emits a helper `fn` that names the field type,
/// which is a compile error when that type mentions a type parameter), so the
/// absent case is a variant of its own instead.
#[cfg(feature = "interpolation")]
#[derive(Facet, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum KeyframeInterpolation<T> {
    /// The keyframe carries no explicit interpolation key.
    Unset,
    /// The keyframe carries an explicit interpolation key.
    Set(Key<T>),
}

#[cfg(feature = "interpolation")]
impl<T> From<Option<Key<T>>> for KeyframeInterpolation<T> {
    fn from(key: Option<Key<T>>) -> Self {
        match key {
            None => KeyframeInterpolation::Unset,
            Some(key) => KeyframeInterpolation::Set(key),
        }
    }
}

#[cfg(feature = "interpolation")]
impl<T> From<KeyframeInterpolation<T>> for Option<Key<T>> {
    fn from(key: KeyframeInterpolation<T>) -> Self {
        match key {
            KeyframeInterpolation::Unset => None,
            KeyframeInterpolation::Set(key) => Some(key),
        }
    }
}

/// One keyframe of a [`KeyDataMap`].
#[derive(Facet, Clone, Debug, PartialEq)]
pub struct KeyframeProxy<K, V> {
    /// The keyframe's position in its domain -- a tick for animated values, a
    /// [`Position`](crate::Position) for curves.
    pub key: K,
    /// The value at that key.
    pub value: V,
    /// The keyframe's interpolation key.
    #[cfg(feature = "interpolation")]
    pub interpolation: KeyframeInterpolation<V>,
}

/// The wire shape of a [`KeyDataMap`].
///
/// `first` is a separate field rather than the head of a list so that the
/// non-emptiness `BTreeMap1` guarantees is a property of the shape. A document
/// that omits it fails to deserialize with a message naming the missing field.
#[derive(Facet, Clone, Debug, PartialEq)]
pub struct KeyDataMapProxy<K, V> {
    /// The lowest-keyed keyframe. Always present.
    pub first: KeyframeProxy<K, V>,
    /// The remaining keyframes, in key order.
    pub rest: Vec<KeyframeProxy<K, V>>,
}

// AIDEV-NOTE: `facet` asks for `TryFrom` in both directions. These are `From`
// impls, and `core`'s blanket `TryFrom<U> for T where U: Into<T>` supplies the
// `TryFrom` with `Error = Infallible`. Totality is then a property of the type
// rather than a claim in a doc comment.
impl<K: Clone + Ord, V: Clone> From<&KeyDataMap<K, V>> for KeyDataMapProxy<K, V> {
    fn from(map: &KeyDataMap<K, V>) -> Self {
        #[cfg(not(feature = "interpolation"))]
        let mut entries = map
            .values
            .as_btree_map()
            .iter()
            .map(|(key, value)| KeyframeProxy {
                key: key.clone(),
                value: value.clone(),
            });
        #[cfg(feature = "interpolation")]
        let mut entries = map
            .values
            .as_btree_map()
            .iter()
            .map(|(key, (value, interpolation))| KeyframeProxy {
                key: key.clone(),
                value: value.clone(),
                interpolation: interpolation.clone().into(),
            });

        // SAFETY: `BTreeMap1` is non-empty by type, so the first entry exists.
        let first = entries.next().unwrap();

        KeyDataMapProxy {
            first,
            rest: entries.collect(),
        }
    }
}

// AIDEV-NOTE: this direction stays a `TryFrom` rather than a `From`. A `From`
// would pull in `core`'s blanket `TryFrom<U> for T where U: Into<T>`, which
// makes every existing `KeyDataMap::try_from(some_btree_map)` call ambiguous.
// The error type is `Infallible`, so nothing is lost but the lint.
#[allow(clippy::infallible_try_from)]
impl<K: Ord, V> TryFrom<KeyDataMapProxy<K, V>> for KeyDataMap<K, V> {
    type Error = std::convert::Infallible;

    fn try_from(proxy: KeyDataMapProxy<K, V>) -> std::result::Result<Self, Self::Error> {
        #[cfg(not(feature = "interpolation"))]
        let values: BTreeMap<K, V> = std::iter::once(proxy.first)
            .chain(proxy.rest)
            .map(|entry| (entry.key, entry.value))
            .collect();
        #[cfg(feature = "interpolation")]
        let values: BTreeMap<K, (V, Option<Key<V>>)> = std::iter::once(proxy.first)
            .chain(proxy.rest)
            .map(|entry| (entry.key, (entry.value, entry.interpolation.into())))
            .collect();

        // SAFETY: `first` is a mandatory field, so the map has at least one entry.
        let values = BTreeMap1::try_from(values).ok().unwrap();

        Ok(KeyDataMap { values })
    }
}

/// The wire stand-in for a math wrapper type's payload.
///
/// A bare array cannot serve as the proxy: several wrapper types already carry
/// a `From<[f32; N]>` impl of their own -- with a different element order --
/// which collides with `core`'s blanket `TryFrom`. A newtype sidesteps that and
/// keeps the wire spelling identical, because it is transparent.
#[derive(Facet, Clone, Debug, PartialEq)]
#[facet(transparent)]
pub struct MathProxy<T>(pub T);

/// Generate the total, infallible proxy conversions of a math wrapper type.
///
/// `$target` is the wrapper, `$proxy` the plain-Rust stand-in named by its
/// `#[facet(proxy = ...)]` attribute, and the two expressions are the accessor
/// and the constructor of the active math backend.
#[allow(unused_macros)]
macro_rules! impl_math_proxy {
    ($target:ty, $proxy:ty, |$out:ident| $to_proxy:expr, |$in:ident| $from_proxy:expr) => {
        impl From<&$target> for MathProxy<$proxy> {
            fn from($out: &$target) -> Self {
                MathProxy($to_proxy)
            }
        }

        impl From<MathProxy<$proxy>> for $target {
            fn from(proxy: MathProxy<$proxy>) -> Self {
                let $in = proxy.0;
                $from_proxy
            }
        }
    };
}

#[cfg(feature = "vector2")]
impl_math_proxy!(
    Vector2,
    [f32; 2],
    |value| *math::vec2_as_ref(&value.0),
    |array| Vector2(math::vec2_from_array(array))
);

#[cfg(feature = "vector3")]
impl_math_proxy!(
    Vector3,
    [f32; 3],
    |value| *math::vec3_as_ref(&value.0),
    |array| Vector3(math::vec3_from_array(array))
);

#[cfg(feature = "normal3")]
impl_math_proxy!(
    Normal3,
    [f32; 3],
    |value| *math::vec3_as_ref(&value.0),
    |array| Normal3(math::vec3_from_array(array))
);

#[cfg(feature = "point3")]
impl_math_proxy!(
    Point3,
    [f32; 3],
    |value| *math::point3_as_ref(&value.0),
    |array| Point3(math::point3_from_array(array))
);

// Matrices are stored column-major, which is the layout every backend's
// `mat*_as_slice` exposes. There is no transpose on the way out and none on the
// way in.
#[cfg(feature = "matrix3")]
impl_math_proxy!(
    Matrix3,
    [f32; 9],
    |value| column_major_9(&value.0),
    |array| Matrix3(math::mat3_from_column_slice(&array))
);

#[cfg(feature = "matrix4")]
impl_math_proxy!(
    Matrix4,
    [f64; 16],
    |value| column_major_16(&value.0),
    |array| Matrix4(math::mat4_from_column_slice(&array))
);

#[cfg(all(feature = "vector2", feature = "vec_variants"))]
impl_math_proxy!(
    Vector2Vec,
    Vec<[f32; 2]>,
    |value| value.0.iter().map(|v| *math::vec2_as_ref(v)).collect(),
    |array| Vector2Vec(array.into_iter().map(math::vec2_from_array).collect())
);

#[cfg(all(feature = "vector3", feature = "vec_variants"))]
impl_math_proxy!(
    Vector3Vec,
    Vec<[f32; 3]>,
    |value| value.0.iter().map(|v| *math::vec3_as_ref(v)).collect(),
    |array| Vector3Vec(array.into_iter().map(math::vec3_from_array).collect())
);

#[cfg(all(feature = "normal3", feature = "vec_variants"))]
impl_math_proxy!(
    Normal3Vec,
    Vec<[f32; 3]>,
    |value| value.0.iter().map(|v| *math::vec3_as_ref(v)).collect(),
    |array| Normal3Vec(array.into_iter().map(math::vec3_from_array).collect())
);

#[cfg(all(feature = "point3", feature = "vec_variants"))]
impl_math_proxy!(
    Point3Vec,
    Vec<[f32; 3]>,
    |value| value.0.iter().map(|p| *math::point3_as_ref(p)).collect(),
    |array| Point3Vec(array.into_iter().map(math::point3_from_array).collect())
);

#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
impl_math_proxy!(
    Matrix3Vec,
    Vec<[f32; 9]>,
    |value| value.0.iter().map(column_major_9).collect(),
    |array| Matrix3Vec(
        array
            .iter()
            .map(|m| math::mat3_from_column_slice(m))
            .collect()
    )
);

#[cfg(all(feature = "matrix4", feature = "vec_variants"))]
impl_math_proxy!(
    Matrix4Vec,
    Vec<[f64; 16]>,
    |value| value.0.iter().map(column_major_16).collect(),
    |array| Matrix4Vec(
        array
            .iter()
            .map(|m| math::mat4_from_column_slice(m))
            .collect()
    )
);

/// Copy a 3x3 matrix out in column-major order.
#[cfg(feature = "matrix3")]
fn column_major_9(matrix: &math::Mat3Impl) -> [f32; 9] {
    let slice = math::mat3_as_slice(matrix);
    // SAFETY: every backend's `mat3_as_slice` yields exactly nine elements.
    slice.try_into().unwrap()
}

/// Copy a 4x4 matrix out in column-major order.
#[cfg(feature = "matrix4")]
fn column_major_16(matrix: &math::Mat4Impl) -> [f64; 16] {
    let slice = math::mat4_as_slice(matrix);
    // SAFETY: every backend's `mat4_as_slice` yields exactly sixteen elements.
    slice.try_into().unwrap()
}
