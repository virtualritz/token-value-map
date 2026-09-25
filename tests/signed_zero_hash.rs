//! `==` compares floats by value, so `-0.0 == 0.0`, and equal values must
//! hash equally. `Data` and `AnimatedData` hashed their floats' raw bits,
//! which differ between the two zeros, so a `HashMap` or `HashSet` keyed on
//! them could hold both and miss on lookup. The wrapper types already hashed
//! `-0.0` as `0.0`; the enums now hash through them.

use std::{
    collections::hash_map::DefaultHasher,
    fmt::Debug,
    hash::{Hash, Hasher},
};
use token_value_map::*;

fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Asserts the two values compare equal and hash equally.
fn assert_equal_and_hash_equally<T: Hash + PartialEq + Debug>(positive: T, negative: T) {
    assert_eq!(positive, negative);
    assert_eq!(hash_of(&positive), hash_of(&negative), "{positive:?}");
}

#[test]
fn data_hashes_signed_zero_as_zero() {
    assert_equal_and_hash_equally(Data::Real(Real(0.0)), Data::Real(Real(-0.0)));
    assert_equal_and_hash_equally(
        Data::Color(Color([0.0, 1.0, 0.0, 1.0])),
        Data::Color(Color([-0.0, 1.0, -0.0, 1.0])),
    );
    assert_equal_and_hash_equally(
        Data::RealVec(RealVec(vec![0.0, 2.0])),
        Data::RealVec(RealVec(vec![-0.0, 2.0])),
    );
    #[cfg(feature = "vector3")]
    assert_equal_and_hash_equally(
        Data::Vector3(Vector3(math::Vec3Impl::new(0.0, 1.0, 2.0))),
        Data::Vector3(Vector3(math::Vec3Impl::new(-0.0, 1.0, 2.0))),
    );
}

#[test]
fn animated_data_hashes_signed_zero_as_zero() {
    let at = |value: f64| {
        AnimatedData::Real(TimeDataMap::from_iter(vec![(
            Time::from_secs(0.0),
            Real(value),
        )]))
    };
    assert_equal_and_hash_equally(at(0.0), at(-0.0));
}

/// The hash still tells values apart that `==` tells apart.
#[test]
fn different_values_still_hash_apart() {
    assert_ne!(
        hash_of(&Data::Real(Real(0.0))),
        hash_of(&Data::Real(Real(1.0)))
    );
}
