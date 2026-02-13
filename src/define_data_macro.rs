//! Macro for defining custom data type systems.
//!
//! This module provides the [`define_data_types!`] macro which generates
//! all the boilerplate needed to create a custom data system compatible
//! with [`GenericValue`](crate::GenericValue) and
//! [`GenericTokenValueMap`](crate::GenericTokenValueMap).

/// Define a custom data type system with full interpolation support.
///
/// This macro generates three enums and implements all necessary traits
/// for use with [`GenericValue`](crate::GenericValue) and
/// [`GenericTokenValueMap`](crate::GenericTokenValueMap):
///
/// 1. A discriminant enum (like `DataType`) with unit variants.
/// 2. A data enum (like `Data`) holding actual values.
/// 3. An animated data enum (like `AnimatedData`) holding `TimeDataMap<T>` values.
///
/// # Example
///
/// ```rust,ignore
/// use token_value_map::{define_data_types, TimeDataMap, Time, DataSystem};
///
/// define_data_types! {
///     /// My custom data types.
///     #[derive(Clone, Debug, PartialEq)]
///     pub MyData / MyAnimatedData / MyDataType {
///         /// A floating point value.
///         Float(MyFloat),
///         /// An integer value.
///         Int(MyInt),
///     }
/// }
///
/// // Now you can use GenericValue<MyData> and GenericTokenValueMap<MyData>.
/// use token_value_map::GenericValue;
///
/// let uniform = GenericValue::<MyData>::uniform(MyData::Float(MyFloat(42.0)));
/// let animated = GenericValue::<MyData>::animated(vec![
///     (Time::default(), MyData::Float(MyFloat(0.0))),
///     (Time::from(10.0), MyData::Float(MyFloat(100.0))),
/// ]).unwrap();
/// ```
///
/// # Generated Types
///
/// Given `MyData / MyAnimatedData / MyDataType`:
///
/// - `MyDataType`: Discriminant enum with unit variants (`Float`, `Int`, `Text`).
/// - `MyData`: Data enum holding values (`Float(f32)`, `Int(i32)`, `Text(String)`).
/// - `MyAnimatedData`: Animated enum holding time maps
///   (`Float(TimeDataMap<f32>)`, etc.).
///
/// # Requirements
///
/// Each variant type must implement:
/// - `Clone + Debug + PartialEq + Eq + Hash + Send + Sync + 'static`
///
/// For interpolation support, types should also implement:
/// - `Add<Output = Self> + Sub<Output = Self>`
/// - `Mul<f32, Output = Self> + Mul<f64, Output = Self>`
/// - `Div<f32, Output = Self> + Div<f64, Output = Self>`
///
/// Types that don't support interpolation will use sample-and-hold behavior.
#[macro_export]
macro_rules! define_data_types {
    (
        $(#[$meta:meta])*
        $vis:vis $data_name:ident / $animated_name:ident / $discriminant_name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident($inner_ty:ty)
            ),+ $(,)?
        }
    ) => {
        // Generate the discriminant enum.
        $(#[$meta])*
        #[derive(Copy, Eq, Hash)]
        $vis enum $discriminant_name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        // Generate the data enum.
        $(#[$meta])*
        #[derive(Eq, Hash)]
        $vis enum $data_name {
            $(
                $(#[$variant_meta])*
                $variant($inner_ty),
            )+
        }

        // Generate the animated data enum.
        $(#[$meta])*
        #[derive(Eq, Hash)]
        $vis enum $animated_name {
            $(
                $(#[$variant_meta])*
                $variant($crate::TimeDataMap<$inner_ty>),
            )+
        }

        // Implement DataSystem for the data enum.
        impl $crate::DataSystem for $data_name {
            type Animated = $animated_name;
            type DataType = $discriminant_name;

            fn discriminant(&self) -> Self::DataType {
                match self {
                    $(
                        $data_name::$variant(_) => $discriminant_name::$variant,
                    )+
                }
            }

            fn variant_name(&self) -> &'static str {
                match self {
                    $(
                        $data_name::$variant(_) => stringify!($variant),
                    )+
                }
            }
        }

        // Implement AnimatedDataSystem for the animated data enum.
        impl $crate::AnimatedDataSystem for $animated_name {
            type Data = $data_name;

            fn keyframe_count(&self) -> usize {
                match self {
                    $(
                        $animated_name::$variant(map) => map.len(),
                    )+
                }
            }

            fn times(&self) -> ::smallvec::SmallVec<[$crate::Time; 10]> {
                match self {
                    $(
                        $animated_name::$variant(map) => {
                            map.iter().map(|(t, _)| *t).collect()
                        }
                    )+
                }
            }

            fn interpolate(&self, time: $crate::Time) -> Self::Data {
                match self {
                    $(
                        $animated_name::$variant(map) => {
                            $data_name::$variant(map.interpolate(time))
                        }
                    )+
                }
            }

            fn sample_at(&self, time: $crate::Time) -> ::core::option::Option<Self::Data> {
                match self {
                    $(
                        $animated_name::$variant(map) => {
                            map.get(&time).cloned().map($data_name::$variant)
                        }
                    )+
                }
            }

            fn try_insert(
                &mut self,
                time: $crate::Time,
                value: Self::Data,
            ) -> $crate::Result<()> {
                match (self, value) {
                    $(
                        ($animated_name::$variant(map), $data_name::$variant(v)) => {
                            map.insert(time, v);
                            Ok(())
                        }
                    )+
                    #[allow(unreachable_patterns)]
                    (this, val) => Err($crate::Error::GenericTypeMismatch {
                        expected: this.variant_name(),
                        got: val.variant_name(),
                    }),
                }
            }

            fn remove_at(&mut self, time: &$crate::Time) -> ::core::option::Option<Self::Data> {
                match self {
                    $(
                        $animated_name::$variant(map) => {
                            map.remove(time).map($data_name::$variant)
                        }
                    )+
                }
            }

            fn discriminant(&self) -> <Self::Data as $crate::DataSystem>::DataType {
                match self {
                    $(
                        $animated_name::$variant(_) => $discriminant_name::$variant,
                    )+
                }
            }

            fn from_single(time: $crate::Time, value: Self::Data) -> Self {
                match value {
                    $(
                        $data_name::$variant(v) => {
                            let mut map = ::std::collections::BTreeMap::new();
                            map.insert(time, v);
                            $animated_name::$variant($crate::TimeDataMap::from(map))
                        }
                    )+
                }
            }

            fn variant_name(&self) -> &'static str {
                match self {
                    $(
                        $animated_name::$variant(_) => stringify!($variant),
                    )+
                }
            }
        }

        // Implement From conversions from inner types to Data.
        $(
            impl ::core::convert::From<$inner_ty> for $data_name {
                fn from(value: $inner_ty) -> Self {
                    $data_name::$variant(value)
                }
            }
        )+
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnimatedDataSystem, DataSystem, GenericValue, Time};
    use std::ops::{Add, Div, Mul, Sub};

    // Wrapper type that implements all required traits for interpolation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct TestFloat(i64); // Store as fixed-point for Eq/Hash.

    impl TestFloat {
        fn new(v: f32) -> Self {
            Self((v * 1000.0) as i64)
        }

        fn value(&self) -> f32 {
            self.0 as f32 / 1000.0
        }
    }

    impl Add for TestFloat {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }

    impl Sub for TestFloat {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self(self.0 - other.0)
        }
    }

    impl Mul<f32> for TestFloat {
        type Output = Self;
        fn mul(self, scalar: f32) -> Self {
            Self((self.0 as f32 * scalar) as i64)
        }
    }

    impl Mul<f64> for TestFloat {
        type Output = Self;
        fn mul(self, scalar: f64) -> Self {
            Self((self.0 as f64 * scalar) as i64)
        }
    }

    impl Div<f32> for TestFloat {
        type Output = Self;
        fn div(self, scalar: f32) -> Self {
            Self((self.0 as f32 / scalar) as i64)
        }
    }

    impl Div<f64> for TestFloat {
        type Output = Self;
        fn div(self, scalar: f64) -> Self {
            Self((self.0 as f64 / scalar) as i64)
        }
    }

    // Integer wrapper that supports interpolation via f32 multiplication.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct TestInt(i64);

    impl Add for TestInt {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }

    impl Sub for TestInt {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            Self(self.0 - other.0)
        }
    }

    impl Mul<f32> for TestInt {
        type Output = Self;
        fn mul(self, scalar: f32) -> Self {
            Self((self.0 as f32 * scalar) as i64)
        }
    }

    impl Mul<f64> for TestInt {
        type Output = Self;
        fn mul(self, scalar: f64) -> Self {
            Self((self.0 as f64 * scalar) as i64)
        }
    }

    impl Div<f32> for TestInt {
        type Output = Self;
        fn div(self, scalar: f32) -> Self {
            Self((self.0 as f32 / scalar) as i64)
        }
    }

    impl Div<f64> for TestInt {
        type Output = Self;
        fn div(self, scalar: f64) -> Self {
            Self((self.0 as f64 / scalar) as i64)
        }
    }

    // Define a simple custom data system for testing.
    define_data_types! {
        /// Test data types.
        #[derive(Clone, Debug, PartialEq)]
        pub TestData / TestAnimatedData / TestDataType {
            /// A float value.
            Float(TestFloat),
            /// An int value.
            Int(TestInt),
        }
    }

    #[test]
    fn test_discriminant() {
        let data = TestData::Float(TestFloat::new(42.0));
        assert_eq!(data.discriminant(), TestDataType::Float);
        assert_eq!(data.variant_name(), "Float");

        let data = TestData::Int(TestInt(42));
        assert_eq!(data.discriminant(), TestDataType::Int);
        assert_eq!(data.variant_name(), "Int");
    }

    #[test]
    fn test_from_conversion() {
        let data: TestData = TestFloat::new(42.0).into();
        assert!(matches!(data, TestData::Float(_)));

        let data: TestData = TestInt(42).into();
        assert!(matches!(data, TestData::Int(TestInt(42))));
    }

    #[test]
    fn test_generic_value_uniform() {
        let value = GenericValue::<TestData>::uniform(TestData::Float(TestFloat::new(42.0)));
        assert!(!value.is_animated());

        if let TestData::Float(f) = value.interpolate(Time::default()) {
            assert!((f.value() - 42.0).abs() < 0.01);
        } else {
            panic!("Expected Float variant");
        }
    }

    #[test]
    fn test_generic_value_animated() {
        let value = GenericValue::<TestData>::animated(vec![
            (Time::default(), TestData::Float(TestFloat::new(0.0))),
            (Time::from(10.0), TestData::Float(TestFloat::new(100.0))),
        ])
        .unwrap();

        assert!(value.is_animated());
        assert_eq!(value.sample_count(), 2);

        // Test interpolation at midpoint.
        let mid = value.interpolate(Time::from(5.0));
        if let TestData::Float(v) = mid {
            assert!((v.value() - 50.0).abs() < 1.0); // Fixed-point has some precision loss.
        } else {
            panic!("Expected Float variant");
        }
    }

    #[test]
    fn test_animated_data_system() {
        let animated =
            TestAnimatedData::from_single(Time::default(), TestData::Float(TestFloat::new(42.0)));
        assert_eq!(animated.keyframe_count(), 1);
        assert_eq!(animated.variant_name(), "Float");

        let sample = animated.sample_at(Time::default());
        assert!(matches!(sample, Some(TestData::Float(_))));
    }
}
