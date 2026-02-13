//! Core traits for generic data systems.
//!
//! This module defines the foundational traits that allow you to define your
//! own data type systems. By implementing these traits, you can replace the
//! built-in [`Data`](crate::Data) and [`AnimatedData`](crate::AnimatedData)
//! types entirely, or extend them with custom types.
//!
//! # Overview
//!
//! The trait hierarchy is:
//! - [`Interpolatable`] -- Types that can be interpolated over time.
//! - [`DataSystem`] -- A complete enum of data variants (like [`Data`](crate::Data)).
//! - [`AnimatedDataSystem`] -- Container for animated data (like [`AnimatedData`](crate::AnimatedData)).
//!
//! # Example
//!
//! ```ignore
//! use token_value_map::{DataSystem, AnimatedDataSystem, GenericValue};
//!
//! // Define your own data enum.
//! #[derive(Clone, Debug, PartialEq, Eq, Hash)]
//! pub enum MyData {
//!     Float(f64),
//!     Quat(MyQuaternion),
//! }
//!
//! // Implement the traits.
//! impl DataSystem for MyData { /* ... */ }
//!
//! // Use with generic types.
//! type MyValue = GenericValue<MyData>;
//! ```

use smallvec::SmallVec;
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Div, Mul, Sub},
};

use crate::{Result, Time};

/// Trait for converting data types to f32 for curve editing.
///
/// Implement this trait on your data enum to enable egui-keyframe integration
/// with [`GenericValue`](crate::GenericValue). Return `None` for types that
/// cannot be displayed as scalar curves (e.g., strings, colors, transforms).
///
/// # Example
///
/// ```ignore
/// impl ToF32 for MyData {
///     fn to_f32(&self) -> Option<f32> {
///         match self {
///             MyData::Float(f) => Some(*f),
///             MyData::Vec3(_) => None, // Can't display as single curve.
///         }
///     }
/// }
/// ```
pub trait ToF32 {
    /// Convert the data value to f32 for curve display.
    ///
    /// Returns `None` for non-numeric types.
    fn to_f32(&self) -> Option<f32>;
}

/// Marker trait for types that support time-based interpolation.
///
/// Types implementing this trait can be used with [`TimeDataMap`](crate::TimeDataMap)
/// for automatic interpolation between keyframes.
///
/// # Required Bounds
///
/// - `Clone` -- Values must be cloneable for interpolation.
/// - `Add`, `Sub` -- Vector arithmetic for blending.
/// - `Mul<f32>`, `Mul<f64>` -- Scalar multiplication for weighting.
/// - `Div<f32>`, `Div<f64>` -- Scalar division for normalization.
/// - `PartialEq` -- Equality comparison for optimization.
/// - `Send + Sync + 'static` -- Thread safety for parallel operations.
pub trait Interpolatable:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f32, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f32, Output = Self>
    + Div<f64, Output = Self>
    + PartialEq
    + Send
    + Sync
    + 'static
{
}

// Blanket implementation for any type meeting the bounds.
impl<T> Interpolatable for T where
    T: Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<f32, Output = T>
        + Mul<f64, Output = T>
        + Div<f32, Output = T>
        + Div<f64, Output = T>
        + PartialEq
        + Send
        + Sync
        + 'static
{
}

/// A complete data type system.
///
/// This trait defines the contract for a "data enum" -- a type that can hold
/// any of several different value types (scalars, vectors, matrices, etc.).
///
/// The built-in [`Data`](crate::Data) type implements this trait. You can
/// define your own enums and implement this trait to create custom type systems.
///
/// # Associated Types
///
/// - `Animated` -- The corresponding animated data container type.
/// - `DataType` -- A discriminant enum for identifying variants.
///
/// # Example
///
/// ```ignore
/// impl DataSystem for MyData {
///     type Animated = MyAnimatedData;
///     type DataType = MyDataType;
///
///     fn data_type(&self) -> MyDataType {
///         match self {
///             MyData::Float(_) => MyDataType::Float,
///             MyData::Quat(_) => MyDataType::Quat,
///         }
///     }
///
///     fn type_name(&self) -> &'static str {
///         match self {
///             MyData::Float(_) => "float",
///             MyData::Quat(_) => "quat",
///         }
///     }
/// }
/// ```
pub trait DataSystem: Clone + Debug + PartialEq + Eq + Hash + Send + Sync + 'static {
    /// The animated data container type for this system.
    type Animated: AnimatedDataSystem<Data = Self>;

    /// The discriminant type for identifying variants.
    type DataType: Clone + Copy + Debug + PartialEq + Eq + Hash + Send + Sync;

    /// Returns the discriminant for this value.
    ///
    /// Named `discriminant()` to avoid conflict with [`DataTypeOps::data_type()`](crate::DataTypeOps::data_type).
    fn discriminant(&self) -> Self::DataType;

    /// Returns a human-readable type name for this value.
    ///
    /// Named `variant_name()` to avoid conflict with [`DataTypeOps::type_name()`](crate::DataTypeOps::type_name).
    fn variant_name(&self) -> &'static str;

    /// Returns the length if this is a vector type, `None` for scalars.
    ///
    /// Override this for data systems that support vector types.
    fn try_len(&self) -> Option<usize> {
        None
    }

    /// Pads a vector value to the target length.
    ///
    /// Override this for data systems that support vector types.
    /// Does nothing by default.
    fn pad_to_length(&mut self, _target_len: usize) {}
}

/// Animated data container for a [`DataSystem`].
///
/// This trait defines the contract for storing and interpolating time-varying
/// data. The built-in [`AnimatedData`](crate::AnimatedData) type implements
/// this trait.
///
/// # Associated Types
///
/// - `Data` -- The corresponding data system type.
///
/// # Implementation Notes
///
/// Implementations typically use [`TimeDataMap<T>`](crate::TimeDataMap) internally
/// to store keyframes for each data type variant.
pub trait AnimatedDataSystem:
    Clone + Debug + PartialEq + Eq + Hash + Send + Sync + 'static
{
    /// The data system type that this animates.
    type Data: DataSystem<Animated = Self>;

    /// Returns the number of keyframes.
    ///
    /// Named `keyframe_count()` to avoid conflict with [`AnimatedDataOps::len()`](crate::AnimatedDataOps::len).
    fn keyframe_count(&self) -> usize;

    /// Returns `true` if there are no keyframes.
    fn is_keyframes_empty(&self) -> bool {
        self.keyframe_count() == 0
    }

    /// Returns `true` if there is more than one keyframe.
    ///
    /// Named `has_animation()` to avoid conflict with [`AnimatedDataOps::is_animated()`](crate::AnimatedDataOps::is_animated).
    fn has_animation(&self) -> bool {
        self.keyframe_count() > 1
    }

    /// Returns all keyframe times.
    fn times(&self) -> SmallVec<[Time; 10]>;

    /// Interpolates the value at the given time.
    fn interpolate(&self, time: Time) -> Self::Data;

    /// Returns the exact sample at a time, or `None` if no keyframe exists.
    fn sample_at(&self, time: Time) -> Option<Self::Data>;

    /// Inserts a value at the given time, checking type compatibility.
    fn try_insert(&mut self, time: Time, value: Self::Data) -> Result<()>;

    /// Removes the keyframe at the given time.
    ///
    /// Returns the removed value if it existed.
    fn remove_at(&mut self, time: &Time) -> Option<Self::Data>;

    /// Returns the data type discriminant for this animated data.
    ///
    /// Named `discriminant()` to avoid conflict with [`DataTypeOps::data_type()`](crate::DataTypeOps::data_type).
    fn discriminant(&self) -> <Self::Data as DataSystem>::DataType;

    /// Creates animated data from a single time-value pair.
    fn from_single(time: Time, value: Self::Data) -> Self;

    /// Returns the type name for this animated data.
    ///
    /// Named `variant_name()` to avoid conflict with [`DataTypeOps::type_name()`](crate::DataTypeOps::type_name).
    fn variant_name(&self) -> &'static str;
}
