//! A time-based data mapping library for animation and interpolation.
//!
//! This crate provides types for storing and manipulating data that changes
//! over time. It supports both uniform (static) and animated (time-varying)
//! values with automatic interpolation between keyframes.
//!
//! # Core Types
//!
//! - [`Value`]: A value that can be either uniform or animated over time.
//! - [`Data`]: A variant enum containing all supported data types.
//! - [`AnimatedData`]: Time-indexed data with interpolation support.
//! - [`TimeDataMap`]: A mapping from time to data values.
//! - [`TokenValueMap`]: A collection of named values indexed by tokens.
//!
//! # Data Types
//!
//! The library supports scalar types ([`Boolean`], [`Integer`], [`Real`],
//! [`String`]), vector types ([`Vector2`], [`Vector3`], [`Color`],
//! [`Matrix3`]), and collections of these types ([`BooleanVec`],
//! [`IntegerVec`], etc.).
//!
//! # Motion Blur Sampling
//!
//! Use the [`Sample`] trait with a [`Shutter`] to generate motion blur samples
//! for animated values during rendering.
//!
//! # Interpolation (Optional Feature)
//!
//! When the `interpolation` feature is enabled, [`TimeDataMap`] supports
//! advanced interpolation modes including bezier curves with tangent control.
//! This enables integration with professional animation systems like Dopamine.
//!
//! # Examples
//!
//! ```rust
//! use frame_tick::Tick;
//! use token_value_map::*;
//!
//! // Create a uniform value.
//! let uniform = Value::uniform(42.0);
//!
//! // Create an animated value.
//! let animated =
//!     Value::animated(vec![(Tick::new(0), 0.0), (Tick::new(10), 10.0)])
//!         .unwrap();
//!
//! // Sample at a specific time.
//! let interpolated = animated.interpolate(Tick::new(5));
//! ```

#[cfg(feature = "facet")]
use facet::Facet;
#[cfg(feature = "builtin-types")]
use function_name::named;
#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "builtin-types")]
use std::hash::Hasher;

// Internal macro module.
#[cfg(feature = "builtin-types")]
mod macros;
#[cfg(feature = "builtin-types")]
use macros::impl_sample_for_value;

// Math backend abstraction.
#[cfg(feature = "builtin-types")]
pub mod math;

// Built-in types (feature-gated).
#[cfg(feature = "builtin-types")]
mod animated_data;
#[cfg(feature = "builtin-types")]
mod data;
#[cfg(feature = "builtin-types")]
mod data_types;
#[cfg(feature = "builtin-types")]
mod token_value_map;
#[cfg(feature = "builtin-types")]
mod value;

// Generic types (always available).
mod define_data_macro;
mod generic_token_value_map;
mod generic_value;
mod traits;

// Token type.
mod token;

// Other modules.
#[cfg(feature = "egui-keyframe")]
mod egui_keyframe_integration;
mod error;
#[cfg(feature = "interpolation")]
mod interpolation;
#[cfg(all(feature = "lua", feature = "builtin-types"))]
mod lua;
mod shutter;
mod time_data_map;

// Re-exports: built-in types (feature-gated).
#[cfg(feature = "builtin-types")]
pub use animated_data::*;
#[cfg(feature = "builtin-types")]
pub use data::*;
#[cfg(feature = "builtin-types")]
pub use data_types::*;
#[cfg(feature = "builtin-types")]
pub use token_value_map::*;
#[cfg(feature = "builtin-types")]
pub use value::*;

// Re-exports: always available.
pub use error::*;
pub use generic_token_value_map::*;
pub use generic_value::*;
#[cfg(feature = "interpolation")]
pub use interpolation::*;
#[cfg(all(feature = "lua", feature = "builtin-types"))]
pub use lua::*;
pub use shutter::*;
pub use time_data_map::*;
pub use token::*;
pub use traits::*;

/// A time value represented as a fixed-point [`Tick`](frame_tick::Tick).
pub type Time = frame_tick::Tick;

/// Trait for getting data type information.
///
/// This trait is only available with the `builtin-types` feature.
#[cfg(feature = "builtin-types")]
pub trait DataTypeOps {
    /// Returns the [`DataType`] variant for this value.
    fn data_type(&self) -> DataType;
    /// Returns a string name for this data type.
    fn type_name(&self) -> &'static str;
}

#[cfg(feature = "builtin-types")]
impl DataTypeOps for Value {
    fn data_type(&self) -> DataType {
        match self {
            Value::Uniform(data) => data.data_type(),
            Value::Animated(animated_data) => animated_data.data_type(),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Value::Uniform(data) => data.type_name(),
            Value::Animated(animated_data) => animated_data.type_name(),
        }
    }
}
