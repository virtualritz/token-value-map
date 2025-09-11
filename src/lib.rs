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

use anyhow::{Result, anyhow};
use function_name::named;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

mod macros;
use macros::impl_sample_for_value;

mod animated_data;
mod data;
mod data_types;
#[cfg(feature = "interpolation")]
mod interpolation;
#[cfg(feature = "lua")]
mod lua;
mod shutter;
mod time_data_map;
mod token_value_map;
mod value;

pub use animated_data::*;
pub use data::*;
pub use data_types::*;
#[cfg(feature = "interpolation")]
pub use interpolation::*;
#[cfg(feature = "lua")]
pub use lua::*;
pub use shutter::*;
pub use time_data_map::*;
pub use token_value_map::*;
pub use value::*;

/// A time value represented as a fixed-point [`Tick`](frame_tick::Tick).
pub type Time = frame_tick::Tick;

/// Trait for getting data type information.
pub trait DataTypeOps {
    /// Returns the [`DataType`] variant for this value.
    fn data_type(&self) -> DataType;
    /// Returns a string name for this data type.
    fn type_name(&self) -> &'static str;
}

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
