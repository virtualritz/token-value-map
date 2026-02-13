//! Math backend implementations.
//!
//! This module provides backend-agnostic math type implementations.
//! The active backend is selected via feature flags.

#[cfg(feature = "glam")]
mod glam_impl;
#[cfg(feature = "nalgebra")]
mod nalgebra_impl;
#[cfg(feature = "ultraviolet")]
mod ultraviolet_impl;

// Re-export the active backend's types.
#[cfg(feature = "glam")]
pub use glam_impl::*;
#[cfg(feature = "nalgebra")]
pub use nalgebra_impl::*;
#[cfg(feature = "ultraviolet")]
pub use ultraviolet_impl::*;
