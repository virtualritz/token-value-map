//! Error types for token-value-map operations.

use crate::Time;

#[cfg(feature = "builtin-types")]
use crate::DataType;

/// Error type for token-value-map operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// Type conversion method called on incompatible type.
    #[cfg(feature = "builtin-types")]
    #[error("{method}: called on incompatible type `{got:?}`")]
    IncompatibleType {
        /// The method that was called.
        method: &'static str,
        /// The actual data type encountered.
        got: DataType,
    },

    /// Conversion between data types is not supported.
    #[cfg(feature = "builtin-types")]
    #[error("cannot convert `{from:?}` to `{to:?}`")]
    ConversionUnsupported {
        /// The source data type.
        from: DataType,
        /// The target data type.
        to: DataType,
    },

    /// Failed to parse string as target type.
    #[error("cannot parse '{input}' as {target_type}")]
    ParseFailed {
        /// The input string that failed to parse.
        input: String,
        /// The target type name.
        target_type: &'static str,
    },

    /// Integer conversion overflow.
    #[error("integer conversion error: {0}")]
    IntegerOverflow(#[from] std::num::TryFromIntError),

    /// Cannot create animated value with no samples.
    #[error("cannot create animated value with no samples")]
    EmptySamples,

    /// Type mismatch in animated samples.
    #[cfg(feature = "builtin-types")]
    #[error("animated sample type mismatch: expected `{expected:?}`, got `{got:?}` at time {time}")]
    AnimatedTypeMismatch {
        /// The expected data type.
        expected: DataType,
        /// The actual data type encountered.
        got: DataType,
        /// The time at which the mismatch occurred.
        time: Time,
    },

    /// Vector length exceeds expected.
    #[error("vector length {actual} exceeds expected {expected} at time {time}")]
    VectorLengthExceeded {
        /// The actual vector length.
        actual: usize,
        /// The expected maximum length.
        expected: usize,
        /// The time at which the error occurred.
        time: Time,
    },

    /// Type mismatch when adding sample to animated value.
    #[cfg(feature = "builtin-types")]
    #[error("cannot add `{got:?}` to animated `{expected:?}`")]
    SampleTypeMismatch {
        /// The expected data type.
        expected: DataType,
        /// The actual data type encountered.
        got: DataType,
    },

    /// Cannot set interpolation on uniform value.
    #[error("cannot set interpolation on uniform value")]
    InterpolationOnUniform,

    /// Bezier handles only supported for Real type.
    #[cfg(feature = "builtin-types")]
    #[error("bezier handles only supported for Real type, got `{got:?}`")]
    BezierNotSupported {
        /// The actual data type encountered.
        got: DataType,
    },

    /// Sample trait called on wrong variant.
    #[cfg(feature = "builtin-types")]
    #[error("Sample<{sample_type}> called on `{got:?}` variant")]
    SampleVariantMismatch {
        /// The expected sample type.
        sample_type: &'static str,
        /// The actual data type encountered.
        got: DataType,
    },

    /// Vector type cannot be empty.
    #[error("{type_name} cannot be empty")]
    EmptyVec {
        /// The vector type name.
        type_name: &'static str,
    },

    /// Key not found in data map.
    #[error("key not found in data map")]
    KeyNotFound,

    /// Cannot extract value from animated data.
    #[error("cannot extract {type_name} from animated Value")]
    AnimatedExtraction {
        /// The type name.
        type_name: &'static str,
    },

    /// Type mismatch (generic version for custom data systems).
    #[error("type mismatch: expected {expected}, got {got}")]
    GenericTypeMismatch {
        /// The expected type name.
        expected: &'static str,
        /// The actual type name.
        got: &'static str,
    },
}

/// Result type alias using the crate's [`Error`] type.
pub type Result<T> = std::result::Result<T, Error>;
