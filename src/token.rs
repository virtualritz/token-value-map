//! A string token type used as a key in token-value maps.

#[cfg(feature = "facet")]
use facet::Facet;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fmt;

/// A string token used as a key in [`TokenValueMap`](crate::TokenValueMap) and
/// [`GenericTokenValueMap`](crate::GenericTokenValueMap).
///
/// When the `ustr` feature is enabled, wraps [`Ustr`](ustr::Ustr) for
/// O(1) equality via interned string comparison. Otherwise wraps [`String`].
///
/// Archives as [`ArchivedString`](https://docs.rs/rkyv/latest/rkyv/string/struct.ArchivedString.html)
/// via manual [`Archive`](https://docs.rs/rkyv/latest/rkyv/trait.Archive.html) impl,
/// so `rkyv` support works regardless of whether the `ustr` backend provides its
/// own rkyv support.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "ustr", derive(Copy))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[cfg_attr(feature = "facet", derive(Facet))]
#[cfg_attr(feature = "facet", facet(opaque))]
#[repr(transparent)]
pub struct Token(
    #[cfg(feature = "ustr")] pub ustr::Ustr,
    #[cfg(not(feature = "ustr"))] pub std::string::String,
);

// AIDEV-NOTE: Manual rkyv impl archives Token as ArchivedString regardless of
// inner type (Ustr or String). This avoids requiring Ustr: Archive, which the
// crates.io ustr (1.x) does not provide.
#[cfg(feature = "rkyv")]
const _: () = {
    use rkyv::{
        Archive, Deserialize, Place, Serialize, SerializeUnsized,
        rancor::{Fallible, Source},
        string::{ArchivedString, StringResolver},
    };

    impl Archive for Token {
        type Archived = ArchivedString;
        type Resolver = StringResolver;

        fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
            ArchivedString::resolve_from_str(self.as_str(), resolver, out);
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for Token
    where
        S::Error: Source,
        str: SerializeUnsized<S>,
    {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            ArchivedString::serialize_from_str(self.as_str(), serializer)
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<Token, D> for ArchivedString {
        fn deserialize(&self, _: &mut D) -> Result<Token, D::Error> {
            Ok(Token::from(self.as_str()))
        }
    }
};

impl Token {
    /// Creates a new `Token` from a string slice.
    pub fn new(s: &str) -> Self {
        Self::from(s)
    }

    /// Returns the underlying string slice.
    pub fn as_str(&self) -> &str {
        #[cfg(feature = "ustr")]
        {
            self.0.as_str()
        }
        #[cfg(not(feature = "ustr"))]
        {
            &self.0
        }
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({:?})", self.as_str())
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::ops::Deref for Token {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Token {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::borrow::Borrow<str> for Token {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for Token {
    fn from(s: &str) -> Self {
        Self(s.into())
    }
}

impl From<std::string::String> for Token {
    fn from(s: std::string::String) -> Self {
        #[cfg(feature = "ustr")]
        {
            Self(ustr::ustr(&s))
        }
        #[cfg(not(feature = "ustr"))]
        {
            Self(s)
        }
    }
}

#[cfg(feature = "ustr")]
impl From<ustr::Ustr> for Token {
    fn from(u: ustr::Ustr) -> Self {
        Self(u)
    }
}

#[cfg(feature = "ustr")]
impl From<Token> for ustr::Ustr {
    fn from(t: Token) -> Self {
        t.0
    }
}
