//! Generic token-value map that works with any [`DataSystem`].

use crate::{Token, generic_value::GenericValue, traits::DataSystem};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A collection of named values indexed by string tokens.
///
/// This is the generic version of [`TokenValueMap`](crate::TokenValueMap) that
/// works with any [`DataSystem`]. Use this when you have a custom data type system.
///
/// For the built-in types, use [`TokenValueMap`](crate::TokenValueMap) which is
/// an alias for `GenericTokenValueMap<Data>`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D: Serialize, D::Animated: Serialize",
        deserialize = "D: Deserialize<'de>, D::Animated: Deserialize<'de>"
    ))
)]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
pub struct GenericTokenValueMap<D: DataSystem> {
    attributes: HashMap<Token, GenericValue<D>>,
}

impl<D: DataSystem> Hash for GenericTokenValueMap<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.attributes.len().hash(state);
        // HashMap iteration order is not deterministic, so sort for consistent hashing.
        let mut items: Vec<_> = self.attributes.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        for (token, value) in items {
            token.hash(state);
            value.hash(state);
        }
    }
}

impl<D: DataSystem> GenericTokenValueMap<D> {
    /// Creates a new empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new map with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            attributes: HashMap::with_capacity(capacity),
        }
    }

    /// Inserts a value with the given token.
    pub fn insert(&mut self, token: impl Into<Token>, value: impl Into<GenericValue<D>>) {
        self.attributes.insert(token.into(), value.into());
    }

    /// Gets a reference to the value for a token.
    pub fn get(&self, token: &Token) -> Option<&GenericValue<D>> {
        self.attributes.get(token)
    }

    /// Gets a mutable reference to the value for a token.
    pub fn get_mut(&mut self, token: &Token) -> Option<&mut GenericValue<D>> {
        self.attributes.get_mut(token)
    }

    /// Removes and returns the value for a token.
    pub fn remove(&mut self, token: &Token) -> Option<GenericValue<D>> {
        self.attributes.remove(token)
    }

    /// Returns `true` if the map contains the given token.
    pub fn contains(&self, token: &Token) -> bool {
        self.attributes.contains_key(token)
    }

    /// Returns the number of entries in the map.
    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    /// Returns `true` if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    /// Removes all entries from the map.
    pub fn clear(&mut self) {
        self.attributes.clear();
    }

    /// Iterates over all token-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Token, &GenericValue<D>)> {
        self.attributes.iter()
    }

    /// Iterates mutably over all token-value pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Token, &mut GenericValue<D>)> {
        self.attributes.iter_mut()
    }

    /// Iterates over all tokens.
    pub fn tokens(&self) -> impl Iterator<Item = &Token> {
        self.attributes.keys()
    }

    /// Iterates over all values.
    pub fn values(&self) -> impl Iterator<Item = &GenericValue<D>> {
        self.attributes.values()
    }

    /// Iterates mutably over all values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut GenericValue<D>> {
        self.attributes.values_mut()
    }

    /// Extends the map with the given iterator.
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Token, GenericValue<D>)>,
    {
        self.attributes.extend(iter);
    }

    /// Retains only the entries for which the predicate returns `true`.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Token, &GenericValue<D>) -> bool,
    {
        self.attributes.retain(|k, v| f(k, v));
    }
}

impl<D: DataSystem> Eq for GenericTokenValueMap<D> {}

impl<D: DataSystem> Default for GenericTokenValueMap<D> {
    fn default() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }
}

impl<D: DataSystem> FromIterator<(Token, GenericValue<D>)> for GenericTokenValueMap<D> {
    fn from_iter<T: IntoIterator<Item = (Token, GenericValue<D>)>>(iter: T) -> Self {
        Self {
            attributes: HashMap::from_iter(iter),
        }
    }
}

impl<D: DataSystem, const N: usize> From<[(Token, GenericValue<D>); N]>
    for GenericTokenValueMap<D>
{
    fn from(arr: [(Token, GenericValue<D>); N]) -> Self {
        Self {
            attributes: HashMap::from(arr),
        }
    }
}
