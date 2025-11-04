use crate::*;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};
use ustr::Ustr;

/// A collection of named values indexed by string tokens.
///
/// [`TokenValueMap`] `struct` stores a mapping from string tokens to [`Value`]
/// instances, allowing efficient lookup of named parameters or attributes by
/// token.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TokenValueMap {
    attributes: HashMap<Ustr, Value>,
}

impl Hash for TokenValueMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.attributes.len().hash(state);
        // Note: HashMap iteration order is not deterministic,
        // so we need to sort for consistent hashing
        let mut items: Vec<_> = self.attributes.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        for (token, value) in items {
            token.hash(state);
            value.hash(state);
        }
    }
}

impl TokenValueMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            attributes: HashMap::with_capacity(capacity),
        }
    }

    pub fn insert<V: Into<Value>>(&mut self, token: impl Into<Ustr>, value: V) {
        self.attributes.insert(token.into(), value.into());
    }

    pub fn get(&self, token: &Ustr) -> Option<&Value> {
        self.attributes.get(token)
    }

    pub fn get_mut(&mut self, token: &Ustr) -> Option<&mut Value> {
        self.attributes.get_mut(token)
    }

    pub fn remove(&mut self, token: &Ustr) -> Option<Value> {
        self.attributes.remove(token)
    }

    pub fn contains(&self, token: &Ustr) -> bool {
        self.attributes.contains_key(token)
    }

    pub fn len(&self) -> usize {
        self.attributes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.attributes.is_empty()
    }

    pub fn clear(&mut self) {
        self.attributes.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Ustr, &Value)> {
        self.attributes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Ustr, &mut Value)> {
        self.attributes.iter_mut()
    }

    pub fn tokens(&self) -> impl Iterator<Item = &Ustr> {
        self.attributes.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.attributes.values()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.attributes.values_mut()
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Ustr, Value)>,
    {
        self.attributes.extend(iter);
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Ustr, &Value) -> bool,
    {
        self.attributes.retain(|k, v| f(k, v));
    }
}

// Manual Eq implementation for TokenValueMap
// This is safe because we handle floating point comparison deterministically
impl Eq for TokenValueMap {}

impl FromIterator<(Ustr, Value)> for TokenValueMap {
    fn from_iter<T: IntoIterator<Item = (Ustr, Value)>>(iter: T) -> Self {
        Self {
            attributes: HashMap::from_iter(iter),
        }
    }
}

impl<const N: usize> From<[(Ustr, Value); N]> for TokenValueMap {
    fn from(arr: [(Ustr, Value); N]) -> Self {
        Self {
            attributes: HashMap::from(arr),
        }
    }
}
