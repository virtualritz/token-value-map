# `token-value-map`

Time-based data mapping library for animation and interpolation.

## Overview

This crate provides types for storing and manipulating data that changes over time, with automatic interpolation between keyframes. It supports uniform (constant) and animated (time-varying) values with multiple interpolation methods.
Think your Maya/Blender/Houdini/whetever Attribute Editor.

## Features

- Scalar types: `Boolean`, `Integer`, `Real`, `String`.
- `Color`
- 2D types: `Vector2`, `Matrix3`,
- 3D types: `Vector3`, `Point3`, `Normal3`, `Matrix4`.
- Collection variants of all types.
- Linear, quadratic, and hermite interpolation.
- Motion blur sampling support.
- Token-based value mapping with `ustr`.
- Optional reflection support via `facet`.

## Feature Flags

- `default` -- Includes `2d` and `vec_variants`.
- `2d` -- Enable 2D types (`Vector2`, `Matrix3`).
- `3d` -- Enable 3D types (`Vector3`, `Matrix4`, `Normal3`, `Point3`).
- `vec_variants` -- Enable vector collection types.
- `interpolation` -- Enable keyframe interpolation with Bezier handles.
- `serde` -- Enable serialization support.
- `lua` -- Enable Lua bindings via `mlua`.
- `facet` -- Enable reflection/introspection via `facet`.

## Example

```rust
use frame_tick::Tick;
use token_value_map::{TokenValueMap, Value};
use ustr::ustr;

// Create a token-value map for animation parameters
let mut params = TokenValueMap::new();

// Add uniform (constant) values
params.insert(ustr("radius"), Value::uniform(5.0));

// Add animated values with keyframes
let animated_position = Value::animated(vec![
    (Tick::new(0), 0.0),
    (Tick::new(30), 100.0),
    (Tick::new(60), 50.0),
]).unwrap();
params.insert(ustr("x_position"), animated_position);

// Interpolate animated value at any time
if let Some(value) = params.get(&ustr("x_position")) {
    let interpolated = value.interpolate(Tick::new(15)); // Returns 50.0
}
```

## License

Licensed under either of

- [Apache License](http://www.apache.org/licenses/LICENSE-2.0)
- [BSD-3-Clause license](https://opensource.org/licenses/BSD-3-Clause)
- [MIT license](http://opensource.org/licenses/MIT)
- [Zlib license](http://opensource.org/licenses/Zlib)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
quad-licensed as above, without any additional terms or conditions.
