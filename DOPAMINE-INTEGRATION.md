# Dopamine Integration with token-value-map

## Executive Summary

This document describes how Dopamine integrates with `token-value-map`'s simplified interpolation system. The design provides minimal interpolation primitives in `token-value-map` while keeping higher-level animation semantics in Dopamine.

### Design Philosophy

- **token-value-map**: Provides only fundamental interpolation primitives (Hold, Linear, Speed-based Bezier)
- **Dopamine**: Handles higher-level concepts (Smooth curves, angle-to-speed conversion, continuity guarantees)

## Architecture

### Core Types in token-value-map

```rust
/// A keyframe's interpolation specification
pub struct Key<T> {
    pub interpolation_in: Interpolation<T>,
    pub interpolation_out: Interpolation<T>,
}

/// Fundamental interpolation modes
pub enum Interpolation<T> {
    Hold,           // Step function
    Linear,         // Linear interpolation
    Speed(T),       // Bezier with speed constraint (derivative)
}
```

Key insights:
- **Generic `T`**: Speed type matches value type (e.g., `Vector3` speed for `Vector3` values)
- **Minimal surface**: Only essential interpolation primitives
- **Type safety**: Compile-time guarantee that speed units match value units

### TimeDataMap Structure

```rust
pub struct TimeDataMap<T> {
    #[cfg(feature = "interpolation")]
    pub values: BTreeMap<Time, (T, Option<Key<T>>)>,
}
```

Each time point stores:
- The value `T`
- Optional interpolation key describing how to enter/leave this keyframe

## Implementation in token-value-map

### Handle Overlap Prevention

When converting speeds to bezier control points, we must ensure the time coordinates remain monotonic. If handles are too long, they can overlap on the time axis, making the knot vector non-monotonic which causes uniform-cubic-splines to fail.

```rust
fn clamp_handle_lengths(
    k1_time: f32, 
    k2_time: f32,
    handle1_length: f32,
    handle2_length: f32,
) -> (f32, f32) {
    let dt = k2_time - k1_time;
    
    // Ensure handles don't overlap in time
    // Each handle can use at most 99% of half the interval
    let max_handle = dt * 0.495; // Just under half to prevent overlap
    
    let clamped_h1 = handle1_length.min(max_handle);
    let clamped_h2 = handle2_length.min(max_handle);
    
    // Additional check: ensure p1.time < p2.time
    let p1_time = k1_time + clamped_h1;
    let p2_time = k2_time - clamped_h2;
    
    if p1_time >= p2_time {
        // Scale both down proportionally
        let scale = (dt * 0.98) / (clamped_h1 + clamped_h2);
        (clamped_h1 * scale, clamped_h2 * scale)
    } else {
        (clamped_h1, clamped_h2)
    }
}
```

### Interpolation Logic

The interpolation implementation should follow Dopamine's approach:

```rust
impl<T> TimeDataMap<T> 
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>
{
    pub fn interpolate(&self, time: Time) -> T {
        // Find surrounding keyframes
        let (k1, k2) = self.find_surrounding_keys(time);
        
        // Get interpolation specs if available
        let key1 = k1.1.as_ref();
        let key2 = k2.1.as_ref();
        
        // Determine interpolation mode
        match (key1.map(|k| &k.interpolation_out), key2.map(|k| &k.interpolation_in)) {
            (Some(Interpolation::Hold), _) | (_, Some(Interpolation::Hold)) => {
                k1.0.clone() // Hold previous value
            }
            (Some(Interpolation::Linear), Some(Interpolation::Linear)) 
            | (None, None) => {
                // Linear interpolation
                let t = (time - k1_time) / (k2_time - k1_time);
                k1.0.clone() + (k2.0.clone() - k1.0.clone()) * t
            }
            (Some(Interpolation::Speed(speed1)), Some(Interpolation::Speed(speed2))) => {
                // Bezier interpolation using speeds
                evaluate_bezier_with_speeds(k1, k2, speed1, speed2, time)
            }
            _ => {
                // Mixed modes: fall back to linear
                let t = (time - k1_time) / (k2_time - k1_time);
                k1.0.clone() + (k2.0.clone() - k1.0.clone()) * t
            }
        }
    }
}
```

### Bezier Evaluation (from Dopamine)

For scalar types, use Dopamine's 2D parametric approach with uniform-cubic-splines:

```rust
use uniform_cubic_splines::{basis::Bezier, spline_inverse_segment, spline_segment};

fn evaluate_bezier_scalar(
    k1_time: f32, k1_value: f32, speed1: f32,
    k2_time: f32, k2_value: f32, speed2: f32,
    time: f32
) -> f32 {
    let dt = k2_time - k1_time;
    
    // Calculate initial handle lengths from speeds
    // Speed defines the derivative at the keyframe
    let base_handle = dt / 3.0;
    let handle1 = base_handle; // Could be affected by speed magnitude
    let handle2 = base_handle; // Could be affected by speed magnitude
    
    // Clamp handles to prevent overlap
    let (h1, h2) = clamp_handle_lengths(k1_time, k2_time, handle1, handle2);
    
    // Calculate control points from clamped handles and speeds
    let p0 = (k1_time, k1_value);
    let p1 = (k1_time + h1, k1_value + speed1 * h1);
    let p2 = (k2_time - h2, k2_value - speed2 * h2);
    let p3 = (k2_time, k2_value);
    
    // Use uniform-cubic-splines for 2D parametric evaluation
    let time_knots = [p0.0, p1.0, p2.0, p3.0];
    let value_knots = [p0.1, p1.1, p2.1, p3.1];
    
    // Find parameter t such that x(t) = time
    let t = spline_inverse_segment::<Bezier, _>(time, &time_knots)?;
    
    // Evaluate the value curve at parameter t
    spline_segment::<Bezier, _, _>(t, &value_knots)
}
```

For vector types, apply component-wise:

```rust
fn evaluate_bezier_vector<T>(
    k1: (Time, T), speed1: T,
    k2: (Time, T), speed2: T,
    time: Time
) -> T 
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>
{
    let dt = (k2.0 - k1.0).as_f32();
    let t = (time - k1.0).as_f32() / dt;
    
    // Calculate control points
    let handle_length = dt / 3.0;
    let p0 = k1.1.clone();
    let p1 = k1.1.clone() + speed1.clone() * handle_length;
    let p2 = k2.1.clone() - speed2.clone() * handle_length;
    let p3 = k2.1.clone();
    
    // Cubic Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    let one_minus_t = 1.0 - t;
    let one_minus_t2 = one_minus_t * one_minus_t;
    let one_minus_t3 = one_minus_t2 * one_minus_t;
    let t2 = t * t;
    let t3 = t2 * t;
    
    p0 * one_minus_t3 + p1 * (3.0 * one_minus_t2 * t) + 
    p2 * (3.0 * one_minus_t * t2) + p3 * t3
}
```

## Dopamine Integration

### Converting Dopamine Types to token-value-map

Dopamine converts its rich interpolation modes to speed-based primitives:

```rust
impl From<&dopamine::Keyframe> for token_value_map::Key<Data> {
    fn from(kf: &dopamine::Keyframe) -> Self {
        Key {
            interpolation_in: convert_interpolation(&kf.interpolation_in, /* context */),
            interpolation_out: convert_interpolation(&kf.interpolation_out, /* context */),
        }
    }
}

fn convert_interpolation(
    interp: &dopamine::Interpolation,
    k_prev: Option<&Keyframe>,
    k_curr: &Keyframe,
    k_next: Option<&Keyframe>,
    frame_base: f32,
) -> token_value_map::Interpolation<Data> {
    match interp {
        dopamine::Interpolation::Hold => Interpolation::Hold,
        dopamine::Interpolation::Linear => Interpolation::Linear,
        
        dopamine::Interpolation::Smooth(tangent_value) => {
            let speed = calculate_speed_from_smooth(
                tangent_value, k_prev, k_curr, k_next, frame_base
            );
            Interpolation::Speed(speed)
        }
        
        dopamine::Interpolation::ExtraSmooth(tangent_value) => {
            // ExtraSmooth uses 2/3 handle length instead of 1/3
            let speed = calculate_speed_from_extra_smooth(
                tangent_value, k_prev, k_curr, k_next, frame_base
            );
            Interpolation::Speed(speed)
        }
        
        // Asymmetric modes affect only one direction
        dopamine::Interpolation::SmoothExtendedOut(tangent_value) => {
            // Extended out, normal in - this affects interpolation_out
            let speed = calculate_speed_from_extra_smooth(
                tangent_value, k_prev, k_curr, k_next, frame_base
            );
            Interpolation::Speed(speed)
        }
        
        dopamine::Interpolation::SmoothExtendedIn(tangent_value) => {
            // Extended in, normal out - this affects interpolation_in
            let speed = calculate_speed_from_extra_smooth(
                tangent_value, k_prev, k_curr, k_next, frame_base
            );
            Interpolation::Speed(speed)
        }
    }
}
```

### Speed Calculation from Dopamine's Tangent Values

```rust
fn calculate_speed_from_smooth(
    tangent_value: &Option<TangentValue>,
    k_prev: Option<&Keyframe>,
    k_curr: &Keyframe,
    k_next: Option<&Keyframe>,
    frame_base: f32,
) -> Data {
    match tangent_value {
        Some(TangentValue::SpeedPerSecond(speed, _modifier)) => {
            // Convert to units per frame (Tick in token-value-map)
            Data::from(speed / frame_base)
        }
        
        Some(TangentValue::SpeedPerFrame(speed, _modifier)) => {
            // Already in correct units
            Data::from(*speed)
        }
        
        Some(TangentValue::Angle(angle, _modifier)) => {
            // Convert angle to speed based on value delta
            let dt = if let Some(next) = k_next {
                next.time - k_curr.time
            } else if let Some(prev) = k_prev {
                k_curr.time - prev.time
            } else {
                1.0 // Default
            };
            
            let dv = if let Some(next) = k_next {
                next.data.clone() - k_curr.data.clone()
            } else if let Some(prev) = k_prev {
                k_curr.data.clone() - prev.data.clone()
            } else {
                k_curr.data.clone() * 0.0 // Zero velocity
            };
            
            // Apply angle transformation
            let angle_rad = angle.to_radians();
            let base_speed = dv / dt;
            base_speed * angle_rad.tan()
        }
        
        None => {
            // Auto-calculate smooth tangent (Catmull-Rom style)
            if let (Some(prev), Some(next)) = (k_prev, k_next) {
                let v_prev = &prev.data;
                let v_next = &next.data;
                let dt_total = next.time - prev.time;
                (v_next.clone() - v_prev.clone()) / dt_total * 0.5
            } else if let Some(next) = k_next {
                // Only next keyframe available
                let dv = next.data.clone() - k_curr.data.clone();
                let dt = next.time - k_curr.time;
                dv / dt
            } else if let Some(prev) = k_prev {
                // Only previous keyframe available
                let dv = k_curr.data.clone() - prev.data.clone();
                let dt = k_curr.time - prev.time;
                dv / dt
            } else {
                // No neighbors: zero speed
                k_curr.data.clone() * 0.0
            }
        }
    }
}
```

## Benefits of This Design

1. **Clean Separation**: Animation semantics stay in Dopamine, math primitives in token-value-map
2. **Type Safety**: Speed type always matches value type
3. **Simplicity**: Only three interpolation modes to implement and test
4. **Flexibility**: Dopamine can implement any high-level animation concept using these primitives
5. **No Circular Dependencies**: token-value-map doesn't know about Dopamine
6. **Proven Algorithm**: Uses the same uniform-cubic-splines approach as Dopamine

## Implementation Status

### Phase 1: Core Implementation (token-value-map)
- [x] Implement `Key<T>` and `Interpolation<T>` types
- [x] Update `TimeDataMap` to use generic keys
- [ ] Implement interpolation logic for Speed variant using uniform-cubic-splines
- [ ] Add tests for all interpolation modes

### Phase 2: Dopamine Integration
- [ ] Implement `From<&Keyframe> for Key<Data>`
- [ ] Add speed calculation from TangentValue variants
- [ ] Handle TangentModifier (multiplier, bias)
- [ ] Test with actual Dopamine animations

### Phase 3: Testing
- [ ] Unit tests for each interpolation mode
- [ ] Integration tests with Dopamine
- [ ] Visual validation of curves
- [ ] Performance benchmarks

## Notes on Implementation

1. **Time Units**: Dopamine uses float seconds/frames, token-value-map uses Tick. Conversion happens at the boundary.

2. **Frame Base**: Dopamine's frame-based speeds need the frame rate (typically 25, 30, or 60 fps) for conversion.

3. **Tangent Modifiers**: Dopamine's `TangentModifier` affects handle length. This translates to scaling the speed value.

4. **Data Type Handling**: For scalar types (Real), use 2D parametric bezier. For vectors, use component-wise cubic bezier.

5. **Zero-Length Tangents**: When interpolation mode is Linear or Hold, the control point coincides with the keyframe point (zero-length tangent).

6. **Handle Overlap Prevention**: Critical for monotonicity! When speeds are high, tangent handles can extend so far they overlap on the time axis, creating a non-monotonic knot vector that breaks uniform-cubic-splines. Always clamp handle lengths to prevent `p1.time >= p2.time`. Dopamine uses a 0.99 multiplier cap to ensure handles never reach the adjacent keyframe.

This design maintains compatibility with Dopamine's proven animation system while providing a clean, minimal API in token-value-map.