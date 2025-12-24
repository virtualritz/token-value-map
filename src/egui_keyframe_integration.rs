//! Integration with egui-keyframe for animation UI.
//!
//! This module provides implementations of egui-keyframe traits for
//! token-value-map types, enabling zero-copy animation curve editing.

use crate::{Data, Time, Value};
use egui_keyframe::{
    BezierHandles, KeyframeId, KeyframeSource, KeyframeType, KeyframeView, TimeTick, uuid,
};

/// Convert a Time to a KeyframeId using a deterministic encoding.
fn time_to_keyframe_id(time: Time) -> KeyframeId {
    let raw: i64 = time.into();
    let mut uuid_bytes = [0u8; 16];
    uuid_bytes[..8].copy_from_slice(&raw.to_le_bytes());
    uuid_bytes[8..16].copy_from_slice(b"tvmtime\0");
    KeyframeId(uuid::Uuid::from_bytes(uuid_bytes))
}

/// Convert a Time to TimeTick.
fn time_to_timetick(time: Time) -> TimeTick {
    TimeTick::new(time.to_secs())
}

/// Convert Data to f32 for curve display.
///
/// Returns None for non-numeric types (String, Color, Vector, etc.)
fn data_to_f32(data: &Data) -> Option<f32> {
    data.to_f32().ok()
}

/// Implementation of KeyframeSource for Value.
///
/// This allows ParameterValue (which wraps Value) to be used directly
/// with the CurveEditor without copying to an intermediate Track<f32>.
impl KeyframeSource for Value {
    fn keyframes_sorted(&self) -> Vec<KeyframeView> {
        match self {
            Value::Uniform(data) => {
                // Uniform values have no keyframes for curve editing
                // (or we could represent as a single keyframe at t=0)
                if let Some(value) = data_to_f32(data) {
                    vec![KeyframeView::new(
                        time_to_keyframe_id(Time::from_secs(0.0)),
                        TimeTick::new(0.0),
                        value,
                        BezierHandles::linear(),
                        true,
                        KeyframeType::Linear,
                    )]
                } else {
                    vec![]
                }
            }
            Value::Animated(animated) => {
                let times = animated.times();
                let mut keyframes: Vec<_> = times
                    .into_iter()
                    .filter_map(|time| {
                        let data = animated.sample_at(time)?;
                        let value = data_to_f32(&data)?;
                        Some(KeyframeView::new(
                            time_to_keyframe_id(time),
                            time_to_timetick(time),
                            value,
                            BezierHandles::linear(), // TODO: extract from Key<T> if interpolation feature
                            true,
                            KeyframeType::Linear, // TODO: extract from Key<T>
                        ))
                    })
                    .collect();

                // Sort by position
                keyframes.sort_by(|a, b| {
                    a.position
                        .partial_cmp(&b.position)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                keyframes
            }
        }
    }

    fn value_range(&self) -> Option<(f32, f32)> {
        match self {
            Value::Uniform(data) => {
                let value = data_to_f32(data)?;
                Some((value, value))
            }
            Value::Animated(animated) => {
                let times = animated.times();
                if times.is_empty() {
                    return None;
                }

                let mut min = f32::MAX;
                let mut max = f32::MIN;

                for time in times {
                    if let Some(data) = animated.sample_at(time) {
                        if let Some(value) = data_to_f32(&data) {
                            min = min.min(value);
                            max = max.max(value);
                        }
                    }
                }

                if min > max { None } else { Some((min, max)) }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Value::Uniform(_) => 1,
            Value::Animated(animated) => animated.times().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Real;

    #[test]
    fn uniform_value_keyframe_source() {
        let value = Value::uniform(Data::Real(Real(42.0)));
        let keyframes = value.keyframes_sorted();
        assert_eq!(keyframes.len(), 1);
        assert_eq!(keyframes[0].value, 42.0);
    }

    #[test]
    fn animated_value_keyframe_source() {
        let value = Value::animated(vec![
            (Time::from_secs(0.0), 0.0),
            (Time::from_secs(1.0), 10.0),
            (Time::from_secs(2.0), 5.0),
        ])
        .unwrap();

        let keyframes = value.keyframes_sorted();
        assert_eq!(keyframes.len(), 3);
        assert_eq!(keyframes[0].value, 0.0);
        assert_eq!(keyframes[1].value, 10.0);
        assert_eq!(keyframes[2].value, 5.0);

        let (min, max) = value.value_range().unwrap();
        assert_eq!(min, 0.0);
        assert_eq!(max, 10.0);
    }
}
