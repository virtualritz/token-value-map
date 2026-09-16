//! Tests for BezierHandle variants and new interpolation modes.
//!
//! This module verifies that all interpolation modes work correctly:
//! - Hold (step function)
//! - Linear
//! - Smooth (Catmull-Rom style)
//! - Bezier with SlopePerSecond
//! - Bezier with SlopePerFrame
//! - Bezier with Delta
//! - Mixed combinations (Bezier/Smooth, etc.)

#[cfg(feature = "interpolation")]
use token_value_map::*;

#[cfg(feature = "interpolation")]
mod hold_interpolation {
    use super::*;

    #[test]
    fn hold_takes_precedence_over_linear() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(5.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Hold,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(15.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        // With Hold on outgoing, value should stay at 5.0 throughout.
        assert_eq!(map.interpolate(Time::from(5.0)), Real(5.0));
        assert_eq!(map.interpolate(Time::from(9.0)), Real(5.0));
    }

    #[test]
    fn hold_on_incoming_side() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(5.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(15.0),
            Key {
                interpolation_in: Interpolation::Hold,
                interpolation_out: Interpolation::Linear,
            },
        );

        // With Hold on incoming, value should stay at first keyframe.
        assert_eq!(map.interpolate(Time::from(5.0)), Real(5.0));
    }

    #[test]
    fn hold_beats_bezier() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Hold,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Hold should take precedence over Bezier.
        assert_eq!(map.interpolate(Time::from(5.0)), Real(0.0));
    }

    #[test]
    fn hold_beats_smooth() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(3.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Hold,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(7.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Linear,
            },
        );

        // Hold beats Smooth.
        assert_eq!(map.interpolate(Time::from(5.0)), Real(3.0));
    }
}

#[cfg(feature = "interpolation")]
mod smooth_interpolation {
    use super::*;

    #[test]
    fn smooth_symmetric_uses_automatic() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        // Smooth/Smooth should use automatic interpolation.
        let mid = map.interpolate(Time::from(5.0));
        // With only 2 points, automatic interpolation should be approximately linear.
        assert!((mid.0 - 5.0).abs() < 0.5, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn smooth_with_multiple_keyframes() {
        // Three keyframes to test Catmull-Rom style tangents.

        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        map.insert_with_interpolation(
            Time::from(20.0),
            Real(20.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        // Middle keyframe should use neighbors for tangent calculation.
        let val1 = map.interpolate(Time::from(5.0));
        let val2 = map.interpolate(Time::from(15.0));

        // Values should be reasonable (monotonic increasing).
        assert!(val1.0 > 0.0 && val1.0 < 10.0);
        assert!(val2.0 > 10.0 && val2.0 < 20.0);
    }
}

#[cfg(feature = "interpolation")]
mod bezier_slope_per_second {
    use super::*;

    #[test]
    fn symmetric_slopes() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Symmetric slopes of 1.0/second should produce approximately linear curve.
        let mid = map.interpolate(Time::from(5.0));
        assert!((mid.0 - 5.0).abs() < 0.5, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn asymmetric_slopes() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(3.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.5))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Fast outgoing (3.0), slow incoming (0.5) creates skewed curve.
        let early = map.interpolate(Time::from(2.0));
        let late = map.interpolate(Time::from(8.0));

        // Early should be higher than linear (fast start).
        assert!(early.0 > 2.0, "Early should be > 2.0, got {:?}", early);

        // Late should be close to endpoint (slow arrival).
        assert!(late.0 < 10.0, "Late should be < 10.0, got {:?}", late);
    }

    #[test]
    fn zero_slope() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Zero slopes create S-curve.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            (mid.0 - 5.0).abs() < 2.0,
            "Mid should be ~5.0 for S-curve, got {:?}",
            mid
        );
    }

    #[test]
    fn negative_slope() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(5.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(-1.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(5.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Curve should dip below 5.0.
        let mid = map.interpolate(Time::from(5.0));
        assert!(mid.0 < 5.0, "Mid should dip below 5.0, got {:?}", mid);
    }
}

#[cfg(feature = "interpolation")]
mod bezier_slope_per_frame {
    use super::*;

    #[test]
    fn slope_per_frame_conversion() {
        // 10 frames between keyframes.

        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerFrame(Real(1.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerFrame(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // With 1.0/frame over 10 frames, slope is effectively 1.0/second.
        let mid = map.interpolate(Time::from(5.0));
        assert!((mid.0 - 5.0).abs() < 0.5, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn asymmetric_frame_slopes() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerFrame(Real(2.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerFrame(Real(0.5))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Fast frame slope creates skewed curve.
        let early = map.interpolate(Time::from(2.0));
        assert!(early.0 > 2.0, "Early should be > 2.0, got {:?}", early);
    }
}

#[cfg(feature = "interpolation")]
mod bezier_delta {
    use super::*;

    #[test]
    fn delta_specification() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from(1.0),
                    value: Real(1.0),
                }),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from(1.0),
                    value: Real(1.0),
                }),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Delta (1.0, 1.0) means slope of 1.0.
        let mid = map.interpolate(Time::from(5.0));
        assert!((mid.0 - 5.0).abs() < 0.5, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn delta_with_zero_dt_falls_back() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from(0.0),
                    value: Real(5.0),
                }),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        // Zero dt should fall back to linear.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            (mid.0 - 5.0).abs() < 0.1,
            "Expected linear ~5.0, got {:?}",
            mid
        );
    }

    #[test]
    fn asymmetric_deltas() {
        let mut map = TimeDataMap::from_single(Time::from(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from(1.0),
                    value: Real(5.0),
                }),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from(2.0),
                    value: Real(1.0),
                }),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Fast outgoing (slope 5.0), slow incoming (slope 0.5).
        let early = map.interpolate(Time::from(1.0));
        assert!(early.0 > 1.0, "Early should be > 1.0, got {:?}", early);
    }

    /// Keys at (0 s, 0) and (1 s, 1) with flat handles reaching `length` seconds.
    fn flat_handles(length: f64) -> TimeDataMap<Real> {
        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from_secs(length),
                    value: Real(0.0),
                }),
            },
        )
        .unwrap();
        map.insert_with_interpolation(
            Time::from_secs(1.0),
            Real(1.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from_secs(length),
                    value: Real(0.0),
                }),
                interpolation_out: Interpolation::Linear,
            },
        );
        map
    }

    fn cubic_bezier(s: f64, p: [f64; 4]) -> f64 {
        let u = 1.0 - s;
        u * u * u * p[0] + 3.0 * u * u * s * p[1] + 3.0 * u * s * s * p[2] + s * s * s * p[3]
    }

    #[test]
    fn delta_handle_length_changes_timing() {
        let length = 0.45;
        let map = flat_handles(length);
        let times = [0.0, length, 1.0 - length, 1.0];
        let values = [0.0, 0.0, 1.0, 1.0];

        for t in [0.1, 0.25, 0.4, 0.6, 0.9] {
            // Independent reference: bisect the time curve for the parameter.
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if cubic_bezier(mid, times) < t {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let expected = cubic_bezier(0.5 * (lo + hi), values);
            // What mapping time linearly onto the parameter would give.
            let linear_timing = cubic_bezier(t, values);

            let got = map.interpolate(Time::from_secs(t)).0;
            assert!(
                (got - expected).abs() < 1e-3,
                "t={t}: expected {expected}, got {got}"
            );
            assert!(
                (expected - linear_timing).abs() > 0.01,
                "t={t}: handle length should change timing ({expected} vs {linear_timing})"
            );
        }
    }

    #[test]
    fn third_length_handles_keep_linear_timing() {
        // Handles at a third of the interval: the time curve is linear, so the
        // result is the value curve at `t`.
        let map = flat_handles(1.0 / 3.0);
        for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let expected = cubic_bezier(t, [0.0, 0.0, 1.0, 1.0]);
            let got = map.interpolate(Time::from_secs(t)).0;
            assert!(
                (got - expected).abs() < 1e-3,
                "t={t}: expected {expected}, got {got}"
            );
        }
    }
}

#[cfg(feature = "interpolation")]
mod mixed_modes {
    use super::*;

    #[test]
    fn bezier_and_smooth_outgoing() {
        // Add extra keyframes for Smooth to calculate tangent.

        let mut map = TimeDataMap::from_single(Time::from(-10.0), Real(-10.0));
        map.set_interpolation_at(
            &Time::from(-10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from_secs(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))),
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        // Bezier handle outgoing, Smooth incoming.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            mid.0 > 0.0 && mid.0 < 10.0,
            "Should interpolate, got {:?}",
            mid
        );
    }

    #[test]
    fn smooth_and_bezier_incoming() {
        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Smooth,
            },
        );

        map.insert_with_interpolation(
            Time::from(20.0),
            Real(20.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        // Smooth outgoing, Bezier incoming.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            mid.0 > 0.0 && mid.0 < 10.0,
            "Should interpolate, got {:?}",
            mid
        );
    }

    #[test]
    fn linear_and_smooth_falls_back() {
        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Linear,
            },
        );

        // Linear vs Smooth should fall back to linear.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            (mid.0 - 5.0).abs() < 0.1,
            "Expected linear ~5.0, got {:?}",
            mid
        );
    }

    #[test]
    fn bezier_and_linear_falls_back() {
        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(0.0));
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        // Bezier vs Linear should fall back to linear.
        let mid = map.interpolate(Time::from(5.0));
        assert!(
            (mid.0 - 5.0).abs() < 0.1,
            "Expected linear ~5.0, got {:?}",
            mid
        );
    }
}

#[cfg(all(feature = "interpolation", feature = "vector3"))]
mod vector_interpolation {
    use super::*;
    use token_value_map::math::Vec3Impl as NVector3;

    #[test]
    fn vector_with_slope_per_second() {
        let v0 = Vector3(NVector3::new(0.0, 0.0, 0.0));
        let v1 = Vector3(NVector3::new(10.0, 10.0, 10.0));
        let slope = Vector3(NVector3::new(1.0, 1.0, 1.0));

        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), v0);
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(
                    slope.clone(),
                )),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            v1,
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(slope)),
                interpolation_out: Interpolation::Linear,
            },
        );

        let mid = map.interpolate(Time::from(5.0));
        // Each component should be approximately 5.0.
        assert!(
            (mid.0.x - 5.0).abs() < 1.0
                && (mid.0.y - 5.0).abs() < 1.0
                && (mid.0.z - 5.0).abs() < 1.0,
            "Expected ~(5,5,5), got {:?}",
            mid
        );
    }

    #[test]
    fn vector_with_smooth() {
        let v0 = Vector3(NVector3::new(0.0, 0.0, 0.0));
        let v1 = Vector3(NVector3::new(10.0, 10.0, 10.0));

        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), v0);
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            v1,
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        );

        let mid = map.interpolate(Time::from(5.0));
        // Should interpolate smoothly.
        assert!(
            mid.0.x > 0.0 && mid.0.x < 10.0,
            "Expected interpolated value, got {:?}",
            mid
        );
    }

    #[test]
    fn vector_with_delta() {
        let v0 = Vector3(NVector3::new(0.0, 0.0, 0.0));
        let v1 = Vector3(NVector3::new(10.0, 10.0, 10.0));
        let dv = Vector3(NVector3::new(1.0, 1.0, 1.0));

        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), v0);
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from_secs(1.0),
                    value: dv.clone(),
                }),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            v1,
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::Delta {
                    time: Time::from_secs(1.0),
                    value: dv,
                }),
                interpolation_out: Interpolation::Linear,
            },
        );

        let mid = map.interpolate(Time::from(5.0));
        // Delta (1.0, (1,1,1)) creates slope of (1,1,1).
        assert!(
            (mid.0.x - 5.0).abs() < 1.0,
            "Expected ~5.0, got {:?}",
            mid.0.x
        );
    }

    #[test]
    fn vector_hold_mode() {
        let v0 = Vector3(NVector3::new(1.0, 2.0, 3.0));
        let v1 = Vector3(NVector3::new(10.0, 20.0, 30.0));

        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), v0.clone());
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Hold,
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            v1,
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        let mid = map.interpolate(Time::from(5.0));
        // Hold should keep initial value.
        assert_eq!(mid, v0);
    }
}

#[cfg(feature = "interpolation")]
mod boundary_cases {
    use super::*;

    #[test]
    // `3.14` here is arbitrary keyframe data, not an approximation of `PI`.
    #[allow(clippy::approx_constant)]
    fn exact_keyframe_values_with_bezier() {
        let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(3.14));
        map.set_interpolation_at(
            &Time::from_secs(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(10.0))),
            },
        )
        .unwrap();

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(42.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.1))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Values at keyframes must be exact.
        assert_eq!(map.interpolate(Time::from_secs(0.0)), Real(3.14));
        assert_eq!(map.interpolate(Time::from(10.0)), Real(42.0));
    }

    #[test]
    fn clamping_before_first_keyframe() {
        let mut map = TimeDataMap::from_single(Time::from(10.0), Real(10.0));
        map.set_interpolation_at(
            &Time::from(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        // Before first keyframe should clamp.
        assert_eq!(map.interpolate(Time::from_secs(0.0)), Real(10.0));
        assert_eq!(map.interpolate(Time::from(-100.0)), Real(10.0));
    }

    #[test]
    fn clamping_after_last_keyframe() {
        let mut map = TimeDataMap::from_single(Time::from(10.0), Real(99.0));
        map.set_interpolation_at(
            &Time::from(10.0),
            Key {
                interpolation_in: Interpolation::Smooth,
                interpolation_out: Interpolation::Smooth,
            },
        )
        .unwrap();

        // After last keyframe should clamp.
        assert_eq!(map.interpolate(Time::from(20.0)), Real(99.0));
        assert_eq!(map.interpolate(Time::from(1000.0)), Real(99.0));
    }

    #[test]
    fn single_keyframe_with_bezier() {
        let mut map = TimeDataMap::from_single(Time::from(5.0), Real(7.0));
        map.set_interpolation_at(
            &Time::from(5.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(100.0))),
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(
                    -100.0,
                ))),
            },
        )
        .unwrap();

        // With single keyframe, always return that value regardless of slopes.
        assert_eq!(map.interpolate(Time::from_secs(0.0)), Real(7.0));
        assert_eq!(map.interpolate(Time::from(5.0)), Real(7.0));
        assert_eq!(map.interpolate(Time::from(10.0)), Real(7.0));
    }
}
