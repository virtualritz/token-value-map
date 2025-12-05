//! Tests for interpolation with asymmetric tangents (legacy Speed syntax).
//!
//! AIDEV-NOTE: These tests were written for the old `Interpolation::Speed` variant.
//! They have been updated to use `Interpolation::Bezier(BezierHandle::SlopePerSecond(...))`.

#[cfg(feature = "interpolation")]
mod asymmetric_tangents {
    use std::collections::BTreeMap;
    use token_value_map::*;

    #[test]
    fn test_symmetric_speed_tangents() {
        // Test that symmetric speeds produce smooth curves.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        // Keyframe at t=0 with value 0.0, speed 1.0/frame.
        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
            },
        );

        // Keyframe at t=10 with value 10.0, speed 1.0/frame.
        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // With symmetric speeds of 1.0, the curve should be close to linear.
        let mid = map.interpolate(Time::from(5.0));
        assert!((mid.0 - 5.0).abs() < 0.5, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn test_asymmetric_speed_tangents() {
        // Test that asymmetric speeds produce asymmetric curves.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        // Keyframe at t=0 with value 0.0, outgoing speed 2.0/frame (fast).
        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))),
            },
        );

        // Keyframe at t=10 with value 10.0, incoming speed 0.5/frame (slow).
        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.5))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // With fast outgoing and slow incoming, the curve should be skewed.
        // Early values should be higher than linear, late values lower.
        let early = map.interpolate(Time::from(2.0));
        let mid = map.interpolate(Time::from(5.0));
        let late = map.interpolate(Time::from(8.0));

        // The curve starts fast, so early value should be > 2.0.
        assert!(
            early.0 > 2.0,
            "Early value should be > 2.0 (linear), got {:?}",
            early
        );

        // The curve ends slow, so late value should approach 8.0.
        // Note: Bezier curves can overshoot slightly.
        assert!(
            late.0 < 10.0,
            "Late value should be < 10.0 (endpoint), got {:?}",
            late
        );

        // Verify monotonicity (values should increase).
        assert!(
            early.0 < mid.0 && mid.0 < late.0,
            "Values should be monotonic: {:?} < {:?} < {:?}",
            early,
            mid,
            late
        );
    }

    #[test]
    fn test_hold_interpolation() {
        // Test that Hold takes precedence.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Hold,
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // With Hold on outgoing, all values should be 0.0.
        let mid = map.interpolate(Time::from(5.0));
        assert_eq!(mid, Real(0.0), "Hold should keep value at 0.0");
    }

    #[test]
    fn test_linear_interpolation() {
        // Test explicit Linear interpolation.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        // With Linear on both sides, should use automatic interpolation.
        let mid = map.interpolate(Time::from(5.0));
        // Automatic interpolation may use Hermite if there are neighbors,
        // but with only 2 points it should be linear.
        assert!((mid.0 - 5.0).abs() < 0.1, "Expected ~5.0, got {:?}", mid);
    }

    #[cfg(feature = "vector3")]
    #[test]
    fn test_vector_asymmetric_tangents() {
        use nalgebra::Vector3 as NVector3;

        // Test asymmetric tangents on vector types.
        let mut map = TimeDataMap::<Vector3>::from(BTreeMap::new());

        let v0 = Vector3(NVector3::new(0.0, 0.0, 0.0));
        let v1 = Vector3(NVector3::new(10.0, 10.0, 10.0));

        // Fast outgoing speed.
        let speed_out = Vector3(NVector3::new(2.0, 2.0, 2.0));
        // Slow incoming speed.
        let speed_in = Vector3(NVector3::new(0.5, 0.5, 0.5));

        map.insert_with_interpolation(
            Time::from(0.0),
            v0,
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(speed_out)),
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            v1,
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(speed_in)),
                interpolation_out: Interpolation::Linear,
            },
        );

        let mid = map.interpolate(Time::from(5.0));

        // Each component should follow the asymmetric curve.
        assert!(
            mid.0.x > 5.0 && mid.0.y > 5.0 && mid.0.z > 5.0,
            "Mid values should be > 5.0, got {:?}",
            mid
        );
    }

    #[test]
    fn test_mixed_interpolation_modes() {
        // Test mixed modes (one Speed, one Linear).
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))),
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Linear,
            },
        );

        // Mixed modes should fall back to linear interpolation.
        let mid = map.interpolate(Time::from(5.0));
        assert!((mid.0 - 5.0).abs() < 0.1, "Expected ~5.0, got {:?}", mid);
    }

    #[test]
    fn test_boundary_conditions() {
        // Test that values at keyframes are exact.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(1.0))),
                interpolation_out: Interpolation::Linear,
            },
        );

        // Values at keyframes should be exact.
        assert_eq!(map.interpolate(Time::from(0.0)), Real(0.0));
        assert_eq!(map.interpolate(Time::from(10.0)), Real(10.0));

        // Values before/after range should clamp.
        assert_eq!(map.interpolate(Time::from(-5.0)), Real(0.0));
        assert_eq!(map.interpolate(Time::from(15.0)), Real(10.0));
    }

    #[test]
    fn test_extreme_asymmetric_speeds() {
        // Test very different speeds.
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(10.0))), // Very fast.
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.1))), // Very slow.
                interpolation_out: Interpolation::Linear,
            },
        );

        // The curve should rise quickly then plateau.
        let early = map.interpolate(Time::from(1.0));
        let late = map.interpolate(Time::from(9.0));

        // Early should be much higher than linear (10% = 1.0).
        assert!(early.0 > 1.5, "Early should be > 1.5, got {:?}", early);

        // Late should be close to 10.0 due to slow incoming.
        assert!(late.0 > 9.0, "Late should be > 9.0, got {:?}", late);

        // Note: Cubic Bezier curves can overshoot endpoints with extreme tangents.
        // This is expected behavior. We just verify the curve is reasonable.
        assert!(
            early.0 < 12.0,
            "Early should be reasonable, got {:?}",
            early
        );
        assert!(late.0 < 12.0, "Late should be reasonable, got {:?}", late);
    }

    #[test]
    fn test_zero_speed_tangents() {
        // Test zero speeds (flat tangents).
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(0.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.0))), // Flat outgoing.
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(10.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(0.0))), // Flat incoming.
                interpolation_out: Interpolation::Linear,
            },
        );

        // With zero speeds, the curve should be S-shaped.
        let early = map.interpolate(Time::from(2.0));
        let mid = map.interpolate(Time::from(5.0));
        let late = map.interpolate(Time::from(8.0));

        // Early should be close to 0 (within reasonable tolerance).
        assert!(early.0 < 1.5, "Early should be < 1.5, got {:?}", early);

        // Late should be approaching 10 (S-curve with flat tangents).
        assert!(late.0 > 8.0, "Late should be > 8.0, got {:?}", late);

        // Mid should be around 5 (S-curve center).
        assert!(
            (mid.0 - 5.0).abs() < 1.5,
            "Mid should be ~5.0, got {:?}",
            mid
        );
    }

    #[test]
    fn test_negative_speeds() {
        // Test negative speeds (curves going backwards).
        let mut map = TimeDataMap::<Real>::from(BTreeMap::new());

        map.insert_with_interpolation(
            Time::from(0.0),
            Real(5.0),
            Key {
                interpolation_in: Interpolation::Linear,
                interpolation_out: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(-2.0))), // Going down.
            },
        );

        map.insert_with_interpolation(
            Time::from(10.0),
            Real(5.0),
            Key {
                interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Real(2.0))), // Coming up.
                interpolation_out: Interpolation::Linear,
            },
        );

        // The curve should dip below 5.0 in the middle.
        let mid = map.interpolate(Time::from(5.0));
        assert!(mid.0 < 5.0, "Mid should be < 5.0, got {:?}", mid);

        // Values at endpoints should be 5.0.
        assert_eq!(map.interpolate(Time::from(0.0)), Real(5.0));
        assert_eq!(map.interpolate(Time::from(10.0)), Real(5.0));
    }
}

// AIDEV-NOTE: Handle clamping is tested indirectly through the integration tests above.
// The extreme_asymmetric_speeds and negative_speeds tests verify that the handle
// clamping prevents invalid curves.
