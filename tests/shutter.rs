use core::num::NonZeroU16;
use token_value_map::*;

fn shutter(start: f64, open_start: f64, open_end: f64, end: f64) -> Shutter {
    Shutter {
        range: Time::from_secs(start)..Time::from_secs(end),
        opening: Time::from_secs(open_start)..Time::from_secs(open_end),
    }
}

fn samples(n: u16) -> NonZeroU16 {
    NonZeroU16::new(n).unwrap()
}

#[test]
fn opening_is_a_clamped_trapezoid() {
    let s = shutter(0.0, 0.25, 0.75, 1.0);
    for (t, expected) in [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.125, 0.5),
        (0.25, 1.0),
        (0.5, 1.0),
        (0.75, 1.0),
        (0.875, 0.5),
        (1.0, 0.0),
        (1.5, 0.0),
    ] {
        let got = s.opening(Time::from_secs(t));
        assert!(
            (got - expected).abs() < 1e-6,
            "opening({t}) = {got}, expected {expected}"
        );
    }
}

#[test]
fn opening_does_not_depend_on_where_the_shutter_sits_in_time() {
    let at_zero = shutter(0.0, 0.25, 0.75, 1.0);
    let later = shutter(10.0, 10.25, 10.75, 11.0);
    for t in [0.0, 0.1, 0.25, 0.6, 0.9, 1.0] {
        let a = at_zero.opening(Time::from_secs(t));
        let b = later.opening(Time::from_secs(10.0 + t));
        assert!((a - b).abs() < 1e-5, "t={t}: {a} vs {b}");
    }
}

#[test]
fn time_at_exposure_spans_the_range() {
    let s = shutter(2.0, 2.5, 3.0, 4.0);
    assert!((s.time_at_exposure(0.0).to_secs() - 2.0).abs() < 1e-6);
    assert!((s.time_at_exposure(1.0).to_secs() - 4.0).abs() < 1e-6);
}

#[test]
fn time_at_exposure_inverts_the_exposure_integral() {
    // Trapezoid: ramp 0..1, plateau 1..3, ramp 3..4. Total exposure 0.5 + 2 + 0.5 = 3.
    let s = shutter(0.0, 1.0, 3.0, 4.0);
    let exposure_until = |t: f64| -> f64 {
        if t <= 1.0 {
            0.5 * t * t
        } else if t <= 3.0 {
            0.5 + (t - 1.0)
        } else {
            let y = t - 3.0;
            2.5 + y - 0.5 * y * y
        }
    };
    for fraction in [0.01, 0.1, 0.16, 0.5, 0.84, 0.9, 0.99] {
        let t = s.time_at_exposure(fraction).to_secs();
        let got = exposure_until(t) / 3.0;
        assert!(
            (got - f64::from(fraction)).abs() < 1e-5,
            "fraction {fraction}: time {t} has exposure fraction {got}"
        );
    }
}

#[test]
fn box_shutter_samples_the_middle_of_equal_slices() {
    let s = shutter(0.0, 0.0, 1.0, 1.0);
    let times: Vec<f64> = s.sample_times(samples(4)).map(|t| t.to_secs()).collect();
    for (got, expected) in times.iter().zip([0.125, 0.375, 0.625, 0.875]) {
        assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
    }
}

#[test]
fn samples_of_a_linear_motion_average_to_the_shutter_centroid() {
    // A plain average of the samples is the exposure-weighted mean, which for a
    // symmetric shutter is its center, and for a one-sided ramp is off-center.
    let mut map = TimeDataMap::from_single(Time::from_secs(0.0), Real(0.0));
    map.insert(Time::from_secs(1.0), Real(1.0));

    let symmetric = shutter(0.0, 0.25, 0.75, 1.0);
    let values = map.sample(&symmetric, samples(64)).unwrap();
    let mean = values.iter().map(|v| v.0).sum::<f64>() / values.len() as f64;
    assert!((mean - 0.5).abs() < 1e-4, "symmetric shutter mean {mean}");

    // Opens over the whole interval, snaps shut at the end: centroid at 2/3.
    let ramp = shutter(0.0, 1.0, 1.0, 1.0);
    let values = map.sample(&ramp, samples(256)).unwrap();
    let mean = values.iter().map(|v| v.0).sum::<f64>() / values.len() as f64;
    assert!((mean - 2.0 / 3.0).abs() < 1e-4, "ramp shutter mean {mean}");
}

#[test]
fn samples_cluster_where_the_shutter_is_open() {
    let s = shutter(0.0, 0.4, 0.6, 1.0);
    let times: Vec<f64> = s.sample_times(samples(100)).map(|t| t.to_secs()).collect();
    let on_plateau = times.iter().filter(|t| (0.4..=0.6).contains(*t)).count();
    // The plateau is 20% of the time but 0.2 / (0.2 + 0.2 + 0.2) = 1/3 of the exposure.
    assert!(
        (32..=35).contains(&on_plateau),
        "{on_plateau} of 100 samples on the plateau"
    );
    assert!(
        times.windows(2).all(|w| w[0] < w[1]),
        "sample times must increase"
    );
}
