use token_value_map::*;

fn make_map(pairs: &[(f64, f64)]) -> TimeDataMap<Real> {
    TimeDataMap::from_iter(pairs.iter().map(|&(k, v)| (Time::from_secs(k), Real(v))))
}

// Note: The test_empty_map test was removed because the Value API ensures
// that TimeDataMaps can never be empty when accessed through proper
// channels. Direct creation of empty TimeDataMaps and calling
// interpolate is no longer a supported use case.

#[test]
fn test_single_entry() {
    let map = make_map(&[(1.0, 42.0)]);
    assert_eq!(interpolate(&map.0, 0.0.into()), Real(42.0));
    assert_eq!(interpolate(&map.0, 2.0.into()), Real(42.0));
}

#[test]
fn test_out_of_bounds() {
    let map = make_map(&[(1.0, 10.0), (1.0, 10.0), (1.0, 10.0), (3.0, 30.0)]);
    assert_eq!(interpolate(&map.0, Time::from_secs(0.5)), Real(10.0));
    assert_eq!(interpolate(&map.0, Time::from_secs(4.0)), Real(30.0));
}

#[test]
fn test_linear_interpolation() {
    let map = make_map(&[(1.0, 10.0), (3.0, 30.0)]);
    assert_eq!(interpolate(&map.0, Time::from_secs(2.0)), Real(20.0));
}

#[test]
fn test_quadratic_interpolation() {
    let map = make_map(&[(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]);
    let result = interpolate(&map.0, Time::from_secs(0.1));
    assert!((result.0 - 1.0).abs() < 1e-6);
}

#[test]
fn test_catmull_rom_interpolation() {
    let map = make_map(&[(1.0, 0.0), (2.0, 1.0), (3.0, 0.0), (4.0, -1.0)]);
    let result = interpolate(&map.0, Time::from_secs(4.0));
    assert!((result.0 + 1.0).abs() < 1e-6);
}

#[test]
fn test_exact_key_match() {
    let map = make_map(&[(1.0, 100.0), (2.0, 200.0), (3.0, 300.0)]);
    assert_eq!(interpolate(&map.0, Time::from_secs(2.0)), Real(200.0));
}

#[cfg(feature = "matrix3")]
#[test]
fn matrix_sampling_is_deterministic() {
    use std::num::NonZeroU16;

    let keyframes = [
        (
            9.0,
            Matrix3::from([
                0.20303795,
                -0.97917086,
                0.0,
                0.97917086,
                0.20303795,
                0.0,
                0.0,
                0.0,
                1.0,
            ]),
        ),
        (
            10.0,
            Matrix3::from([
                0.17364822, -0.9848077, 0.0, 0.9848077, 0.17364822, 0.0, 0.0, 0.0, 1.0,
            ]),
        ),
        (
            11.0,
            Matrix3::from([
                0.14496485, -0.9894366, 0.0, 0.9894366, 0.14496485, 0.0, 0.0, 0.0, 1.0,
            ]),
        ),
    ];

    let map = TimeDataMap::from_iter(keyframes.into_iter().map(|(t, m)| (Time::from_secs(t), m)));

    let shutter = Shutter {
        range: Time::from_secs(9.75)..Time::from_secs(10.25),
        opening: Time::from_secs(9.75)..Time::from_secs(10.25),
    };

    let samples_a = map
        .sample(&shutter, NonZeroU16::new(8).unwrap())
        .expect("sampling should succeed");
    let samples_b = map
        .sample(&shutter, NonZeroU16::new(8).unwrap())
        .expect("sampling should succeed");

    assert_eq!(samples_a, samples_b);
}

#[cfg(feature = "matrix3")]
#[test]
fn rotation_decomposition_roundtrip() {
    use nalgebra::Matrix3;

    let angle = std::f32::consts::PI / 3.0;
    let rot = Matrix3::new(
        angle.cos(),
        -angle.sin(),
        0.0,
        angle.sin(),
        angle.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let matrix = Matrix3(rot);

    let decomposed = super::super::time_data_map::tests::expose_decompose(&matrix.0);
    let recomposed = super::super::time_data_map::tests::expose_recompose(
        &decomposed.translation,
        &decomposed.rotation,
        &decomposed.stretch,
    );

    for (a, b) in matrix.0.iter().zip(recomposed.iter()) {
        assert!((a - b) == 0.0);
    }
}
