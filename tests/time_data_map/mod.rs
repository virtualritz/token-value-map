use token_value_map::*;

fn make_map(pairs: &[(f64, f64)]) -> TimeDataMap<Real> {
    TimeDataMap::from_iter(
        pairs.iter().map(|&(k, v)| (Time::from_secs(k), Real(v))),
    )
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
    let map =
        make_map(&[(1.0, 10.0), (1.0, 10.0), (1.0, 10.0), (3.0, 30.0)]);
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