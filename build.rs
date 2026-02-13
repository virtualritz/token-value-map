// AIDEV-NOTE: Compile-time feature validation for math backend configuration.
// Ensures mutually exclusive backends (glam vs nalgebra vs ultraviolet) and
// prevents incompatible feature combinations (rkyv + nalgebra/ultraviolet).

fn main() {
    let has_glam = cfg!(feature = "glam");
    let has_nalgebra = cfg!(feature = "nalgebra");
    let has_ultraviolet = cfg!(feature = "ultraviolet");

    let count = has_glam as u8 + has_nalgebra as u8 + has_ultraviolet as u8;

    if count > 1 {
        panic!(
            "Only one math backend can be enabled at a time. \
             Choose one of: 'glam' (default), 'nalgebra', or 'ultraviolet'."
        );
    }

    // rkyv only works with glam backend.
    let has_rkyv = cfg!(feature = "rkyv");
    if has_rkyv && !has_glam {
        panic!("The 'rkyv' feature requires the 'glam' backend.");
    }

    if count == 0 {
        println!(
            "cargo:warning=No math backend selected. \
             Enable one of: 'glam' (default), 'nalgebra', or 'ultraviolet'."
        );
    }
}
