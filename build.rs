use std::env;
use std::path::PathBuf;

#[cfg(target_os = "linux")]
const R_H: &str = "/usr/lib64/R/include/R.h";
#[cfg(target_os = "linux")]
const RINTERNALS_H: &str = "/usr/lib64/R/include/Rinternals.h";
#[cfg(target_os = "linux")]
const INCLUDE_PATH: &str = "-I/usr/lib64/R/include";

#[cfg(target_os = "macos")]
const R_H: &str = "/usr/local/include/R.h";
#[cfg(target_os = "macos")]
const RINTERNALS_H: &str = "/usr/local/include/Rinternals.h";
#[cfg(target_os = "macos")]
const INCLUDE_PATH: &str = "-I/usr/local/include";

fn main() {
    // println!("cargo:rerun-if-changed=src/bindings.rs");
    println!("cargo:rustc-link-lib=R");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=/usr/local/lib/R/lib/");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-search=/usr/lib64/R/lib");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bind_file = crate_dir.join("src/bindings.rs");

    if !bind_file.exists() {
        let bindings = bindgen::Builder::default()
            .clang_arg(INCLUDE_PATH)
            // .clang_arg("-I/usr/local/include")
            .header(R_H)
            .header(RINTERNALS_H)
            .blacklist_item("FP_.*")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate()
            .expect("Failed to parse headers");

        bindings
            .write_to_file(bind_file)
            .expect("Failed to write R bindings");
    }
}
