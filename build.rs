use std::path::PathBuf;
use std::env;

fn main() {
    // println!("cargo:rerun-if-changed=src/bindings.h");
    // let r_inc_path = Path::new("/usr/lib64/R/include");
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bindings = bindgen::Builder::default()
        .clang_arg("-I/usr/lib64/R/include")
        .header("/usr/lib64/R/include/R.h")
        .header("/usr/lib64/R/include/Rinternals.h")
        .blacklist_item("FP_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Failed to parse headers");

    bindings.write_to_file(crate_dir.join("src/bindings.rs"))
        .expect("Failed to write R bindings");

}

