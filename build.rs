fn main() {
    println!("cargo::rerun-if-changed=native/hipblaslt.cpp");
    println!("cargo::rerun-if-env-changed=QUASAR_HIPBLASLT_SKIP_NATIVE");

    if std::env::var_os("CARGO_FEATURE_HIPBLASLT").is_none()
        || std::env::var_os("QUASAR_HIPBLASLT_SKIP_NATIVE").as_deref()
            == Some(std::ffi::OsStr::new("1"))
    {
        return;
    }

    cc::Build::new()
        .cpp(true)
        .compiler("/opt/rocm/bin/hipcc")
        .include("/opt/rocm/include")
        .file("native/hipblaslt.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .warnings(true)
        .warnings_into_errors(true)
        .compile("quasar_hipblaslt");

    println!("cargo::rustc-link-search=native=/opt/rocm/lib");
    println!("cargo::rustc-link-lib=dylib=hipblaslt");
    println!("cargo::rustc-link-lib=dylib=amdhip64");
    println!("cargo::rustc-link-lib=dylib=stdc++");
}
