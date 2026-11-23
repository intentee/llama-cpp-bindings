#[cfg(all(feature = "shared-stdcxx", feature = "static-stdcxx"))]
compile_error!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");
