let
  pinnedNixpkgs = fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/a9e6d84f9c2f9012f5fe7d964a7851352300e61a.tar.gz";
    sha256 = "1fs3yf53flp3yj8wnp2izxhxqwkzmiq2wnd29lhfj15ppzdi6xss";
  };
in

{ pkgs ? import pinnedNixpkgs {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    rustup
    clang-tools
    cmake
    gcc
    pkg-config
    cargo-llvm-cov
    ccache
  ];

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
}
