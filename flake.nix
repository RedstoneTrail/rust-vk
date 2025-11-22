{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ fenix, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      perSystem =
        { pkgs, system, ... }:

        let
          rust = fenix.packages.${system};
          toolchain =
            with rust;
            combine [
              stable.toolchain
            ];
          packageList = [
            pkgs.cmake
            pkgs.pkg-config
            pkgs.shaderc
            pkgs.spirv-tools
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.vulkan-tools-lunarg
            pkgs.vulkan-validation-layers
            pkgs.wayland
            pkgs.libxkbcommon
            rust.rust-analyzer
          ];
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
          };

          devShells.default = pkgs.mkShell {
            nativeBuildInputs = [ toolchain ];

            packages = packageList;

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packageList;
            SHADERC_LIB_DIR = pkgs.lib.makeLibraryPath [ pkgs.shaderc ];
            VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
          };
        };
    };
}
