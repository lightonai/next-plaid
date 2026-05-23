{
  description = "colgrep";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = [ "x86_64-linux" ];
      forAllSystems = lib.genAttrs systems;

      pkgsFor = { system, cudaSupport, cudaCapabilities ? [] }:
        import nixpkgs {
          inherit system;
          config = {
            inherit cudaSupport cudaCapabilities;
            allowUnfree = cudaSupport;
            cudaForwardCompat = true;
          };
        };

      mkColgrep = pkgs:
        let
          cudaSupport = pkgs.config.cudaSupport or false;
          runtimeLibraryPath = lib.makeLibraryPath ([
            pkgs.onnxruntime
          ] ++ lib.optionals cudaSupport [
            pkgs.linuxPackages.nvidia_x11
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ]);
        in
        pkgs.rustPlatform.buildRustPackage {
          __structuredAttrs = true;
          strictDeps = true;
          
          pname = "colgrep";
          version = (lib.importTOML ./Cargo.toml).workspace.package.version;

          src = ./.;

          buildAndTestSubdir = "colgrep";
          cargoLock.lockFile = ./Cargo.lock;

          buildNoDefaultFeatures = true;
          buildFeatures = lib.optionals cudaSupport [ "cuda" ];
          doCheck = false;

          nativeBuildInputs = [
            pkgs.pkg-config
            pkgs.makeWrapper
          ];

          buildInputs = [
            pkgs.onnxruntime
            pkgs.openssl
            pkgs.curl
          ] ++ lib.optionals cudaSupport [
            pkgs.linuxPackages.nvidia_x11
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];

          env = {
            ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
            ORT_SKIP_DOWNLOAD = "1";
          } // lib.optionalAttrs cudaSupport {
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          };

          postFixup = ''
            wrapProgram $out/bin/colgrep \
              --prefix LD_LIBRARY_PATH : ${runtimeLibraryPath} \
              --set-default ORT_DYLIB_PATH ${pkgs.onnxruntime}/lib/libonnxruntime.so \
              --set-default ORT_LIB_LOCATION ${pkgs.onnxruntime}/lib \
              --set-default ORT_PREFER_DYNAMIC_LINK 1 \
              --set-default ORT_SKIP_DOWNLOAD 1
          '';

          meta = {
            description = "Semantic code search powered by ColBERT";
            homepage = "https://github.com/lightonai/next-plaid/tree/main/colgrep";
            license = lib.licenses.asl20;
            mainProgram = "colgrep";
          };
        };

      mkDevShell = pkgs:
        let
          cudaSupport = pkgs.config.cudaSupport or false;
          libraryPath = lib.makeLibraryPath ([
            pkgs.onnxruntime
          ] ++ lib.optionals cudaSupport [
            pkgs.linuxPackages.nvidia_x11
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ]);
        in
        pkgs.mkShell {
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
            pkgs.clippy
            pkgs.pkg-config
            pkgs.onnxruntime
            pkgs.openssl
            pkgs.curl
          ] ++ lib.optionals cudaSupport [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];

          env = {
            ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
            ORT_SKIP_DOWNLOAD = "1";
          } // lib.optionalAttrs cudaSupport {
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
            RUSTFLAGS = "-L native=${pkgs.linuxPackages.nvidia_x11}/lib";
          };

          shellHook = ''
            export LD_LIBRARY_PATH="${libraryPath}:''${LD_LIBRARY_PATH:-}"
          '';
        };
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = pkgsFor { inherit system; cudaSupport = false; };
          pkgsCuda = pkgsFor { inherit system; cudaSupport = true; };
        in
        {
          colgrep = mkColgrep pkgs;
          colgrep-cuda = mkColgrep pkgsCuda;

          default = self.packages.${system}.colgrep;
        });

      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor { inherit system; cudaSupport = false; };
          pkgsCuda = pkgsFor { inherit system; cudaSupport = true; };
        in
        {
          default = mkDevShell pkgs;
          cuda = mkDevShell pkgsCuda;
        });

      lib = {
        mkPackagesWithCudaCapabilities = capabilities: forAllSystems (system:
          let
            pkgs = pkgsFor { inherit system; cudaSupport = true; cudaCapabilities = capabilities; };
          in
          {
            colgrep = mkColgrep pkgs;
          });
      };
    };
}
