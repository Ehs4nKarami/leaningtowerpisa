{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    packages = [
        (pkgs.python311.withPackages (ps: with ps; [
            pip
            setuptools
            wheel
            virtualenv
        ]))
    ];
    
    
    env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.libz
    ];
}
