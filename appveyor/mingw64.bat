@ECHO ON
SET MINGW_PREFIX=C:/mingw-w64/x86_64-7.2.0-posix-seh-rt_v5-rev1/bin

mkdir build
pushd build

:: -DCMAKE_SH="CMAKE_SH-NOTFOUND" makes CMake ignore the fact that sh.exe
:: is on the path of the appveyor build workers.
cmake ^
    -DCMAKE_C_COMPILER=%MINGW_PREFIX%/gcc.exe ^
    -DCMAKE_CXX_COMPILER=%MINGW_PREFIX%/gcc.exe ^
    -DCMAKE_VERBOSE_MAKEFILE=ON ^
    -DCMAKE_CXX_FLAGS="-Wno-error" ^
    -DCMAKE_SH="CMAKE_SH-NOTFOUND" ^
    -G "MinGW Makefiles" ^
    ..

mingw32-make -j4
"tests/test_libdynd.exe"
popd
