SET PATH=C:\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev1\bin;%PATH%
mkdir build
pushd build
:: -DCMAKE_SH="CMAKE_SH-NOTFOUND" makes CMake ignore the fact that sh.exe
:: is on the path of the appveyor build workers.
cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_FLAGS="-Wno-error" -DCMAKE_SH="CMAKE_SH-NOTFOUND" -G "MinGW Makefiles" ..
mingw32-make -j4
"tests/test_libdynd.exe"
popd
