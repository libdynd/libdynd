SET PATH=C:\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev1\bin;%PATH%
mkdir build
pushd build
cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_FLAGS="-Wno-error" -G "MinGW Makefiles" ..
mingw32-make -j4
"tests/test_libdynd.exe"
popd
