@ECHO ON

mkdir build
pushd build

cmake -G "Visual Studio 14 Win64" -DCMAKE_BUILD_TYPE=%MSVC_BUILD_TYPE% ..

cmake --build . --config %MSVC_BUILD_TYPE%

"tests/%MSVC_BUILD_TYPE%/test_libdynd.exe"

popd
