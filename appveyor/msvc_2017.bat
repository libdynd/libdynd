@ECHO ON

mkdir build
pushd build

cmake -G "Visual Studio 14 Win64" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ..

cmake --build . --config %BUILD_TYPE%

"tests/%BUILD_TYPE%/test_libdynd.exe"

popd
