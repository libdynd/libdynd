@ECHO ON

mkdir build
pushd build

cmake -G "Visual Studio 14 Win64" -DCMAKE_BUILD_TYPE=%MSVC_BUILD_TYPE% -DCMAKE_CXX_FLAGS="-MP3" .. || exit /b 1

cmake --build . --config %MSVC_BUILD_TYPE% || exit /b 1

"tests/%MSVC_BUILD_TYPE%/test_libdynd.exe" || exit /b 1

popd
