@ECHO ON

Powershell -Command "Start-FileDownload \"https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe\" C:\Miniconda.exe"
C:\Miniconda.exe /S /D=C:\Py || exit /b 1
SET PATH=C:\Py;C:\Py\Scripts;C:\Py\Library\bin;%PATH%

conda config --set always_yes yes || exit /b 1

conda install -c conda-forge clangdev=%CLANG_VERSION% ninja || exit /b 1

mkdir build
pushd build

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

cmake ^
    -DCMAKE_VERBOSE_MAKEFILE=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang-cl ^
    -DCMAKE_CXX_COMPILER=clang-cl ^
    -G "Ninja" ^
    .. ^
    || exit /b 1

ninja || exit /b 1

"tests/test_libdynd.exe" || exit /b 1

popd
