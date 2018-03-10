@ECHO ON

SET MINICONDA_ROOT=C:\Miniconda3-x64
SET PATH=%MINICONDA_ROOT%;%MINICONDA_ROOT%\Scripts;%MINICONDA_ROOT%\Library\bin;%PATH%

conda config --set always_yes yes
conda update --all

conda install -c conda-forge clangdev=%CLANG_VERSION% ninja

mkdir build
pushd build

cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -G "Ninja" ..

ninja

"tests/test_libdynd.exe"

popd
