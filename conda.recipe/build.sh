#!/bin/bash
set -ex

cd $RECIPE_DIR

echo Setting the compiler...
if [ `uname` == Linux ]; then
    export CC="$PREFIX/bin/gcc"
    export CXX="$PREFIX/bin/g++"
elif [ `uname` == Darwin ]; then
    export CC="$PREFIX/bin/gcc"
    export CXX="$PREFIX/bin/g++"
#    CPPFLAGS="-stdlib=libc++"
    EXTRAOPTIONS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"
    MACOSX_DEPLOYMENT_TARGET=10.9
else
    CPPFLAGS=
    EXTRAOPTIONS=
fi

echo Creating build directory...
cd ..
mkdir build
cd build
pwd
echo Configuring build with cmake...
cmake \
    $EXTRAOPTIONS \
    -DCMAKE_CXX_FLAGS="$CPPFLAGS" \
    -DDYND_INSTALL_LIB=ON \
    -DDYND_BUILD_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX .. || exit 1
echo Building with make...
make || exit 1
echo Installing the build...
make install || exit 1

