#!/bin/bash
set -ex

cd $RECIPE_DIR

conda install -c stuarteberg gcc=4.8.4.99

echo Setting the compiler...
if [ `uname` == Linux ]; then
    export CC="$PREFIX/bin/gcc -D__USE_XOPEN2K8"
    export CXX="$PREFIX/bin/g++ -D__USE_XOPEN2K8"
elif [ `uname` == Darwin ]; then
    CPPFLAGS="-stdlib=libc++"
    EXTRAOPTIONS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.8"
    MACOSX_DEPLOYMENT_TARGET=10.8
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
    -DCMAKE_BUILD_TYPE=Release  \
    -DDYND_SHARED_LIB=ON \
    -DDYND_INSTALL_LIB=ON \
    -DDYND_BUILD_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX .. || exit 1
echo Building with make...
make || exit 1
echo Installing the build...
make install || exit 1

