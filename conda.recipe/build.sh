#!/bin/bash
set -ex

cd $RECIPE_DIR

echo Setting the compiler...
CC=cc
CXX=c++

if [ `uname` == Linux ]; then
    CC=gcc48
    CXX=g++48
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
    -DCMAKE_CXX_FLAGS="$CPPFLAGS" \
    -DCMAKE_BUILD_TYPE=Release  \
    -DDYND_SHARED_LIB=ON \
    -DDYND_INSTALL_LIB=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX .. || exit 1
echo Building with make...
make || exit 1
echo Installing the build...
make install || exit 1

