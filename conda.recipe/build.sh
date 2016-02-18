#!/bin/bash
set -ex

cd $RECIPE_DIR

echo Setting the compiler flags...
if [ `uname` == Linux ]; then
    SHARED_LINKER_FLAGS='-static-libstdc++'
elif [ `uname` == Darwin ]; then
    SHARED_LINKER_FLAGS=''
fi

#elif [ `uname` == Darwin ]; then
 #   export CC="$PREFIX/bin/gcc"
  #  export CXX="$PREFIX/bin/g++"
#    CPPFLAGS="-stdlib=libc++"
  #  EXTRAOPTIONS="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"
   # MACOSX_DEPLOYMENT_TARGET=10.9
#else
 #   CPPFLAGS=
  #  EXTRAOPTIONS=
#fi

echo Creating build directory...
cd ..
mkdir build
cd build
pwd
echo Configuring build with cmake...
cmake \
    $EXTRAOPTIONS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.9 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SHARED_LINKER_FLAGS=$SHARED_LINKER_FLAGS \
    -DDYND_INSTALL_LIB=ON \
    -DDYND_BUILD_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX .. || exit 1
echo Building with make...
# We should use the CPU_COUNT environment variable set by conda-build, but Travis CI has issues with that.
make -j3 || exit 1
echo Installing the build...
make install || exit 1
