REM
REM Copyright (C) 2011-15 DyND Developers
REM BSD 2-Clause License, see LICENSE.txt
REM
REM This is the master windows build + test script for building
REM libdynd.
REM
REM Jenkins Requirements:
REM   - Use a jenkins build matrix for multiple
REM     platforms/python versions
REM   - Use the XShell plugin to launch this script
REM   - Call the script from the root workspace
REM     directory as ./jenkins/jenkins-build
REM   - Use a user-defined axis to select compiler versions with COMPILER_VERSION
REM

REM Require a compiler version to be selected
if "%COMPILER_VERSION%" == "" exit /b 1

REM Determine 32/64-bit based on the machine name, or allow it to be already
REM be specified from the COMPILER_3264 variable
if not "%COMPILER_3264%" == "" goto compiler_3264_done
REM Check if '32' or '64' is a substring in COMPUTERNAME, by using search/replace
if not "%COMPUTERNAME:32=XX%" == "%COMPUTERNAME%" set COMPILER_3264=32
if not "%COMPUTERNAME:64=XX%" == "%COMPUTERNAME%" set COMPILER_3264=64
REM Require that COMPILER_3264 be selected
if "%COMPILER_3264%" == "" exit /b 1
:compiler_3264_done

REM Jenkins has '/' in its workspace. Fix it to '\' to simplify the DOS commands.
set WORKSPACE=%WORKSPACE:/=\%

REM Determine the MSVC version from the compiler version
set CMAKE_BUILD_TARGET=
if "%COMPILER_VERSION%" == "MSVC2013" set CMAKE_BUILD_TARGET=Visual Studio 12
if "%CMAKE_BUILD_TARGET%" == "" exit /b 1

REM Create variables for the various pieces
if NOT "%COMPILER_3264%" == "64" goto :notamd64
 set CMAKE_BUILD_TARGET=%CMAKE_BUILD_TARGET% Win64
:notamd64

REM Remove the build subdirectory from last time
rd /q /s build

REM Create a fresh visual studio solution with cmake, and do the build/install
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=install -G "%CMAKE_BUILD_TARGET%" .. || exit /b 1
cmake --build . --config RelWithDebInfo || exit /b 1

REM Run gtests
.\tests\RelWithDebInfo\test_libdynd --gtest_output=xml:../test_results.xml || exit /b 1

exit /b 0
