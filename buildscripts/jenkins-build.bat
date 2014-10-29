REM
REM Copyright (C) 2011-14 DyND Developers
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
set MSVC_VERSION=
if "%COMPILER_VERSION%" == "MSVC2010" set MSVC_VERSION=10.0
if "%COMPILER_VERSION%" == "MSVC2012" set MSVC_VERSION=11.0
if "%COMPILER_VERSION%" == "MSVC2013" set MSVC_VERSION=12.0
if "%MSVC_VERSION%" == "" exit /b 1

REM Create variables for the various pieces
if "%COMPILER_3264%" == "64" goto :amd64
 set MSVC_VCVARS_PLATFORM=x86
 set MSVC_BUILD_PLATFORM=Win32
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11"
 if "%MSVC_VERSION%" == "12.0" set CMAKE_BUILD_TARGET="Visual Studio 12"
goto :notamd64
:amd64
 set MSVC_VCVARS_PLATFORM=amd64
 set MSVC_BUILD_PLATFORM=x64
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10 Win64"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11 Win64"
 if "%MSVC_VERSION%" == "12.0" set CMAKE_BUILD_TARGET="Visual Studio 12 Win64"
:notamd64

REM Configure the appropriate visual studio command line environment
if "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES%\Microsoft Visual Studio %MSVC_VERSION%\VC
if NOT "%PROGRAMFILES(X86)%" == "" set VCDIR=%PROGRAMFILES(X86)%\Microsoft Visual Studio %MSVC_VERSION%\VC
call "%VCDIR%\vcvarsall.bat" %MSVC_VCVARS_PLATFORM%
IF %ERRORLEVEL% NEQ 0 exit /b 1
echo on

REM Remove the build subdirectory from last time
rd /q /s build

REM Create a fresh visual studio solution with cmake, and do the build/install
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=install -G %CMAKE_BUILD_TARGET% ..
IF %ERRORLEVEL% NEQ 0 exit /b 1
devenv libdynd.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%"
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Run gtests
.\tests\RelWithDebInfo\test_libdynd --gtest_output=xml:../test_results.xml
IF %ERRORLEVEL% NEQ 0 exit /b 1

exit /b 0
