REM
REM Copyright (C) 2011-13, DyND Developers
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

REM Jenkins has '/' in its workspace. Fix it to '\' to simplify the DOS commands.
set WORKSPACE=%WORKSPACE:/=\%

REM Determine the MSVC version from the compiler version
set MSVC_VERSION=
if "%COMPILER_VERSION%" == "MSVC2008" set MSVC_VERSION=9.0
if "%COMPILER_VERSION%" == "MSVC2010" set MSVC_VERSION=10.0
if "%COMPILER_VERSION%" == "MSVC2012" set MSVC_VERSION=11.0
if "%MSVC_VERSION%" == "" exit /b 1

REM Create variables for the various pieces
if "%PROCESSOR_ARCHITECTURE%" == "AMD64" goto :amd64
 set MSVC_VCVARS_PLATFORM=x86
 set MSVC_BUILD_PLATFORM=Win32
 if "%MSVC_VERSION%" == "9.0" set CMAKE_BUILD_TARGET="Visual Studio 9 2008"
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11"
goto :notamd64
:amd64
 set MSVC_VCVARS_PLATFORM=amd64
 set MSVC_BUILD_PLATFORM=x64
 if "%MSVC_VERSION%" == "9.0" set CMAKE_BUILD_TARGET="Visual Studio 9 2008 Win64"
 if "%MSVC_VERSION%" == "10.0" set CMAKE_BUILD_TARGET="Visual Studio 10 Win64"
 if "%MSVC_VERSION%" == "11.0" set CMAKE_BUILD_TARGET="Visual Studio 11 Win64"
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
devenv dynd-python.sln /Build "RelWithDebInfo|%MSVC_BUILD_PLATFORM%"
IF %ERRORLEVEL% NEQ 0 exit /b 1

REM Run gtests
.\tests\RelWithDebInfo\test_dynd --gtest_output=xml:../test_results.xml
IF %ERRORLEVEL% NEQ 0 exit /b 1

exit /b 0