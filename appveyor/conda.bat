@ECHO ON

IF "%PYTHON_ARCH%"=="x86_64" SET MINICONDA_ROOT=C:\Miniconda3-x64
IF "%PYTHON_ARCH%"=="x86" SET MINICONDA_ROOT=C:\Miniconda3
SET PATH=%MINICONDA_ROOT%;%MINICONDA_ROOT%\Scripts;%MINICONDA_ROOT%\Library\bin;%PATH%

conda config --set always_yes yes || exit /b 1
conda update --all || exit /b 1

conda install conda-build || exit /b 1
if not defined APPVEYOR_PULL_REQUEST_NUMBER conda install anaconda-client || exit /b 1

conda-build conda.recipe || exit /b 1
