@ECHO ON
IF "%PYTHON_ARCH%"=="x86_64" SET MINICONDA_ROOT=C:\Miniconda3-x64
IF "%PYTHON_ARCH%"=="x86" SET MINICONDA_ROOT=C:\Miniconda3
SET PATH=%MINICONDA_ROOT%;%MINICONDA_ROOT%\Scripts;%MINICONDA_ROOT%\Library\bin;%PATH%
conda config --set always_yes yes
conda update --all
conda install conda-build
if not defined APPVEYOR_PULL_REQUEST_NUMBER conda install anaconda-client
conda-build conda.recipe
