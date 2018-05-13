@ECHO ON

Powershell -Command "Start-FileDownload \"https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-$env:PYTHON_ARCH.exe\" C:\Miniconda.exe"
C:\Miniconda.exe /S /D=C:\Py
SET PATH=C:\Py;C:\Py\Scripts;C:\Py\Library\bin;%PATH%

conda config --set always_yes yes || exit /b 1

conda install conda-build || exit /b 1
if not defined APPVEYOR_PULL_REQUEST_NUMBER conda install anaconda-client || exit /b 1

conda-build conda.recipe || exit /b 1
