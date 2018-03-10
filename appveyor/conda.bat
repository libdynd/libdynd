Powershell -Command "Start-FileDownload \"https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-$env:PYTHON_ARCH.exe\" C:\Miniconda.exe"
C:\Miniconda.exe /S /D=C:\Py
SET PATH=C:\Py;C:\Py\Scripts;C:\Py\Library\bin;%PATH%
conda config --set always_yes yes
conda update conda
conda install conda-build
if not defined APPVEYOR_PULL_REQUEST_NUMBER conda install anaconda-client
conda-build conda.recipe
