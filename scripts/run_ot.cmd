@echo off
setlocal
set "OT_PREFIX=D:\miniconda_envs\OT"
set "PATH=%OT_PREFIX%;%OT_PREFIX%\Library\bin;%OT_PREFIX%\Scripts;%PATH%"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
"%OT_PREFIX%\python.exe" %*
exit /b %ERRORLEVEL%
