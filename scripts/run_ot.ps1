# Invoke Python in the user-selected OT environment with its native DLL paths.
$otPrefix = 'D:\miniconda_envs\OT'
$env:PATH = "$otPrefix;$otPrefix\Library\bin;$otPrefix\Scripts;" + $env:PATH
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& "$otPrefix\python.exe" @args
exit $LASTEXITCODE
