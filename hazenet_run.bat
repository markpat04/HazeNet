@echo off
REM HazeNet pipeline launcher (Windows).
REM Activates the hazenet conda env so matplotlib/GDAL DLLs (Library\bin) load,
REM then runs the CLI. Usage:
REM   hazenet_run.bat --config configs\local.yaml --stage all
REM   hazenet_run.bat --config configs\local.yaml --stage datacube,train,eval
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
call conda run -n hazenet --no-capture-output python -m hazenet.cli %*
