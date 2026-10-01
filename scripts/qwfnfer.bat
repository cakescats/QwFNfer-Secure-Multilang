@echo off
REM qwfnfer: the one command (Windows).
REM Starts the console (or finds the one already running) and opens it.
REM Works from the release bundle (this file next to bin\ and tools\) and
REM from a source checkout (this file under scripts\).
REM
REM   qwfnfer                 open the console at http://127.0.0.1:8090
REM   qwfnfer --no-browser    just run the console in this terminal
REM   qwfnfer --port 8091     another console port (--server-port N for the engine's)
REM   qwfnfer --start --model D:\models\flash-next\UD-Q4_K_XL\...-00001-of-00004.gguf
REM                           serve that shard's model right away, even outside the
REM                           scanned folders (its folder is saved as a location)
setlocal EnableDelayedExpansion
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
if not exist "%HERE%\tools\qwfn_console.py" (
  for %%I in ("%HERE%\..") do set "HERE=%%~fI"
)
if not exist "%HERE%\tools\qwfn_console.py" (
  echo qwfnfer: tools\qwfn_console.py not found next to %0 1>&2
  exit /b 1
)
set "port=8090"
set "browser=1"
set "args="
:parse
if "%~1"=="" goto done
if "%~1"=="--no-browser" ( set "browser=0" ) else if "%~1"=="--port" ( set "port=%~2" & set "args=%args% --port %~2" & shift ) else ( set "args=%args% %~1" )
shift
goto parse
:done
set "url=http://127.0.0.1:%port%"
REM Already running? Just open it.
python -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1', int('%port%'))) == 0 else 1)" 2>nul
if %ERRORLEVEL%==0 (
  echo qwfnfer console is already running at %url%
  if "%browser%"=="1" start "" "%url%"
  exit /b 0
)
REM The bundle keeps its DLLs next to the engine.
if exist "%HERE%\bin\ggml-base.dll" set "PATH=%HERE%\bin;%PATH%"
if "%browser%"=="1" start "" "%url%"
cd /d "%HERE%"
python tools\qwfn_console.py %args%
