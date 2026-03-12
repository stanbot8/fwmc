@echo off
:: Launch FWMC Brain Viewer
:: Builds if needed, then runs

setlocal enabledelayedexpansion

:: Default build dir is sibling to source
set BUILD_DIR=%~dp0..\fwmc-build
if defined FWMC_BUILD_DIR set BUILD_DIR=%FWMC_BUILD_DIR%

:: Check both possible output locations
set VIEWER_EXE=
if exist "%BUILD_DIR%\viewer\Release\fwmc-viewer.exe" (
    set "VIEWER_EXE=%BUILD_DIR%\viewer\Release\fwmc-viewer.exe"
)
if exist "%BUILD_DIR%\Release\fwmc-viewer.exe" (
    set "VIEWER_EXE=%BUILD_DIR%\Release\fwmc-viewer.exe"
)

if "!VIEWER_EXE!"=="" (
    echo Viewer not found, building...
    cmake --build "%BUILD_DIR%" --config Release --target fwmc-viewer
    if errorlevel 1 (
        echo Build failed.
        pause
        exit /b 1
    )
    if exist "%BUILD_DIR%\viewer\Release\fwmc-viewer.exe" (
        set "VIEWER_EXE=%BUILD_DIR%\viewer\Release\fwmc-viewer.exe"
    ) else (
        set "VIEWER_EXE=%BUILD_DIR%\Release\fwmc-viewer.exe"
    )
)

echo Launching FWMC Brain Viewer...
start "" "!VIEWER_EXE!"
