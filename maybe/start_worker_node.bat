@echo off
REM Batch file wrapper to keep PowerShell window open
REM This allows you to double-click the file to run it

cd /d "%~dp0"
powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0start_worker_node.ps1"
pause

