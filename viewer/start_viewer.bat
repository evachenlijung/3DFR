@echo off
rem Double-click to compare meshes under C:\atop\data (edit the path below if needed).
rem Needs a Python with numpy, e.g. run it from an activated conda prompt.
cd /d "%~dp0"
python serve.py --root "C:\atop\data" %*
pause
