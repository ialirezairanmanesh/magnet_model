@echo off
echo Compiling thesis for Windows...
echo.

REM Clean previous files
del *.aux *.log *.out *.toc *.lof *.lot *.loa *.bbl *.bcf *.blg *.run.xml *.synctex.gz 2>nul

echo Step 1: First compilation...
xelatex SBUKThesis-main.tex
if errorlevel 1 (
    echo Error in first compilation!
    pause
    exit /b 1
)

echo Step 2: Bibliography compilation...
biber SBUKThesis-main
if errorlevel 1 (
    echo Error in bibliography compilation!
    pause
    exit /b 1
)

echo Step 3: Second compilation...
xelatex SBUKThesis-main.tex
if errorlevel 1 (
    echo Error in second compilation!
    pause
    exit /b 1
)

echo Step 4: Third compilation...
xelatex SBUKThesis-main.tex
if errorlevel 1 (
    echo Error in third compilation!
    pause
    exit /b 1
)

echo.
echo Compilation completed successfully!
echo Output file: SBUKThesis-main.pdf
echo.
pause
