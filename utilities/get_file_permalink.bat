@echo off
rem Print (and copy) a GitHub permalink to a file in this repository.
rem
rem Asks for a path - absolute, or relative to the directory the script is run
rem from - and builds <origin>/blob/<commit sha>/<path in repo> for it. The SHA
rem (not a branch name) is what makes it a permalink: it keeps pointing at the
rem file as it is now, even after later commits move things around.
setlocal

set "INPUT="
set /p "INPUT=Enter local file path: "
if not defined INPUT (
    echo No path entered.
    goto :end
)

rem Drop the quotes "Copy as path" in Explorer wraps the path in
set "INPUT=%INPUT:"=%"

rem Absolute path of the target; also normalizes / to \
for %%I in ("%INPUT%") do set "ABSFILE=%%~fI"

if not exist "%ABSFILE%" (
    echo File not found: "%ABSFILE%"
    goto :end
)

rem The directory holding the file, WITHOUT its trailing backslash: in
rem git -C "C:\dir\" the \" reads as an escaped quote, so git gets one mangled
rem argument and fails. (A drive root keeps its backslash - "C:" alone means
rem "the current directory on C:", which is not the same place.)
for %%I in ("%ABSFILE%") do set "FILEDIR=%%~dpI"
if "%FILEDIR:~-1%"=="\" set "FILEDIR=%FILEDIR:~0,-1%"
if "%FILEDIR:~-1%"==":" set "FILEDIR=%FILEDIR%\"

rem The repository the file belongs to
set "REPOROOT="
for /f "delims=" %%R in ('git -C "%FILEDIR%" rev-parse --show-toplevel 2^>nul') do set "REPOROOT=%%R"

if not defined REPOROOT (
    echo Could not find a Git repository for this file:
    echo   "%ABSFILE%"
    goto :end
)

rem Its path inside the repository
set "RELFILE="
for /f "delims=" %%P in ('git -C "%REPOROOT%" ls-files --full-name -- "%ABSFILE%" 2^>nul') do set "RELFILE=%%P"

if not defined RELFILE (
    echo File is not tracked by Git ^(commit it first^):
    echo   "%ABSFILE%"
    goto :end
)

set "REPOURL="
for /f "delims=" %%U in ('git -C "%REPOROOT%" remote get-url origin 2^>nul') do set "REPOURL=%%U"

if not defined REPOURL (
    echo This repository has no 'origin' remote, so it has no web address.
    goto :end
)

set "COMMIT="
for /f "delims=" %%C in ('git -C "%REPOROOT%" rev-parse HEAD 2^>nul') do set "COMMIT=%%C"

rem git@github.com:user/repo.git and ssh://git@github.com/user/repo.git
rem both have to become https://github.com/user/repo
set "BASE=%REPOURL%"
set "BASE=%BASE:ssh://git@=https://%"
set "BASE=%BASE:git@github.com:=https://github.com/%"
if /i "%BASE:~-4%"==".git" set "BASE=%BASE:~0,-4%"

set "PERMALINK=%BASE%/blob/%COMMIT%/%RELFILE%"

echo.
echo %PERMALINK%
rem <nul set /p keeps the trailing space and newline that "echo x | clip" would
rem otherwise put on the clipboard
<nul set /p "=%PERMALINK%" | clip
echo.
echo Permalink copied to clipboard.

rem A commit that was never pushed has no page on GitHub, so say so rather than
rem handing over a link that 404s.
set "PUSHED="
for /f "delims=" %%B in ('git -C "%REPOROOT%" branch -r --contains HEAD 2^>nul') do set "PUSHED=1"
if not defined PUSHED (
    echo.
    echo WARNING: commit %COMMIT:~0,7% is not on any remote branch yet -
    echo          push it, or the link will not resolve.
)

rem Uncommitted edits are not in the linked commit: the reader sees the file as
rem it was at HEAD, not as it is on disk.
git -C "%REPOROOT%" diff --quiet HEAD -- "%ABSFILE%"
if errorlevel 1 (
    echo.
    echo NOTE: this file has uncommitted changes; the link shows it as of
    echo       commit %COMMIT:~0,7%, without them.
)

:end
echo.
pause
