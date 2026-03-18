@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m phase_motion_app.app.main
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if not errorlevel 1 (
    python -m phase_motion_app.app.main
    exit /b %ERRORLEVEL%
)

echo Python was not found. Install Python 3 or ensure ^`py^` or ^`python^` is on PATH.
exit /b 1

REM ######################################################################################################################
REM
REM
REM                                         AAAAAAAA
REM                                       AAAA    AAAAA              AAAAAAAA
REM                                     AAA          AAA           AAAA    AAA
REM                                     AA            AA          AAA       AAA
REM                                     AA            AAAAAAAAAA  AAA       AAAAAAAAAA
REM                                     AAA                  AAA  AAA               AA
REM                                      AAA                AAA    AAAAA            AA
REM                                       AAAAA            AAA        AAA           AA
REM                                          AAA          AAA                       AA
REM                                          AAA         AAA                        AA
REM                                          AA         AAA                         AA
REM                                          AA        AAA                          AA
REM                                         AAA       AAAAAAAAA                     AA
REM                                         AAA       AAAAAAAAA                     AA
REM                                         AA                   AAAAAAAAAAAAAA     AA
REM                                         AA  AAAAAAAAAAAAAAAAAAAAAAAA    AAAAAAA AA
REM                                        AAAAAAAAAAA                           AA AA
REM                                                                            AAA  AA
REM                                                                          AAAA   AA
REM                                                                       AAAA      AA
REM                                                                    AAAAA        AA
REM                                                                AAAAA            AA
REM                                                             AAAAA               AA
REM                                                         AAAAAA                  AA
REM                                                     AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
REM
REM
REM ######################################################################################################################
REM
REM                                                 Copyright (c) JoeShade
REM                               Licensed under the GNU Affero General Public License v3.0
REM
REM ######################################################################################################################
REM
REM                                         +44 (0) 7356 042702 | joe@jshade.co.uk
REM
REM ######################################################################################################################
