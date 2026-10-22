@echo off
chcp 65001 >nul
title SPECTRUM (이 창을 닫으면 앱이 꺼집니다)
rem 더블클릭하면 이 PC에서 웹앱을 켜고 브라우저로 연다. 학습은 웹앱의 [자동 태깅 - 학습 현황]에서 한다.
rem 이 파일은 UTF-8 + CRLF로 저장해야 한글이 깨지지 않는다 (.gitattributes *.bat eol=crlf).
cd /d "%~dp0"

set "PY=python"
%PY% --version >nul 2>nul
if errorlevel 1 set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python을 찾을 수 없습니다. README.md의 "처음 한 번: 설치"를 먼저 해 주세요.
  echo.
  pause
  exit /b 1
)

echo.
echo SPECTRUM 웹앱을 켜는 중입니다. 잠시 뒤 브라우저가 열립니다.
echo 안 열리면 브라우저 주소창에 http://localhost:8501 을 입력하세요.
echo 이 검은 창을 닫으면 앱이 꺼집니다. 학습 중이었다면 다시 켜고 학습 시작을 누르면 이어서 합니다.
echo.
%PY% -m streamlit run app.py
echo.
echo 앱이 꺼졌습니다. 위에 오류가 있다면 README.md의 "문제가 생기면"을 확인하세요.
pause
