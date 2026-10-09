@echo off
chcp 65001 >nul
echo ============================================================
echo   VTUBER 3D AVATAR - DONG GOI UNG DUNG THANH FILE .EXE
echo ============================================================
echo.

:: 1. Kiem tra va cai dat PyInstaller neu chua co
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Dang cai dat PyInstaller...
    python -m pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [LOI] Khong the cai dat PyInstaller. Vui long kiem tra ket noi mang.
        pause
        exit /b 1
    )
) else (
    echo [1/3] PyInstaller da duoc cai dat san.
)

:: 2. Thuc hien dong goi ung dung bang PyInstaller spec
echo.
echo [2/3] Dang dong goi thanh file .exe (co the mat 1-2 phut)...
python -m PyInstaller --clean --noconfirm vtuber.spec

if %errorlevel% neq 0 (
    echo.
    echo [LOI] Qua trinh dong goi gap loi!
    pause
    exit /b 1
)

:: 3. Sao chep file config.txt vao thu muc dist de nguoi dung co the tuy chinh
echo.
echo [3/3] Dang sao chep config.txt vao thu muc dist...
if exist config.txt (
    copy /y config.txt dist\config.txt >nul
)

echo.
echo ============================================================
echo   DONG GOI THANH CONG!
echo   File chay nam tai: dist\VTuberAvatar.exe
echo   File cau hinh nam tai: dist\config.txt
echo ============================================================
echo.
pause
