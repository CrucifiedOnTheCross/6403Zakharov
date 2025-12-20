$ErrorActionPreference = "Stop"
$Venv = ".venv_simple"

# 1. Пересоздаем окружение
if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
python -m venv $Venv

# 2. Ищем WHL файл
$Whl = Get-ChildItem "dist\*.whl" | Select-Object -First 1
if (-not $Whl) { throw "WHL файл не найден. Сделайте: python -m build" }

# 3. Устанавливаем пакет
& ".\$Venv\Scripts\pip" install $Whl.FullName | Out-Null

# 4. Получаем полный путь к Python
$PyPath = (Resolve-Path ".\$Venv\Scripts\python.exe").Path

# 5. Выходим из папки проекта и проверяем импорт
Push-Location .. 
try {
    $Out = & $PyPath -c "import cat_api; print(cat_api.__file__)"
    
    if ($Out -match "site-packages") {
        Write-Output "OK: Пакет загружен из $Out"
    } else {
        Write-Output "FAIL: Пакет загружен из $Out"
    }
} finally {
    Pop-Location
}