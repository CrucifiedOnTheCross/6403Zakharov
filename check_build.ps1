$ErrorActionPreference = "Stop"
$Venv = ".venv_simple"

# 1. Recreate environment
if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
python -m venv $Venv

# 2. Find WHL file
$Whl = Get-ChildItem "dist\*.whl" | Select-Object -First 1
if (-not $Whl) { throw "WHL file not found. Run: python -m build" }

# 3. Install package
& ".\$Venv\Scripts\pip" install $Whl.FullName | Out-Null

# 4. Get full path to Python
$PyPath = (Resolve-Path ".\$Venv\Scripts\python.exe").Path

# 5. Exit project folder and check import
Push-Location .. 
try {
    $Out = & $PyPath -c "import cat_api; print(cat_api.__file__)"
    
    if ($Out -match "site-packages") {
        Write-Output "OK: Package loaded from $Out"
    } else {
        Write-Output "FAIL: Package loaded from $Out"
    }
} finally {
    Pop-Location
}
