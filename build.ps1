$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolchain = 'C:\msys64\ucrt64\bin'
$compiler = Join-Path $toolchain 'g++.exe'
$windres = Join-Path $toolchain 'windres.exe'
$resourceObject = Join-Path $workspace 'resources.o'
$outputDir = Join-Path $workspace 'dist'
$output = Join-Path $outputDir 'NARC-Next.exe'

if (-not (Test-Path -LiteralPath $compiler)) { throw "Compiler not found: $compiler" }
if (-not (Test-Path -LiteralPath $windres)) { throw "Resource compiler not found: $windres" }

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$env:PATH = "$toolchain;$env:PATH"
& python (Join-Path $workspace 'tests\embed_net.py') `
    (Join-Path $workspace 'networks\narc-next.nnue') `
    (Join-Path $workspace 'src\nnue_net.h')
if ($LASTEXITCODE -ne 0) { throw 'NNUE embedding failed.' }
& python (Join-Path $workspace 'tests\embed_policy.py') `
    (Join-Path $workspace 'networks\narc-next-context-policy.bin') `
    (Join-Path $workspace 'src\policy_net.h')
if ($LASTEXITCODE -ne 0) { throw 'Policy embedding failed.' }
& $windres (Join-Path $workspace 'src\resources.rc') -O coff -o $resourceObject
if ($LASTEXITCODE -ne 0) { throw 'Windows resource compilation failed.' }

try {
    & $compiler -O3 -march=native -flto -static -std=c++20 -pthread `
        -Wall -Wextra -Wshadow -DNDEBUG `
        -o $output (Join-Path $workspace 'src\main.cpp') $resourceObject
    if ($LASTEXITCODE -ne 0) { throw 'C++ compilation failed.' }
}
finally {
    Remove-Item -LiteralPath $resourceObject -Force -ErrorAction SilentlyContinue
}

Write-Host "Build complete: $output"
