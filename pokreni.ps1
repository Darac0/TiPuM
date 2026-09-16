# Pogreške pripreme LLM-a obrađuju se prelaskom na sažetak temeljen na pravilima.
$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$model = Join-Path $PSScriptRoot "tools\llm\Qwen3-4B-Q4_K_M.gguf"
$server = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "tools\llama.cpp") `
    -Recurse -Filter "llama-server.exe" -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
$healthUrl = "http://127.0.0.1:8081/health"
$startedServer = $null

# Kratka provjera spremnosti lokalnog servisa bez učitavanja drugog modela.
function Test-LlmServer {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

# Provjeri instalaciju prije pokretanja procesa koji zauzima memoriju.
if (-not (Test-Path -LiteralPath $python)) {
    throw "Nedostaje .venv. Najprije pokrenite postavi_okruzenje.cmd."
}
$previousDisableLlm = $env:TIPUM_DISABLE_LLM
$applicationExitCode = 1

try {
    try {
    if ($env:TIPUM_DISABLE_LLM -in @('1', 'true', 'yes')) {
        Write-Host "Odabran je sazetak temeljen na pravilima."
    }
    elseif (-not (Test-LlmServer)) {
        if (-not (Test-Path -LiteralPath $model) -or $null -eq $server) {
            throw "Nedostaje lokalni model ili llama-server."
        }
        Write-Host "Pokretanje lokalnog LLM-a..."
        # CPU način ostavlja grafičku memoriju Whisperu i diarizaciji.
        $arguments = @(
            "--model", ('"' + $model + '"'),
            "--alias", "Qwen/Qwen3-4B-GGUF:Q4_K_M",
            "--host", "127.0.0.1",
            "--port", "8081",
            "--ctx-size", "8192",
            "--n-gpu-layers", "0",
            "--reasoning", "off",
            "--reasoning-format", "none",
            "--jinja"
        )
        $startedServer = Start-Process -FilePath $server.FullName `
            -ArgumentList $arguments -WorkingDirectory $server.DirectoryName `
            -WindowStyle Hidden -PassThru

        # Čekanje je ograničeno kako se aplikacija ne bi trajno zaglavila pri startu.
        $deadline = (Get-Date).AddSeconds(60)
        while (-not (Test-LlmServer)) {
            if ($startedServer.HasExited) {
                throw "Lokalni LLM nije se uspio pokrenuti."
            }
            if ((Get-Date) -ge $deadline) {
                throw "Lokalni LLM nije postao spreman unutar 60 sekundi."
            }
            Start-Sleep -Seconds 1
        }
        Write-Host "Lokalni LLM je spreman."
    }
    }
    catch {
        Write-Warning "LLM nije spreman: $($_.Exception.Message) Nastavljam sa sazetkom temeljenim na pravilima."
        $env:TIPUM_DISABLE_LLM = '1'
    }

    & $python -X utf8 (Join-Path $PSScriptRoot "main.py")
    $applicationExitCode = $LASTEXITCODE
}
finally {
    $env:TIPUM_DISABLE_LLM = $previousDisableLlm
    # Ugasi samo proces koji je pokrenula ova skripta, ne već postojeći servis.
    if ($null -ne $startedServer -and -not $startedServer.HasExited) {
        Stop-Process -Id $startedServer.Id -Force
    }
}

exit $applicationExitCode
