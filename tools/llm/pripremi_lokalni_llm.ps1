# Jednokratno preuzimanje runtimea i modela. Ne pokreće transkripciju.
param(
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"
$projectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
# Instalacijski paketi i metapodaci ne pripadaju izvršivoj mapi aplikacije.
$downloadDir = Join-Path (Split-Path $projectDir) "development\downloads"
$runtimeDir = Join-Path $projectDir "tools\llama.cpp"
$modelDir = $PSScriptRoot
$modelPath = Join-Path $modelDir "Qwen3-4B-Q4_K_M.gguf"
$modelUrl = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true"

New-Item -ItemType Directory -Force -Path $downloadDir, $runtimeDir, $modelDir | Out-Null

# Nastavi prekinuto preuzimanje i ponovno koristi već prisutne datoteke.
function Download-File([string]$Url, [string]$Destination) {
    if ((Test-Path -LiteralPath $Destination) -and -not $ForceDownload) {
        Write-Host "Već postoji: $Destination"
        return
    }
    Write-Host "Preuzimanje: $Url"
    & curl.exe --fail --location --retry 5 --retry-delay 3 --continue-at - --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Preuzimanje nije uspjelo (curl exit code $LASTEXITCODE): $Url"
    }
}

Write-Host "Traženje najnovijeg službenog llama.cpp izdanja s CUDA 12.4 paketima..."
$releaseMetadataPath = Join-Path $downloadDir "github_releases.json"
& curl.exe --fail --location --retry 5 `
    --header "User-Agent: zavrsni-lokalni-llm-setup" `
    --output $releaseMetadataPath `
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"
if ($LASTEXITCODE -ne 0) {
    throw "Nije moguće dohvatiti popis službenih llama.cpp izdanja."
}
$releases = Get-Content -LiteralPath $releaseMetadataPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$selectedRelease = $null
$runtimeAsset = $null
$cudartAsset = $null
# Oba CUDA paketa moraju potjecati iz odabranog kompatibilnog izdanja.
foreach ($release in $releases) {
    $candidateRuntime = @($release.assets) | Where-Object {
        $_.name -match '^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$'
    } | Select-Object -First 1
    $candidateCudart = @($release.assets) | Where-Object {
        $_.name -eq 'cudart-llama-bin-win-cuda-12.4-x64.zip'
    } | Select-Object -First 1
    if ($null -ne $candidateRuntime -and $null -ne $candidateCudart) {
        $selectedRelease = $release
        $runtimeAsset = $candidateRuntime
        $cudartAsset = $candidateCudart
        break
    }
}
if ($null -eq $selectedRelease) {
    throw "U zadnjih 20 službenih izdanja nisu pronađena oba Windows CUDA 12.4 paketa."
}
Write-Host "Odabrano llama.cpp izdanje: $($selectedRelease.tag_name)"

$runtimeZip = Join-Path $downloadDir $runtimeAsset.name
$cudartZip = Join-Path $downloadDir $cudartAsset.name
Download-File $runtimeAsset.browser_download_url $runtimeZip
Download-File $cudartAsset.browser_download_url $cudartZip

Write-Host "Raspakiravanje llama.cpp runtimea..."
Expand-Archive -LiteralPath $runtimeZip -DestinationPath $runtimeDir -Force
Expand-Archive -LiteralPath $cudartZip -DestinationPath $runtimeDir -Force

$server = Get-ChildItem -LiteralPath $runtimeDir -Recurse -Filter "llama-server.exe" |
    Select-Object -First 1
if ($null -eq $server) {
    throw "llama-server.exe nije pronađen nakon raspakiravanja."
}

Download-File $modelUrl $modelPath

Write-Host "Provjera SHA-256 modela prema Hugging Face metapodacima..."
$modelMetadataPath = Join-Path $downloadDir "qwen3_4b_metadata.json"
& curl.exe --fail --location --retry 5 `
    --header "User-Agent: zavrsni-lokalni-llm-setup" `
    --output $modelMetadataPath `
    "https://huggingface.co/api/models/Qwen/Qwen3-4B-GGUF?blobs=true"
if ($LASTEXITCODE -ne 0) {
    throw "Nije moguće dohvatiti službene metapodatke Qwen modela."
}
$modelInfo = Get-Content -LiteralPath $modelMetadataPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$modelEntry = @($modelInfo.siblings) | Where-Object {
    $_.rfilename -eq "Qwen3-4B-Q4_K_M.gguf"
} | Select-Object -First 1
if ($null -eq $modelEntry -or [string]::IsNullOrWhiteSpace($modelEntry.lfs.sha256)) {
    throw "Službeni SHA-256 modela nije pronađen."
}
# Usporedi cijeli preuzeti model s objavljenim otiskom prije prijave uspjeha.
$actualHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedHash = ([string]$modelEntry.lfs.sha256).ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "SHA-256 modela nije ispravan. Očekivano $expectedHash, dobiveno $actualHash."
}

$installation = @{
    llama_cpp_release = $selectedRelease.tag_name
    llama_server = $server.FullName
    model = $modelPath
    model_sha256 = $actualHash
    prepared_on = (Get-Date).ToString("o")
}
$installation | ConvertTo-Json | Set-Content `
    -LiteralPath (Join-Path $downloadDir "instalacija.json") -Encoding UTF8

Write-Host "Lokalni LLM je pripremljen."
Write-Host "Server: $($server.FullName)"
Write-Host "Model:  $modelPath"
