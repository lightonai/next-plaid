param(
    [string]$ModelDir = "C:\Users\cepera\.cache\huggingface\hub\models--lightonai--LateOn-Code-edge\snapshots\07ef20f406c86badca122464808f4cac2f6e4b25",
    [string]$OrtDylibPath = "C:\Users\cepera\.cache\colgrep\onnxruntime\1.23.0\gpu\onnxruntime.dll",
    [int[]]$BatchSizes = @(1, 2, 4, 8, 16, 32, 64),
    [int[]]$DocumentLengths = @(1, 8, 64, 128, 256, 512, 1024, 2048),
    [string]$OutputRoot = "captures\batch-planner-grid",
    [switch]$Int8
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputBase = Join-Path $repoRoot $OutputRoot
$runDir = Join-Path $outputBase $timestamp
$logsDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$summaryPath = Join-Path $runDir "summary.csv"
$jsonlPath = Join-Path $runDir "summary.jsonl"

"batch_size,document_length,num_docs,exit_code,observed_incremental_peak_mib,static_reference_batch_incremental_mib,windows_peak_dedicated_mib,windows_peak_shared_mib,windows_peak_total_committed_mib,windows_incremental_shared_peak_mib,fits_dedicated_vram,log_path" | Set-Content $summaryPath
if (Test-Path $jsonlPath) { Remove-Item $jsonlPath -Force }

Write-Host "Building probe binary..."
cargo build --release -p next-plaid-onnx --bin batch-planner-probe --features cuda | Out-Host

$probeExe = Join-Path $repoRoot "target\release\batch-planner-probe.exe"
if (-not (Test-Path $probeExe)) {
    throw "Probe binary not found at $probeExe"
}

foreach ($batchSize in $BatchSizes) {
    foreach ($documentLength in $DocumentLengths) {
        $numDocs = $batchSize
        $logName = "b${batchSize}-l${documentLength}.log"
        $logPath = Join-Path $logsDir $logName
        $args = @(
            "--model", $ModelDir,
            "--cuda",
            "--batch-size", "$batchSize",
            "--document-length", "$documentLength",
            "--num-docs", "$numDocs",
            "--exact"
        )
        if ($Int8) {
            $args += "--int8"
        }

        Write-Host "Running batch=$batchSize len=$documentLength ..."

        $env:ORT_DYLIB_PATH = $OrtDylibPath
        $stdoutPath = "$logPath.stdout"
        $stderrPath = "$logPath.stderr"
        $process = Start-Process -FilePath $probeExe -ArgumentList $args -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $exitCode = $process.ExitCode
        $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath } else { @() }
        $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath } else { @() }
        $output = @($stdout + $stderr)
        $output | Set-Content $logPath
        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }

        $observed = ""
        $static = ""
        $windowsDedicated = ""
        $windowsShared = ""
        $windowsTotalCommitted = ""
        $windowsIncrementalShared = ""
        $fitsDedicated = ""
        $summaryLine = $output | Select-String "PROBE_SUMMARY" | Select-Object -Last 1
        if ($summaryLine) {
            if ($summaryLine -match "observed_incremental_peak_mib=(\d+)") {
                $observed = $Matches[1]
            }
            if ($summaryLine -match "static_reference_batch_incremental_mib=(\d+)") {
                $static = $Matches[1]
            }
            if ($summaryLine -match "windows_peak_dedicated_mib=(\d+)") {
                $windowsDedicated = $Matches[1]
            }
            if ($summaryLine -match "windows_peak_shared_mib=(\d+)") {
                $windowsShared = $Matches[1]
            }
            if ($summaryLine -match "windows_peak_total_committed_mib=(\d+)") {
                $windowsTotalCommitted = $Matches[1]
            }
            if ($summaryLine -match "windows_incremental_shared_peak_mib=(\d+)") {
                $windowsIncrementalShared = $Matches[1]
            }
            if ($summaryLine -match "fits_dedicated_vram=(true|false)") {
                $fitsDedicated = $Matches[1]
            }
        }

        "$batchSize,$documentLength,$numDocs,$exitCode,$observed,$static,$windowsDedicated,$windowsShared,$windowsTotalCommitted,$windowsIncrementalShared,$fitsDedicated,$logPath" | Add-Content $summaryPath

        $json = [ordered]@{
            batch_size = $batchSize
            document_length = $documentLength
            num_docs = $numDocs
            exit_code = $exitCode
            observed_incremental_peak_mib = if ($observed -ne "") { [int]$observed } else { $null }
            static_reference_batch_incremental_mib = if ($static -ne "") { [int]$static } else { $null }
            windows_peak_dedicated_mib = if ($windowsDedicated -ne "") { [int]$windowsDedicated } else { $null }
            windows_peak_shared_mib = if ($windowsShared -ne "") { [int]$windowsShared } else { $null }
            windows_peak_total_committed_mib = if ($windowsTotalCommitted -ne "") { [int]$windowsTotalCommitted } else { $null }
            windows_incremental_shared_peak_mib = if ($windowsIncrementalShared -ne "") { [int]$windowsIncrementalShared } else { $null }
            fits_dedicated_vram = if ($fitsDedicated -ne "") { [bool]::Parse($fitsDedicated) } else { $null }
            log_path = $logPath
        } | ConvertTo-Json -Compress
        Add-Content $jsonlPath $json
    }
}

Write-Host "Grid complete."
Write-Host "Summary CSV: $summaryPath"
Write-Host "Summary JSONL: $jsonlPath"
