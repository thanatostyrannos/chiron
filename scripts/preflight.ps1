<#
  preflight.ps1 — environment check for a gfx1151 (Strix Halo) research lab
  on NATIVE WINDOWS.

  Check-only by default. Emits ENVIRONMENT.md and a REMEDIATION list.
  Run:  .\scripts\preflight.ps1
        .\scripts\preflight.ps1 -Verbose
#>
[CmdletBinding()]
param(
    [string]$OutFile = "ENVIRONMENT.md"
)

$ErrorActionPreference = "Continue"
$results = @()
$remediation = @()

function Check {
    param([string]$Name, [scriptblock]$Test, [string]$Expected, [string]$Fix)
    $val = try { & $Test } catch { "ERROR: $($_.Exception.Message)" }
    $ok = $val -and ($val -notmatch '^(ERROR|NOT FOUND|MISSING)')
    $script:results += [pscustomobject]@{ Check=$Name; Value="$val"; Expected=$Expected; OK=$ok }
    $color = if ($ok) { "Green" } else { "Red" }
    Write-Host ("  [{0}] {1}: {2}" -f $(if($ok){"OK"}else{"--"}), $Name, $val) -ForegroundColor $color
    if (-not $ok -and $Fix) { $script:remediation += "**$Name** — $Fix" }
}

Write-Host "`n=== PLATFORM ===" -ForegroundColor Cyan
Check "OS" { (Get-CimInstance Win32_OperatingSystem).Caption + " " + (Get-CimInstance Win32_OperatingSystem).Version } "Windows 11" $null
Check "CPU" { (Get-CimInstance Win32_Processor).Name } "Ryzen AI MAX+ 395" $null
Check "GPU" { (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1).Name } "Radeon 8060S" $null
Check "GPU driver" { (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "Radeon" } | Select-Object -First 1).DriverVersion } "Adrenalin 26.x" "Update AMD Adrenalin to the latest release."
Check "System RAM (GB)" { [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0) } "~128 minus UMA reservation" $null

Write-Host "`n=== TOOLCHAIN ===" -ForegroundColor Cyan
Check "git" { (git --version) 2>$null } "any" "winget install Git.Git"
Check "gh CLI" { (gh --version 2>$null | Select-Object -First 1) } "any" "winget install GitHub.cli"
Check "gh auth" { $s = gh auth status 2>&1 | Out-String; if ($s -match "Logged in") { "authenticated" } else { "NOT AUTHENTICATED" } } "authenticated" "Run: gh auth login   (needed to create the remote repo in the Scaffold phase)"
Check "Python" { (python --version 2>&1) } "3.11 or 3.12" "winget install Python.Python.3.12"
Check "uv" { (uv --version 2>$null) } "any" "pip install uv   (or: winget install astral-sh.uv)"

Write-Host "`n=== PATH SANITY (known gfx1151 breakage) ===" -ForegroundColor Cyan
Check "Repo path has no spaces" {
    if ($PWD.Path -match ' ') { "MISSING — path contains spaces: $($PWD.Path)" } else { "clean: $($PWD.Path)" }
} "no spaces" "Move the repo to a space-free path (e.g. c:\projects\school\<name>). Conda/pip packages break on paths with spaces on this stack."
Check "Python env path has no spaces" {
    $p = (python -c "import sys; print(sys.prefix)" 2>$null)
    if (-not $p) { "MISSING — python not found" } elseif ($p -match ' ') { "MISSING — env path has spaces: $p" } else { "clean: $p" }
} "no spaces" "Create the venv under a space-free path. If using conda: conda config --add envs_dirs c:\conda_envs"

Write-Host "`n=== ROCM / PYTORCH (gfx1151) ===" -ForegroundColor Cyan
$rocmFix = @'
Install AMD's gfx1151 nightlies into a clean venv:
    python -m venv c:\venvs\lab && c:\venvs\lab\Scripts\activate
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio
'@
Check "torch importable" { python -c "import torch; print(torch.__version__)" 2>$null } "2.9+ with rocm tag" $rocmFix
Check "torch sees GPU" { python -c "import torch; print('True' if torch.cuda.is_available() else 'MISSING - no device')" 2>$null } "True" $rocmFix
Check "device name" { python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'MISSING')" 2>$null } "AMD Radeon 8060S" $rocmFix
Check "arch is gfx1151" {
    python -c "import torch; p=torch.cuda.get_device_properties(0); print(getattr(p,'gcnArchName','unknown'))" 2>$null
} "gfx1151" "If arch is wrong or unknown, you have a generic build. Reinstall from the gfx1151 index above."
Check "hipBLASLt configured" {
    $e = $env:HIPBLASLT_TENSILE_LIBPATH
    if ($e) { "set: $e" } else { "MISSING - unset (GEMM may run ~5x slow)" }
} "set" "Locate the hipblaslt library dir in your ROCm install and set HIPBLASLT_TENSILE_LIBPATH. Also set TORCH_BLAS_PREFER_HIPBLASLT=1. A misconfigured path drops GEMM from ~33 to ~6 TFLOPS."

Write-Host "`n=== CAPACITY PROBE (the decisive number) ===" -ForegroundColor Cyan
$probe = @'
import torch, sys
if not torch.cuda.is_available():
    print("MISSING - no GPU"); sys.exit(0)
lo, hi, best = 1, 100, 0
while lo <= hi:
    mid = (lo + hi) // 2
    try:
        x = torch.empty(int(mid * 1024**3 // 2), dtype=torch.float16, device="cuda")
        del x; torch.cuda.empty_cache(); best = mid; lo = mid + 1
    except Exception:
        torch.cuda.empty_cache(); hi = mid - 1
print(f"{best} GB")
'@
$probe | Out-File -Encoding utf8 "$env:TEMP\cap_probe.py"
Check "Max GPU allocation" { python "$env:TEMP\cap_probe.py" 2>$null } "close to BIOS UMA FB Size (target 96GB)" "If far below the BIOS UMA setting: raise UMA FB Size in BIOS, confirm the driver exposes it, and re-run. This number is the ceiling for every long-context experiment — record it."

Write-Host "`n=== OPTIONAL ===" -ForegroundColor Cyan
Check "huggingface-cli" { (huggingface-cli version 2>$null) } "any" "pip install huggingface_hub[cli] — needed only for gated model downloads. Then: huggingface-cli login"

# ---- report ----
$pass = ($results | Where-Object OK).Count
$total = $results.Count

$md = @()
$md += "# Environment Report"
$md += ""
$md += "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm')  |  **$pass / $total checks passed**"
$md += ""
$md += "Platform target: native Windows, AMD Strix Halo (gfx1151). Not WSL2 — the ROCm"
$md += "pool under WSL2 is clamped to the .wslconfig memory value and cannot reach"
$md += "dedicated VRAM (ROCm issue #6022)."
$md += ""
$md += "| Check | Value | Expected | OK |"
$md += "|---|---|---|:--:|"
foreach ($r in $results) {
    $v = ($r.Value -replace '\|','\|' -replace "`r?`n",' ')
    $md += "| $($r.Check) | ``$v`` | $($r.Expected) | $(if($r.OK){'YES'}else{'**NO**'}) |"
}
if ($remediation.Count) {
    $md += ""
    $md += "## Remediation required"
    $md += ""
    foreach ($f in $remediation) { $md += "- $f"; $md += "" }
} else {
    $md += ""
    $md += "## All checks passed"
    $md += ""
    $md += "Record the max-allocation number above in ``ASSUMPTIONS.md`` tagged ``[M]`` with today's date."
}
[System.IO.File]::WriteAllLines((Join-Path $PWD $OutFile), $md)

Write-Host ""
Write-Host ("RESULT: {0}/{1} passed. Wrote {2}" -f $pass, $total, $OutFile) -ForegroundColor $(if($pass -eq $total){"Green"}else{"Yellow"})
if ($remediation.Count) {
    Write-Host "`nREMEDIATION NEEDED:" -ForegroundColor Yellow
    $remediation | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
}
Write-Host ""
