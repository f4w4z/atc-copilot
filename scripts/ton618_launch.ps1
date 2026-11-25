# Start a script detached from the SSH session. Processes started directly from an
# OpenSSH session on Windows are killed when it closes; ones created through WMI are not.
#   powershell -ExecutionPolicy Bypass -File scripts\ton618_launch.ps1 [-Script scripts\ton618_week1.ps1] [-ScriptArgs "..."]
# Check progress:  Get-Content logs\week1.log
# Stop it:         Stop-Process -Id (Get-Content logs\<Name>.pid)
param(
    [string]$Script = "scripts\ton618_week1.ps1",
    [string]$ScriptArgs = "",
    [string]$Name = "week1"   # pid goes to logs\<Name>.pid
)
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force (Join-Path $root logs) | Out-Null
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $root $Script)`" $ScriptArgs"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root }
if ($r.ReturnValue -ne 0) { Write-Error "launch failed: $($r.ReturnValue)"; exit 1 }
Set-Content (Join-Path $root "logs\$Name.pid") $r.ProcessId
"launched pid $($r.ProcessId): $cmd"
