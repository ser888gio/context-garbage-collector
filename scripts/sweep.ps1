param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(ValueFromRemainingArguments)][string[]]$AgentArgs
)
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\results" | Out-Null
# Start-Process detaches the child, so it outlives this shell and the Claude session
$p = Start-Process python -ArgumentList (@("-u", "-m", "cgc.agent") + $AgentArgs) -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$root\results\$Name.log" -RedirectStandardError "$root\results\$Name.err"
"PID $($p.Id), logs: results\$Name.log, results\$Name.err"
