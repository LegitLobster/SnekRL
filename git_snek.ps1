param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$gitdir = Join-Path $env:TEMP 'SnekRL.git'
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path $gitdir)) {
  Write-Error "Git dir not found: $gitdir"
  exit 1
}
& git --git-dir $gitdir --work-tree $work @Args
