param([string]$Fen,[int]$Depth)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = 'D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe'
$psi.WorkingDirectory = 'D:\stockfish-windows-x86-64-avx2\stockfish'
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null
$stdin = $p.StandardInput
$stdout = $p.StandardOutput
$stdin.WriteLine('uci'); $stdin.Flush(); Start-Sleep -Milliseconds 100
$stdin.WriteLine('isready'); $stdin.Flush(); Start-Sleep -Milliseconds 100
$stdin.WriteLine('setoption name Threads value 1')
$stdin.WriteLine('setoption name Hash value 32')
$stdin.WriteLine('ucinewgame')
$stdin.WriteLine('position fen ' + $Fen)
$stdin.WriteLine('go depth ' + $Depth)
$stdin.Flush(); Start-Sleep -Milliseconds 400
$stdin.WriteLine('quit'); $stdin.Flush()
$lines = @()
while (($line = $stdout.ReadLine()) -ne $null) {
  $lines += $line
  if ($line -like 'bestmove*') { break }
}
$p.WaitForExit(2000) | Out-Null
$lines | Select-Object -Last 20 | ConvertTo-Json -Compress
