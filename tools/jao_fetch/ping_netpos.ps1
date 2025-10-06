$from='2024-01-01T00:00:00Z'
$to='2024-01-02T00:00:00Z'
$qs = "fromUtc=$from`&toUtc=$to`&format=CSV"

Write-Host '--- maxNetPos ---'
try {
    $c = (Invoke-WebRequest -Uri "https://publicationtool.jao.eu/core/api/data/maxNetPos`?$qs" -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop).Content
    $c -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host $_ }
} catch { Write-Host '[ERR maxNetPos]' $_.Exception.Message }

Write-Host '--- netPos ---'
try {
    $c = (Invoke-WebRequest -Uri "https://publicationtool.jao.eu/core/api/data/netPos`?$qs" -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop).Content
    $c -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host $_ }
} catch { Write-Host '[ERR netPos]' $_.Exception.Message }
