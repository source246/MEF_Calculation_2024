$from = '2024-01-01T00:00:00Z'
$to   = '2024-01-02T00:00:00Z'
$maxUrl = 'https://publicationtool.jao.eu/core/api/data/maxNetPos'
$npUrl  = 'https://publicationtool.jao.eu/core/api/data/netPosition'
$qs = "fromUtc=$from`&toUtc=$to`&format=CSV"

Write-Host '--- maxNetPos ---'
try {
    $r = Invoke-WebRequest -Uri "$maxUrl`?$qs" -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
    $txt = $r.Content -split "`n" | Select-Object -First 5
    $txt | ForEach-Object { Write-Host $_ }
} catch { Write-Host '[ERR maxNetPos]' $_.Exception.Message }

Write-Host '--- netPosition ---'
try {
    $r = Invoke-WebRequest -Uri "$npUrl`?$qs" -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
    $txt = $r.Content -split "`n" | Select-Object -First 5
    $txt | ForEach-Object { Write-Host $_ }
} catch { Write-Host '[ERR netPosition]' $_.Exception.Message }
