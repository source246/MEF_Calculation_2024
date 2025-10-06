param(
    [string]$From = '2024-01-01T00:00:00Z',
    [string]$To   = '2024-01-03T00:00:00Z',
    [string]$Hub  = 'DE',
    [string]$Out  = 'input/fb/fb_core_DE_2024_inc_chunk.csv'
)

$qs = "fromUtc=$From`&toUtc=$To`&format=CSV"
$maxUrl = "https://publicationtool.jao.eu/core/api/data/maxNetPos`?$qs"
$npUrl  = "https://publicationtool.jao.eu/core/api/data/netPos`?$qs"
Write-Host "[TEST] Fetching maxNetPos: $maxUrl"
try { $maxContent = (Invoke-WebRequest -Uri $maxUrl -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop).Content } catch { Write-Error "Failed fetching maxNetPos: $($_.Exception.Message)"; exit 1 }
Write-Host "[TEST] Fetching netPos: $npUrl"
try { $npContent = (Invoke-WebRequest -Uri $npUrl -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop).Content } catch { Write-Warning "netPos fetch failed (may be absent): $($_.Exception.Message)"; $npContent = $null }

# parse max piece
$maxChunkList = @()
if ($maxContent) {
    $s = $maxContent.Trim()
    if ($s.StartsWith('{') -or $s.StartsWith('[')) {
        $j = $s | ConvertFrom-Json -ErrorAction Stop
        if ($j -and $j.PSObject.Properties.Name -contains 'data') { foreach ($it in $j.data) { $maxChunkList += $it } } elseif ($j -is [System.Array]) { foreach ($it in $j) { $maxChunkList += $it } } else { $maxChunkList += $j }
    } else {
        $csv = $s | ConvertFrom-Csv -ErrorAction Stop
        foreach ($row in $csv) { $maxChunkList += $row }
    }
}

# parse np piece
$npChunkList = @()
if ($npContent) {
    $s = $npContent.Trim()
    if ($s.StartsWith('{') -or $s.StartsWith('[')) {
        $j = $s | ConvertFrom-Json -ErrorAction Stop
        if ($j -and $j.PSObject.Properties.Name -contains 'data') { foreach ($it in $j.data) { $npChunkList += $it } } elseif ($j -is [System.Array]) { foreach ($it in $j) { $npChunkList += $it } } else { $npChunkList += $j }
    } else {
        $csv = $s | ConvertFrom-Csv -ErrorAction Stop
        foreach ($row in $csv) { $npChunkList += $row }
    }
}

# expand max wide
$maxExpanded = @()
if ($maxChunkList.Count -gt 0) {
    $firstProps = $maxChunkList[0].PSObject.Properties.Name
    $minProps = $firstProps | Where-Object { $_ -match '^(?i)min' }
    $maxProps = $firstProps | Where-Object { $_ -match '^(?i)max' }
    if ($minProps.Count -gt 0 -and $maxProps.Count -gt 0 -and -not ($firstProps -contains 'hub' -or $firstProps -contains 'Hub')) {
        foreach ($rec in $maxChunkList) {
            $ts = $null
            foreach ($c in $firstProps) { if ($c -match '(?i)date|time|utc|timestamp') { try { $ts = [datetime]::Parse($rec.$c).ToUniversalTime() } catch { $ts = $null }; break } }
            if (-not $ts) { continue }
            foreach ($mp in $minProps) {
                $suffix = $mp.Substring(3)
                $matchingMax = $maxProps | Where-Object { $_.ToLower() -eq ("max" + $suffix).ToLower() }
                if (-not $matchingMax) { continue }
                try { $minVal = [double]($rec.$mp -as [double]) } catch { $minVal = $null }
                try { $maxVal = [double]($rec.$matchingMax[0] -as [double]) } catch { $maxVal = $null }
                $maxExpanded += [PSCustomObject]@{ timestamp_utc = $ts; hub = $suffix; minNP = $minVal; maxNP = $maxVal }
            }
        }
    } else {
        foreach ($rec in $maxChunkList) {
            $ts = $null; try { $ts = [datetime]::Parse($rec.dateTimeUtc -as $rec.Date -as $rec.Timestamp -as [string]).ToUniversalTime() } catch { $ts = $null }
            $hub = $null; try { $hub = $rec.hub -or $rec.Hub -or $rec.zone -or $rec.Zone } catch { $hub = $null }
            $minv = $null; $maxv = $null
            try { $minv = [double]($rec.minNP -as [double]) } catch { $minv = $null }
            try { $maxv = [double]($rec.maxNP -as [double]) } catch { $maxv = $null }
            if (-not $minv) { try { $minv = [double]($rec.minDE -as [double]) } catch { } }
            if (-not $maxv) { try { $maxv = [double]($rec.maxDE -as [double]) } catch { } }
            if ($ts -ne $null -and $hub) { $maxExpanded += [PSCustomObject]@{ timestamp_utc = $ts; hub = $hub; minNP = $minv; maxNP = $maxv } }
        }
    }
}

# expand np wide
$npExpandedChunk = @()
if ($npChunkList.Count -gt 0) {
    $first = $npChunkList[0]
    $props = $first.PSObject.Properties.Name
    $timeCols = $props | Where-Object { $_ -match '(?i)date|time|utc|timestamp' }
    $hubProps = $props | Where-Object { $_ -match '(?i)^hub[_]?' }
    if ($hubProps.Count -gt 0) {
        foreach ($rec in $npChunkList) {
            $ts = $null
            foreach ($tc in $timeCols) { try { $ts = [datetime]::Parse($rec.$tc).ToUniversalTime(); break } catch { $ts = $null } }
            if (-not $ts) { continue }
            foreach ($hp in $hubProps) {
                $hubcode = $hp -replace '^(?i)hub[_]?',''
                try { $val = [double]($rec.$hp -as [double]) } catch { $val = $null }
                $npExpandedChunk += [PSCustomObject]@{ timestamp_utc = $ts; hub = $hubcode; NetPosition = $val }
            }
        }
    } else {
        foreach ($rec in $npChunkList) {
            try { $ts = [datetime]::Parse($rec.dateTimeUtc -as $rec.Date -as $rec.Timestamp -as [string]).ToUniversalTime() } catch { $ts = $null }
            $hub = $null; try { $hub = $rec.hub -or $rec.Hub -or $rec.zone -or $rec.Zone } catch { $hub = $null }
            $val = $null; try { $val = [double]($rec.NetPosition -as [double]) } catch { $val = $null }
            if ($ts -ne $null -and $hub) { $npExpandedChunk += [PSCustomObject]@{ timestamp_utc = $ts; hub = $hub; NetPosition = $val } }
        }
    }
}

# build daily rows
$maxHubFilteredChunk = $maxExpanded | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }
$npHubFilteredChunk  = $npExpandedChunk  | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }

$groupedChunk = $maxHubFilteredChunk | Group-Object @{ Expression = { $_.timestamp_utc.ToString('yyyy-MM-dd') } }
$dailyChunk = @()
foreach ($g in $groupedChunk) {
    $best = $g.Group | Sort-Object @{Expression = { ($_.maxNP - $_.minNP) }} | Select-Object -First 1
    $date = [datetime]::ParseExact($g.Name,'yyyy-MM-dd',$null).ToUniversalTime()
    $dailyChunk += [PSCustomObject]@{ timestamp_utc = $date.ToString('yyyy-MM-dd 00:00:00'); minNP = $best.minNP; maxNP = $best.maxNP }
}

$npLookupChunk = @{}
foreach ($r in $npHubFilteredChunk) { $npLookupChunk[$r.timestamp_utc.ToString('yyyy-MM-dd')] = $r.NetPosition }
foreach ($d in $dailyChunk) {
    $k = ([datetime]::ParseExact($d.timestamp_utc,'yyyy-MM-dd 00:00:00',$null)).ToString('yyyy-MM-dd')
    $npVal = $null
    if ($npLookupChunk.ContainsKey($k)) { $npVal = $npLookupChunk[$k] }
    $d | Add-Member -NotePropertyName NetPosition -NotePropertyValue $npVal
}

# compute final
$tolerance = 100.0
$finalChunk = $dailyChunk | ForEach-Object {
    $npv = $_.NetPosition
    if ($npv -ne $null) {
        $slMin = [math]::Max(0.0, $npv - $_.minNP)
        $slMax = [math]::Max(0.0, $_.maxNP - $npv)
        $isBoundary = (($npv - $_.minNP).Magnitude -le $tolerance) -or (($_.maxNP - $npv).Magnitude -le $tolerance)
    } else {
        $slMin = $null; $slMax = $null; $isBoundary = $false
    }
    [PSCustomObject]@{ timestamp_utc = $_.timestamp_utc; minNP = $_.minNP; maxNP = $_.maxNP; NetPosition = $npv; slack_to_min = $slMin; slack_to_max = $slMax; fb_boundary = $isBoundary }
}

if ($finalChunk.Count -gt 0) {
    if (-not (Test-Path $Out)) {
        $finalChunk | Sort-Object timestamp_utc | Export-Csv -NoTypeInformation -Encoding UTF8 -Force -Path $Out
        Write-Host "[TEST] Created $Out with $($finalChunk.Count) rows"
    } else {
        $finalChunk | Sort-Object timestamp_utc | Export-Csv -NoTypeInformation -Encoding UTF8 -Append -Path $Out
        Write-Host "[TEST] Appended $($finalChunk.Count) rows to $Out"
    }
} else { Write-Host "[TEST] No rows to write" }

Write-Host "[TEST] Done"
