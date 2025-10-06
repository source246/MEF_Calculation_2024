param(
    [int]$Year = 2024,
    [string]$Hub = "DE",
    [string]$OutCsv = "input/fb/fb_core_DE_2024.csv",
    [string]$BaseMax = "https://publicationtool.jao.eu/core/api/data/maxNetPos",
    [string]$BaseNP  = "https://publicationtool.jao.eu/core/api/data/netPosition",
    [int]$ChunkDays = 2,
    # Optional: limit number of chunks processed (0 = no limit). Useful for quick tests.
    [int]$LimitChunks = 0,
    # Optional extra query string to append when requesting netPos (copy from the 'Try-it' tab, e.g. resolution=PT60M)
    [string]$NetPosExtra = "",
    [int]$TimeoutSec = 60,
    [int]$Retries = 3,
    [int]$SleepSec = 2
)

# Ensure output dir
$OutDir = Split-Path $OutCsv -Parent
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

# --- Endpoint / parameter variants ---
$endpoints = @($BaseMax, $BaseMax.Replace('maxNetPos','maxNetPositions'))
$npEndpoints = @(
    'https://publicationtool.jao.eu/core/api/data/netPos',
    'https://publicationtool.jao.eu/core/api/data/netPosition',
    'https://publicationtool.jao.eu/core/api/data/netPositions'
)

$paramVariants = @(
    @{ kFrom='fromUtc'; kTo='toUtc'; kFmt='format'; vFmt='CSV' },
    @{ kFrom='FromUtc'; kTo='ToUtc'; kFmt='Format'; vFmt='CSV' }
)

function Invoke-Jao([string]$url,[datetime]$from,[datetime]$to,[string]$extraQS=$null) {
    foreach ($pv in $paramVariants) {
        $qs = "$($pv.kFrom)=$($from.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss'Z'"))&$($pv.kTo)=$($to.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss'Z'"))&$($pv.kFmt)=$($pv.vFmt)"
        if ($extraQS -and $extraQS.Trim() -ne '') { $qs = "$qs&$extraQS" }
        $u  = "$url`?$qs"
        for ($i=1; $i -le $Retries; $i++) {
            try {
                Write-Host "[INFO] GET $u (attempt $i of variant $($pv.kFrom))"
                $r = Invoke-WebRequest -UseBasicParsing -Uri $u -Method GET -TimeoutSec $TimeoutSec -ErrorAction Stop
                if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300 -and $r.Content -match ',') { return $r.Content }
                if ($r.StatusCode -eq 429) {
                    # Respect Retry-After header if present, otherwise back off more aggressively
                    $ra = $null
                    try { $ra = $r.Headers['Retry-After'] } catch { $ra = $null }
                    if ($ra) { Write-Warning "429 received; sleeping Retry-After: $ra s"; Start-Sleep -Seconds ([int]$ra) } else { Write-Warning "429 received; sleeping longer"; Start-Sleep -Seconds ($SleepSec * $i * 5) }
                } else {
                    Write-Warning "Non-200 or empty result: $($r.StatusCode)"
                    Start-Sleep -Seconds ($SleepSec * $i)
                }
            } catch {
                # Check for HTTP response inside exception (e.g. 404, 429)
                if ($_.Exception.Response -ne $null) {
                    try { $sc = $_.Exception.Response.StatusCode.Value__ } catch { $sc = $null }
                    if ($sc -eq 429) {
                        $ra = $null
                        try { $ra = $_.Exception.Response.Headers['Retry-After'] } catch { $ra = $null }
                        if ($ra) { Write-Warning "Attempt $($i) failed with 429; sleeping Retry-After: $ra s"; Start-Sleep -Seconds ([int]$ra) } else { Write-Warning "Attempt $($i) failed with 429; sleeping longer"; Start-Sleep -Seconds ($SleepSec * $i * 5) }
                    } else {
                        Write-Warning "Attempt $($i) failed for $($u): HTTP $sc"
                        Start-Sleep -Seconds ($SleepSec * $i)
                    }
                } else {
                    Write-Warning "Attempt $($i) failed for $($u): $($_.Exception.Message)"
                    Start-Sleep -Seconds ($SleepSec * $i)
                }
            }
        }
    }
    return $null
}

$fmt = 'yyyy-MM-ddTHH:mm:ssZ'
$start = [datetime]::ParseExact("$Year-01-01T00:00:00Z", $fmt, $null, [System.Globalization.DateTimeStyles]::AssumeUniversal)
$end   = [datetime]::ParseExact(("$($Year+1)-01-01T00:00:00Z"), $fmt, $null, [System.Globalization.DateTimeStyles]::AssumeUniversal)

$maxPieces = New-Object System.Collections.Generic.List[System.String]
$npPieces  = New-Object System.Collections.Generic.List[System.String]

$chunks = @()
$t = $start
while ($t -lt $end) {
    $u = $t.AddDays($ChunkDays)
    if ($u -gt $end) { $u = $end }
    $chunks += ,@($t,$u)
    $t = $u
}

Write-Host "[INFO] Will process $($chunks.Count) chunks from $start to $end (chunkDays=$ChunkDays)"

# processed chunks counter (used when -LimitChunks is set)
$processedChunks = 0

foreach ($c in $chunks) {
    $from = $c[0]; $to = $c[1]
    Write-Host "[INFO] Chunk: $($from.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm')) -> $($to.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm'))"

    # Try max endpoints (prefer correct maxNetPos, fallback to alias)
    $m = $null
    foreach ($ep in $endpoints) {
        try { $m = Invoke-Jao $ep $from $to; if ($m) { break } } catch { }
    }
    if ($m) { $maxPieces.Add($m) } else { Write-Warning "MaxNP chunk empty or all endpoints failed: $from - $to" }

    # NetPosition endpoints: try several candidate slugs (netPos is the correct Core slug)
    $n = $null
    foreach ($ep in $npEndpoints) {
        try { $n = Invoke-Jao $ep $from $to $NetPosExtra; if ($n) { break } } catch { }
    }
    if ($n) { $npPieces.Add($n) } else { Write-Warning "NetPos chunk empty or failed: $from - $to" }

    # Small polite pause between chunks to reduce rate-limit pressure
    Start-Sleep -Seconds 1

    # -----------------------
    # Incremental processing for this chunk: parse $m (max) and $n (netpos) and append daily rows to CSV
    # -----------------------
    try {
        $maxChunkList = @()
        if ($m) {
            $s = $m.ToString().Trim()
            if ($s.StartsWith('{') -or $s.StartsWith('[')) {
                $j = $s | ConvertFrom-Json -ErrorAction Stop
                if ($j -and $j.PSObject.Properties.Name -contains 'data') { foreach ($it in $j.data) { $maxChunkList += $it } }
                elseif ($j -is [System.Array]) { foreach ($it in $j) { $maxChunkList += $it } }
                else { $maxChunkList += $j }
            } else {
                $csv = $s | ConvertFrom-Csv -ErrorAction Stop
                foreach ($row in $csv) { $maxChunkList += $row }
            }
        }

        $npChunkList = @()
        if ($n) {
            $s = $n.ToString().Trim()
            if ($s.StartsWith('{') -or $s.StartsWith('[')) {
                $j = $s | ConvertFrom-Json -ErrorAction Stop
                if ($j -and $j.PSObject.Properties.Name -contains 'data') { foreach ($it in $j.data) { $npChunkList += $it } }
                elseif ($j -is [System.Array]) { foreach ($it in $j) { $npChunkList += $it } }
                else { $npChunkList += $j }
            } else {
                $csv = $s | ConvertFrom-Csv -ErrorAction Stop
                foreach ($row in $csv) { $npChunkList += $row }
            }
        }

        # Expand wide maxNetPos if necessary (minDE/maxDE columns)
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
                # normalize long format names
                foreach ($rec in $maxChunkList) {
                    $ts = $null; try { $ts = [datetime]::Parse($rec.dateTimeUtc -as $rec.Date -as $rec.Timestamp -as [string]).ToUniversalTime() } catch { $ts = $null }
                    $hub = $null; try { $hub = $rec.hub -or $rec.Hub -or $rec.zone -or $rec.Zone } catch { $hub = $null }
                    $minv = $null; $maxv = $null
                    try { $minv = [double]($rec.minNP -as [double]) } catch { $minv = $null }
                    try { $maxv = [double]($rec.maxNP -as [double]) } catch { $maxv = $null }
                    # attempt other common names
                    if (-not $minv) { try { $minv = [double]($rec.minDE -as [double]) } catch { } }
                    if (-not $maxv) { try { $maxv = [double]($rec.maxDE -as [double]) } catch { } }
                    if ($ts -ne $null -and $hub) { $maxExpanded += [PSCustomObject]@{ timestamp_utc = $ts; hub = $hub; minNP = $minv; maxNP = $maxv } }
                }
            }
        }

    # Expand netPos wide to long if necessary
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
                # assume already long
                foreach ($rec in $npChunkList) {
                    try { $ts = [datetime]::Parse($rec.dateTimeUtc -as $rec.Date -as $rec.Timestamp -as [string]).ToUniversalTime() } catch { $ts = $null }
                    $hub = $null; try { $hub = $rec.hub -or $rec.Hub -or $rec.zone -or $rec.Zone } catch { $hub = $null }
                    $val = $null; try { $val = [double]($rec.NetPosition -as [double]) } catch { $val = $null }
                    if ($ts -ne $null -and $hub) { $npExpandedChunk += [PSCustomObject]@{ timestamp_utc = $ts; hub = $hub; NetPosition = $val } }
                }
            }
        }

        # Filter to selected hub and build daily rows for this chunk
        $maxHubFilteredChunk = $maxExpanded | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }
        $npHubFilteredChunk  = $npExpandedChunk  | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }

        $groupedChunk = $maxHubFilteredChunk | Group-Object @{ Expression = { $_.timestamp_utc.ToString('yyyy-MM-dd') } }
        $dailyChunk = @()
        foreach ($g in $groupedChunk) {
            $best = $g.Group | Sort-Object @{Expression = { ($_.maxNP - $_.minNP) }} | Select-Object -First 1
            $date = [datetime]::ParseExact($g.Name,'yyyy-MM-dd',$null).ToUniversalTime()
            $dailyChunk += [PSCustomObject]@{ timestamp_utc = $date.ToString('yyyy-MM-dd 00:00:00'); minNP = $best.minNP; maxNP = $best.maxNP }
        }

        # Attach NetPosition by date (if available)
        $npLookupChunk = @{}
        foreach ($r in $npHubFilteredChunk) { $npLookupChunk[$r.timestamp_utc.ToString('yyyy-MM-dd')] = $r.NetPosition }
        foreach ($d in $dailyChunk) {
            $k = ([datetime]::ParseExact($d.timestamp_utc,'yyyy-MM-dd 00:00:00',$null)).ToString('yyyy-MM-dd')
            $npVal = $null
            if ($npLookupChunk.ContainsKey($k)) { $npVal = $npLookupChunk[$k] }
            $d | Add-Member -NotePropertyName NetPosition -NotePropertyValue $npVal
        }

        # Compute slacks and fb_boundary
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

        # Append to CSV incrementally (create file with header if missing)
        if ($finalChunk.Count -gt 0) {
            if (-not (Test-Path $OutCsv)) {
                $finalChunk | Sort-Object timestamp_utc | Export-Csv -NoTypeInformation -Encoding UTF8 -Force -Path $OutCsv
                Write-Host "[INC] Created $OutCsv with $($finalChunk.Count) rows"
            } else {
                $finalChunk | Sort-Object timestamp_utc | Export-Csv -NoTypeInformation -Encoding UTF8 -Append -Path $OutCsv
                Write-Host "[INC] Appended $($finalChunk.Count) rows to $OutCsv"
            }
        } else {
            Write-Host "[INC] No rows to append for chunk $($from.ToString('yyyy-MM-dd')) -> $($to.ToString('yyyy-MM-dd'))"
        }
    } catch {
        Write-Warning "Incremental processing failed for chunk $($from) - $($to): $($_.Exception.Message)"
    }

    # increment processed counter and check limit
    $processedChunks = $processedChunks + 1
    if ($LimitChunks -gt 0 -and $processedChunks -ge $LimitChunks) { Write-Host "[INFO] Reached LimitChunks=$LimitChunks; stopping after $processedChunks chunks."; break }
}

if ($maxPieces.Count -eq 0 -and $npPieces.Count -eq 0) {
    Write-Error "No data downloaded from JAO endpoints. Aborting."
    exit 1
}

# Consolidate downloaded pieces and parse each piece robustly (JSON or CSV)
$maxText = ($maxPieces -join "`n")
$npText  = ($npPieces  -join "`n")

$maxObjList = @()
foreach ($piece in $maxPieces) {
    $s = $piece.ToString().Trim()
    if ([string]::IsNullOrWhiteSpace($s)) { continue }
    if ($s.StartsWith('{') -or $s.StartsWith('[')) {
        try {
            $j = $s | ConvertFrom-Json -ErrorAction Stop
            if ($j -and $j.PSObject.Properties.Name -contains 'data') {
                foreach ($it in $j.data) { $maxObjList += $it }
            } elseif ($j -is [System.Array]) {
                foreach ($it in $j) { $maxObjList += $it }
            } else {
                $maxObjList += $j
            }
        } catch {
            Write-Error "Failed to parse piece as JSON: $($_.Exception.Message)"
            exit 1
        }
    } else {
        try {
            $csv = $s | ConvertFrom-Csv -ErrorAction Stop
            foreach ($row in $csv) { $maxObjList += $row }
        } catch {
            Write-Error "Failed to parse piece as CSV: $($_.Exception.Message)"
            exit 1
        }
    }

        # Detect whether netPos returned hourly timestamps (preferred) or only daily
        if ($npExpandedChunk.Count -gt 0) {
            $hourlyCount = ($npExpandedChunk | Where-Object { $_.timestamp_utc.Hour -ne 0 }) | Measure-Object | Select-Object -ExpandProperty Count
            if ($hourlyCount -eq 0) {
                Write-Warning "NetPosition data appears to have only daily timestamps (no hourly values). If you need hourly NetPosition, pass the Try-it URL extra query string via -NetPosExtra (example: resolution=PT60M or mtu=60)."
            } else {
                Write-Host "[INFO] NetPosition appears hourly (sample rows: $($npExpandedChunk.Count))."
            }
        }
}
$maxObj = $maxObjList

$npObjList = @()
foreach ($piece in $npPieces) {
    $s = $piece.ToString().Trim()
    if ([string]::IsNullOrWhiteSpace($s)) { continue }
    if ($s.StartsWith('{') -or $s.StartsWith('[')) {
        try {
            $j = $s | ConvertFrom-Json -ErrorAction Stop
            if ($j -and $j.PSObject.Properties.Name -contains 'data') {
                foreach ($it in $j.data) { $npObjList += $it }
            } elseif ($j -is [System.Array]) {
                foreach ($it in $j) { $npObjList += $it }
            } else {
                $npObjList += $j
            }
        } catch {
            Write-Warning "Failed to parse NetPosition piece as JSON: $($_.Exception.Message); skipping piece."
        }
    } else {
        try {
            $csv = $s | ConvertFrom-Csv -ErrorAction Stop
            foreach ($row in $csv) { $npObjList += $row }
        } catch {
            Write-Warning "Failed to parse NetPosition piece as CSV: $($_.Exception.Message); skipping piece."
        }
    }
}
$npObj = $npObjList

# If netPos came back as wide JSON (hub_... or hub... columns), expand to long form: timestamp_utc, hub, NetPosition
function Expand-NetPosWide($rawList) {
    $expanded = @()
    if (-not $rawList) { return $expanded }
    # determine property names from first record
    $first = $rawList | Select-Object -First 1
    if (-not $first) { return $expanded }
    $props = $first.PSObject.Properties.Name
    $timeCols = $props | Where-Object { $_ -match '(?i)date|time|utc|timestamp' }
    $hubProps = $props | Where-Object { $_ -match '(?i)^hub[_]?' }
    if ($hubProps.Count -eq 0) { return $expanded }
    foreach ($rec in $rawList) {
        # find timestamp
        $ts = $null
        foreach ($tc in $timeCols) {
            try { $ts = [datetime]::Parse($rec.$tc).ToUniversalTime(); break } catch { $ts = $null }
        }
        if (-not $ts) { continue }
        foreach ($hp in $hubProps) {
            # hub property names like 'hub_DE' or 'hubDE' -> extract hub code
            $hubcode = $hp -replace '^(?i)hub[_]?',''
            try { $val = [double]($rec.$hp -as [double]) } catch { $val = $null }
            $expanded += [PSCustomObject]@{
                timestamp_utc = $ts
                hub = $hubcode
                NetPosition = $val
            }
        }
    }
    return $expanded
}

$npExpanded = Expand-NetPosWide $npObj
if ($npExpanded.Count -gt 0) { $npObj = $npExpanded }

# Helper: find column names
function Find-Col($obj, $pattern) {
    if (-not $obj) { return $null }
    $first = $obj | Select-Object -First 1
    foreach ($p in $first.PSObject.Properties.Name) {
        if ($p -match $pattern) { return $p }
    }
    return $null
}


# Detect columns. JAO 'maxNetPos' may return WIDE JSON (minDE/maxDE per country) or LONG CSV/JSON with hub column.
$timeColMax = Find-Col $maxObj 'Date|Time|UTC|timestamp'
$hubColMax  = Find-Col $maxObj 'Hub|Zone|Area|Bidding'
$minColMax  = Find-Col $maxObj 'Min.*Pos|MinNet|MinNetPos|min'
$maxColMax  = Find-Col $maxObj 'Max.*Pos|MaxNet|MaxNetPos|max'

# If hub column missing but there are many min*/max* columns (wide format), expand to long
if (-not $hubColMax) {
    # inspect property names of first record
    $firstProps = @()
    if ($maxObj -and $maxObj.Count -gt 0) { $firstProps = $maxObj[0].PSObject.Properties.Name }
    $minProps = $firstProps | Where-Object { $_ -match '^(?i)min' }
    $maxProps = $firstProps | Where-Object { $_ -match '^(?i)max' }
    if ($minProps.Count -gt 0 -and $maxProps.Count -gt 0) {
        $expanded = @()
        foreach ($rec in $maxObj) {
            # find timestamp
            $ts = $null
            foreach ($c in $firstProps) { if ($c -match '(?i)date|time|utc|timestamp') { try { $ts = [datetime]::Parse($rec.$c).ToUniversalTime() } catch { $ts = $null }; break } }
            if (-not $ts) { continue }
            foreach ($mp in $minProps) {
                $suffix = $mp.Substring(3) # after 'min'
                $matchingMax = $maxProps | Where-Object { $_.ToLower() -eq ("max" + $suffix).ToLower() }
                if (-not $matchingMax) { continue }
                $minVal = $null; $maxVal = $null
                try { $minVal = [double]($rec.$mp -as [double]) } catch { $minVal = $null }
                try { $maxVal = [double]($rec.$matchingMax[0] -as [double]) } catch { $maxVal = $null }
                $expanded += [PSCustomObject]@{
                    timestamp_utc = $ts
                    hub = $suffix
                    minNP = $minVal
                    maxNP = $maxVal
                }
            }
        }
        $maxObj = $expanded
        # re-detect columns in expanded structure
        $timeColMax = 'timestamp_utc'
        $hubColMax  = 'hub'
        $minColMax  = 'minNP'
        $maxColMax  = 'maxNP'
    }
}

if (-not ($timeColMax -and $minColMax -and $maxColMax)) {
    Write-Error "Could not detect required columns in maxNetPositions result. Available: $($maxObj | Select-Object -First 1 | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name -ErrorAction SilentlyContinue)"
    exit 1
}

Write-Host "[INFO] Detected columns in maxNetPositions: time=$timeColMax hub=$hubColMax min=$minColMax max=$maxColMax"

# Normalize max data to objects with timestamp_utc, hub, minNP, maxNP
$maxNormalized = $maxObj | ForEach-Object {
    try {
        $ts = [datetime]::Parse($_.$timeColMax).ToUniversalTime()
    } catch { $ts = $null }
    [PSCustomObject]@{
        timestamp_utc = $ts
        hub = $_.$hubColMax
        minNP = [double]($_.$minColMax -as [double])
        maxNP = [double]($_.$maxColMax -as [double])
    }
} | Where-Object { $_.timestamp_utc -ne $null }

# Normalize NetPos if present
$npNormalized = @()
if ($npObj -and $npObj.Count -gt 0) {
    $timeColNp = Find-Col $npObj 'Date|Time|UTC|timestamp'
    $hubColNp  = Find-Col $npObj 'Hub|Zone|Area|Bidding'
    $npCol     = Find-Col $npObj 'Net.*Pos|NetPosition|NetPos|Net'
    if ($timeColNp -and $hubColNp -and $npCol) {
        Write-Host "[INFO] Detected columns in netPosition: time=$timeColNp hub=$hubColNp np=$npCol"
        $npNormalized = $npObj | ForEach-Object {
            try { $ts = [datetime]::Parse($_.$timeColNp).ToUniversalTime() } catch { $ts = $null }
            [PSCustomObject]@{
                timestamp_utc = $ts
                hub = $_.$hubColNp
                NetPosition = [double]($_.$npCol -as [double])
            }
        } | Where-Object { $_.timestamp_utc -ne $null }
    } else {
        Write-Warning "NetPosition CSV present but could not detect columns; skipping NetPosition." 
    }
}

# Final check: does NetPosition look hourly across normalized data?
if ($npNormalized.Count -gt 0) {
    $hourlySample = ($npNormalized | Where-Object { $_.timestamp_utc.Hour -ne 0 }) | Measure-Object | Select-Object -ExpandProperty Count
    if ($hourlySample -eq 0) { Write-Warning "Final NetPosition dataset appears daily only. To fetch hourly NetPosition, run with -NetPosExtra '<params from Try-it tab>' (e.g. resolution=PT60M)." }
}

# Filter to selected hub (case-insensitive). JAO 'DE' usually represents DE-LU.
$maxHubFiltered = $maxNormalized | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }
$npHubFiltered  = $npNormalized  | Where-Object { ($_.hub -ne $null) -and ($_.hub.ToString().ToUpper() -eq $Hub.ToUpper()) }

if (-not $maxHubFiltered) { Write-Warning "No maxNetPositions records matched hub $Hub" }

# Convert to arrays for grouping
# Group by date and pick the row with minimal range (maxNP-minNP) per day
$grouped = $maxHubFiltered | Group-Object @{ Expression = { $_.timestamp_utc.ToString('yyyy-MM-dd') } }

$daily = @()
foreach ($g in $grouped) {
    $best = $g.Group | Sort-Object @{Expression = { ($_.maxNP - $_.minNP) }} | Select-Object -First 1
    $date = [datetime]::ParseExact($g.Name,'yyyy-MM-dd',$null).ToUniversalTime()
    $daily += [PSCustomObject]@{
        timestamp_utc = $date.ToString('yyyy-MM-dd 00:00:00')
        minNP = $best.minNP
        maxNP = $best.maxNP
    }
}

# Attach NetPosition if available: join by date/time equality (if hourly, match hour), prefer exact timestamp match
if ($npHubFiltered -and $npHubFiltered.Count -gt 0) {
    # Build lookup by timestamp string
    $npLookup = @{}
    foreach ($r in $npHubFiltered) {
        $k = $r.timestamp_utc.ToString('yyyy-MM-dd')
        # if multiple entries per day, keep last (or average?) – keep last
        $npLookup[$k] = $r.NetPosition
    }
    foreach ($d in $daily) {
        $k = ([datetime]::ParseExact($d.timestamp_utc,'yyyy-MM-dd 00:00:00',$null)).ToString('yyyy-MM-dd')
        $npVal = $null
        if ($npLookup.ContainsKey($k)) { $npVal = $npLookup[$k] }
        $d | Add-Member -NotePropertyName NetPosition -NotePropertyValue $npVal
    }
} else {
    foreach ($d in $daily) { $d | Add-Member -NotePropertyName NetPosition -NotePropertyValue $null }
}

# Compute slacks and fb_boundary using tolerance 100 MW (and 2% rule as fallback)
$tolerance = 100.0
$final = $daily | ForEach-Object {
    $np = $_.NetPosition
    if ($np -ne $null) {
        $slMin = [math]::Max(0.0, $np - $_.minNP)
        $slMax = [math]::Max(0.0, $_.maxNP - $np)
        $rng = [math]::Max(1.0, $_.maxNP - $_.minNP)
        $isBoundary = (($np - $_.minNP).Magnitude -le $tolerance) -or (($_.maxNP - $np).Magnitude -le $tolerance)
    } else {
        $slMin = $null; $slMax = $null; $isBoundary = $false
    }
    [PSCustomObject]@{
        timestamp_utc = $_.timestamp_utc
        minNP = $_.minNP
        maxNP = $_.maxNP
        NetPosition = $np
        slack_to_min = $slMin
        slack_to_max = $slMax
        fb_boundary = $isBoundary
    }
}

# Export CSV
$finalSorted = $final | Sort-Object timestamp_utc
$finalSorted | Export-Csv -NoTypeInformation -Encoding UTF8 -Force -Path $OutCsv
Write-Host "[OK] Wrote $OutCsv with $($finalSorted.Count) rows"

# Quick checks
Write-Host "Head:"; $finalSorted | Select-Object -First 5 | Format-Table -AutoSize
Write-Host "Tail:"; $finalSorted | Select-Object -Last 5 | Format-Table -AutoSize
$valid = ($finalSorted | Where-Object { $_.minNP -le $_.maxNP }).Count
Write-Host "minNP<=maxNP rows: $valid / $($finalSorted.Count)"
$bd = ($finalSorted | Where-Object { $_.fb_boundary -eq $true }).Count
Write-Host "fb_boundary TRUE count: $bd"
