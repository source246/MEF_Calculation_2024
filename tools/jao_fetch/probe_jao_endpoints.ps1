$urls = @(
    'https://publicationtool.jao.eu/core/api/data/maxNetPos?fromUtc=2024-01-01T00:00:00Z&toUtc=2024-01-03T00:00:00Z&format=CSV',
    'https://publicationtool.jao.eu/core/api/data/maxNetPositions?fromUtc=2024-01-01T00:00:00Z&toUtc=2024-01-03T00:00:00Z&format=CSV',
    'https://publicationtool.jao.eu/core/api/data/netPosition?fromUtc=2024-01-01T00:00:00Z&toUtc=2024-01-03T00:00:00Z&format=CSV'
)

foreach ($u in $urls) {
    Write-Host "--- URL: $u"
    try {
        $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
        Write-Host "[OK] Status: $($r.StatusCode) Len: $($r.Content.Length)"
        $sampleLen = [Math]::Min(300, $r.Content.Length)
        if ($sampleLen -gt 0) {
            $sample = $r.Content.Substring(0, $sampleLen)
            Write-Host "Sample (first $sampleLen chars):"
            Write-Host $sample
        }
    } catch {
        if ($_.Exception.Response -ne $null) {
            try { $sc = $_.Exception.Response.StatusCode.Value__ } catch { $sc = 'unknown' }
            Write-Host "[HTTP_ERR] StatusCode: $sc"
        } else {
            Write-Host "[ERR] $_.Exception.Message"
        }
    }
}
