$ProgressPreference = 'SilentlyContinue'
$base = "http://127.0.0.1:8000/api"

function Post($path, $body) {
    $json = $body | ConvertTo-Json -Depth 10
    try {
        $r = Invoke-WebRequest -Uri "$base$path" -Method Post -Body $json `
            -ContentType "application/json" -UseBasicParsing
        return @{ status = $r.StatusCode; data = ($r.Content | ConvertFrom-Json) }
    } catch {
        $resp = $_.Exception.Response
        $code = if ($resp) { [int]$resp.StatusCode } else { 0 }
        $content = if ($_.ErrorDetails.Message) { $_.ErrorDetails.Message | ConvertFrom-Json } else { $null }
        return @{ status = $code; data = $content }
    }
}

Write-Host "=== FAILURE PREDICTION (3 unseen inputs) ==="
$failCases = @(
    @{ machine_type = "L"; air_temperature_k = 300.5; process_temperature_k = 310.6; rotational_speed_rpm = 1500; torque_nm = 40.0; tool_wear_min = 10 },
    @{ machine_type = "M"; air_temperature_k = 320.0; process_temperature_k = 345.0; rotational_speed_rpm = 2400; torque_nm = 55.0; tool_wear_min = 200 },
    @{ machine_type = "H"; air_temperature_k = 295.0; process_temperature_k = 305.0; rotational_speed_rpm = 1200; torque_nm = 25.0; tool_wear_min = 5 }
)
foreach ($c in $failCases) {
    $r = Post "/predict/failure" $c
    Write-Host ("  HTTP {0}: p={1} risk={2} mode={3} features={4}" -f `
        $r.status, $r.data.failure_probability, $r.data.risk_category, `
        $r.data.predicted_failure_type, $r.data.features.Count)
}

Write-Host "=== ANOMALY DETECTION (3 unseen inputs) ==="
$anomCases = @(
    @{ machine_type = "L"; air_temperature_k = 300.5; process_temperature_k = 310.6; rotational_speed_rpm = 1500; torque_nm = 40.0; tool_wear_min = 10 },
    @{ machine_type = "M"; air_temperature_k = 250.0; process_temperature_k = 355.0; rotational_speed_rpm = 29000; torque_nm = 59.0; tool_wear_min = 255 },
    @{ machine_type = "H"; air_temperature_k = 310.0; process_temperature_k = 330.0; rotational_speed_rpm = 1800; torque_nm = 45.0; tool_wear_min = 150 }
)
foreach ($c in $anomCases) {
    $r = Post "/detect/anomaly" $c
    Write-Host ("  HTTP {0}: score={1} is_anomaly={2} severity={3}" -f `
        $r.status, $r.data.anomaly_score, $r.data.is_anomaly, $r.data.severity)
}

Write-Host "=== RUL (engine unit from C-MAPSS) ==="
$r = Post "/predict/rul" @{ engine_unit = "42"; cycle = 120; dataset_subset = "FD001"; model_family = "xgboost"; persist = $false }
Write-Host ("  HTTP {0}: rul={1} family={2} curve={3} version={4}" -f `
    $r.status, $r.data.predicted_rul, $r.data.model_family, $r.data.degradation_curve.Count, $r.data.model_version)

Write-Host "=== RUL (LSTM family) ==="
$r = Post "/predict/rul" @{ engine_unit = "7"; cycle = 100; dataset_subset = "FD001"; model_family = "lstm"; persist = $false }
Write-Host ("  HTTP {0}: rul={1} family={2} curve={3}" -f `
    $r.status, $r.data.predicted_rul, $r.data.model_family, $r.data.degradation_curve.Count)

Write-Host "=== RUL (custom unseen window) ==="
$window = [System.Collections.ArrayList]::new()
for ($i = 0; $i -lt 30; $i++) { [void]$window.Add(@((0.1*$i), -0.2, 0.5, 1.0, 0.0, 0.3, -0.5, 0.8, 0.2, -0.1, 0.4, 0.6, -0.3, 0.9, 0.1, 0.7, -0.4)) }
$r = Post "/predict/rul" @{ engine_unit = "unseen_999"; cycle = 250; window = $window.ToArray(); dataset_subset = "FD001"; persist = $false }
Write-Host ("  HTTP {0}: rul={1}" -f $r.status, $r.data.predicted_rul)

Write-Host "=== RUL (short window - padding path) ==="
$short = [System.Collections.ArrayList]::new()
for ($i = 0; $i -lt 5; $i++) { [void]$short.Add(@(0.1, -0.2, 0.5, 1.0, 0.0, 0.3, -0.5, 0.8, 0.2, -0.1, 0.4, 0.6, -0.3, 0.9, 0.1, 0.7, -0.4)) }
$r = Post "/predict/rul" @{ engine_unit = "unseen_998"; cycle = 99; window = $short.ToArray(); dataset_subset = "FD001"; persist = $false }
Write-Host ("  HTTP {0}: rul={1}" -f $r.status, $r.data.predicted_rul)

Write-Host "=== VALIDATION (invalid input -> 422) ==="
$r = Post "/predict/failure" @{ machine_type = "L"; air_temperature_k = 9999; process_temperature_k = 310.6; rotational_speed_rpm = 1500; torque_nm = 40.0; tool_wear_min = 10 }
Write-Host ("  out-of-range temp -> HTTP {0} error={1}" -f $r.status, $r.data.error)
$r = Post "/predict/failure" @{ machine_type = "L" }
Write-Host ("  missing fields -> HTTP {0} error={1}" -f $r.status, $r.data.error)

Write-Host "=== MODEL METRICS ==="
$m = Invoke-WebRequest -Uri "$base/models/metrics" -UseBasicParsing
$metrics = $m.Content | ConvertFrom-Json
Write-Host ("  HTTP {0}: {1} trained models" -f $m.StatusCode, $metrics.total)
foreach ($item in $metrics.items) {
    Write-Host ("    - {0} / {1} v{2} ({3} samples)" -f $item.task, $item.algorithm, $item.model_version, $item.training_samples)
}

Write-Host "=== OTHER ENDPOINTS ==="
foreach ($path in @("/dashboard/summary", "/analytics", "/machines", "/agents/catalog", "/knowledge/documents", "/vision/categories", "/settings", "/datasets", "/alerts", "/maintenance/recommendations", "/models/status", "/models/trained")) {
    try {
        $r = Invoke-WebRequest -Uri "$base$path" -UseBasicParsing
        Write-Host ("  GET {0} -> {1}" -f $path, $r.StatusCode)
    } catch {
        $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        Write-Host ("  GET {0} -> {1} ERROR" -f $path, $code)
    }
}
Write-Host "DONE"
