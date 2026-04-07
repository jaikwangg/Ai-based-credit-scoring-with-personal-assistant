param(
    [string]$FrontendBase = "http://127.0.0.1:3000",
    [string]$BackendBase = "http://127.0.0.1:8000",
    [string]$PlannerBase = "http://127.0.0.1:8001",
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"

function Invoke-JsonGet {
    param([string]$Url)
    return Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec $TimeoutSec
}

function Invoke-JsonPost {
    param(
        [string]$Url,
        [hashtable]$Body
    )
    $json = $Body | ConvertTo-Json -Depth 10
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $json -TimeoutSec $TimeoutSec
    $sw.Stop()
    return @{
        elapsed_ms = $sw.ElapsedMilliseconds
        body = $result
    }
}

Write-Host "== Health checks =="
$plannerHealth = Invoke-JsonGet "$PlannerBase/health"
$backendHealth = Invoke-JsonGet "$BackendBase/health"
$frontendHealth = Invoke-JsonGet "$FrontendBase/api/health"
Write-Host "planner:  $($plannerHealth.status)"
Write-Host "backend:  $($backendHealth.status)"
Write-Host "frontend: $($frontendHealth.status)"

Write-Host "`n== Backend route checks =="
$openapi = Invoke-JsonGet "$BackendBase/openapi.json"
$paths = @($openapi.paths.PSObject.Properties.Name)
$required = @("/predict", "/rag/query")
$missing = $required | Where-Object { $_ -notin $paths }
if ($missing.Count -gt 0) {
    throw "Backend is missing required routes: $($missing -join ', '). Restart backend with latest code."
}
Write-Host "routes: ok ($($required -join ', '))"

$predictPayload = @{
    input_text = "production-smoke"
    extra_features = @{
        Sex = "Male"
        Occupation = "Salaried_Employee"
        Salary = "85000"
        Marital_status = "Single"
        credit_score = "690"
        credit_grade = "Good"
        outstanding = "120000"
        overdue = "0"
        loan_amount = "900000"
        Coapplicant = "No"
        Interest_rate = "6.2"
    }
}

Write-Host "`n== Predict flow checks =="
$backendPredict = Invoke-JsonPost -Url "$BackendBase/predict" -Body $predictPayload
$frontendPredict = Invoke-JsonPost -Url "$FrontendBase/api/predict" -Body $predictPayload

Write-Host ("backend /predict:      {0} ms | planner_error={1}" -f $backendPredict.elapsed_ms, $backendPredict.body.planner_error)
Write-Host ("frontend /api/predict: {0} ms | planner_error={1}" -f $frontendPredict.elapsed_ms, $frontendPredict.body.planner_error)

$ragPayload = @{
    question = "What documents are required for a home loan application?"
    top_k = 3
}

Write-Host "`n== Assistant RAG flow check =="
$frontendRag = Invoke-JsonPost -Url "$FrontendBase/api/rag/query" -Body $ragPayload
$hasAnswer = -not [string]::IsNullOrWhiteSpace([string]$frontendRag.body.answer)
Write-Host ("frontend /api/rag/query: {0} ms | has_answer={1}" -f $frontendRag.elapsed_ms, $hasAnswer)

Write-Host "`n== Summary =="
Write-Host "Production smoke test completed."
