# ============================================
# run_local.ps1 — 本地一键启动所有 Serverless 实例 (Option 2: 4 Experts)
# ============================================

Write-Host ">>> Activating virtual environment..."
$venv = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
} else {
    Write-Host "[ERROR] Virtual environment not found! Expected at $venv"
    exit
}

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host ">>> Project root = $ROOT"

# --------------------------------------------
# 全局环境变量
# --------------------------------------------
$env:COMM_SIM_DIR        = "comm_sim"
$env:INSTANCES_FILE      = "instances.json"
$env:FUNC_MAP_FILE       = "func_map.json"
$env:CUDA_VISIBLE_DEVICES = ""      # CPU only

if (-not (Test-Path "$ROOT\comm_sim"))      { New-Item "$ROOT\comm_sim" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\hot"))  { New-Item "$ROOT\comm_sim\hot" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\cold")) { New-Item "$ROOT\comm_sim\cold" -ItemType Directory | Out-Null }

function Start-Window {
    param ( [string]$command )
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd `"$ROOT`"; . .\.venv\Scripts\Activate.ps1; $command"
    )
}

# ============================================
# 1. 启动 pre_fn (8001-8003)
# ============================================
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='4';
uvicorn pre_fn:app --host 127.0.0.1 --port 8001
"@
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='4';
uvicorn pre_fn:app --host 127.0.0.1 --port 8002
"@
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='4';
uvicorn pre_fn:app --host 127.0.0.1 --port 8003
"@

# ============================================
# 2. 启动 post_fn (8101-8103)
# ============================================
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8101
"@
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8102
"@
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8103
"@

# ============================================
# 3. 启动 Expert 0 (8201-8203)
# ============================================
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8201
"@
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8202
"@
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8203
"@

# ============================================
# 4. 启动 Expert 1 (8211-8213)
# ============================================
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8211
"@
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8212
"@
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8213
"@

# ============================================
# 5. [新增] 启动 Expert 2 (8221-8222)
# ============================================
Start-Window -command @"
`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8221
"@
Start-Window -command @"
`$env:LOGICAL_EID='2'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8222
"@

# ============================================
# 6. [新增] 启动 Expert 3 (8231-8232)
# ============================================
Start-Window -command @"
`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8231
"@
Start-Window -command @"
`$env:LOGICAL_EID='3'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8232
"@

# ============================================
# 7. 启动 Controller
# ============================================
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='4';  # 4个专家选2个 -> 50% 激活率
`$env:BATCH_SIZE='8'; `$env:BLOCK_SIZE='64'; # Batch Size 调小点，因为进程多了
`$env:MAX_STEPS='500'; `$env:VAL_INTERVAL='100'; `$env:LOG_TRAIN_EVERY='10';
`$env:MICRO_BATCHES='4';
python controller.py
"@

Write-Host ">>> All services started!"
Write-Host ">>> Experts: 0, 1, 2, 3 (Total 4 logical experts)"