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
`$env:TOP_K='2'; `$env:NUM_EXPERTS='4';  # 给 pre_fn 用
`$env:VOCAB_SIZE='2000';                # 建议同步 controller vocab，避免口径不一致
`$env:EMB_DIM='256';                    # 建议同步（若 controller 读 MOE_CONFIG.d_model 也行）

`$env:BATCH_SIZE='8'; `$env:BLOCK_SIZE='64';
`$env:MAX_STEPS='2200'; `$env:VAL_INTERVAL='100'; `$env:LOG_TRAIN_EVERY='10';

`$env:MICRO_BATCHES="4"
`$env:PARALLEL_DEGREE="4"

# 让冷热变化更明显（论文现象更清楚）
`$env:HOTSPOT_DRIFT_EVERY="20"
`$env:HOTSPOT_SPAN="3"
`$env:HOT_PROB="0.80"
`$env:WARM_PROB="0.10"

# 让 grad_apply 出现 hot/cold/http 混合分布
`$env:GRAD_HOT_PROB="0.85"
`$env:GRAD_COLD_PROB="0.85"

# 开启 autoscale（模拟平台弹性扩容，减少 queue）
`$env:AUTOSCALE_ENABLE="1"
`$env:AUTOSCALE_QUEUE_TH_MS="30"
`$env:AUTOSCALE_MAX_REPLICA="6"
`$env:AUTOSCALE_COOLDOWN_STEPS="8"

# deadline 改为自适应（p95 * 1.1），避免全 miss
`$env:DEADLINE_WARMUP_STEPS="30"
`$env:DEADLINE_PCTL="95"
`$env:DEADLINE_SAFETY="1.10"
`$env:DEADLINE_MIN_MS="200"

python controller.py
"@


Write-Host ">>> All services started!"
Write-Host ">>> Experts: 0, 1, 2, 3 (Total 4 logical experts)"