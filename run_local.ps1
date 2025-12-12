# ============================================
# run_local.ps1 — 本地一键启动所有 Serverless 实例
# 实例来源：instances.json（你提供的那份）
# 启动：
#   - pre_fn: 8001, 8002
#   - post_fn: 8101, 8102
#   - expert0: 8201, 8202
#   - expert1: 8211, 8212
#   - controller: 本地训练控制器
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
# 全局环境变量（所有新窗口都会继承）
# --------------------------------------------
$env:COMM_SIM_DIR        = "comm_sim"
$env:INSTANCES_FILE      = "instances.json"
$env:FUNC_MAP_FILE       = "func_map.json"
$env:CUDA_VISIBLE_DEVICES = ""      # 强制屏蔽 GPU，保证只用 CPU

# 若通信目录不存在，则自动创建
if (-not (Test-Path "$ROOT\comm_sim"))      { New-Item "$ROOT\comm_sim" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\hot"))  { New-Item "$ROOT\comm_sim\hot" -ItemType Directory | Out-Null }
if (-not (Test-Path "$ROOT\comm_sim\cold")) { New-Item "$ROOT\comm_sim\cold" -ItemType Directory | Out-Null }

# --------------------------------------------
# 定义函数：启动一个新的 PowerShell 窗口（移除 title 逻辑）
# --------------------------------------------
function Start-Window {
    param (
        [string]$command  # 移除 title 参数
    )

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd `"$ROOT`";
         . .\.venv\Scripts\Activate.ps1;
         $command"
    )
}

# ============================================
# 1. 启动 pre_fn 的两个实例（8001 / 8002）
# ============================================

# fn_pre_py_torch_gpu_1  -> 端口 8001
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='2';
uvicorn pre_fn:app --host 127.0.0.1 --port 8001

"@

# fn_pre_py_torch_cpu_1  -> 端口 8002
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='2';
uvicorn pre_fn:app --host 127.0.0.1 --port 8002
"@

# [新增] fn_pre_instance_3
Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='2';
uvicorn pre_fn:app --host 127.0.0.1 --port 8003
"@

# ============================================
# 2. 启动 post_fn 的两个实例（8101 / 8102）
# ============================================

# fn_post_py_torch_gpu_1 -> 端口 8101
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8101
"@

# fn_post_py_torch_cpu_1 -> 端口 8102
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8102
"@

# [新增] fn_post_instance_3
Start-Window -command @"
`$env:VOCAB_SIZE='2000'; `$env:EMB_DIM='256';
uvicorn post_fn:app --host 127.0.0.1 --port 8103
"@

# ============================================
# 3. 启动 expert0 的两个实例（8201 / 8202）
#    来自：
#      fn_exp0_py_torch_gpu_1  -> 8201
#      fn_exp0_py_torch_cpu_1  -> 8202
# ============================================

# expert0 - 实例 1（py torch gpu 名字，实际跑在 CPU）
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8201
"@

# expert0 - 实例 2（py torch cpu 名字）
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8202
"@

# [新增] expert0 - 实例 3
Start-Window -command @"
`$env:LOGICAL_EID='0'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8203
"@
# ============================================
# 4. 启动 expert1 的两个实例（8211 / 8212）
#    来自：
#      fn_exp1_py_torch_gpu_1     -> 8211
#      fn_exp1_cpp_onnx_cpu_1     -> 8212
# ============================================

# expert1 - 实例 1（py torch gpu 名字）
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8211
"@

# expert1 - 实例 2（cpp onnx 名字，这里仍用 expert_app 模拟）
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8212
"@

# [新增] expert1 - 实例 3
Start-Window -command @"
`$env:LOGICAL_EID='1'; `$env:EMB_DIM='256';
uvicorn expert_app:app --host 127.0.0.1 --port 8213
"@

# ============================================
# 5. 启动 controller（训练控制器）
# ============================================

Start-Window -command @"
`$env:TOP_K='2'; `$env:NUM_EXPERTS='2';  # 这里配置为 2 个逻辑专家：0 和 1
`$env:BATCH_SIZE='4'; `$env:BLOCK_SIZE='64';
`$env:MAX_STEPS='4200'; `$env:VAL_INTERVAL='50'; `$env:LOG_TRAIN_EVERY='100';
`$env:MICRO_BATCHES='2';  # 将 Batch Size 拆分为 2 个微批次
python controller.py
"@

Write-Host ">>> All services started!"
Write-Host ">>> 已自动启动：pre(8001/8002) / post(8101/8102) / expert0(8201/8202) / expert1(8211/8212) / controller"