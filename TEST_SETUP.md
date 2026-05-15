# SparseAttnSharedKV 测试环境搭建指南

> 本文档记录从零开始把 `sparse_attn_sharedkv` 算子的精度测试跑通的完整流程，按本仓库这个版本（torch 2.10.0+cpu / torch_npu 2.10.0rc2 / CANN 910_93）实测踩坑后整理。

## 0. 适用范围

| 项 | 要求 |
|---|---|
| 硬件 | Atlas A3 / Ascend 910_93（NPU 必需） |
| 操作系统 | openEuler 24.03 LTS-SP2（其它 Linux 也行，包管理器命令需自行调整） |
| Python | 3.11.x |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0rc2 |

> **强调**：本文档不是给 macOS / 无 NPU 的机器用的。`torch_npu`、CANN、自定义算子 `.run` 包都需要真实 NPU 硬件。

## 1. 路径约定

为方便阅读，本文档把以下路径写死，请按你实际机器替换：

| 占位符 | 含义 | 示例 |
|---|---|---|
| `$REPO_ROOT` | `ops-transformer` 仓库根 | `/sdb/yq/ops-transformer` |
| `$RECIPES_ROOT` | `cann-recipes-infer` 仓库根 | `/sdb/yq/cann-recipes-infer` |
| `$CANN_CUSTOM` | 自定义算子 `.run` 包的安装父目录 | `/sdb/yq/cann_custom` |
| `$PY_SITE` | Python site-packages | `/usr/local/python3.11.14/lib/python3.11/site-packages` |

## 2. 完整步骤

### 2.1 系统层依赖

```bash
# python 头文件 + 编译工具
sudo dnf install -y python3-devel gcc gcc-c++ make cmake ninja-build

# 如果走代理 / 跳 SSL（按需）
# sudo dnf install -y python3-devel gcc gcc-c++ make cmake ninja-build \
#   --setopt=proxy=http://your-proxy:port \
#   --setopt=sslverify=false
```

**验证**：

```bash
which python3-config gcc g++ cmake ninja
python3-config --includes
```

### 2.2 Python 层依赖

```bash
# 关键：setuptools 必须 < 81，否则 torch_npu 里 torchair 用的 pkg_resources 会缺
pip3 install "setuptools<81" --upgrade

# pytest 测试框架 + xlsx 写出
pip3 install pytest pandas openpyxl numpy ninja
```

走代理 / 跳 SSL 时附加：

```bash
pip3 install ... \
  --proxy http://your-proxy:port \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

**验证**：

```bash
python3 -c "import pkg_resources; print(pkg_resources.__file__)"   # 不报错且打印路径
python3 -c "import pandas, openpyxl, pytest; print('ok')"
```

### 2.3 torch + torch_npu

必须装 **CPU 版 torch**（不是 GPU 版），版本与 torch_npu 严格对齐。

```bash
# CPU 版 torch（参考 Ascend 文档对应版本，本环境是 2.10.0）
pip3 install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# torch_npu（从 Ascend gitcode 下载对应 wheel，再 pip 安装）
# pip3 install torch_npu-2.10.0rc2-cp311-cp311-linux_$(uname -m).whl
```

> **为什么是 "CPU 版"？** "CPU 版" 不是说计算在 CPU 跑，而是 PyTorch 框架本身不内置任何硬件加速后端。GPU 版 torch（如 +cu121）会内置一个 npu 占位符，跟 `torch_npu` 撞名导致 "Two accelerators" 报错。

**验证**：

```bash
python3 -c "import torch; print(torch.__version__)"
# 期望: 2.10.0+cpu

python3 -c "import torch; print('cuda:', torch.version.cuda)"
# 期望: cuda: None

python3 -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
# 期望: True
```

### 2.4 CANN 与自定义算子包

#### 2.4.1 CANN 基础环境

确认 CANN 包已装，环境变量已 source：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
npu-smi info   # 能看到 NPU 设备就 OK
```

#### 2.4.2 编译自定义算子包

在 `$REPO_ROOT` 下编译：

```bash
cd $REPO_ROOT
bash build.sh --pkg --experimental --soc=ascend910_93 \
  --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata
```

编译产物在 `$REPO_ROOT/build_out/cann-ops-transformer-*_linux-*.run`。

#### 2.4.3 安装自定义算子包

```bash
mkdir -p $CANN_CUSTOM
./build_out/cann-ops-transformer-*_linux-*.run --install-path=$CANN_CUSTOM
source $CANN_CUSTOM/vendors/custom_transformer/bin/set_env.bash
```

> **注意**：`--install-path` 跟的是 `vendors` 的父目录，`.run` 会自动在它下面创建 `vendors/custom_transformer/...`，**不要把 `vendors` 也写进路径**。

**验证**：

```bash
ls $CANN_CUSTOM/vendors/custom_transformer/op_api/lib/
# 应该能看到 .so 文件
echo $ASCEND_CUSTOM_OPP_PATH
# 应该包含 $CANN_CUSTOM 路径
```

### 2.5 torch_ops_extension

提供 `torch.ops.custom.npu_sparse_attn_sharedkv` 这个 Python 入口的关键包。它在 `cann-recipes-infer` 仓库里。

```bash
# source 好前面的环境（否则编译找不到头文件 / so）
source ${ASCEND_HOME_PATH}/set_env.sh
source $CANN_CUSTOM/vendors/custom_transformer/bin/set_env.bash

cd $RECIPES_ROOT/ops/ascendc/torch_ops_extension
bash build_and_install.sh
```

> 脚本逻辑：`python3 setup.py build_ext && bdist_wheel` → `pip3 install dist/*.whl -I`。无参数。
> 如果想限制并发避免占满 CPU：`MAX_JOBS=8 bash build_and_install.sh`。

**验证**：

```bash
python3 -c "import custom_ops; print(custom_ops.__file__)"
```

如果报 `libc10.so` 或 `libtorch_npu.so` 找不到，看下一节 **2.6**。

### 2.6 LD_LIBRARY_PATH 配置（必需）

`custom_ops_lib.so` 运行时需要找到 `libc10.so`（torch 自带）和 `libtorch_npu.so`（torch_npu 自带），但 wheel 没把 RPATH 写正确，得手动加：

```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import torch_npu,os;print(os.path.join(os.path.dirname(torch_npu.__file__),'lib'))"):$LD_LIBRARY_PATH
```

**验证**：

```bash
python3 -c "import custom_ops; print(custom_ops.__file__)"
python3 -c "import torch, custom_ops; print(torch.ops.custom.npu_sparse_attn_sharedkv)"
# 期望输出: <OpOverloadPacket(op='custom.npu_sparse_attn_sharedkv')>
```

## 3. 启动脚本（一键搞定所有 source）

每次开新 shell 都重复 source 太烦，写个一键脚本：

```bash
cat > $HOME/sas_env.sh <<'EOF'
#!/bin/bash
# === CANN ===
source ${ASCEND_HOME_PATH}/set_env.sh
# === 自定义算子 ===
source /sdb/yq/cann_custom/vendors/custom_transformer/bin/set_env.bash
# === torch / torch_npu 动态库 ===
export LD_LIBRARY_PATH=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import torch_npu,os;print(os.path.join(os.path.dirname(torch_npu.__file__),'lib'))"):$LD_LIBRARY_PATH
EOF

# 以后每次跑测试前
source $HOME/sas_env.sh
```

也可以写进 `~/.bashrc` 永久生效。

## 4. 运行测试

### 4.1 最小冒烟测试（推荐第一次）

只跑 1 个 decode 用例验证环境（几十秒出结果）：

```bash
cd $REPO_ROOT/experimental/attention/sparse_attn_sharedkv/tests/pytest

# 编辑 paramset.py 最后一行
vi sparse_attn_sharedkv_paramset.py
```

把最后那行 `ENABLED_PARAMS = ...` 改成：

```python
ENABLED_PARAMS = [TEST_PARAMS["swa_decode"]]
```

然后跑：

```bash
bash test_run.sh single
```

### 4.2 全部 6 个内置用例

恢复 paramset 默认行，或改为：

```python
ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
```

```bash
bash test_run.sh single
```

预计耗时 **30 ~ 60 分钟**（3 个 prefill 用例每个跑 CPU golden 要 5~15 分钟，3 个 decode 几秒）。

### 4.3 看精度结果

```bash
python3 -c "
import pandas as pd
df = pd.read_excel('./result/sas_result.xlsx')
print(df.to_string())
"
```

期望每条 case `Result` 列都是 `Pass`、`PctRlt` ≥ 99.5%。

### 4.4 自定义长序列用例

内置 6 个用例 KV 最长 8K。要测更长序列：

**方式 A：改 paramset**（最简单）

```python
"scfa_prefill":{
    "S1": [32768],
    "T1": [32768],
    "cu_seqlens_q": [[0, 32768]],
    "seqused_kv": [[32768]],
    "block_num1": [<新值>],  # ≥ ceil(S / block_size) * B
    "block_num2": [<新值>],  # ≥ ceil(S/cmp_ratio / block_size) * B
    ...
}
```

**方式 B：Excel → pt → 批量跑**

```bash
# 1. 写 excel 用例表
# 2. 生成 pt
bash test_run.sh save -E ./excel/my_cases.xlsx -S Sheet1 -P ./data
# 3. 批量跑
bash test_run.sh load -P ./data -R ./result/long_seq.xlsx
```

## 5. 常见问题 (Troubleshooting)

按报错关键字索引：

### 5.1 `Two accelerators cannot be used at the same time in PyTorch: npu and npu`

**根因 A**：setuptools 太新（≥81），`pkg_resources` 被移除，torch_npu 内部 torchair 加载失败，副作用表现成这条错误。

**解法**：

```bash
pip3 install "setuptools<81" --upgrade
```

**根因 B**：import 顺序问题——`import custom_ops` 在 `import torch_npu`（或 `import torch` 触发 autoload）之前。

**解法**：保证代码里 `import torch_npu` 或 `import torch` 在 `import custom_ops` 之前。本仓库测试代码顺序是对的。

**兜底**：设置环境变量绕过：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

### 5.2 `ImportError: libc10.so: cannot open shared object file`

`LD_LIBRARY_PATH` 没包含 torch 的 lib 目录。见 **2.6**。

### 5.3 `ImportError: libtorch_npu.so: cannot open shared object file`

`LD_LIBRARY_PATH` 没包含 torch_npu 的 lib 目录。见 **2.6**。

### 5.4 `ModuleNotFoundError: No module named 'pkg_resources'`

setuptools ≥81 不再自带。降版本：

```bash
pip3 install "setuptools<81" --upgrade
```

### 5.5 `ModuleNotFoundError: No module named 'openpyxl'`

pandas 写 xlsx 需要 openpyxl，pytest 写结果文件时报错。

```bash
pip3 install openpyxl
```

### 5.6 `cannot import name 'custom_ops_lib' from partially initialized module 'custom_ops'`

误导性报错。真实原因是当前 cwd 在 `torch_ops_extension/` 里，Python 优先 import 了源码目录（无 `.so`）而不是 site-packages 里装好的包。

**解法**：`cd` 到任何其它目录再跑 python。

### 5.7 `Operator SparseAttnSharedkv ... not found` / `aclnnXxx ... not registered`

自定义算子 `.run` 包没装，或者 `set_env.bash` 没 source。

```bash
source $CANN_CUSTOM/vendors/custom_transformer/bin/set_env.bash
```

### 5.8 `RuntimeError: ACL ... error` / `device not available`

NPU 被别人占了，或没初始化。

```bash
npu-smi info   # 看设备占用
```

### 5.9 pytest 报 FAILED 但精度对比 Result=Pass

通常是写结果 xlsx 失败 → 见 5.5（缺 openpyxl）或 `./result/` 目录不存在（`mkdir -p result`）。

### 5.10 编译 torch_ops_extension 时 `fatal error: Python.h: No such file`

```bash
sudo dnf install -y python3-devel
```

### 5.11 dnf 装包遇到代理 / SSL

```bash
sudo dnf install -y <pkg> \
  --setopt=proxy=http://your-proxy:port \
  --setopt=sslverify=false
```

或写进 `/etc/dnf/dnf.conf` 持久化：

```ini
[main]
proxy=http://your-proxy:port
sslverify=False
```

## 6. 完整必装清单

回顾从零起步、能跑通 `bash test_run.sh single` 所需的全部步骤：

| # | 操作 | 验证命令 |
|---|---|---|
| 1 | `sudo dnf install python3-devel gcc gcc-c++ make cmake ninja-build` | `which python3-config gcc cmake ninja` |
| 2 | `pip3 install "setuptools<81" pandas openpyxl pytest numpy ninja` | `python3 -c "import pkg_resources, openpyxl, pandas"` |
| 3 | 装 CPU 版 torch + torch_npu，版本对齐 | `python3 -c "import torch; print(torch.__version__)"` → `2.x+cpu` |
| 4 | `source ${ASCEND_HOME_PATH}/set_env.sh` + `npu-smi info` 验证 | `npu-smi info` 能看到 NPU |
| 5 | `bash build.sh --pkg --experimental --soc=ascend910_93 --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata` | `ls build_out/*.run` |
| 6 | `./build_out/*.run --install-path=$CANN_CUSTOM` + source set_env.bash | `ls $CANN_CUSTOM/vendors/custom_transformer/op_api/lib/` |
| 7 | `cd $RECIPES_ROOT/ops/ascendc/torch_ops_extension && bash build_and_install.sh` | `python3 -c "import custom_ops"` |
| 8 | `export LD_LIBRARY_PATH=<torch/lib>:<torch_npu/lib>:$LD_LIBRARY_PATH` | `python3 -c "import torch, custom_ops; print(torch.ops.custom.npu_sparse_attn_sharedkv)"` |
| 9 | `cd .../tests/pytest && bash test_run.sh single` | terminal 出现 `Result: Pass` |

## 7. 一些算子背景（备查）

- **算子类型**：FlashAttention 变体，支持 Sliding Window / Compressed / Sparse Compressed Attention 及它们的组合
- **数据类型**：输入 Q/KV 是 FP16/BF16，中间矩阵乘累加 FP32，softmax / rescale 在 FP32 上做，输出 FP16/BF16
- **支持平台**：Atlas A3（910_93）非量化版本。本仓库不包含 950PR/DT 伪量化分支
- **核间配比**：1 cube : 2 vector（`KERNEL_TYPE_MIX_AIC_1_2`）
- **基本块**：L0A/L0B (128,128)，L0C 累加器 FP32，L0A/L0B 各 2×32KB DoubleBuffer，L0C 2×64KB
- **AICPU metadata 表**：按 36 个 cube 核 + 72 个 vec 核排布（A3 上限），通过 enable bit 兼容小核数变体
- **算子签名**：详见 [README.md](./README.md) 中"参数说明"小节

## 8. 参考资料

- 算子接口文档：[README.md](./README.md)
- 仓库总文档：[../Attention融合算子Experimental使用说明.md](../Attention融合算子Experimental使用说明.md)
- DeepSeek-V4 算子设计：<https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md>
- torch_ops_extension 编译：<https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc>
