# MHC head-compute-mix —— GM→UB 搬运验证

把 DeepSeek TileKernels 的 MHC `head_compute_mix` 算子从 GPU 移植到
Ascend NPU 时,`input_mix` 这个 tile 从 Global Memory（GM）搬到 Unified
Buffer（UB）这一步有 **32 字节对齐**的风险。本目录在写完整 kernel
**之前**,先把这一步搬运单独验证清楚:确认在真机上逐 bit 正确,再往下写。

GPU 原始算子(DeepSeek TileKernels 仓,<https://github.com/deepseek-ai/TileKernels>):

- kernel:`tile_kernels/mhc/head_compute_mix_kernel.py`
- test:`tests/mhc/test_head_compute_mix.py`

## 文件

```
verify_mhc_copy.py   搬运 kernel + 验证脚本(单文件,可直接运行)
README.md            本文件
```

## 这是在验证什么

GPU 的 forward kernel 把 `input_mix`（形状 `(num_tokens, mhc_mult)`）按
`token_block_size` 个 token 一块地处理,每块需要一个
`[token_block_size, mhc_mult]` 的 float32 tile 驻留在 UB 里。GPU 测试里
`mhc_mult` 只取 `4`。

**问题**:在 Ascend 上,`[token_block_size, 4]` 这个 tile 的内层一行 =
`4 × 4 字节 = 16 字节`,低于 UB / MTE 的 **32 字节对齐下限**。直接照搬
GPU 的 2D tile 形状,要么 vector 单元访问 row≥1 时 `ADDR_MISALIGN` trap,
要么 strided DMA 的行 stride(16B)非法。

**解法**:`token_block_size` 个**连续**的 token 行,在内存里就是一段连续
的 `token_block_size × mhc_mult` 个 float32。以 `(32, 4)` 为例 = 128 个
float32 = **512 字节,正好是 32B 的整数倍**。把 `input_mix` reshape 成
2D 的 `[M, N]`:

- `N = token_block_size × mhc_mult` —— 一个对齐的扁平行;
- `M = num_tokens // token_block_size` —— 扁平行的数量。

reshape 本身零成本(`input_mix` 连续,只改 shape / stride 元数据,不搬
数据)。之后每个核搬若干整行 —— 单次对齐 burst,全程不出现 16 字节的
2D 行。

## 怎么运行

在 **NPU host** 上运行(需要 `tilelang-ascend` + `torch_npu`):

```bash
python3 mhc_tilelang/verify_mhc_copy.py
```

没有 NPU 环境时,脚本会打印提示并直接退出,不报错。脚本没有相对 import,
从仓库根目录或 `mhc_tilelang/` 目录里跑都可以。

可选参数:`--mhc-mult`(默认 4)、`--token-block-size`(默认 32)、
`--seed`(默认 0)。

## kernel 怎么来的

`verify_mhc_copy.py` 里的 `copy_kernel` 是 tilelang-ascend 自带的、已验证
示例 `examples/reduce/example_reduce_min.py` 的**克隆**,只把其中的 reduce
换成一次 copy-back —— `@tilelang.jit`、`T.Kernel(..., is_npu=True)`、
`T.Scope("V")`、`T.copy`、`T.barrier_all()` 这些结构和那个示例完全一致。
这样做是为了最大化"能在真机上编译通过"的把握:照搬一个确定能跑的骨架,
只改数据流。

kernel 做的事:`[M, N]` 的输入,M 行按 `block_M` 切给各个核、再在 2 个
vector 子核间对半分,每个子核把自己那几行 DMA 进 UB(`T.copy` GM→UB)、
`T.barrier_all()` 等 DMA 落地、再 DMA 回 GM(`T.copy` UB→GM)。是一次纯
GM→UB→GM 往返。

## 验证细节

- **覆盖的 shape**:`num_tokens ∈ {1024, 2048, 4096, 8192}` —— 对应 GPU
  测试里 `n0 ∈ {1, 2}` × `n1 ∈ {1024, 4096}`。
- **两种输入**:
  - `ramp`:等差数列(`input[i, j] = i * mhc_mult + j`)。任何"错位一格 /
    用 lane 位置冒充索引值"之类的搬运 bug,都会在数值上直接暴露。
  - `randn`:随机 float32。
- **比对方式**:GM→UB→GM 往返一圈后,和输入逐 bit 比对(`torch.equal`)。
  纯拷贝不做任何运算,结果必须 **bit-exact**,有一个 bit 不一样就是 `FAIL`。

预期输出:每个 shape 打印 `[PASS] ramp ...` 和 `[PASS] randn ...` 两行,
最后一行 `RESULT: flat-row GM->UB->GM copy is bit-exact ...`。任何 `[FAIL]`
或 `RESULT: ... MISMATCH` 都说明这个搬运在真机上有问题。

## 这个验证之外

本验证只覆盖 `input_mix` 这一个 tile 的搬运。写完整 MHC kernel 时还有两点
要单独处理:

- `mhc_base`(形状 `(mhc_mult,)` = 16 字节)和 `mhc_scale`(`(1,)` = 4 字节)
  是另外两个小搬运,同样卡 32B 对齐,需要 pad 到 `[1, 8]` 的 UB buffer
  或做 broadcast。
- tile 在 UB 里是扁平的 `[..., token_block_size × mhc_mult]`,所以
  `output = sigmoid(input * scale + base[j])` 的逐元素计算要按
  `buf[..., i1 * mhc_mult + j]` 索引,`base[j]` 需按长度 `mhc_mult` 的
  模式广播。
