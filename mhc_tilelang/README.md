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
verify_mhc_copy.py   两个搬运 kernel + 验证脚本(单文件,可直接运行)
README.md            本文件
```

## 这是在验证什么

GPU 的 forward kernel 把 `input_mix`（形状 `(num_tokens, mhc_mult)`）按
`token_block_size` 个 token 一块地处理,每块需要一个
`[token_block_size, mhc_mult]` 的 float32 tile 驻留在 UB 里。GPU 测试里
`mhc_mult` 只取 `4`。

**问题**:在 Ascend 上,`[token_block_size, 4]` 这个 tile 的内层一行 =
`4 × 4 字节 = 16 字节`,低于 UB / MTE 的 **32 字节对齐下限**:

- 2D `[token_block_size, 4]` 的 UB buffer,vector 单元访问第 1 行及以后时,
  行 stride 16B < 32B,触发 `ADDR_MISALIGN` 硬件 trap;
- strided 的 `[token_block_size, 4]` DMA,行 stride 16B 不是 32B 的整数倍,
  描述符非法。

**解法**:`token_block_size` 个**连续**的 token 行,在内存里就是一段连续的
`token_block_size × mhc_mult` 个 float32。以 `(32, 4)` 为例 = 128 个
float32 = **512 字节,正好是 32B 的整数倍**。把 `input_mix` reshape 成
`[num_blocks, token_block_size × mhc_mult]`,每个核搬自己一整行 —— 一次
对齐的连续 burst,全程不出现任何 16 字节的 2D 行。

reshape 本身零成本:`input_mix` 是连续的,`(num_tokens, 4)` 和
`(num_blocks, 128)` 是同一段字节、同样的行优先顺序,reshape 只改 shape /
stride 元数据,不搬数据。

## 怎么运行

在 **NPU host** 上运行(需要 `tilelang-ascend` + `torch_npu`):

```bash
python3 mhc_tilelang/verify_mhc_copy.py            # 只跑推荐方案,应全 PASS
python3 mhc_tilelang/verify_mhc_copy.py --naive    # 额外跑 naive 2D 对照
```

没有 NPU 环境时,脚本会打印提示并直接退出,不报错。脚本没有相对 import,
从仓库根目录或 `mhc_tilelang/` 目录里跑都可以。

可选参数:`--mhc-mult`(默认 4)、`--token-block-size`(默认 32)、
`--seed`(默认 0)。

## 为什么这样跑

脚本里有两个 kernel:

| kernel | 写法 | 状态 |
| --- | --- | --- |
| `build_copy_flat` | 扁平连续行,每核一次对齐 burst | **推荐**,必须逐 bit PASS |
| `build_copy_naive_2d` | 2D tile,16 字节内层行 | 对照组,`--naive` 才跑 |

**默认只跑推荐方案** —— 它给的是干净、确定的结论:四个 shape 全 `[PASS]`,
就说明这个搬运方式在真机上逐 bit 正确,可以放心用进 MHC kernel。

**naive 对照是 opt-in(`--naive`)**,原因:16 字节行可能触发
`ADDR_MISALIGN` 硬件 trap,而 trap 会直接 abort 整个进程。如果默认就跑它、
它崩了,你连推荐方案的 PASS 都看不到。所以 naive 放在 `--naive` 后面、且
排在推荐方案**之后**跑 —— 即使它把进程搞崩,推荐方案的结论你也已经拿到了。

naive 的三种结局都有信息量:

- **进程 abort** → 印证 16 字节行确实过不了硬件对齐;
- **跑完但 FAIL** → 印证编译器没把它 lower 成连续 burst,数据搬错了;
- **意外 PASS** → 说明当前 tilelang 版本的 lowering 恰好认出了这块连续性。
  但仍不推荐 naive 写法 —— 它依赖 lowering 的具体行为,换版本 / 换 shape
  就可能崩。

## 验证细节

- **覆盖的 shape**:`num_tokens ∈ {1024, 2048, 4096, 8192}` —— 对应 GPU
  测试里 `n0 ∈ {1, 2}` × `n1 ∈ {1024, 4096}`。
- **两种输入**:
  - `ramp`:等差数列(`input[i, j] = i * mhc_mult + j`)。任何"错位一格 /
    用 lane 位置冒充索引值"之类的搬运 bug,都会在数值上直接暴露出来。
  - `randn`:随机 float32。
- **比对方式**:GM→UB→GM 往返一圈后,和输入逐 bit 比对(`torch.equal`)。
  纯拷贝不做任何运算,所以结果必须 **bit-exact**,有一个 bit 不一样就是
  `FAIL`。

预期输出:推荐方案下,每个 shape 打印 `[PASS] flat ramp ...` 和
`[PASS] flat randn ...` 两行,最后一行
`RESULT: flat-row GM->UB->GM copy is bit-exact ...`。任何 `[FAIL]` 或
`RESULT: ... MISMATCH` 都说明这个搬运在真机上有问题。

## 这个验证之外

本验证只覆盖 `input_mix` 这一个 tile 的搬运。写完整 MHC kernel 时还有两点
要单独处理:

- `mhc_base`(形状 `(mhc_mult,)` = 16 字节)和 `mhc_scale`(`(1,)` = 4 字节)
  是另外两个小搬运,同样卡 32B 对齐,需要 pad 到 `[1, 8]` 的 UB buffer
  或做 broadcast。
- tile 在 UB 里是扁平的 `[1, token_block_size × mhc_mult]`,所以
  `output = sigmoid(input * scale + base[j])` 的逐元素计算要按
  `buf[0, i1 * mhc_mult + j]` 索引,`base[j]` 需按长度 `mhc_mult` 的模式
  广播。
