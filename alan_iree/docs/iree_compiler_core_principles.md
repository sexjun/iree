# IREE 编译器核心原理

## 目录

1. [指令执行顺序如何确定](#1-指令执行顺序如何确定)
2. [Stream 编译器深入解析](#2-stream-编译器深入解析)
3. [完整编译流程](#3-完整编译流程)
4. [SSA 值在 Stream 中的转换](#4-ssa-值在-stream-中的转换)

---

## 1. 指令执行顺序如何确定

IREE 中指令的执行顺序由**多层机制**共同决定，从编译期到运行时逐层衔接。

### 1.1 VM 字节码层：程序计数器 (PC) 顺序执行

**文件**: `runtime/src/iree/vm/bytecode/dispatch.c`

VM 字节码指令**严格按 PC 顺序**逐条执行，除非遇到显式分支/调用/返回：

- 每条指令执行后 `pc++`，跳到下一条
- 分支指令直接修改 `pc` 值
- 函数调用保存当前 `pc` 到栈帧，返回时恢复
- 支持 `vm.call.yieldable` 实现异步挂起/恢复

### 1.2 HAL 命令缓冲区层：录制顺序 + Barrier 构建 DAG

**文件**: `runtime/src/iree/hal/command_buffer.c`

命令缓冲区内的指令**默认按录制顺序**执行，但可以通过 `iree_hal_command_buffer_execution_barrier` 显式切分：

```
dispatch A → dispatch B → [barrier] → dispatch C → dispatch D
```

barrier 定义了 stage 之间的依赖关系（如 `DISPATCH → TRANSFER`），barrier 前的操作必须完成后才能执行 barrier 后的操作。

**执行阶段** (`iree_hal_execution_stage_t`):
- `COMMAND_ISSUE` — 流水线顶端
- `COMMAND_PROCESS` — Dispatch 参数消费
- `DISPATCH` — Dispatch 执行
- `TRANSFER` — 传输操作
- `COMMAND_RETIRE` — 命令退役
- `HOST` — 主机侧操作

### 1.3 队列提交层：信号量 Wait→Issue→Retire 三段 DAG

**文件**: `runtime/src/iree/hal/drivers/local_task/task_queue.c:19-50`

每次队列提交构成一个三阶段 DAG：

```
sema_wait → issue_cmd → retire_cmd
```

跨提交的顺序通过**时间线信号量 (timeline semaphore)** 串联：前一个提交的 `retire_cmd` 信号后一个提交的 `wait` 条件。

### 1.4 编译期 Stream 方言：SSA 依赖驱动拓扑排序

**文件**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleExecution.cpp`

编译期通过 **SSA 数据依赖**构建执行 DAG：

- 操作按 `mlir::sortTopologically()` 拓扑排序
- 每个分区 (`AsyncExecuteOp`) 有明确的 `ins`/`outs`
- 通过 `TimepointAwaitOp` 用信号量链执行区域，实现异步流水线

### 1.5 信号量/围栏层：跨阶段同步原语

**文件**: `runtime/src/iree/hal/semaphore.h`, `fence.h`

**时间线信号量**是核心同步原语（类似 Vulkan timeline semaphore）：
- 单调递增的 `uint64_t` payload 值
- Device→Device：拼接命令缓冲区提交为 DAG
- Host→Device：主机填充数据后发信号
- Device→Host：设备完成后通知主机取结果

**Fence** 是多信号量的 wait-all join。

### 总结：各层如何决定"谁先谁后"

| 层级 | 决定机制 |
|------|---------|
| VM 字节码 | PC 顺序递增，显式分支改变流向 |
| 命令缓冲区 | 录制顺序；barrier 创建 DAG 分叉/汇合点 |
| 队列提交 | 信号量等待→执行→退休三段 DAG |
| Stream 编译期 | SSA 拓扑排序 + TimepointAwait |
| 信号量/围栏 | 时间线 payload 单调递增，fence = 多信号量 join |

**核心原则**：默认按顺序（PC 递增/录制顺序），显式 barrier/信号量建立跨域依赖，SSA 数据流决定编译期调度。

---

## 2. Stream 编译器深入解析

### 2.1 定位：Stream 解决了什么问题？

Stream 方言位于 **Flow 方言** 和 **HAL 层**之间，是整个 IREE 编译器中负责**执行分区与调度**的核心桥梁：

```
Tensor Program → Flow → [Stream] → HAL → VM Bytecode
```

在进入 Stream 之前，编译器拥有的是 **隐式顺序执行的 Tensor 程序**。Stream 的任务是把这段程序转成一个**显式异步调度模型**：

| 问题 | 解决方案 |
|------|---------|
| 操作应该在哪个设备上执行？ | Affinity 分析 |
| 哪些操作可以并发？ | 分区 + ScheduleConcurrency |
| 不同内存空间之间如何传递数据？ | stream.async.transfer |
| 资源何时创建/何时销毁？ | 生命周期 + ARC |
| Tensor 如何编码为具体格式？ | MaterializeEncodings |
| 操作的执行顺序如何保证？ | Timepoint + DAG |

### 2.2 Stream 的三类核心操作

**1. Tensor 阶段 (`stream.tensor.*`)**：接收 Flow 方言的 dispatch，仍保留 Tensor 语义

**2. Async 阶段 (`stream.async.*`)**：核心抽象

| Op | 含义 |
|----|------|
| `stream.async.execute` | 在单个设备上原子执行的区域，有明确的输入/输出和 timepoint |
| `stream.async.concurrent` | 一个 execute 区域内的并发 wave |
| `stream.async.dispatch` | 将工作分发到可执行对象 |
| `stream.async.copy/transfer` | 资源间的传输 |

**3. Command 阶段 (`stream.cmd.*`)**：最接近硬件的抽象

| Op | 含义 |
|----|------|
| `stream.cmd.execute` | 命令缓冲区执行区域 |
| `stream.cmd.dispatch` | 绑定并分发 |
| `stream.cmd.copy/fill/discard` | 底层命令 |

### 2.3 资源生命周期

Stream 中的 `!stream.resource<lifetime>` 携带明确的生命周期信息：

| 生命周期 | 含义 |
|---------|------|
| `external` | 外部管理（API 边界传入） |
| `staging` | 用于上传/下载的暂存缓冲 |
| `transient` | 短生命周期，跨 stream 使用 |
| `variable` | 长生命周期，全局变量持有 |
| `constant` | 不可变，程序生命周期 |

由 `RefineUsage` pass（`ResourceUsageAnalysis`）根据分析结果赋值。

### 2.4 关键 Pass 流水线

Stream 的转换分三条子流水线依次执行：

**Tensor Pipeline**（Flow → stream.tensor）：

```
verify-input → cleanup → clone-to-consumers → conversion → inliner → cleanup
```

**Async Pipeline**（stream.tensor → stream.async）：

```
specialize-encodings → encode-tensors → materialize-encodings
→ materialize-copy-on-write → elide-async-copies → refine-usage（定生命周期）
→ ★ schedule-execution（形成 execute 区域）
→ ★ schedule-concurrency（形成并发 wave）
→ propagate-timepoints → materialize-builtins
```

**Cmd Pipeline**（stream.async → stream.cmd）：

```
schedule-allocation → emplace-transients → pack-constants
→ automatic-reference-counting → verify-lowering-to-cmd
```

**Optimization Pipeline**（后优化）：

```
reuse-allocations → elide-timepoints → fuse-dispatch-bindings
→ pack-dispatch-operands → fold-uniform-operands
```

### 2.5 分区算法详解

#### ScheduleExecution

**文件**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleExecution.cpp`
**数据结构**: `compiler/src/iree/compiler/Dialect/Stream/Analysis/Partitioning.h`

**Partition 结构**：

```cpp
struct Partition {
  AffinityAttr affinity;              // 执行亲和性
  SmallVector<Value> ins;             // 从外部消费的 SSA 值
  SmallVector<Value> outs;            // 被外部使用的产出 SSA 值
  SmallVector<Operation *> ops;       // 包含的操作集合
};
```

**算法流程**（逆向遍历 + 危险追踪）：

1. 逆向遍历 Block，对每个 streamable 操作：
2. 用 **ResourceHazardAnalysis** 追踪"哪些分区依赖这个操作"
3. 如果操作和已有分区 **affinity 兼容** 且**无危险冲突**，并入该分区
4. 如果操作标记了 `preferCloneToConsumers()`（如常量、splat），**克隆到多个消费分区**
5. 否则**创建新分区**
6. 每个分区最终变成一个 `stream.async.execute` 区域

#### ScheduleConcurrency

**文件**: `compiler/src/iree/compiler/Dialect/Stream/Transforms/ScheduleConcurrency.cpp`

在每个 `execute` 区域内部，再次分区形成"波"（wave）：

1. 无危险依赖的操作放入同一 wave
2. 同一 wave 内的操作包在 `stream.async.concurrent` 中，可以**真正并发**

### 2.6 Timepoint：执行顺序的运行时表达

`!stream.timepoint` 是 Stream 对"何时完成"的显式建模：

```mlir
%A = stream.async.execute ... await(%tp_in) => %tp_A
%B = stream.timepoint.await %tp_A   // B 等 A 完成
%C = stream.async.execute ... await(%tp_B) => %tp_C
```

编译期通过 `propagate-timepoints` pass 在 DAG 中传播 timepoint，最终运行时通过 **HAL 信号量** 实现等待。

### 2.7 关键文件索引

| 内容 | 路径 |
|------|------|
| Dialect 定义 | `compiler/.../Stream/IR/StreamOps.td` |
| 类型定义 | `compiler/.../Stream/IR/StreamTypes.td` |
| 分区数据结构 | `compiler/.../Stream/Analysis/Partitioning.h` |
| 分区算法 | `compiler/.../Stream/Analysis/Partitioning/ReferencePartitioning.cpp` |
| 执行调度 | `compiler/.../Stream/Transforms/ScheduleExecution.cpp` |
| 并发调度 | `compiler/.../Stream/Transforms/ScheduleConcurrency.cpp` |
| 资源使用分析 | `compiler/.../Stream/Analysis/ResourceUsage.h` |
| 流水线注册 | `compiler/.../Stream/Transforms/Passes.cpp` |

---

## 3. 完整编译流程

### 3.1 总览

```
前端模型 → IREE 编译器 → .vmfb 可执行文件
```

编译管线按 **12 个阶段** 依次执行：

```
Input → ABI → Preprocessing → GlobalOptimization → DispatchCreation
  → Flow → Stream → HAL → VM → 输出
```

### 3.2 各阶段详解

#### Input（输入转换）

**接收**：StableHLO / Torch MLIR / TOSA / 原生 MLIR
**产出**：统一的核心 IR（linalg + tensor + arith）

| 前端 | 转换流程 |
|------|---------|
| StableHLO | 反序列化 VHLO → 拉平 tuple → 转为 Linalg |
| Torch | torch-mlir 转换 → Shape 绑定 → 转为 Linalg |
| TOSA | TOSA→Linalg → 去除 signedness → 类型提升/降级 |

关键文件：`compiler/plugins/input/*/Conversion/Passes.cpp`

#### ABI（接口适配）

根据目标运行时生成绑定包装：
- **Native ABI**：生成 C/Python 可直接调用的函数签名
- **TFLite ABI**：TFLite 兼容接口

#### Preprocessing（用户自定义预处理）

三种机制，优先级从高到低：
1. `--iree-preprocessing-pass-pipeline` 命令行自定义
2. Transform Dialect 规范
3. PDL（模式匹配）规范

#### GlobalOptimization（全局优化）

| Pass | 作用 |
|------|------|
| `ExpandTensorShapesPass` | 动态 shape 计算转为 SSA |
| `PropagateLinalgTransposePass` | 推动 transpose 到输入端以便融合 |
| `GeneralizeLinalgNamedOpsPass` | 将具名 Linalg 转为通用形式 |
| 常量求值（JIT const-eval） | 编译期执行不可变计算 |
| `InferNumericNarrowingPass` | 精度压缩推断 |

#### DispatchCreation（计算区域形成）

**这是 Fusion 发生的地方**，决定哪些操作打包到一个 dispatch 中：

```
元素级操作 → 融合 → reshape 传播 → 多生产者融合
  → 标量 dispatch 形成 → ★ FormDispatchRegionsPass（核心）
  → 克隆生产者 → 维度折叠 → 数据分片编码
  → 转为 workgroups → 物化 workgroup 计数
```

关键 Pass：`FormDispatchRegionsPass` — 围绕融合根（root）构建 `flow.dispatch.region`

#### Flow（Flow 方言成型）

1. `OutlineDispatchRegionsPass` — 将 dispatch 区域**提取为独立函数**（`hal.executable`）
2. `DeduplicateExecutablesPass` — 去重相同 executable
3. `OutlineConstantsPass` — 常量转为全局变量
4. 固定点迭代：IPO + CSE + DCE

至此，程序变成了**一组 dispatch + 数据流图**。

#### Stream（异步调度）

三条子流水线依次执行：

```
Tensor Pipeline   →  Async Pipeline  →  Cmd Pipeline  →  Optimization
flow.* → stream   →  分区+调度+并发   →  命令缓冲区     →  后优化
```

#### HAL（硬件抽象层）

**这是代码生成发生的地方**：

1. `MaterializeInterfacesPass` — 创建 HAL interface
2. `ConfigureExecutablesPass` — 选择后端翻译策略
3. **`TranslateAllExecutablesPass`** — 调用后端代码生成器：

   | 后端 | 产物 |
   |------|------|
   | LLVMCPU | 机器码 (ELF) |
   | SPIRV | SPIR-V 二进制 |
   | CUDA | PTX |
   | ROCm | HSACO |
   | WGSL | WebGPU 着色器 |

4. `ConvertToHALPass` — Stream → HAL 转换
5. `SerializeAllExecutablesPass` — 序列化可执行文件

#### VM（虚拟机字节码）

1. **内联** — 所有函数内联（后续 pass 需要局部性）
2. **循环展开** — SCF → CFG
3. **方言转换** — std/util → VM 方言
4. **常量表物化** — `ReifyRodataTablesPass`
5. **全局初始化** — 生成 `__init` / `__deinit`
6. **可 yield 调用转换** — 支持异步挂起

### 3.3 输出打包

最终产物 `.vmfb` 文件是一个 **FlatBuffer + ZIP 多态归档**：

| 内容 | 格式 |
|------|------|
| VM 字节码 | `iree_vm_BytecodeModuleDef` FlatBuffer |
| HAL 可执行文件 | ELF / SPIR-V / PTX / HSACO（base64 嵌入） |
| 常量数据 | 内联 rodata 段或外部文件 |
| 调试信息 | 可选源码映射 |

### 3.4 完整流程图

```
┌─────────────────────────────────────────────────────┐
│  输入: .mlir / .tflite / StableHLO / Torch          │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Input     │  前端 → Linalg/Tensor/Arith            │
│  ABI       │  接口包装生成                           │
│  Preproc   │  用户自定义预处理                       │
│  GlobalOpt │  常量求值/Transpose传播/精度压缩        │
│  Dispatch  │  Fusion → 形成 dispatch regions         │
│  Flow      │  提取 executable / IPO / DCE            │
│  Stream    │  分区 → 异步调度 → 并发 → 命令缓冲区     │
│  HAL       │  Interface → 代码生成 → 序列化          │
│  VM        │  内联 → 转换 → 字节码编码               │
└────────────────────────┬────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  输出: .vmfb (ZIP + FlatBuffer)                     │
│  ├── VM 字节码                                      │
│  ├── HAL 可执行二进制 (ELF/SPIR-V/PTX/...)          │
│  └── 常量数据 / 调试信息                             │
└─────────────────────────────────────────────────────┘
```

### 3.5 命令行用法

```bash
iree-compile input.mlir -o output.vmfb                  # 完整流程
iree-compile --compile-from=stream input.mlir            # 从 Stream 阶段开始
iree-compile --compile-to=flow input.mlir                # 只跑到 Flow，输出中间 IR
iree-compile --compile-from=flow --compile-to=stream     # 仅运行 Flow→Stream
```

关键入口文件：`compiler/src/iree/compiler/Tools/iree_compile_lib.cc`

---

## 4. SSA 值在 Stream 中的转换

### 4.1 问题描述

假设原始 IR 中：

```mlir
%t = "flow.dispatch" () : () -> tensor<4x8xf32>           // Op A 产出 %t
%r = "flow.dispatch" (%t) : (tensor<4x8xf32>) -> tensor<4x8xf32>  // Op B 消费 %t
```

这里 `%t` 是 SSA 值：Op A 的定义点（definition），Op B 的使用点（use）。

### 4.2 转为 Stream 形式

**在同一 execute 区域内**（affinity 相同、无危险冲突）：

```mlir
%r, %tp_r = stream.async.execute with(%tp_in) -> (!stream.resource<*>, !stream.timepoint) {
  // Op A
  %resource_a = stream.resource.alloc ... : !stream.resource<*>
  %tensor_a = "flow.dispatch" () : () -> tensor<4x8xf32>
  %tp_a = stream.tensor.export %tensor_a -> %resource_a

  // Op B 直接使用同一区域内的结果
  %tensor_b = stream.tensor.import %resource_a : !stream.resource<*> -> tensor<4x8xf32>
  %result_tensor = "flow.dispatch" (%tensor_b) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %r = stream.tensor.export %result_tensor -> !stream.resource<*>

  stream.yield %r : !stream.resource<*>
}
```

**跨分区时**（affinity 不同或被 barrier 隔开）：

```mlir
// 分区 1: Op A
%r_a, %tp_a = stream.async.execute with(%tp_in) {
  %resource = stream.resource.alloc ...
  %t = "flow.dispatch" () : () -> tensor<4x8xf32>
  %r_a = stream.tensor.export %t -> %resource
  stream.yield %r_a : !stream.resource<*>
} => %tp_a

// 分区 2: Op B —— 等待 Op A 的 timepoint
%r_b, %tp_b = stream.async.execute await(%tp_a) {
  %tensor = stream.tensor.import %r_a : !stream.resource<*> -> tensor<4x8xf32>
  %r_b = "flow.dispatch" (%tensor) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  stream.yield %r_b : !stream.resource<*>
} => %tp_b
```

关键：`await(%tp_a)` 建立了执行顺序。

### 4.3 核心要点

1. **SSA 值 `%t` 变成 `!stream.resource`**：Tensor 不再直接传递，而是先 `export` 到 resource，再从 resource `import`。

2. **依赖通过 Timepoint 显式化**：SSA 的 use-def 链 → `await` 上游 timepoint。不需要 SSA 的 use-def 来隐式保证顺序了。

3. **同一分区内不需要 await**：如果 Op A 和 Op B 在同一个 `execute` 区域内，它们仍然按 IR 顺序执行，不需要显式等待。

4. **跨分区才需要 await**：只有被拆分成不同 `execute` 区域的分区之间，才需要 `await(%tp)` 建立 DAG 边。

### 4.4 总结

```
SSA use-def 链（编译期驱动分区构建）
        │
        ▼
  同一分区内 → IR 顺序执行，无需显式等待
  跨分区    → await(%tp) 建立 DAG 边
        │
        ▼
  Timepoint → HAL 信号量（运行时等待）
```

SSA 的 use-def 链在编译期驱动分区构建，分区之间用 Timepoint 显式表达依赖，分区内部仍然靠 IR 顺序保证执行顺序。
