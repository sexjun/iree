# IREE 调试技巧指南

本文档汇总 IREE 模型编译和运行过程中常用的调试技巧。

## 环境路径

| 工具 | 路径 |
|------|------|
| `iree-compile` | `/Users/alanchen/workspace/iree/build/tools/iree-compile` |
| `iree-run-module` | `/Users/alanchen/workspace/iree/build/tools/iree-run-module` |

---

## 1. 编译阶段调试

### 1.1 查看各 Pass 耗时

```bash
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --time-passes \
    -o model.vmfb
```

输出每个编译 Pass 的耗时，定位性能瓶颈。

### 1.2 打印 Pass 前后的 IR

```bash
# 打印所有 Pass 前后的 IR（输出量大，建议重定向到文件）
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --mlir-print-ir-before-all \
    -o /dev/null 2>&1 | head -500

# 打印所有 Pass 后的 IR
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --mlir-print-ir-after-all \
    -o /dev/null 2>&1 | tail -500
```

### 1.3 只打印特定 Pass 的 IR

```bash
# 先查看有哪些 Pass
iree-compile model.mlir --help | grep pass

# 打印特定 Pass 前后的 IR
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --mlir-print-ir-before="iree-codegen-linalg-to-loops" \
    -o /dev/null 2>&1
```

### 1.4 导出各阶段 IR 到文件

```bash
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --dump-compilation-phases-to=dump/ \
    -o model.vmfb
```

生成 `dump/` 目录，包含各编译阶段的 IR 快照，方便对比差异。

### 1.5 启用详细日志

```bash
# 打印所有操作日志
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --log-actions-to=- \
    -o model.vmfb
```

### 1.6 禁用多线程便于复现问题

```bash
iree-compile model.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --ir-thread-pool-size=1 \
    -o model.vmfb
```

### 1.7 验证 IR 格式

```bash
# 检查 MLIR 文件是否合法
iree-opt model.mlir --mlir-print-op-generic
```

### 1.8 debug日志

编译为debug模式,并且iree-compiler 添加 debug 的参数,则可以打印出来日志, IREE使用了LLVM的日志系统
---

## 2. ONNX 导入调试

### 2.1 验证 ONNX 模型

```bash
python -c "
import onnx
from onnx import checker

model = onnx.load('model.onnx')
try:
    checker.check_model(model)
    print('ONNX model is valid')
except Exception as e:
    print(f'ONNX model invalid: {e}')
"
```

### 2.2 检查输入输出 Shape

```bash
python -c "
import onnx

model = onnx.load('model.onnx')
print('=== Inputs ===')
for inp in model.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.HasField('dim_value'):
            dims.append(str(d.dim_value))
        elif d.HasField('dim_param'):
            dims.append(d.dim_param)
        else:
            dims.append('?')
    print(f'  {inp.name}: [{', '.join(dims)}]')

print('=== Outputs ===')
for out in model.graph.output:
    dims = []
    for d in out.type.tensor_type.shape.dim:
        if d.HasField('dim_value'):
            dims.append(str(d.dim_value))
        elif d.HasField('dim_param'):
            dims.append(d.dim_param)
        else:
            dims.append('?')
    print(f'  {out.name}: [{', '.join(dims)}]')
"
```

### 2.3 统计模型中的节点类型

```bash
python -c "
import onnx
from collections import Counter

model = onnx.load('model.onnx')
op_counts = Counter(node.op_type for node in model.graph.node)
print('Node types:')
for op, count in op_counts.most_common():
    print(f'  {op}: {count}')
print(f'Total nodes: {sum(op_counts.values())}')
"
```

### 2.4 列出所有节点名称

```bash
python -c "
import onnx

model = onnx.load('model.onnx')
for i, node in enumerate(model.graph.node):
    print(f'  [{i}] {node.op_type}: {node.name if node.name else \"(no name)\"}')
"
```

### 2.5 调试 ONNX 导入失败

```bash
# 导入时保存中间 MLIR
python -m iree.compiler.tools.import_onnx \
    model.onnx \
    -o model.mlir \
    --opset-version 17 \
    2>&1 | grep -i error
```

如果导入失败，检查：

- ONNX opset 版本是否支持（常用 `--opset-version 17`）
- 是否有不支持的算子
- 动态 Shape 是否正确

---

## 3. 运行时调试

### 3.1 查看 VMFB 中的函数列表

```bash
# 使用 iree-run-module 不指定函数名，会列出可用函数
iree-run-module \
    --module=model.vmfb \
    --device=local-task

# 或用 Python 查看
python -c "
from iree.runtime import VmModule, Config, SystemContext

with open('model.vmfb', 'rb') as f:
    vm_module = VmModule.from_buffer(SystemContext().instance, f.read())
print('Functions:', list(vm_module.function_names))
"
```

### 3.2 输入数据格式

`iree-run-module` 支持多种输入格式：

```bash
# 全 1 输入
--input=1x3x224x224xf32=1

# 全 0 输入
--input=1x3x224x224xf32=0

# 从二进制文件读取（raw binary）
--input=1x3x224x224xf32=@input.bin

# 从 .npy 文件读取
--input=1x3x224x224xf32=@input.npy

# 显式指定每个元素（小张量）
--input=1x2xf32=[1.0 2.0]
```

### 3.3 内存统计

```bash
iree-run-module \
    --module=model.vmfb \
    --device=local-task \
    --function=mxnet_converted_model \
    --input=1x3x224x224xf32=1 \
    --print_statistics=true
```

输出：

```
[[ iree_hal_allocator_t memory statistics ]]
  HOST_LOCAL:         3200B peak /         3200B allocated
  DEVICE_LOCAL:      4400000B peak /      4400000B allocated
```

### 3.4 控制输出量

```bash
# 只打印前 N 个元素
--print-max-element-count=10

# 打印全部元素（小张量）
--print-max-element-count=0
```

### 3.5 验证计算结果

```bash
python -c "
import numpy as np

# 用 ONNX Runtime 计算参考结果
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
onnx_result = sess.run(None, {input_name: np.ones((1,3,224,224), dtype=np.float32)})[0]

# 与 IREE 输出对比
print(f'ONNX result shape: {onnx_result.shape}')
print(f'ONNX result first 5: {onnx_result.flatten()[:5]}')
print(f'ONNX result min/max/mean: {onnx_result.min():.6f} / {onnx_result.max():.6f} / {onnx_result.mean():.6f}')
"
```

### 3.6 多设备测试

```bash
# local-task（多线程 CPU）
--device=local-task

# local-sync（单线程同步）
--device=local-sync

# vulkan（GPU）
--device=vulkan
```

---

## 4. Python Runtime API 调试

### 4.1 基本调试模板

```python
from iree.runtime import SystemContext, Config, VmModule
import numpy as np

# 初始化
ctx = SystemContext(config=Config(driver_name="local-task"))

# 加载模块
with open("model.vmfb", "rb") as f:
    vm_module = VmModule.from_buffer(ctx.instance, f.read())
ctx.add_vm_module(vm_module)

# 查看模块信息
print("Module name:", vm_module.name)
print("Functions:", list(vm_module.function_names))

# 获取函数
bound = ctx.modules.get(vm_module.name)
func = getattr(bound, "function_name")

# 准备输入
inputs = [np.ones((1, 3, 224, 224), dtype=np.float32)]

# 执行
print("Executing...")
results = func(*inputs)

# 查看输出
for i, out in enumerate(results):
    arr = np.asarray(out)
    print(f"Output[{i}]: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  first 5: {arr.flatten()[:5]}")
```

### 4.2 查看函数签名

```python
vm_func = func.vm_function
print("Function:", vm_func)
# 注意：不同 IREE 版本的 API 可能有差异
# 如果 vm_func.argument_count 不可用，使用硬编码已知 shape
```

---

## 5. 常见问题排查

### 5.1 VMFB 文件为 0 字节

**原因**：使用了废弃的编译参数。

**解决**：确保使用正确的编译参数：

```bash
--iree-hal-target-device=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-cpu=host
```

旧的 `--iree-hal-target-backends=llvm-cpu` 已废弃。

### 5.2 "no callable functions found"

**原因**：函数名不对。ONNX 转换后的模型通常使用原始 ONNX 模型名。

**排查**：

```bash
python -c "
from iree.runtime import VmModule, SystemContext
with open('model.vmfb', 'rb') as f:
    vm = VmModule.from_buffer(SystemContext().instance, f.read())
print(list(vm.function_names))
"
```

常见函数名：

- `mxnet_converted_model`（完整 ResNet50）
- `mxnet_converted_model_extracted`（提取的层）

跳过 `__init` 和包含 `$async` 的函数。

### 5.3 输入 Shape 不匹配

**错误**：`Input shape mismatch: expected [1, 3, 224, 224], got [1, 3, 299, 299]`

**排查**：

```bash
# 查看 ONNX 模型实际输入 shape
python -c "
import onnx
m = onnx.load('model.onnx')
for inp in m.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'{inp.name}: {dims}')
"
```

### 5.4 iree-compile 无日志输出

**原因**：成功编译时默认无输出，这是正常行为。

**获取详细信息**：

```bash
# 添加时间统计
--time-passes

# 添加详细日志
--log-actions-to=-
```

### 5.5 iree.runtime 未安装

```bash
pip install /Users/alanchen/workspace/iree/build/runtime
```

确保使用的是本地编译的 runtime，而非 pip 下载的。

---

## 6. 快速参考表

| 需求 | 命令/方法 |
|------|-----------|
| 查看 Pass 耗时 | `--time-passes` |
| 打印 IR | `--mlir-print-ir-before-all` |
| 导出各阶段 IR | `--dump-compilation-phases-to=dir/` |
| 验证 ONNX | `onnx.checker.check_model()` |
| 查看输入 Shape | `inp.type.tensor_type.shape.dim` |
| 列出 VMFB 函数 | `vm_module.function_names` |
| 打印内存统计 | `--print_statistics=true` |
| 控制输出量 | `--print-max-element-count=N` |
| 二进制输入 | `--input=shape=@file.bin` |
| NPY 输入 | `--input=shape=@file.npy` |
| 单线程调试 | `--ir-thread-pool-size=1` |
