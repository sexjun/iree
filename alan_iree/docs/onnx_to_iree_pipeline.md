# IREE ONNX 模型端到端流程

本文档记录从下载 ONNX 模型，到固化 Shape，再到转换为 MLIR IR，编译为 VMFB，最后执行推理的完整流程。

## 环境准备

```bash
# 激活 conda 环境
conda activate llm

# 安装 ONNX 依赖
pip install onnx
```

**编译工具**: 全部使用本地编译的 IREE，不依赖外部安装包。

| 工具 | 路径 |
|------|------|
| `iree-compile` | `/Users/alanchen/workspace/iree/build/tools/iree-compile` |
| `iree-run-module` | `/Users/alanchen/workspace/iree/build/tools/iree-run-module` |
| Python 编译绑定 | `/Users/alanchen/workspace/iree/build/compiler/bindings/python` |
| Python Runtime 绑定 | `/Users/alanchen/workspace/iree/build/runtime/bindings/python` |

安装本地 Runtime 绑定：
```bash
pip install /Users/alanchen/workspace/iree/build/runtime
```

## 步骤 1: 固化 Batch Size

ONNX 模型的输入通常使用动态 batch size（用 `N` 或 `?` 表示）。为简化部署，需要先固定 batch size。

```bash
python cds/fix_batch_size.py
```

**输入**: `cds/resnet50-v1-12.onnx`（动态 batch）
**输出**: `cds/resnet50-v1-12-bs1.onnx`（batch=1）

验证模型输入 Shape：

```python
import onnx
m = onnx.load("cds/resnet50-v1-12-bs1.onnx")
for i in m.graph.input:
    dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in i.type.tensor_type.shape.dim]
    print(f"  Input: {i.name} -> {dims}")
# 输出: Input: data -> [1, 3, 224, 224]
```

## 步骤 2: ONNX → MLIR

使用 IREE 的 ONNX 导入工具，将 ONNX protobuf 模型转换为 MLIR 中间表示：

```bash
python -m iree.compiler.tools.import_onnx \
    cds/resnet50-v1-12-bs1.onnx \
    -o cds/resnet50-v1-12-bs1.mlir \
    --opset-version 17
```

**输出**: `cds/resnet50-v1-12-bs1.mlir`（约 196MB）

## 步骤 3: MLIR → VMFB（编译）

使用本地编译的 `iree-compile` 将 MLIR 编译为 IREE 可执行的 VMFB 格式：

```bash
/Users/alanchen/workspace/iree/build/tools/iree-compile \
    cds/resnet50-v1-12-bs1.mlir \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    -o cds/resnet50-v1-12-bs1.vmfb
```

**输出**: `cds/resnet50-v1-12-bs1.vmfb`（约 113MB）

**关键参数说明**:
| 参数 | 作用 |
|------|------|
| `--iree-hal-target-device=local` | 目标设备为本地 CPU |
| `--iree-hal-local-target-device-backends=llvm-cpu` | 使用 LLVM CPU 后端 |
| `--iree-llvmcpu-target-cpu=host` | 针对当前主机 CPU 优化（非 generic） |

其他可用后端：`vulkan-spirv`、`cuda`、`rocm` 等。

## 步骤 4: 执行推理

### 方式 A: 使用一键脚本（推荐）

```bash
# 全 1 输入
cds/run_pipeline.sh cds/resnet50-v1-12-bs1.onnx --input-data ones

# 随机输入
cds/run_pipeline.sh cds/resnet50-v1-12-bs1.onnx --input-data random
```

脚本自动完成：ONNX→MLIR→VMFB→`iree-run-module` 执行。

### 方式 B: 使用 iree-run-module CLI

```bash
# 全 1 输入
/Users/alanchen/workspace/iree/build/tools/iree-run-module \
    --module=cds/resnet50-v1-12-bs1.vmfb \
    --device=local-task \
    --function=mxnet_converted_model \
    --input=1x3x224x224xf32=1

# 随机输入（先生成二进制文件）
python -c "import numpy as np; np.random.randn(1,3,224,224).astype(np.float32).tofile('random.bin')"
/Users/alanchen/workspace/iree/build/tools/iree-run-module \
    --module=cds/resnet50-v1-12-bs1.vmfb \
    --device=local-task \
    --function=mxnet_converted_model \
    --input=1x3x224x224xf32=@random.bin
```

**输入格式**:
| 格式 | 说明 |
|------|------|
| `shape=值` | 填充全部元素，如 `1x3x224x224xf32=1` |
| `shape=@file.bin` | 从二进制文件读取 |
| `shape=[v1 v2 ...]` | 显式指定每个元素 |

### 方式 C: 使用 Python Runtime API

```bash
python cds/run_model.py cds/resnet50-v1-12-bs1.vmfb random
python cds/run_model.py cds/resnet50-v1-12-bs1.vmfb ones
```

使用 `SystemContext + Config(driver_name='local-task')`，等价于 `iree-run-module --device=local-task`。

**输出示例**:
```
EXEC @mxnet_converted_model
result[0]: hal.buffer_view
1x1000xf32=[-1.13853 0.343612 -1.10569 ...]
[[ iree_hal_allocator_t memory statistics ]]
  HOST_LOCAL:         4000B peak /         4000B allocated
DEVICE_LOCAL:      9443168B peak /      9443168B allocated
```

## 提取单个 Layer

使用 `extract_layer.py` 从完整模型中提取指定层作为独立 ONNX 模型：

```bash
# 列出可用的 layer name
python cds/extract_layer.py cds/resnet50-v1-12.onnx <不存在的名称>

# 提取指定层
python cds/extract_layer.py cds/resnet50-v1-12-bs1.onnx resnetv17_pool0_fwd
# 输出: cds/resnet50-v1-12-bs1_resnetv17_pool0_fwd.onnx
```

提取后的模型可以走相同的 ONNX→MLIR→VMFB→执行流程。

## 工具路径

```bash
export IREE_COMPILE=/Users/alanchen/workspace/iree/build/tools/iree-compile
export IREE_RUN_MODULE=/Users/alanchen/workspace/iree/build/tools/iree-run-module
```

脚本默认使用上述路径，可通过环境变量覆盖。

## 故障排查

### 编译失败（0 字节 VMFB）
确保使用正确的编译参数。旧的 `--iree-hal-target-backends=llvm-cpu` 已废弃，改用：
```
--iree-hal-target-device=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-cpu=host
```

### `iree.runtime` 未安装
```bash
pip install /Users/alanchen/workspace/iree/build/runtime
```
