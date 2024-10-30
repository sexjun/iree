# IREE教程

## 环境搭建
- tflite 的环境
```
python -m pip install iree-tools-tf iree-tools-tflite -f https://iree.dev/pip-release-links.html
```


# 配置环境
```
pip install nanobind
pip install pybind11
pip install numpy
```

## 编译指令
> 开启了python 语言绑定。
cmake -G Ninja -B ./build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python)" \
    -DCMAKE_INSTALL_PREFIX=/usr/local/cds_bin \
    -DIREE_EMBED_RELEASE_INFO=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON


cmake --build  ./build/

- python设置
编译结束后，在build文件夹有一个文件 `/Users/chendongsheng/github/iree/build/.env` 该文件里里的路径，需要添加到python包加载路径。


## iree demo
- 编译
    ```bash
    # 编译
    iree-compile --iree-hal-target-backends=llvm-cpu ./model.mlir -o model.vmfb
    # 将mlir转为文本格式
    iree-opt ./model.mlir > txt.mlir

    ```
- 运行
    ```
    (tf) (base) ➜  offcial_guide_tf git:(20241017) ✗ iree-run-module --device=local-task --module=module.vmfb --input="1x192x192x3xi8=0"
    EXEC @main
    result[0]: hal.buffer_view
    1x1x17x3xf32=[[[0.499816 0.44246 0.057356][0.397395 0.462945 0.0245811][0.446557 0.344136 0.0737434][0.372814 0.471138 0.0368717][0.43017 0.381007 0.0245811][0.487526 0.516204 0.0409685][0.44246 0.36462 0.0409685][0.475235 0.577656 0.0409685][0.487526 0.397395 0.0368717][0.495719 0.557172 0.0245811][0.516204 0.44246 0.0368717][0.692368 0.5203 0.0409685][0.655497 0.483429 0.0409685][0.716949 0.5203 0.0245811][0.725143 0.487526 0.0368717][0.860339 0.5203 0.057356][0.831661 0.553075 0.0942276]]]
    ```

- 关于输入
```
# Name of a function contained in the module specified by --module= to run.
--function=""

# An input (a) value or (b) buffer of the format:
#   (a) scalar value
#      value
#      e.g.: --input="3.14"
#   (b) buffer:
#      [shape]xtype=[value]
#      e.g.: --input="2x2xi32=1 2 3 4"
# Optionally, brackets may be used to separate the element values:
#   2x2xi32=[[1 2][3 4]]
# Raw binary files can be read to provide buffer contents:
#   2x2xi32=@some/file.bin
# Numpy npy files from numpy.save can be read to provide 1+ values:
#   @some.npy
# Each occurrence of the flag indicates an input in the order they were
# specified on the command line.
# --input=...

```


## debug skill
> 参考官网文档: https://iree.dev/developers/general/developer-tips/#dumping-compilation-phases
1. vmfb文件

- 这个是flatbuffer文件,可以使用`unzip`命令打开

```
$ unzip -d simple_abs_cpu ./simple_abs_cpu.vmfb

Archive:  ./simple_abs_cpu.vmfb
  extracting: simple_abs_cpu/module.fb
  extracting: simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so
```

- 嵌入的二进制文件（此处是带有 CPU 代码的 ELF 共享对象）可以通过标准工具进行解析：
```
$ readelf -Ws ./simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so

Symbol table '.dynsym' contains 2 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000001760    17 FUNC    GLOBAL DEFAULT    7 iree_hal_executable_library_query

Symbol table '.symtab' contains 42 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abs_dispatch_0
    2: 0000000000001730    34 FUNC    LOCAL  DEFAULT    7 abs_dispatch_0_generic
    3: 00000000000034c0    80 OBJECT  LOCAL  DEFAULT    8 iree_hal_executable_library_query_v0
    4: 0000000000001780   111 FUNC    LOCAL  DEFAULT    7 iree_h2f_ieee
    5: 00000000000017f0   207 FUNC    LOCAL  DEFAULT    7 iree_f2h_ieee
    ...
```

- iree-dump-module 工具还可用于查看有关给定 .vmfb 文件的信息：
 > 类似于 --mlir-print-ir-after= 标志，但在明确定义的管道阶段。
```
$ iree-dump-module simple_abs.vmfb
```
- 您可以使用 --compile-to=<phase name> 标志在中间阶段输出程序快照：
```
 iree-compile simple_abs.mlir --compile-to=abi
```

- 或使用 --compile-from=<phase name> 从中间阶段显式恢复：
```
$ iree-compile simple_exp_abi.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --compile-from=abi \
  -o simple_exp_cpu.vmfb
```

- --dump-compilation-phases-to 标志可用于在每个阶段后转储程序 IR
```
$ iree-compile simple_abs.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --dump-compilation-phases-to=/tmp/iree/simple_abs \
  -o /tmp/iree/simple_abs/simple_abs_cpu.vmfb

$ ls /tmp/iree/simple_abs -1v

simple_abs.1.input.mlir
simple_abs.2.abi.mlir
simple_abs.3.preprocessing.mlir
simple_abs.4.global-optimization.mlir
simple_abs.5.dispatch-creation.mlir
simple_abs.6.flow.mlir
simple_abs.7.stream.mlir
simple_abs.8.executable-sources.mlir
simple_abs.9.executable-configurations.mlir
simple_abs.10.executable-targets.mlir
simple_abs.11.hal.mlir
simple_abs.12.vm.mlir
```
## IREE的编译结果工具
- iree-opt
iree-opt是测试pass的工具,类似 mlir-opt
- iree-comple
编译工具
- iree-run-module
run time
- iree-check-module
将已翻译的 IREE 模块作为输入，并将其作为一系列 googletest 测试执行。这是 IREE 检查框架的测试运行程序。
- iree-run-mlir
将 .mlir 文件作为输入，将其转换为 IREE 字节码模块，然后执行该模块。
专为测试和调试而设计，而不是用于生产用途，
- iree-dump-module
iree-dump-module 程序打印 IREE 模块 FlatBuffer 文件的内容。


## MLIR 教程

https://github.com/j2kun/mlir-tutorial
