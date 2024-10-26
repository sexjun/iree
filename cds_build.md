conda install --file runtime/bindings/python/iree/runtime/build_requirements.txt


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

## python设置
编译结束后，在build文件夹有一个文件 `/Users/chendongsheng/github/iree/build/.env` 该文件里里的路径，需要添加到python包加载路径。


