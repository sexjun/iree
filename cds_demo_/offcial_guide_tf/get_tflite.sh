WORKDIR="./"
TFLITE_URL="https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
TFLITE_PATH=${WORKDIR}/model.tflite
IMPORT_PATH=${WORKDIR}/model.mlir
MODULE_PATH=${WORKDIR}/module.vmfb

# Fetch the sample model
# wget ${TFLITE_URL} -O ${TFLITE_PATH}

# Import the sample model to an IREE compatible form
iree-import-tflite ${TFLITE_PATH} -o ${IMPORT_PATH}

# Compile for the CPU backend
iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    ${IMPORT_PATH} \
    -o ${MODULE_PATH}

