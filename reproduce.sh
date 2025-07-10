set -e
ONNX_VERSION="1.22.0"
WORK_DIR="/tmp/ort_example"
mkdir -p ${WORK_DIR}
echo "Downloading ONNX runtime"
curl -L -o ${WORK_DIR}/onnx_runtime.tgz  https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz
echo "Extracting ONNX runtime"
tar --directory=${WORK_DIR} -xf ${WORK_DIR}/onnx_runtime.tgz 
export LD_LIBRARY_PATH=${WORK_DIR}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}/lib/
echo "Building example"
cargo build
echo "Running example"
for i in {1..100}
do
    echo "Repeat ${i}"
    ./target/debug/ort_example data/copy_input_output.onnx > ${WORK_DIR}/log_${i}.txt 2>&1
done
