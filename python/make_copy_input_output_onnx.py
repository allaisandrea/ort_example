import sys
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model

if __name__ == "__main__":
    output_path = sys.argv[1]
    tensor = make_tensor_value_info("tensor", TensorProto.FLOAT, shape=[None])
    graph = make_graph([], "graph", [tensor], [tensor])
    onnx_model = make_model(graph, ir_version=10, opset_imports=[make_opsetid("", 21)])
    check_model(onnx_model)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
