#!/usr/bin/env python3

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_model() -> onnx.ModelProto:
    query_length = 4
    embedding_dim = 4
    vocab_size = 10

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, [1, query_length]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [1, query_length]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, query_length, embedding_dim]
    )

    embedding_table = helper.make_tensor(
        "embedding_table",
        TensorProto.FLOAT,
        [vocab_size, embedding_dim],
        [
            # [PAD], [UNK], [CLS], [SEP], [MASK], [unused0], [unused1]
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            # alpha, beta, gamma
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
    )
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [2])

    nodes = [
        helper.make_node("Gather", ["embedding_table", "input_ids"], ["token_embeddings"], axis=0),
        helper.make_node("Cast", ["attention_mask"], ["mask_f32"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_f32", "axes"], ["mask_expanded"]),
        helper.make_node("Mul", ["token_embeddings", "mask_expanded"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_encoder_proof",
        [input_ids, attention_mask],
        [output],
        initializer=[embedding_table, axes],
    )
    opset = [helper.make_operatorsetid("", 13)]
    model = helper.make_model(
        graph,
        producer_name="next-plaid-browser",
        producer_version="0.1.0",
        opset_imports=opset,
    )
    model.ir_version = onnx.IR_VERSION
    onnx.checker.check_model(model)
    return model


def main() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    output_path = workspace_root / "fixtures" / "encoder-proof" / "tiny-encoder.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), output_path)
    print(output_path)


if __name__ == "__main__":
    main()
