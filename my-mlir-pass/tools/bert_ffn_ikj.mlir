// Loop schedule for the two tensor matmuls in the exported BERT-Tiny FFN.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ffn = transform.structured.match ops{["func.func"]}
        attributes {sym_name = "main"} in %root
        : (!transform.any_op) -> !transform.any_op

    // Each filter lists the activation, transposed weight, and accumulator.
    %expand = transform.structured.match ops{["linalg.matmul"]}
        filter_operand_types = [tensor<64x128xf32>, tensor<128x512xf32>, tensor<64x512xf32>]
        in %ffn : (!transform.any_op) -> !transform.any_op
    %project = transform.structured.match ops{["linalg.matmul"]}
        filter_operand_types = [tensor<64x512xf32>, tensor<512x128xf32>, tensor<64x128xf32>]
        in %ffn : (!transform.any_op) -> !transform.any_op

    %matmuls = transform.merge_handles %expand, %project : !transform.any_op
    %generics = transform.structured.generalize %matmuls
        : (!transform.any_op) -> !transform.any_op
    // Keep the token loop first and move the output-feature loop innermost.
    %reordered = transform.structured.interchange %generics iterator_interchange = [0, 2, 1]
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
