func.func @test(%arg0: tensor<4x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<4x3xf32> {
  %0 = tensor.broadcast %arg1 : tensor<1x3xf32> -> tensor<4x3xf32>
  %1 = arith.addf %arg0, %0 : tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}
