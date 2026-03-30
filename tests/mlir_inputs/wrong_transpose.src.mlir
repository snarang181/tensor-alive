func.func @test(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %0 = tensor.transpose %arg0 [2, 0, 1] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}
