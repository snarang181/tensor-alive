func.func @test(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tensor.transpose %arg0 [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  %1 = tensor.transpose %0 [1, 0] : tensor<3x2xf32> -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
