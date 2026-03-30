func.func @test(%arg0: tensor<2x3xf32>, %init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = tensor.transpose %arg0 [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
