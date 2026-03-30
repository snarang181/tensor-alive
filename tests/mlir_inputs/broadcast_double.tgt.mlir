func.func @test(%arg0: tensor<1x1xf32>) -> tensor<4x3xf32> {
  %0 = tensor.broadcast %arg0 : tensor<1x1xf32> -> tensor<1x3xf32>
  %1 = tensor.broadcast %0 : tensor<1x3xf32> -> tensor<4x3xf32>
  return %1 : tensor<4x3xf32>
}
