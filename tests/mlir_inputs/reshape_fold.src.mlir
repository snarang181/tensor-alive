func.func @test(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] : tensor<6xf32> into tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
