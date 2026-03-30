func.func @test(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>, %C: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>) outs(%C : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
