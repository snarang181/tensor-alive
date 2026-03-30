func.func @test(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
