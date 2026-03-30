func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
