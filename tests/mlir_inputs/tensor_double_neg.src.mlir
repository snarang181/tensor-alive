func.func @test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.negf %arg0 : tensor<8xf32>
  %1 = arith.negf %0 : tensor<8xf32>
  return %1 : tensor<8xf32>
}
