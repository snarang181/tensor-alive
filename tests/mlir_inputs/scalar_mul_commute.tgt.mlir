func.func @test(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.mulf %arg1, %arg0 : f32
  return %0 : f32
}
