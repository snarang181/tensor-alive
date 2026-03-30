func.func @test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  %0 = arith.addf %arg1, %arg2 : f32
  %1 = arith.mulf %arg0, %0 : f32
  return %1 : f32
}
