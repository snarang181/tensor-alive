func.func @test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg1 : f32
  %1 = arith.mulf %arg0, %arg2 : f32
  %2 = arith.addf %0, %1 : f32
  return %2 : f32
}
