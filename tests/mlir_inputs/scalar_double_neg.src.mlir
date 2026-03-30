func.func @test(%arg0: f32) -> f32 {
  %0 = arith.negf %arg0 : f32
  %1 = arith.negf %0 : f32
  return %1 : f32
}
