func.func @test(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>, %C: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>) outs(%C : tensor<2x4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
  } -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
