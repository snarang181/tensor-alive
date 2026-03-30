func.func @test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%init : tensor<4x4xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
