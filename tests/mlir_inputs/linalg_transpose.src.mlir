func.func @test(%arg0: tensor<2x3xf32>, %init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
  } -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
