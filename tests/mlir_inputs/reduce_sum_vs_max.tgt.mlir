func.func @test(%arg0: tensor<3x4xf32>, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<3x4xf32>) outs(%init : tensor<3xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %max = arith.maximumf %acc, %a : f32
    linalg.yield %max : f32
  } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
