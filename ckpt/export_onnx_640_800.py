import torch
from SPNet.src.networks import V2Net

dims = [192, 384, 768, 1536]
depths = [3, 3, 27, 3]
dp_rate = 0.2
norm_type = 'CNX'
model_dir = 'Large_300.pth'

net = V2Net(dims, depths, dp_rate, norm_type).cuda().eval()
net.load_state_dict(torch.load(model_dir)['network'])

# odin1 half-res: 1600x1296 raw -> 800x640.
# SPNet has 4 effective stride-2 stages so both dims must be divisible by 16.
# 800/16=50 and 640/16=40 — both clean. 648 (the naive 1296/2) fails because
# 648/8=81 is odd: the 4th stride-2 floors to 40, ConvTranspose gives 80≠81.
rgb = torch.randn(1, 3, 640, 800).cuda()
depth = torch.randn(1, 1, 640, 800).cuda()
mask = torch.ones_like(depth).cuda()
mask[depth == 0] = 0

torch.onnx.export(
    net,
    (rgb, depth, mask),
    "spnet_640_800.onnx",
    input_names=["rgb", "depth", "mask"],
    output_names=["pred"],
    dynamic_axes={
        "rgb": {0: "batch", 2: "height", 3: "width"},
        "depth": {0: "batch", 2: "height", 3: "width"},
        "mask": {0: "batch", 2: "height", 3: "width"},
        "pred": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
    dynamo=False,   # torch 2.11 defaults to the dynamo exporter (needs onnxscript
                    # and changes dynamic-shape semantics); force the legacy
                    # TorchScript exporter these scripts target.
)

print("ONNX Export ok.")
