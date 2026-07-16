import inspect
import torch
from SPNet.src.networks import V2Net

dims = [192, 384, 768, 1536]
depths = [3, 3, 27, 3]
dp_rate = 0.2
norm_type = 'CNX'
model_dir = 'Large_300.pth'

net = V2Net(dims, depths, dp_rate, norm_type).cuda().eval()
net.load_state_dict(torch.load(model_dir)['network'])

# odin1 half-res: 1600x1296 raw -> 800x648. Both dims stay divisible by 8, which
# is all SPNet's 3 stride-2 downsampling stages require, so the fully
# convolutional net runs natively at this size.
rgb = torch.randn(1, 3, 648, 800).cuda()
depth = torch.randn(1, 1, 648, 800).cuda()
mask = torch.ones_like(depth).cuda()
mask[depth == 0] = 0

# torch >= 2.5 defaults to the dynamo exporter (needs onnxscript, different
# dynamic-shape semantics); force the legacy TorchScript exporter these scripts
# target. The `dynamo` kwarg does not exist on torch < 2.5 (e.g. x86's 2.0.1),
# so only pass it where supported — keeps one script working on both stacks.
export_kwargs = dict(opset_version=17)
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    export_kwargs["dynamo"] = False

torch.onnx.export(
    net,
    (rgb, depth, mask),
    "spnet_648_800.onnx",
    input_names=["rgb", "depth", "mask"],
    output_names=["pred"],
    dynamic_axes={
        "rgb": {0: "batch", 2: "height", 3: "width"},
        "depth": {0: "batch", 2: "height", 3: "width"},
        "mask": {0: "batch", 2: "height", 3: "width"},
        "pred": {0: "batch", 2: "height", 3: "width"},
    },
    **export_kwargs,
)

print("ONNX Export ok.")
