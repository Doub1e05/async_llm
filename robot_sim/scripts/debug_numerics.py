import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from robot_sim.transformers_policy import TransformersPolicy

root = Path(__file__).resolve().parents[2]
snapshots = Path(__import__("os").environ.get("ROBOT_HF_HOME", str(Path.home()/".cache/huggingface")))/"hub/models--Qwen--Qwen3.5-9B/snapshots"
args = SimpleNamespace(model=str(next(snapshots.iterdir())),gpu_memory_gib=12,metadata=None)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
policy = TransformersPolicy(args,"Answer exactly one color: red, blue, green, or stop.","Choose forward, left, right or stop.")


def check(name):
    def hook(module,inputs,output):
        tensors = output if isinstance(output,(tuple,list)) else (output,)
        for tensor in tensors:
            if isinstance(tensor,torch.Tensor) and tensor.is_floating_point():
                if not torch.isfinite(tensor).all().item():
                    print("FIRST_NONFINITE_MODULE",name,type(module).__name__,"dtype",tensor.dtype,flush=True)
                    raise FloatingPointError(name)
    return hook


for name,module in policy.model.named_modules():
    if "language_model.layers" in name and len(name.split(".")) == 4:
        def stage(module, inputs, output, name=name):
            tensor = output[0] if isinstance(output,tuple) else output
            if isinstance(tensor,torch.Tensor):
                print("STAGE",name,"absmax",float(tensor.abs().max()),"finite",bool(torch.isfinite(tensor).all()),flush=True)
        module.register_forward_hook(stage)
for name,parameter in policy.model.named_parameters():
    if name in {"lm_head.weight","model.language_model.norm.weight","model.language_model.layers.0.input_layernorm.weight","model.language_model.embed_tokens.weight"}:
        print("WEIGHT",name,"dtype",parameter.dtype,"range",float(parameter.min()),float(parameter.max()),flush=True)
try:
    result = policy.select_action({"task":"control","target_direction":"ahead","distance":1.0})
    print("TEXT",json.dumps(result),flush=True)
except Exception as error:
    print("TEXT_ERROR",repr(error),flush=True)
image = root / "robot_sim/results/qwen9b_evaluation_20261026/invalid_backend_nan/warehouse_red_blue/vision_goal_0.png"
try:
    result=policy.select_action({"task":"navigation","instruction":"Go to the blue marker.",
        "goal_candidates":{"red":[-1.7,-2,0],"blue":[1,0,0],"green":[4,2.3,0]},
        "image":base64.b64encode(image.read_bytes()).decode()})
    print("VISION",json.dumps(result),flush=True)
except Exception as error:
    print("VISION_ERROR",repr(error),flush=True)
