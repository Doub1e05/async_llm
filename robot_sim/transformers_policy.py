"""BF16 layer-sharded reference backend with the existing action protocol."""
import base64
import hashlib
import io
import json
import os
import time

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from robot_sim.control import ACTIONS
from robot_sim.direct_control import DIRECT_PROMPT, DIRECT_STATE_PROMPT


class TransformersPolicy:
    def __init__(self, args, navigation_prompt, control_prompt):
        self.navigation_prompt = navigation_prompt
        self.control_prompt = control_prompt
        count = torch.cuda.device_count()
        if count not in (1, 2):
            raise ValueError("Exactly one or two visible GPUs required")
        started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="balanced",
            max_memory={index: f"{args.gpu_memory_gib}GiB" for index in range(count)},
            attn_implementation="sdpa", local_files_only=True).eval()
        self.bridge_hooks = []
        if count == 2:
            from accelerate.utils import recursively_apply
            for module in self.model.modules():
                hook = getattr(module, "_hf_hook", None)
                if hook is None:
                    continue
                if hasattr(hook, "io_same_device"):
                    hook.io_same_device = False
                execution = getattr(hook, "execution_device", None)
                if execution is None:
                    continue
                destination = torch.device(execution)

                def stage(module, inputs, keywords, destination=destination):
                    def transfer(tensor):
                        if tensor.device.type == "cuda" and tensor.device != destination:
                            return tensor.to("cpu")
                        return tensor
                    return recursively_apply(transfer, inputs), recursively_apply(transfer, keywords)

                self.bridge_hooks.append(module.register_forward_pre_hook(stage, with_kwargs=True))
        mapping = self.model.hf_device_map
        if any(str(device) in {"cpu", "disk"} for device in mapping.values()):
            raise RuntimeError("CPU/disk weight offload is not allowed for this experiment")
        self.input_device = self.model.get_input_embeddings().weight.device
        self.metadata = {"model": args.model, "backend": "transformers_accelerate_layer_sharding",
                         "dtype": "bfloat16", "quantization": None,
                         "inter_gpu_transfer": "cpu_staging" if count == 2 else "none",
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "device_map": mapping, "cpu_affinity": sorted(os.sched_getaffinity(0)),
                         "load_seconds": time.monotonic() - started,
                         "torch_version": torch.__version__}
        if args.metadata:
            args.metadata.parent.mkdir(parents=True, exist_ok=True)
            args.metadata.write_text(json.dumps(self.metadata, indent=2))
        print("MODEL_METADATA " + json.dumps(self.metadata), flush=True)

    @torch.inference_mode()
    def select_action(self, observation):
        navigation = observation.get("task") == "navigation"
        names = ("red", "blue", "green", "stop") if navigation else tuple(ACTIONS)
        state = {key: value for key, value in observation.items() if key != "image"}
        content = []
        image = None
        if observation.get("image"):
            image = Image.open(io.BytesIO(base64.b64decode(observation["image"], validate=True))).convert("RGB")
            if image.width > 640 or image.height > 480:
                raise ValueError("Camera frame exceeds 640x480")
            content.append({"type": "image", "image": image})
        if navigation and image is None:
            raise ValueError("Navigation requires a camera image")
        content.append({"type": "text", "text": json.dumps(state)})
        control_prompt = DIRECT_STATE_PROMPT if observation.get("control_protocol") == "state_v2" else DIRECT_PROMPT
        messages = [{"role": "system", "content": self.navigation_prompt if navigation else control_prompt if observation.get("task") == "direct_driving" else self.control_prompt},
                    {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        if inputs["input_ids"].shape[-1] > 4096:
            raise ValueError("Input exceeds the experiment context limit")
        tokens = [self.tokenizer.encode(name, add_special_tokens=False) for name in names]
        if any(len(token) != 1 for token in tokens):
            raise ValueError("Actions must be single-token labels")
        inputs = inputs.to(self.input_device)
        output = self.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1].float()
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite model logits; no action may be executed")
        scores = logits[[token[0] for token in tokens]].softmax(dim=0).cpu().tolist()
        index = max(range(len(names)), key=lambda position: scores[position])
        return {"action": names[index], "multimodal": image is not None,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest() if image else None,
                "reasoning": "", "probabilities": dict(zip(names, scores)),
                "input_tokens": int(inputs["input_ids"].shape[-1]), "backend": "transformers",
                "logit_range": [float(logits.min()), float(logits.max())],
                "top_tokens": [self.tokenizer.decode([int(token)]) for token in logits.topk(5).indices.cpu()],
                "gpu_peak_allocated_mib": [round(torch.cuda.max_memory_allocated(device)/2**20, 1)
                                            for device in range(torch.cuda.device_count())]}

    async def close(self):
        del self.model
        torch.cuda.empty_cache()
