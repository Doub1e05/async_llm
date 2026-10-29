"""AsyncLLM worker for telemetry actions and camera-based semantic Nav2 goals."""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
RUNTIME = ROOT / "robot_sim/.runtime"
for name, path in {"HF_HOME": "hf", "TRITON_CACHE_DIR": "triton",
                   "TORCH_EXTENSIONS_DIR": "torch_extensions", "CUDA_CACHE_PATH": "cuda_cache"}.items():
    os.environ.setdefault(name, str(RUNTIME / path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from PIL import Image
from robot_sim.control import ACTIONS
from robot_sim.direct_control import DIRECT_PROMPT, DIRECT_STATE_PROMPT
from robot_sim.evaluation import validate_decision

PROMPT = """You control a differential drive robot. Observation gives target_direction,
distance and lidar ranges in meters. If direction is ahead choose forward; if left choose left;
if right choose right. Stop for an obstacle closer than 0.3 m. Explain in one short line."""
NAV_PROMPT = """Select a destination for a delivery robot from its camera and known map.
Follow the user's instruction. Answer exactly one color: red, blue, green, or stop.
You select a destination, not a steering command. Nav2 plans a safe path around obstacles.
An obstacle in front does not prevent choosing a known destination.
Use stop only when no requested destination exists in goal_candidates."""


async def choose(llm, observation, max_tokens):
    if hasattr(llm, "select_action"):
        return llm.select_action(observation)
    navigation = observation.get("task") == "navigation"
    names = ("red", "blue", "green", "stop") if navigation else tuple(ACTIONS)
    token_ids = []
    for name in names:
        encoded = llm.tokenizer.encode(name, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Action is not one token: {name}")
        token_ids.append(encoded[0])
    state = {key: value for key, value in observation.items() if key != "image"}
    image = None
    if observation.get("image"):
        payload = base64.b64decode(observation["image"], validate=True)
        image = Image.open(io.BytesIO(payload))
        if image.width > 640 or image.height > 480:
            raise ValueError("Camera frame exceeds 640x480")
        image = image.convert("RGB")
    if navigation and image is None:
        raise ValueError("Navigation vision requires a camera image")
    block = await llm.create_block()
    generated = []
    try:
        control_prompt = DIRECT_STATE_PROMPT if observation.get("control_protocol") == "state_v2" else DIRECT_PROMPT
        system = NAV_PROMPT if navigation else control_prompt if observation.get("task") == "direct_driving" else PROMPT
        if image is not None:
            inputs = llm.processor.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": [
                     {"type": "image", "image": image},
                     {"type": "text", "text": json.dumps(state)}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", enable_thinking=False)
        else:
            prompt = llm.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": json.dumps(state)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = llm.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs.pop("attention_mask", None)
        for _ in range(0 if navigation or observation.get("task") == "direct_driving" else max_tokens):
            output = await llm(**dict(inputs), cache_view=[block])
            token = int(await llm.sample(output))
            generated.append(token)
            if token == llm.tokenizer.eos_token_id or "\n" in llm.tokenizer.decode([token]):
                break
            inputs = {"input_ids": torch.tensor([token], dtype=torch.int32)}
        if navigation or observation.get("task") == "direct_driving":
            logits = (await llm(**dict(inputs), cache_view=[block])).logits.float()
        else:
            suffix = llm.tokenizer("\nAction: ", return_tensors="pt", add_special_tokens=False)
            suffix.pop("attention_mask", None)
            logits = (await llm(**dict(suffix), cache_view=[block])).logits.float()
        scores = logits[token_ids].softmax(dim=0).cpu().tolist()
        index = int(logits[token_ids].argmax())
        return {"action": names[index], "multimodal": image is not None,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest() if image else None,
                "reasoning": llm.tokenizer.decode(generated, skip_special_tokens=True),
                "probabilities": dict(zip(names, scores))}
    finally:
        await llm.free_block(block)


async def serve(args, llm):
    try:
        print("WARMUP", flush=True)
        await choose(llm, {"target_direction": "ahead", "distance": 0.6,
                          "ranges": {"front": 2, "left": 2, "right": 2}}, args.reasoning_tokens)
        if args.vision:
            buffer = io.BytesIO()
            Image.new("RGB", (320, 240), (190, 40, 40)).save(buffer, format="PNG")
            await choose(llm, {"task": "navigation", "instruction": "Go to the red delivery marker.",
                               "image": base64.b64encode(buffer.getvalue()).decode(),
                               "goal_candidates": {"red": [-1.7, -2, 0]}}, args.reasoning_tokens)
        lock = asyncio.Lock()

        async def handle(reader, writer):
            try:
                payload = await asyncio.wait_for(reader.readline(), timeout=5)
                observation = json.loads(payload)
                async with lock:
                    response = await choose(llm, observation, args.reasoning_tokens)
                    validate_decision(response)
            except Exception as error:
                response = {"error": str(error), "action": "stop"}
            try:
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", args.port, limit=2_000_000)
        print(f"READY 127.0.0.1:{args.port}", flush=True)
        async with server:
            await server.serve_forever()
    finally:
        await llm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["async", "transformers"], default="async")
    parser.add_argument("--gpu-memory-gib", type=int, default=12)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument("--memory-ratio", type=float, default=0.4)
    parser.add_argument("--reasoning-tokens", type=int, default=16)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.memory_ratio <= 1 or args.reasoning_tokens < 1:
        parser.error("invalid memory ratio or token limit")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.backend == "transformers":
        from robot_sim.transformers_policy import TransformersPolicy
        llm = TransformersPolicy(args, NAV_PROMPT, PROMPT)
    else:
        from minisgl.llm import AsyncLLM
        llm = AsyncLLM(args.model, dtype=torch.bfloat16,
                       quantization=None if args.quantization == "none" else args.quantization,
                       max_running_req=2, cuda_graph_max_bs=1, memory_ratio=args.memory_ratio,
                       max_seq_len_override=4096, page_size=1)
    asyncio.run(serve(args, llm))


if __name__ == "__main__":
    main()
