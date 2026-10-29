from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import torch
import vizdoom.gymnasium_wrapper  # noqa: F401 - registers VizDoom environments
from minisgl.llm import AsyncLLM
from PIL import Image, ImageDraw

DEFAULT_MODEL = os.environ.get("MINISGL_DOOM_MODEL", "Qwen/Qwen3.5-0.8B")
ACTION_NAMES = ("wait", "fire", "right", "left")
EXPECTED_BUTTON_MAP = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))
PROMPT = """
This is a first-person video game called Doom. Which action should I choose
next: `wait`, `fire`, `right`, or `left`? Use one of these actions verbatim.
Your shot hits if the enemy is at the center of the screen, precisely above
the gun barrel; otherwise the shot misses. If the enemy is to the right, move
right. If it is to the left, move left. If the enemy is centered, fire.
Analyze the image for only one paragraph, then write
"Action: `your_choice`".
""".strip()


@dataclass(frozen=True)
class Config:
    model: str
    quantization: str | None
    max_game_steps: int
    max_reasoning_tokens: int
    frame_skip: int
    memory_ratio: float
    seed: int
    output_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run the AsyncLLM multimodal agent in VizdoomBasic-v1."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local snapshot.")
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none")
    parser.add_argument("--max-game-steps", type=int, default=20)
    parser.add_argument("--max-reasoning-tokens", type=int, default=48)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--memory-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/doom_basic/output"))
    args = parser.parse_args()
    if args.max_game_steps < 1 or args.max_reasoning_tokens < 1:
        parser.error("max-game-steps and max-reasoning-tokens must be positive")
    if not 0 < args.memory_ratio <= 1:
        parser.error("memory-ratio must be in (0, 1]")
    values = vars(args)
    values["quantization"] = None if values["quantization"] == "none" else values["quantization"]
    return Config(**values)




def validate_actions(env: gymnasium.Env) -> None:
    button_map = tuple(tuple(int(v) for v in action) for action in env.unwrapped.button_map)
    if button_map != EXPECTED_BUTTON_MAP:
        raise RuntimeError(
            f"Unexpected VizdoomBasic action map {button_map}; expected {EXPECTED_BUTTON_MAP}"
        )


def candidate_token_ids(llm: AsyncLLM) -> list[int]:
    result = []
    for name in ACTION_NAMES:
        ids = llm.tokenizer.encode(name, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Action {name!r} must be one token, got {ids}")
        result.append(int(ids[0]))
    return result


def prepare_frame_inputs(llm: AsyncLLM, frame: np.ndarray) -> dict[str, torch.Tensor]:
    inputs = llm.processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(frame)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs.pop("attention_mask", None)
    return dict(inputs)


async def choose_action(
    llm: AsyncLLM,
    frame: np.ndarray,
    action_token_ids: list[int],
    max_reasoning_tokens: int,
) -> tuple[int, str, dict[str, float]]:
    block = await llm.create_block()
    generated: list[int] = []
    eos_id = llm.tokenizer.eos_token_id
    try:
        inputs = prepare_frame_inputs(llm, frame)
        for _ in range(max_reasoning_tokens):
            output = await llm(**inputs, cache_view=[block])
            token_id = int(await llm.sample(output))
            generated.append(token_id)
            token = llm.tokenizer.decode([token_id])
            if token_id == eos_id or "\n" in token:
                break
            inputs = {"input_ids": torch.tensor([token_id], dtype=torch.int32)}

        suffix = llm.tokenizer("\nAction: `", return_tensors="pt", add_special_tokens=False)
        suffix.pop("attention_mask", None)
        logits = (await llm(**dict(suffix), cache_view=[block])).logits.float()
        candidate_logits = logits[action_token_ids]
        probabilities = candidate_logits.softmax(dim=0).cpu().tolist()
        action_index = int(candidate_logits.argmax())
        scores = dict(zip(ACTION_NAMES, (float(p) for p in probabilities)))
        reasoning = llm.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return action_index, reasoning, scores
    finally:
        await llm.free_block(block)


def annotate(
    frame: np.ndarray, step: int, action: str, reward: float, total_reward: float
) -> Image.Image:
    image = Image.fromarray(frame).resize((640, 480), Image.Resampling.NEAREST)
    label = (
        f"step={step} action={action} reward={reward:.1f} "
        f"total={total_reward:.1f}"
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 24), fill=(0, 0, 0))
    draw.text((7, 6), label, fill=(255, 255, 255))
    return image


async def play(config: Config, llm: AsyncLLM) -> dict[str, object]:
    env = gymnasium.make(
        "VizdoomBasic-v1",
        render_mode="rgb_array",
        frame_skip=config.frame_skip,
        max_buttons_pressed=1,
    )
    validate_actions(env)
    action_ids = candidate_token_ids(llm)
    frames: list[Image.Image] = []
    decisions: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    total_reward = 0.0

    try:
        observation, _ = env.reset(seed=config.seed)
        for step in range(1, config.max_game_steps + 1):
            frame = observation["screen"]
            action_index, reasoning, scores = await choose_action(
                llm, frame, action_ids, config.max_reasoning_tokens
            )
            observation, reward, terminated, truncated, _ = env.step(action_index)
            reward = float(reward)
            total_reward += reward
            action = ACTION_NAMES[action_index]
            action_counts[action] += 1
            frames.append(annotate(frame, step, action, reward, total_reward))
            record = {
                "step": step,
                "action": action,
                "action_index": action_index,
                "reward": reward,
                "total_reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "candidate_probabilities": scores,
                "reasoning": reasoning,
            }
            decisions.append(record)
            print(
                f"step={step} action={action:<5} reward={reward:>6.1f} "
                f"total={total_reward:>6.1f} scores={scores}"
            )
            if reasoning:
                print(f"  reasoning: {reasoning}")
            if terminated or truncated:
                break
    finally:
        env.close()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = config.output_dir / "decisions.jsonl"
    trace_path.write_text(
        "".join(json.dumps(item, ensure_ascii=True) + "\n" for item in decisions),
        encoding="utf-8",
    )
    gif_path = config.output_dir / "doom_run.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=350,
            loop=0,
            optimize=False,
        )
    return {
        "total_reward": total_reward,
        "action_counts": dict(action_counts),
        "decisions": len(decisions),
        "gif": str(gif_path),
        "trace": str(trace_path),
    }


async def run_async(config: Config, llm: AsyncLLM) -> None:
    try:
        summary = await play(config, llm)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    finally:
        await llm.close()



def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Doom demo requires a CUDA-capable GPU")
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print(f"model: {config.model}")
    print("quantization:", config.quantization or "none")
    print("Loading tokenizer, vision processor, and engine...")
    llm = AsyncLLM(
        config.model,
        quantization=config.quantization,
        dtype=torch.bfloat16,
        max_running_req=2,
        cuda_graph_max_bs=1,
        memory_ratio=config.memory_ratio,
        max_seq_len_override=4096,
        page_size=1,
    )
    torch.cuda.manual_seed_all(config.seed)
    print(f"visible CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        asyncio.run(run_async(config, llm))
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
