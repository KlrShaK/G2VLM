import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image

from g2vlm_utils import load_model_and_tokenizer, build_transform, process_conversation


def load_images_for_scene(scene, scenes_dir):
    scene_id = scene["scene_id"]
    input_type = scene["input_type"]
    scene_dir = os.path.join(scenes_dir, scene_id)

    if input_type == "photos":
        paths = sorted(glob.glob(os.path.join(scene_dir, "*.jpg")))
        if not paths:
            raise FileNotFoundError(f"No .jpg files found in {scene_dir}")
        return [Image.open(p).convert("RGB") for p in paths]

    elif input_type == "frames":
        frames_dir = os.path.join(scene_dir, "frames")
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        n = scene.get("num_frames", 12)
        if len(paths) <= n:
            indices = list(range(len(paths)))
        else:
            indices = np.linspace(0, len(paths) - 1, n, dtype=int).tolist()
        return [Image.open(paths[i]).convert("RGB") for i in indices]

    else:
        raise ValueError(f"Unknown input_type '{input_type}' for scene {scene_id}")


def save_results(output_file, metadata, results):
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    os.replace(tmp, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/cluster/work/igp_psr/spanwar/G2VLM/models/G2VLM-2B-MoT")
    parser.add_argument("--prompts-file", type=str, required=True)
    parser.add_argument("--scenes-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts_data = json.load(f)

    metadata = {
        "model_path": args.model_path,
        "prompts_file": args.prompts_file,
        "scenes_dir": args.scenes_dir,
        "timestamp": datetime.now().isoformat(),
    }

    print("Loading model...")
    model, tokenizer, new_token_ids, vit_image_transform, dino_transform = load_model_and_tokenizer(args)
    image_transform = build_transform(pixel=768, model_path=args.model_path)
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {total_params:.2f}B params")

    results = []
    scenes = prompts_data["scenes"]
    total_prompts = sum(len(s["prompts"]) for s in scenes)
    done = 0

    for scene in scenes:
        scene_id = scene["scene_id"]
        print(f"\n[{scene_id}] Loading images ({scene['input_type']})...")

        try:
            images = load_images_for_scene(scene, args.scenes_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(images)} image(s) loaded")

        for prompt in scene["prompts"]:
            done += 1
            prompt_id = prompt["id"]
            question = prompt["question"]
            gt = prompt["ground_truth"]

            print(f"  [{done}/{total_prompts}] {prompt_id}: {question[:80]}...")

            imgs, conversation = process_conversation(images, question)
            answer = model.chat_with_recon(
                tokenizer,
                new_token_ids,
                image_transform,
                dino_transform,
                images=imgs,
                prompt=conversation,
                max_length=200,
            )

            print(f"    -> {answer}")

            results.append({
                "scene_id": scene_id,
                "capture_type": scene.get("capture_type"),
                "location": scene.get("location"),
                "prompt_id": prompt_id,
                "question": question,
                "model_answer": answer,
                "ground_truth": gt,
            })

            save_results(args.output_file, metadata, results)

    print(f"\nDone. {len(results)} results saved to {args.output_file}")
