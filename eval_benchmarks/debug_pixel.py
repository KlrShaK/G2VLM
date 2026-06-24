"""Diagnostic: at fixed 32 frames, does lowering per-frame resolution (fewer ViT
tokens/frame) restore coherent generation? vit_max_num_patch_per_side=36 -> max
pixel 504, so sweep 224/336/448."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2vlm_utils import build_transform, load_model_and_tokenizer, process_conversation
from eval_g2vlm_bench import build_prompt, load_frames


class A:
    model_path = "/cluster/work/igp_psr/spanwar/G2VLM/models/G2VLM-2B-MoT"


def main():
    docs = json.load(open(os.path.join(os.path.dirname(__file__), "vsi_random200.json")))
    mca = next(d for d in docs if d["options"])
    na = next(d for d in docs if not d["options"])

    model, tok, ntk, _vt, dino = load_model_and_tokenizer(A)
    NF = 32
    for doc in (na, mca):
        print("\n" + "=" * 80)
        print(f"id={doc['id']} type={doc['question_type']} gt={doc['ground_truth']} mc={doc['mc_answer']}")
        prompt = build_prompt(doc, "vsi")
        frames = load_frames(doc["video_path"], NF)
        for px in (224, 336, 448):
            itf = build_transform(pixel=px, model_path=A.model_path)
            imgs, conv = process_conversation(frames, prompt)
            out = model.chat_with_recon(tok, ntk, itf, dino, images=imgs,
                                        prompt=conv, max_length=48, do_sample=False)
            print(f"  pixel={px} (nframes={NF}) -> {out!r}")


if __name__ == "__main__":
    main()
