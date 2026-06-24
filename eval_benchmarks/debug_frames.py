"""Diagnostic: does G2VLM degenerate as #frames grows? Run one MCA + one NA
sample at several frame counts and print raw outputs."""
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
    itf = build_transform(pixel=768, model_path=A.model_path)

    for doc in (na, mca):
        print("\n" + "=" * 80)
        print(f"id={doc['id']} type={doc['question_type']} gt={doc['ground_truth']} mc={doc['mc_answer']}")
        prompt = build_prompt(doc, "vsi")
        print("PROMPT:", repr(prompt[:200]))
        for nf in (1, 4, 8, 16, 32):
            frames = load_frames(doc["video_path"], nf)
            imgs, conv = process_conversation(frames, prompt)
            out = model.chat_with_recon(tok, ntk, itf, dino, images=imgs,
                                        prompt=conv, max_length=48, do_sample=False)
            print(f"  nframes={nf:>2} -> {out!r}")


if __name__ == "__main__":
    main()
