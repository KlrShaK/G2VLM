import argparse
import os
import re
import json
from PIL import Image
from tqdm import tqdm
import torch
from g2vlm_utils import load_model_and_tokenizer, build_transform, process_conversation

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='InternRobotics/G2VLM-2B-MoT')
    parser.add_argument('--image-path', type=str, default='examples/25_0.jpg')
    parser.add_argument('--question', type=str, default='')
    args = parser.parse_args()
    enable_template = True

    model, tokenizer, new_token_ids , vit_image_transform, dino_transform = load_model_and_tokenizer(args)
    image_transform = build_transform(pixel=768, model_path=args.model_path)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    img_path = 'examples/25_0.jpg'
    question = "If the table (red point) is positioned at 2.6 meters, estimate the depth of the clothes (blue point).  Calculate or judge based on the 3D center points of these objects. The unit is meter. Submit your response as one numeric value only."

    post_prompt = "Please answer the question using a single word or phrase."
    templated_question =  "\n" + question + "\n" + post_prompt

    if args.question is not None: 
        templated_question = args.question
    
    print(question)

    images = [Image.open(img_path).convert('RGB') ]
    images, conversation = process_conversation(images, templated_question)

    response = model.chat_with_recon(
        tokenizer, 
        new_token_ids,
        image_transform,
        dino_transform,
        images=images,
        prompt=conversation,
        max_length=100,
    )
    print('answer: ',response)
