# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import str2bool

EXAMPLE_PROMPT = {
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert "ti2v" in args.task, f"Only ti2v tasks are supported, current task: {args.task}"
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    if args.prompt_file is None and args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./syn_latents",
        help="The directory to save the generated latents.")
    parser.add_argument(
        "--save_latents_only",
        action="store_true",
        default=False,
        help="Whether to save the generated latents only.")
    parser.add_argument(
        "--task",
        type=str,
        default="ti2v-5B",
        choices=["ti2v-5B"],
        help="The task to run (only ti2v-5B supported).")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from (used when prompt_file is not specified).")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="The file containing prompts, one per line.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def save_latents_to_parquet_batch(prompts: list, vae_latent_list: list, prompt_embeds_list: list, 
                                 prompt_attention_masks_list: list, output_dir: str, chunk_idx: int):
    """Save a batch of prompts and their corresponding latents, text embeddings, and attention masks to Parquet file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare batch data
    batch_data = []
    
    for prompt, vae_latent, prompt_embed, prompt_attention_mask in zip(prompts, vae_latent_list, prompt_embeds_list, prompt_attention_masks_list):
        # Convert tensors to numpy arrays
        vae_latent_np = vae_latent.cpu().numpy()
        prompt_embed_np = prompt_embed.cpu().numpy()
        prompt_attention_mask_np = prompt_attention_mask.cpu().numpy()
        
        record = {
            "caption": prompt,
            "vae_latent_bytes": vae_latent_np.tobytes(),
            "vae_latent_shape": list(vae_latent_np.shape),
            "vae_latent_dtype": str(vae_latent_np.dtype),
            "text_embedding_bytes": prompt_embed_np.tobytes(),
            "text_embedding_shape": list(prompt_embed_np.shape),
            "text_embedding_dtype": str(prompt_embed_np.dtype),
            "text_attention_mask_bytes": prompt_attention_mask_np.tobytes(),
            "text_attention_mask_shape": list(prompt_attention_mask_np.shape),
            "text_attention_mask_dtype": str(prompt_attention_mask_np.dtype)
        }
        batch_data.append(record)
    
    # Convert batch data to PyArrow arrays
    arrays = [
        pa.array([record["caption"] for record in batch_data]),
        pa.array([record["vae_latent_bytes"] for record in batch_data], type=pa.binary()),
        pa.array([record["vae_latent_shape"] for record in batch_data], type=pa.list_(pa.int64())),
        pa.array([record["vae_latent_dtype"] for record in batch_data]),
        pa.array([record["text_embedding_bytes"] for record in batch_data], type=pa.binary()),
        pa.array([record["text_embedding_shape"] for record in batch_data], type=pa.list_(pa.int64())),
        pa.array([record["text_embedding_dtype"] for record in batch_data]),
        pa.array([record["text_attention_mask_bytes"] for record in batch_data], type=pa.binary()),
        pa.array([record["text_attention_mask_shape"] for record in batch_data], type=pa.list_(pa.int64())),
        pa.array([record["text_attention_mask_dtype"] for record in batch_data])
    ]
    
    # Define schema
    schema = pa.schema([
        ("caption", pa.string()),
        ("vae_latent_bytes", pa.binary()),
        ("vae_latent_shape", pa.list_(pa.int64())),
        ("vae_latent_dtype", pa.string()),
        ("text_embedding_bytes", pa.binary()),
        ("text_embedding_shape", pa.list_(pa.int64())),
        ("text_embedding_dtype", pa.string()),
        ("text_attention_mask_bytes", pa.binary()),
        ("text_attention_mask_shape", pa.list_(pa.int64())),
        ("text_attention_mask_dtype", pa.string())
    ])
    
    # Create table
    table = pa.Table.from_arrays(arrays, schema=schema)
    
    # Generate filename with chunk index
    parquet_file = os.path.join(output_dir, f"latents_chunk_{chunk_idx:04d}.parquet")
    
    # Write to Parquet file with no compression
    pq.write_table(table, parquet_file, compression='none')
    
    logging.info(f"Saved batch of {len(prompts)} latents to {parquet_file}")
    return parquet_file





def generate_single_prompt(args, prompt, img, rank, device, cfg, prompt_expander=None):
    """Generate video for a single prompt using ti2v."""
    # Apply prompt extension if enabled
    if args.use_prompt_extend and prompt_expander is not None:
        logging.info(f"Extending prompt: {prompt}")
        prompt_output = prompt_expander(
            prompt,
            image=img,
            tar_lang=args.prompt_extend_target_lang,
            seed=args.base_seed)
        if prompt_output.status == False:
            logging.info(f"Extending prompt failed: {prompt_output.message}")
            logging.info("Falling back to original prompt.")
            prompt = prompt
        else:
            prompt = prompt_output.prompt
        logging.info(f"Extended prompt: {prompt}")
    
    # Generate video using ti2v
    logging.info("Creating WanTI2V pipeline.")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    logging.info(f"Generating video for prompt: {prompt}")
    video, prompt_embed = wan_ti2v.generate(
        prompt,
        img=img,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        save_latents_only=args.save_latents_only)
    
    return video, prompt_embed, prompt


def generate(args):
    device = 0  # Single GPU
    _init_logging(0)

    if args.offload_model is None:
        args.offload_model = True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    # Load prompts from file or use single prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f.readlines() if line.strip()]
        logging.info(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}")
    else:
        if args.prompt is None:
            raise ValueError("Either --prompt or --prompt_file must be specified")
        all_prompts = [args.prompt]
        logging.info(f"Using single prompt: {args.prompt}")

    # Check cache for already processed prompts
    cache_file = os.path.join(args.save_dir, "processed_prompts.txt")
    processed_prompts = set()
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            processed_prompts = set(line.strip() for line in f.readlines() if line.strip())
        logging.info(f"Found cache file with {len(processed_prompts)} already processed prompts")
    
    # Filter out already processed prompts
    prompts = [prompt for prompt in all_prompts if prompt not in processed_prompts]
    skipped_count = len(all_prompts) - len(prompts)
    
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} already processed prompts")
    
    if len(prompts) == 0:
        logging.info("All prompts have been processed. Exiting.")
        return
    
    logging.info(f"Will process {len(prompts)} new prompts")

    # Load image if specified
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # Initialize prompt expander if needed
    prompt_expander = None
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=device)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    # Process prompts in batches
    logging.info(f"Processing {len(prompts)} prompts in batches of 8")
    
    # Initialize batch storage
    batch_size = 8
    current_batch_prompts = []
    current_batch_vae_latent = []
    current_batch_prompt_embeds = []
    current_batch_prompt_attention_masks = []
    chunk_idx = 0
    
    # Create progress bar
    pbar = tqdm(total=len(prompts), desc="Processing prompts", unit="prompt")
    
    for i, prompt in enumerate(prompts):
        try:
            logging.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
            
            # Generate for current prompt
            video, prompt_embed, processed_prompt = generate_single_prompt(
                args, prompt, img, 0, device, cfg, prompt_expander)
            
            # Prepare data for saving
            vae_latent = video.to(torch.float32)
            prompt_embed = prompt_embed.to(torch.float32)
            text_seq_len = prompt_embed.shape[0]
            prompt_attention_mask = torch.ones(text_seq_len).to(torch.long)
            
            # Add to current batch
            current_batch_prompts.append(processed_prompt)
            current_batch_vae_latent.append(vae_latent)
            current_batch_prompt_embeds.append(prompt_embed)
            current_batch_prompt_attention_masks.append(prompt_attention_mask)
            
            # Save batch if it's full or if it's the last prompt
            if len(current_batch_prompts) == batch_size or i == len(prompts) - 1:
                parquet_file = save_latents_to_parquet_batch(
                    current_batch_prompts,
                    current_batch_vae_latent,
                    current_batch_prompt_embeds,
                    current_batch_prompt_attention_masks,
                    args.save_dir,
                    chunk_idx
                )
                logging.info(f"Saved batch {chunk_idx} with {len(current_batch_prompts)} prompts to {parquet_file}")
                
                # Update cache with processed prompts (original prompts, not extended ones)
                with open(cache_file, 'a', encoding='utf-8') as f:
                    # Calculate the start index of this batch in the original prompts list
                    batch_start_idx = i - len(current_batch_prompts) + 1
                    for j in range(len(current_batch_prompts)):
                        original_prompt = prompts[batch_start_idx + j]
                        f.write(f"{original_prompt}\n")
                
                # Clear batch storage
                current_batch_prompts = []
                current_batch_vae_latent = []
                current_batch_prompt_embeds = []
                current_batch_prompt_attention_masks = []
                chunk_idx += 1
            
            # Clean up memory
            del video
            if 'prompt_embed' in locals():
                del prompt_embed
            torch.cuda.empty_cache()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "Current": f"{i+1}/{len(prompts)}",
                "Batch": f"{chunk_idx}",
                "Prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt
            })
            
        except Exception as e:
            logging.error(f"Error processing prompt '{prompt}': {str(e)}")
            pbar.update(1)  # Still update progress bar even if there's an error
            continue
    
    # Close progress bar
    pbar.close()
    
    logging.info(f"Finished processing all {len(prompts)} prompts")

    torch.cuda.synchronize()
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
