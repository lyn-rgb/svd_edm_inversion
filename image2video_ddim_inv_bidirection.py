import os
import sys
sys.path.append("../")

from tqdm import tqdm
import argparse

import numpy as np

import torch

from pipeline_stable_video_diffusion_video2video_bidirection import StableVideoDiffusionVideo2VideoPipeline
from diffusers.utils import load_image, export_to_video
from torchvision.io import write_video


parser = argparse.ArgumentParser()
parser.add_argument("--model_id_or_path", type=str, default="/cfs/zhlin/projects/aigc_engine/pretrained_models/stable-video-diffusion-img2vid-xt-1-1")
parser.add_argument("--image_start", type=str, required=True)
parser.add_argument("--image_end", type=str, required=True)
parser.add_argument("--frames", type=str, default="/cfs/zhlin/projects/aigc_engine/samples/wolf")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--fps", type=int, default=7)
parser.add_argument("--nframes", type=int, default=14)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--strength", type=float, default=1.0)
parser.add_argument("--save_path", type=str, default="/cfs/zhlin/projects/aigc_engine/demos/results")
args = parser.parse_args()

if __name__ == "__main__":
    # load pipeline
    pipe = StableVideoDiffusionVideo2VideoPipeline.from_pretrained(
        args.model_id_or_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # load the conditioning images
    image_start = load_image(args.image_start)
    image_start = image_start.resize((args.width, args.height))
    
    image_end = load_image(args.image_end)
    image_end = image_end.resize((args.width, args.height))

    # load the original frames 
    frames = []
    frame_list = sorted(os.listdir(args.frames))
    frame_list = frame_list[:args.nframes]
    for frame_name in tqdm(frame_list, desc="loading frames", colour="green"):
        frame_path = os.path.join(args.frames, frame_name)
        frame = load_image(frame_path)
        frame = frame.resize((args.width, args.height))
        frames.append(frame)

    # ddim inversion
    generator = torch.manual_seed(123)
    
    # get vae reconstruction as a reference
    print(f"Get VAE reconstruction result ...")
    vae_rec_frames = pipe.vae_reconstruction(
        frames = frames,
        height = args.height,
        width  = args.width,
        decode_chunk_size = 4,
    ).frames[0]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    vae_rec_path = os.path.join(args.save_path, "vae_rec.mp4")
    vae_rec_frames = [torch.from_numpy(np.array(rec_frame)) for rec_frame in vae_rec_frames]
    vae_rec_frames = torch.stack(vae_rec_frames)
    write_video(vae_rec_path, vae_rec_frames, fps=args.fps)
    print(f"VAE rec result has been saved into {vae_rec_path}")
    
    # 1st forward pass
    pipe.register_spatial_attention_injection(1.0)
    latents_list, image_embeddings, image_latents, added_time_ids, needs_upcasting, num_frames = pipe.edm_inversion(
        frames=frames,
        height= args.height,
        width=args.width,
        fps=args.fps, 
        num_videos_per_prompt=1,
        motion_bucket_id=127,
        noise_aug_strength=0.00,
        num_inference_steps=args.steps,
        strength=args.strength,
        timesteps_to_save=None,
        generator=generator,
    )
 
    # reconstruction
    rec_frames = pipe.edm_sample(
        latents=latents_list[-1],
        image_embeddings=image_embeddings,
        image_latents=image_latents,    
        added_time_ids=added_time_ids,
        needs_upcasting=needs_upcasting,
        num_frames=num_frames,
        num_inference_steps=args.steps,
        strength=args.strength,
        decode_chunk_size=8,
    ).frames

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    forward_rec_path = os.path.join(args.save_path, "rec.mp4")
    forward_rec_frames = [torch.from_numpy(np.array(rec_frame)) for rec_frame in rec_frames[0]]
    forward_rec_frames = torch.stack(forward_rec_frames)
    
    write_video(forward_rec_path, forward_rec_frames, fps=args.fps)
    print(f"Rec result has been saved into {forward_rec_path}")
    
    inverse_rec_path = os.path.join(args.save_path, "inv_rec.mp4")
    inverse_rec_frames = [torch.from_numpy(np.array(rec_frame)) for rec_frame in rec_frames[-1]]
    inverse_rec_frames = torch.stack(inverse_rec_frames)
    
    write_video(inverse_rec_path, inverse_rec_frames, fps=args.fps)
    print(f"Inversed rec result has been saved into {inverse_rec_path}")
    
    # edit with spatial-temporal attention injection
    pipe.register_spatialtemporal_attention_injection(1.0)
    edit_frames = pipe.edit_injection(
        image                = image_start,
        latents              = latents_list[-1][:1],
        inv_latents          = latents_list,
        src_image_embeddings = image_embeddings,
        src_image_latents    = image_latents,
        frames               = frames,
        strength             = args.strength,
        height               = args.height,
        width                = args.width,
        num_frames           = num_frames,
        num_inference_steps  = args.steps,
        min_guidance_scale   = 1.0,
        max_guidance_scale   = 3.0,
        fps                  = args.fps,
        motion_bucket_id     = 127,
    ).frames[0]
    
    edit_path = os.path.join(args.save_path, "edited.mp4")
    edit_frames = [torch.from_numpy(np.array(edit_frame)) for edit_frame in edit_frames]
    edit_frames = torch.stack(edit_frames)
    write_video(edit_path, edit_frames, fps=args.fps)
    print(f"Finish. Edit result has been saved into {edit_path}")

