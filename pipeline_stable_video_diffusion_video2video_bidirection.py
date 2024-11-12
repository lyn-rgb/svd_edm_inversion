#import sys
#sys.path.append("../../")

import PIL 
from PIL import Image
from typing import Callable, Dict, List, Optional, Union
from tqdm import tqdm
import inspect
import torch

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline, 
    _append_dims,
    StableVideoDiffusionPipelineOutput, 
    tensor2vid
)
#from diffusers.schedulers import DDIMScheduler
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def register_time(model, t):
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].temporal_transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].temporal_transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].temporal_transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_temporal_attention_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward
    
    for _, module in model.unet.named_modules():
        if isinstance_str(module, "TemporalBasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])
        
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].temporal_transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class IdentityAttnProcessor:
    r"""
    Processor for implementing Identity.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        #hidden_states = attn.to_v(encoder_hidden_states)
        hidden_states = encoder_hidden_states
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttnInjectProcessor2_0:
    r"""
    Processor for implementing injected scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnInjectProcessor2_0o0k1o requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        # batch_size is set as 2 for rec and edit routes sepearately
        #print(f"batch_size: {batch_size}")
        #assert(batch_size == 2)
        nframes = batch_size // 2
        
        # construct attention mask
        # q_mask = torch.ones((batch_size, sequence_length), dtype=hidden_states.dtype, device=hidden_states.device)
        # nframes = batch_size // 2
        # k_mask = torch.linspace(0, 1, nframes, dtype=hidden_states.dtype, device=hidden_states.device)
        # k_mask = torch.stack([1 - k_mask, k_mask])                       # 2, batch_size // 2
        # k_mask = k_mask.unsqueeze(-1).expand(-1, -1, sequence_length)    # 2, batch_size // 2, sequence_length
        # q_mask = q_mask.view(2, batch_size // 2, sequence_length).transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, 1)
        # k_mask = k_mask.transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, 1)
        # attention_mask = torch.matmul(q_mask, k_mask.transpose(1, 2)).unsqueeze(1)   # (batch, 1, sequence_length, sequence_length)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # inject srouce attention
        #query = query.view(2, batch_size // 2, sequence_length, -1)
        query[1*nframes:2*nframes] = query[:1*nframes]
        #query = query.view(batch_size, sequence_length, -1)
        
        #key = key.view(2, batch_size // 2, sequence_length, -1)
        key[1*nframes:2*nframes] = key[:1*nframes]
        #key = key.view(batch_size, sequence_length, -1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

def register_spatialtemporal_attention_efficient(model, injection_schedule):    
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         #module.attn1.forward = sa_forward(module.attn1)
    #         module.attn1.set_processor(AttnInjectProcessor2_0())
    #         setattr(module.attn1, 'injection_schedule', [])
        
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            #set spatial attention processer
            spatial_module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            spatial_module.set_processor(AttnInjectProcessor2_0())
            setattr(spatial_module, 'injection_schedule', injection_schedule)
            #set temporal attention processer
            temporal_module = model.unet.up_blocks[res].attentions[block].temporal_transformer_blocks[0].attn1
            temporal_module.set_processor(AttnInjectProcessor2_0())
            setattr(temporal_module, 'injection_schedule', injection_schedule)
            
    # res_dict = {3: [2]}
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.set_processor(IdentityAttnProcessor())
    #         setattr(module, 'injection_schedule', injection_schedule)

            
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # construct attention mask
        q_mask = torch.ones((batch_size, sequence_length), dtype=hidden_states.dtype, device=hidden_states.device)
        nframes = batch_size // 2
        k_mask = torch.linspace(0, 1, nframes, dtype=hidden_states.dtype, device=hidden_states.device)
        k_mask = torch.stack([1 - k_mask, k_mask])                       # 2, batch_size // 2
        k_mask = k_mask.unsqueeze(-1).expand(-1, -1, sequence_length)    # 2, batch_size // 2, sequence_length
        q_mask = q_mask.view(2, batch_size // 2, sequence_length).transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, 1)
        k_mask = k_mask.transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, 1)
        attention_mask = torch.matmul(q_mask, k_mask.transpose(1, 2)).unsqueeze(1)   # (batch, 1, sequence_length, sequence_length)
        
        # modify 
        forward_hidden_states, inverse_hidden_states = hidden_states.chunk(2)
        inverse_hidden_states = inverse_hidden_states.flip(0)
        hidden_states = torch.cat([forward_hidden_states, inverse_hidden_states])
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        hidden_states = hidden_states.view(2, batch_size // 2, sequence_length, -1).transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, -1)
        encoder_hidden_states = encoder_hidden_states.view(2, batch_size // 2, sequence_length, -1).transpose(0, 1).reshape(batch_size // 2, sequence_length * 2, -1)
        batch_size, sequence_length = batch_size // 2, sequence_length * 2
        #print(f"batch_size, sequence_length = {batch_size}, {sequence_length}")
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # modify
        hidden_states = hidden_states.view(batch_size, 2, sequence_length // 2, -1).transpose(0, 1).reshape(batch_size * 2, sequence_length // 2, -1)
        batch_size, sequence_length = batch_size * 2, sequence_length // 2
        
        forward_hidden_states, inverse_hidden_states = hidden_states.chunk(2)
        inverse_hidden_states = inverse_hidden_states.flip(0)
        hidden_states = torch.cat([forward_hidden_states, inverse_hidden_states])
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        #print(f"using modified attention processer")
        return hidden_states


def register_spatial_attention_efficient(model, injection_schedule):    
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         #module.attn1.forward = sa_forward(module.attn1)
    #         module.attn1.set_processor(AttnProcessor2_0())
    #         setattr(module.attn1, 'injection_schedule', [])
        
    # res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.set_processor(AttnProcessor2_0())
    #         #setattr(module, 'injection_schedule', injection_schedule)
            
    # res_dict = {3: [2]}
    # for res in res_dict:
    #     for block in res_dict[res]:
    #         module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         module.set_processor(IdentityAttnProcessor())
    #         setattr(module, 'injection_schedule', injection_schedule)
    pass
    

class StableVideoDiffusionVideo2VideoPipeline(StableVideoDiffusionPipeline):

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    
    def reset_scheduler(
        self
    ):
        self.scheduler = self.backup_scheduler

    @torch.no_grad()
    def encode_frames(
        self, 
        frames: Union[List[Image.Image], torch.FloatTensor],
        dtype: torch.dtype,
        device: torch.device,
        height: int=1024,
        width: int=576,
        
    ):
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        init_latents = []
        if isinstance(frames[0], Image.Image):
            for frame in tqdm(frames, desc="encoding frames"):
                frame = self.image_processor.preprocess(frame, height=height, width=width)
                init_latent = self._encode_vae_image(frame, device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
                init_latents.append(init_latent)
            init_latents = torch.cat(init_latents, dim=0)
        else:
            frames = self.image_processor.preprocess(frames, height=height, width=width)
            init_latents = self._encode_vae_image(frames, device, num_videos_per_prompt=1, do_classifier_free_guidance=False)

        init_latents = init_latents.to(dtype)
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents.unsqueeze_(0)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        return init_latents
    
    @torch.no_grad()
    def decode_frames(
        self,
        latents: torch.FloatTensor,
        decode_chunk_size: int = 8,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            num_frames = latents.shape[1]
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
    
    @torch.no_grad()
    def add_noise(
        self,
        init_latents: torch.FloatTensor,
        strength: float=1.0,
        num_inference_steps: int=20,
        num_images_per_prompt: int=1,
        generator:  Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=init_latents.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, init_latents.device)
        latent_timestep = timesteps[:1].repeat(init_latents.shape[0] * num_images_per_prompt)

        print(latent_timestep)
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=init_latents.device, dtype=init_latents.dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents
        return latents
    
    def prepare_latents(
        self,
        timestep,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        frames=None
    ):
        init_latents = None
        if frames is not None:
            if isinstance(frames, list):
                num_frames = len(frames)
            else:
                num_frames = frames.shape[0]
            
            init_latents = self.encode_frames(
                frames, 
                dtype=dtype,
                device=device,
                height=height,
                width=width
            )
            if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                # expand init_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)
      
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if init_latents is None:
                # from a pure noise, for image2video
                latents = latents * self.scheduler.init_noise_sigma
            else:
                # from latents + noise, for video2video
                latents = self.scheduler.add_noise(init_latents, latents, timestep)
        else:
            # for video2video
            latents = latents.to(device)
        
        return latents

    def scale_model_input(
        self, sample: torch.FloatTensor, step_index: Union[int, torch.LongTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """

        sigma = self.scheduler.sigmas[step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        return sample

    def register_temporal_attention_injection(self, pnp_attn_t):
        self.n_timesteps = len(self.scheduler.timesteps)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.qk_injection_timesteps = self.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
        register_temporal_attention_efficient(self, self.qk_injection_timesteps)
    
    def register_spatialtemporal_attention_injection(self, pnp_attn_t):
        self.n_timesteps = len(self.scheduler.timesteps)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.qk_injection_timesteps = self.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
        register_spatialtemporal_attention_efficient(self, self.qk_injection_timesteps)
    
    def register_spatial_attention_injection(self, pnp_attn_t):
        self.n_timesteps = len(self.scheduler.timesteps)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.qk_injection_timesteps = self.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
        register_spatial_attention_efficient(self, self.qk_injection_timesteps)

    def inversion_step(
        self,
        model_output: torch.FloatTensor,
        step_index: Union[int, torch.LongTensor],
        prev_sample: torch.FloatTensor,
    ):
        
        # Upcast to avoid precision issues when computing sample
        prev_sample = prev_sample.to(torch.float32)        # x_t
        prev_sigma = self.scheduler.sigmas[step_index + 1] # sigma_t

        sigma = self.scheduler.sigmas[step_index]      # sigma_{t + 1}
        #print(f"step index {step_index}, prev sigma {prev_sigma}, sigma {sigma}")

        c_skip = 1 / (sigma**2 + 1)
        c_out = (-sigma /(sigma**2 + 1) ** 0.5)
        dt = prev_sigma - sigma
        sample = (prev_sample * sigma + c_out * model_output * dt) / ((1 - c_skip) * dt + sigma)

        #sample = (prev_sample * (sigma**2 + 1) - model_output * (prev_sigma - sigma) * ((sigma**2 + 1)**0.5)) / (prev_sigma * sigma)

        # Cast sample back to model compatible dtype
        sample = sample.to(model_output.dtype)

        return sample
    
    @torch.no_grad()
    def vae_reconstruction(
        self,
        frames: List[Image.Image],
        height: int = 576,
        width: int = 1024,
        decode_chunk_size: int = 8,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        if isinstance(frames, list):
            num_frames = len(frames)
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(frames, height, width)
        
        # 2. get vae latents
        device = self._execution_device
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        latents = self.encode_frames(
            frames, torch.float16, device, height=height, width=width
        )
        # 3. decode to videos
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
    
    @torch.no_grad()
    def edm_inversion(
        self, 
        frames: List[Image.Image],
        height: int = 576,
        width: int = 1024,
        fps: int = 7, 
        num_videos_per_prompt: int = 1,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        num_inference_steps: int = 20,
        strength: float = 1.0,
        timesteps_to_save: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        if isinstance(frames, list):
            num_frames = len(frames)
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(frames, height, width)

        # 2. Define call parameters
        batch_size = 2
        
        device = self._execution_device

        # 3. Encode input image
        # 3.0 forward image
        forward_image_embeddings = self._encode_image(frames[0], device, num_videos_per_prompt, do_classifier_free_guidance=False)
        # 3.1 inverse image
        inverse_image_embeddings = self._encode_image(frames[-1], device, num_videos_per_prompt, do_classifier_free_guidance=False)
        # 3.2 batch image
        image_embeddings = torch.cat([forward_image_embeddings, inverse_image_embeddings])

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        # 4.1 forward image
        forward_image = self.image_processor.preprocess(frames[0], height=height, width=width)
        forward_noise = randn_tensor(forward_image.shape, generator=generator, device=forward_image.device, dtype=forward_image.dtype)
        forward_image = forward_image + noise_aug_strength * forward_noise
        # 4.2 inverse image
        inverse_image = self.image_processor.preprocess(frames[-1], height=height, width=width)
        inverse_noise = randn_tensor(inverse_image.shape, generator=generator, device=inverse_image.device, dtype=inverse_image.dtype)
        inverse_image = inverse_image + noise_aug_strength * inverse_noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        # 4.3 forward image
        forward_image_latents = self._encode_vae_image(forward_image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
        forward_image_latents = forward_image_latents.to(forward_image_embeddings.dtype)
        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        forward_image_latents = forward_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        # 4.4 inverse image
        inverse_image_latents = self._encode_vae_image(inverse_image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
        inverse_image_latents = inverse_image_latents.to(inverse_image_embeddings.dtype)
        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        inverse_image_latents = inverse_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        # 4.5 batch image
        image_latents = torch.cat([forward_image_latents, inverse_image_latents])

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            forward_image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance=False,
        )
        added_time_ids = added_time_ids.to(device)
        
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        #timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        start_step_index = self.scheduler.index_for_timestep(timesteps[0])
        step_indices = [start_step_index + i for i in range(len(timesteps))]

        # 7. Prepare latent variables
        # encode frames into latents with vae encoder
        forward_init_latents = self.encode_frames(
            frames, image_latents.dtype, device, height=height, width=width
        )
        inverse_init_latents = forward_init_latents.flip(1)
        init_latents = torch.cat([forward_init_latents, inverse_init_latents])
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 8. EDM Inv
        latents = init_latents
        timesteps = reversed(timesteps)
        step_indices = list(reversed(step_indices))
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        latents_list = [init_latents]
        for i, t in enumerate(tqdm(timesteps, desc="EDM Inv", colour="red")):
            # Concatenate image_latents over channels dimention
            latent_model_input = latents
            latent_model_input = self.scale_model_input(latent_model_input, step_indices[i])
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids
            ).sample
            
            latents = self.inversion_step(noise_pred, step_indices[i], latents)

            latents_list.append(latents)

        return latents_list, image_embeddings, image_latents, added_time_ids, needs_upcasting, num_frames
    
    def sample_step(
        self,
        model_output: torch.FloatTensor,
        step_index: Union[int, torch.LongTensor],
        sample: torch.FloatTensor,
    ):
        #print(f"modified sampler, step_index {step_index}")

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        sigma = self.scheduler.sigmas[step_index]                 # t + 1
        sigma_hat = sigma

        # v_prediction
        # denoised = model_output * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + sample * (1 / (sigma**2 + 1))
        
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = self.scheduler.sigmas[step_index + 1] - sigma_hat     # t
        #print(f"step index {step_index}, prev sigma {self.scheduler.sigmas[step_index + 1]}, sigma {sigma_hat}")
        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
 
        return (prev_sample, pred_original_sample)
        
    @torch.no_grad()
    def edm_sample(
        self, 
        latents: torch.FloatTensor, 
        image_embeddings: torch.FloatTensor, 
        image_latents: torch.FloatTensor, 
        added_time_ids: torch.FloatTensor,
        needs_upcasting: bool,
        num_frames: int,
        num_inference_steps: int = 20,
        strength: float = 1.0,
        decode_chunk_size: int = 8,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        #timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, "cuda")
        
        start_step_index = self.scheduler.index_for_timestep(timesteps[0])
        step_indices = [start_step_index + i for i in range(len(timesteps))]
        for i, t in enumerate(tqdm(timesteps, desc="Rec", colour="blue")):
            latent_model_input = latents
            latent_model_input = self.scale_model_input(latent_model_input, step_indices[i])
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids
            ).sample
            
            latents = self.sample_step(noise_pred, step_indices[i], latents)[0]

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    @torch.no_grad()
    def edit(
        self,
        images: List[Image.Image],
        latents: Optional[torch.FloatTensor],
        strength: float = 0.8,
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = 8,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        assert(len(images) == 2)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        
        batch_size, num_frames = latents.shape[:2]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        # 1. Define call parameters
        device = self._execution_device
        self._guidance_scale = max_guidance_scale
        
        # 2. Encode input image for cross attention 
        forward_image_embeddings = self._encode_image(images[0], device, num_videos_per_prompt, False)
        inverse_image_embeddings = self._encode_image(images[1], device, num_videos_per_prompt, False)

        image_embeddings = torch.cat([forward_image_embeddings, inverse_image_embeddings])
        uncond_image_embeddings = torch.zeros_like(image_embeddings)
        
        # 3. Encode input image using VAE
        forward_image = self.image_processor.preprocess(images[0], height=height, width=width)
        forward_noise = randn_tensor(forward_image.shape, generator=generator, device=forward_image.device, dtype=forward_image.dtype)
        forward_image = forward_image + noise_aug_strength * forward_noise
        
        inverse_image = self.image_processor.preprocess(images[1], height=height, width=width)
        inverse_noise = randn_tensor(inverse_image.shape, generator=generator, device=inverse_image.device, dtype=inverse_image.dtype)
        inverse_image = inverse_image + noise_aug_strength * inverse_noise
        
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            
        forward_image_latents = self._encode_vae_image(forward_image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
        forward_image_latents = forward_image_latents.to(image_embeddings.dtype)
        forward_image_latents = forward_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        inverse_image_latents = self._encode_vae_image(inverse_image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
        inverse_image_latents = inverse_image_latents.to(image_embeddings.dtype)
        inverse_image_latents = inverse_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        image_latents = torch.cat([forward_image_latents, inverse_image_latents])
        uncond_image_latents = torch.zeros_like(image_latents)
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # 4. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            forward_image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            False,
        )
        added_time_ids = added_time_ids.to(device)
        uncond_added_time_ids = added_time_ids
        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, "cuda")
        start_step_index = self.scheduler.index_for_timestep(timesteps[0])
        step_indices = [start_step_index + i for i in range(len(timesteps))]
        
        # 6. Prepare latent variables
        init_latents = latents
        
        # 7. Prepare guidance scale
        if max_guidance_scale > min_guidance_scale:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        else:
            guidance_scale = torch.ones(num_frames).unsqueeze(0) * min_guidance_scale
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
            
        self._guidance_scale = guidance_scale
        
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        for i, t in enumerate(tqdm(timesteps, desc="Edit", colour="yellow")):
            # 8.1 condition 
            latent_model_input = latents
            latent_model_input = self.scale_model_input(latent_model_input, step_indices[i])
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            
            noise_pred_cond = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids
            )[0]
            
            # 8.2 uncondition
            if self.do_classifier_free_guidance and max_guidance_scale > min_guidance_scale:
                uncond_latent_model_input = latents
                uncond_latent_model_input = self.scale_model_input(uncond_latent_model_input, step_indices[i])
                uncond_latent_model_input = torch.cat([uncond_latent_model_input, uncond_image_latents], dim=2)
                
                noise_pred_uncond = self.unet(
                    uncond_latent_model_input, 
                    t, 
                    encoder_hidden_states=uncond_image_embeddings,
                    added_time_ids=uncond_added_time_ids
                )[0]
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            # 8.3 compute the previous noisy sample x_t -> x_t-1
            latents = self.sample_step(noise_pred, step_indices[i], latents)[0]

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
    
    @torch.no_grad()
    def edit_injection(
        self,
        image: Image.Image,
        inv_latents: Optional[List[torch.FloatTensor]],
        src_image_embeddings: torch.FloatTensor, 
        src_image_latents: torch.FloatTensor, 
        frames: List[Image.Image] = None,
        latents: Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        height: int = 576,
        width: int = 1024,
        num_frames: int = 25,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = 8,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        #assert (frames is not None or latents is not None)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        if latents is not None:
            batch_size, num_frames = latents.shape[:2]
        elif frames is not None:
            batch_size = 1
            num_frames = len(frames)
        else:
            batch_size = 1
            
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        # 1. Define call parameters
        device = self._execution_device
        self._guidance_scale = max_guidance_scale
        
        # 2. Encode input image for cross attention 
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, False)
        
        uncond_image_embeddings = torch.zeros_like(image_embeddings)
        # only remain the forward features
        src_image_embeddings = src_image_embeddings[:1]
        image_embeddings = torch.cat([src_image_embeddings, image_embeddings])
        uncond_image_embeddings = torch.cat([src_image_embeddings, uncond_image_embeddings])
        
        # 3. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise
        
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            
        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance=False)
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        uncond_image_latents = torch.zeros_like(image_latents)
        
        # only remain the forward features
        src_image_latents = src_image_latents[:1]
        image_latents = torch.cat([src_image_latents, image_latents])
        uncond_image_latents = torch.cat([src_image_latents, uncond_image_latents])
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # 4. Get Added Time IDs, batch_size is set as 2
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            2,
            num_videos_per_prompt,
            False,
        )
        added_time_ids = added_time_ids.to(device)
        uncond_added_time_ids = added_time_ids
        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, "cuda")
        start_step_index = self.scheduler.index_for_timestep(timesteps[0])
        step_indices = [start_step_index + i for i in range(len(timesteps))]
        
        # 6. Prepare latent variables
        if latents is None:
            latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                latent_timestep,
                batch_size * num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                height,
                width,
                image_embeddings.dtype,
                device,
                generator,
                latents,
                frames=frames,
            )
        
        # 7. Prepare guidance scale
        if max_guidance_scale > min_guidance_scale:
            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        else:
            guidance_scale = torch.ones(num_frames).unsqueeze(0) * min_guidance_scale
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
            
        self._guidance_scale = guidance_scale
        
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        for i, t in enumerate(tqdm(timesteps, desc="Edit", colour="yellow")):
            # only remain the forward features
            inv_latent = inv_latents[-i-1][:1]
            # 8.1 condition 
            latent_model_input = latents
            latent_model_input = torch.cat([inv_latent, latent_model_input])
            latent_model_input = self.scale_model_input(latent_model_input, step_indices[i])
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            
            noise_pred_cond = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids
            )[0]
            
            # 8.2 uncondition
            if self.do_classifier_free_guidance and max_guidance_scale > min_guidance_scale:
                uncond_latent_model_input = latents
                uncond_latent_model_input = torch.cat([inv_latent, uncond_latent_model_input])
                uncond_latent_model_input = self.scale_model_input(uncond_latent_model_input, step_indices[i])
                uncond_latent_model_input = torch.cat([uncond_latent_model_input, uncond_image_latents], dim=2)
                
                noise_pred_uncond = self.unet(
                    uncond_latent_model_input, 
                    t, 
                    encoder_hidden_states=uncond_image_embeddings,
                    added_time_ids=uncond_added_time_ids
                )[0]
                _, noise_pred_cond = noise_pred_cond.chunk(2)
                _, noise_pred_uncond = noise_pred_uncond.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                _, noise_pred = noise_pred_cond.chunk(2)
            # 8.3 compute the previous noisy sample x_t -> x_t-1
            latents = self.sample_step(noise_pred, step_indices[i], latents)[0]

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        frames: Optional[Union[List[Image.Image], torch.FloatTensor]] = None,
        strength: float = 0.8,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        src_image: Optional[Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor]]=None,
        inv_latents: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)
        if frames is not None:
            if isinstance(frames, list):
                num_frames = len(frames)
            else:
                num_frames = frames.shape[0]
            self.check_inputs(frames, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        src_image_embeddings = self._encode_image(src_image, device, num_videos_per_prompt, False)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        src_image = self.image_processor.preprocess(src_image, height=height, width=width)
        src_noise = randn_tensor(src_image.shape, generator=generator, device=src_image.device, dtype=src_image.dtype)
        src_image = src_image + noise_aug_strength * src_noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        image_latents = image_latents.to(image_embeddings.dtype)
        src_image_latents = self._encode_vae_image(src_image, device, num_videos_per_prompt, False)
        src_image_latents = src_image_latents.to(src_image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        src_image_latents = src_image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        start_step_index = self.scheduler.index_for_timestep(timesteps[0])
        step_indices = [start_step_index + i for i in range(len(timesteps))]
        #print(f"length of sigmas in scheduler {len(self.scheduler.sigmas)}")
        #print(f"step_indices {step_indices}")
        #print(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            latent_timestep,
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
            frames=frames,
        )

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        use_inv_latents = False
        if inv_latents is not None:
            use_inv_latents = True
            print(f"src image embeddings shape: {src_image_embeddings.shape}, image embeddings shape: {image_embeddings.shape}")
            print(f"src image embeddings shape: {src_image_latents.shape}, image embeddings shape: {image_latents.shape}")
            src_added_time_ids, _ = added_time_ids.chunk(2)
            added_time_ids = torch.cat([src_added_time_ids, added_time_ids], dim=0)
            image_embeddings = torch.cat([src_image_embeddings, image_embeddings], dim=0)
            image_latents = torch.cat([src_image_latents, image_latents], dim=0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                register_time(self, t.item())
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                if use_inv_latents:
                    latent_model_input = torch.cat([inv_latents[-i-1], latent_model_input], dim=0)

                #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = self.scale_model_input(latent_model_input, step_indices[i])

                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    if use_inv_latents:
                        noise_pred_src, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
                    else:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred_src, noise_pred = noise_pred.chunk(2)

                # compute the previous noisy sample x_t -> x_t-1
                #latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                latents = self.sample_step(noise_pred, step_indices[i], latents)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)