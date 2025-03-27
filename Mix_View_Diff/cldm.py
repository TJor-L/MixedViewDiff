import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.models as models

from cldm.mix_view_diff import MixedViewDiff

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Pretrained ResNet50 for satellite CNN
        resnet_satellite = models.resnet50(pretrained=True)
        self.satellite_cnn = nn.Sequential(
            *list(resnet_satellite.children())[:-2]  # Remove the FC layer and the avgpool
        )

        # Pretrained ResNet50 for structure CNN, adjusted for single-channel input
        resnet_structure = models.resnet50(pretrained=True)
        self.structure_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Adjust input channel to 1
            *list(resnet_structure.children())[1:-2]  # Skip the first conv layer
        )

        # Decoder adjusted for 1024-dimensional embeddings
        self.combined_hint_decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 2048),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, satellite_input, structure_input):
        # Forward pass through satellite CNN
        satellite_features = self.satellite_cnn(satellite_input)  # Shape: (batch_size, 2048, H/32, W/32)

        # Forward pass through structure CNN
        structure_features = self.structure_cnn(structure_input)  # Shape: (batch_size, 2048, H/32, W/32)

        # Combine the features
        combined_features = torch.cat([satellite_features, structure_features], dim=1)  # Shape: (batch_size, 4096, H/32, W/32)

        # Decode to final output
        decoded_output = self.combined_hint_decoder(combined_features)  # Shape: (batch_size, 3, 512, 512)
        return decoded_output

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        
        # self.resnet = CustomModel()

        self.mixview_diff = MixedViewDiff()
        
        self.satellite_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        )
        
        self.structure_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        )
        
        
        self.combined_hint_decoder = nn.Sequential(
                nn.GroupNorm(16, 128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Tanh()
        )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        # mixview_encoder = MixedViewEncoder()
        # decoder = Decoder(hidden_dim=64)
        # self.mixviewattn = MixedViewDiff(mixview_encoder, decoder)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        
        #TODO: parse hint tesnsor, apply mixview encoder
        #----------------------#
        #channels order
        # overhead * 3
        # target_structure * 1
        # target_position * 2
        # near_structures * 3
        # near_streetviews * 9
        # near_positions * 6
        # total 24 channels
        #----------------------#
        target_satellite = hint[:, :3,...]
        #print("target_satellite", target_satellite.shape)
        target_structure = hint[:, 3:4,...]
        #print("target_structure", target_structure.shape)
        target_pos = hint[:,4:6,...]
        #print("target_pos", target_pos.shape)
        surround_structure = hint[:, 6:9,...].unsqueeze(2)
        #print("surround_structure", surround_structure.shape)
        surround_panoramas1 = hint[:, 9:12,...].unsqueeze(1)
        surround_panoramas2 = hint[:, 12:15,...].unsqueeze(1)
        surround_panoramas3 = hint[:, 15:18,...].unsqueeze(1)
        surround_panoramas = th.cat([surround_panoramas1, surround_panoramas2, surround_panoramas3], dim=1)
        #print("surround_panoramas", surround_panoramas.shape)
        surround_pos = hint[:, 18:,...]
        surround_pos=surround_pos[:, :, 0:1, 0:1]
        #print("surround_pos", surround_pos.shape)
        # exit()
        # print("hint", hint.shape)
        # bs = 3
        # T = 3
        # target_pos = torch.randn(bs, 2, 512, 512).to('cuda')       # [batch_size,2,512,512]
        # target_structure = torch.randn(bs, 1, 512, 512).to('cuda')  # [batch_size,1,512,512]
        # target_satellite = torch.randn(bs, 3, 512,512).to('cuda')    # [batch_size,3,512,512]
        # surround_pos = torch.randn(bs, T * 2,1,1).to('cuda')          # [batch_size,T * 2,1,1]
        # surround_structure = torch.randn(bs, T, 1,512,512).to('cuda')# [batch_size,T,1,512,512]
        # surround_panoramas = torch.randn(bs, T, 3,512,512).to('cuda')# [batch_size,T,3,512,512]

        
        
        
        hint_real = self.mixview_diff(target_pos, target_structure, target_satellite, 
                                      surround_pos, surround_structure, surround_panoramas,  verbose=False, view_result=True)
        
        
        #save hint_real as jpg
        # import numpy as np
        # from PIL import Image
        # image = hint_real[0].detach().cpu().numpy()
        # image = np.transpose(image, (1, 2, 0))
        # image = (image + 1) / 2
        # image = (image * 255).astype(np.uint8)
        # image = Image.fromarray(image)
        # image.save('hint_real.jpg')
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint_real, emb, context)

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


# class PositionEncoder(nn.Module):  # Nerf position encoder
#     def __init__(self, output_dim, input_dim = 2): #L_embed is output_dim/2
#         super(PositionEncoder, self).__init__()
#         self.input_dim = input_dim
#         L_embed = int(output_dim / 2)
#         self.L_embed = L_embed

#         # Compute frequencies (2^i) for encoding
#         self.freq_bands = 2.0 ** torch.linspace(0, L_embed - 1, L_embed)

#     def forward(self, pos):
#         rets = []
#         for freq in self.freq_bands:
#             rets.append(torch.sin(freq * pos)) 
#             rets.append(torch.cos(freq * pos))
#         # Concatenate all encodings along the last dimension
#         pos_enc = torch.cat(rets, dim=-1)
#         return pos_enc


# class StructureEncoder(nn.Module):
#     def __init__(self, output_dim, img_height, img_width):
#         super(StructureEncoder, self).__init__()
#         self.img_height = img_height
#         self.img_width = img_width

#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(128, output_dim)

#     def forward(self, x):
#         # x: [batch_size, 1, img_height, img_width]
#         features = self.conv(x)
#         features = features.view(features.size(0), -1)
#         out = self.fc(features)
#         return out


# class SatelliteEncoder(nn.Module):
#     def __init__(self, hidden_dim):
#         super(SatelliteEncoder, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # Output: 256x256
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # Output: 128x128
#             nn.ReLU(),
#             nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # Output: 64x64
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         # x: [batch_size, 3, 512, 512]
#         out = self.conv(x)
#         return out  # [batch_size, hidden_dim, 64, 64]


# class WeightedVCNN(nn.Module):
#     def __init__(self, hidden_dim):
#         super(WeightedVCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # Output: 256x256
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # Output: 128x128
#             nn.ReLU(),
#             nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # Output: 64x64
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.conv(x)  # [batch_size, hidden_dim, 64, 64]


# class Decoder(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # Output: 128x128
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),  # Output: 256x256
#             nn.ReLU(),
#             nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),  # Output: 512x512
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim // 4, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # Assuming output image pixel values between 0 and 1
#         )

#     def forward(self, x):
#         # x: [batch_size, 2 * hidden_dim, 64, 64]
#         out = self.decoder(x)
#         return out  # [batch_size, 3, 512, 512]


# class MixedViewEncoder(nn.Module):
#     def __init__(self, position_dim=64, structure_dim=128, hidden_dim=64, img_height=512, img_width=512):
#         super(MixedViewEncoder, self).__init__()
#         self.img_height = img_height
#         self.img_width = img_width

#         self.position_encoder = PositionEncoder(position_dim)
#         self.structure_encoder = StructureEncoder(structure_dim, img_height, img_width)

#         self.satellite_cnn = SatelliteEncoder(hidden_dim)
#         self.weighted_V_cnn = WeightedVCNN(hidden_dim)

#         self.query_mlp = nn.Linear(2*position_dim + structure_dim, hidden_dim)
#         self.key_mlp = nn.Linear(2*position_dim + structure_dim, hidden_dim)

#         self.hidden_dim = hidden_dim

#     def forward(self, target_pos, target_structure, target_satellite, surround_pos, surround_structure,
#                 surround_panoramas):
#         batch_size = target_pos.size(0)
#         t = int(surround_pos.size(1)/2)

#         # === Target ===
#         target_pos_cut = target_pos[:, :, 0, 0]
#         print(target_pos.shape, target_structure.shape)
#         target_pos_emb = self.position_encoder(target_pos_cut)
#         # target_pos_emb: [batch_size, position_dim]

#         target_structure_emb = self.structure_encoder(target_structure)
#         # target_structure_emb: [batch_size, structure_dim]

#         print(target_pos_emb.shape, target_structure_emb.shape)


#         target_features = torch.cat([target_pos_emb, target_structure_emb], dim=1)
#         # target_features: [batch_size, position_dim + structure_dim]

#         Q = self.query_mlp(target_features)
#         # Q: [batch_size, hidden_dim]
#         Q = Q.unsqueeze(1)
#         # Q: [batch_size, 1, hidden_dim]

#         # === Surroundings ===
#         surround_pos_flat = surround_pos[:, :, 0, 0].reshape(batch_size * t, 2)
#         # surround_pos_flat: [batch_size * t, 2]
#         surround_pos_emb = self.position_encoder(surround_pos_flat)
#         # surround_pos_emb: [batch_size * t, position_dim*2]

#         surround_structure_flat = surround_structure.reshape(
#             batch_size * t, 1, 512, 512)
#         # surround_structure_flat: [batch_size * t, 1, img_height, img_width]
#         surround_structure_emb = self.structure_encoder(surround_structure_flat)
#         print("surround_structure_emb", surround_structure_emb.shape)
#         # surround_structure_emb: [batch_size * t, structure_dim]

#         surround_features = torch.cat([surround_pos_emb, surround_structure_emb], dim=1)
#         # surround_features: [batch_size * t, position_dim + structure_dim]
#         K = self.key_mlp(surround_features)
#         # K: [batch_size * t, hidden_dim]
#         K = K.view(batch_size, t, -1)
#         # K: [batch_size, t, hidden_dim]

#         surround_panoramas_flat = surround_panoramas.reshape(
#             batch_size, t, 3, 512, 512)
#         # surround_panoramas_flat: [batch_size, t, 3, img_height, img_width]
#         V = surround_panoramas_flat
#         # V: [batch_size, t, 3, img_height, img_width]
#         print("V", V.shape)
#         # === Cross Attention Calculation ===
#         attention_scores = torch.matmul(Q, K.transpose(1, 2))
#         # attention_scores: [batch_size, 1, t]
#         attention_weights = F.softmax(attention_scores / (self.hidden_dim ** 0.5), dim=-1)
#         # attention_weights: [batch_size, 1, t]
#         attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
#         # attention_weights: [batch_size, 1, t, 1, 1]

#         print("attention_weights", attention_weights.shape)

#         V = V.permute(0, 2, 1, 3, 4)

#         print("V", V.shape)
#         # V: [batch_size, 3, t, img_height, img_width]
#         weighted_V = torch.sum(V * attention_weights, dim=2)
#         # weighted_V: [batch_size, 3, img_height, img_width]

#         print("weighted_V", weighted_V.shape)

#         # === Satellite Processing ===
#         target_satellite_emb = self.satellite_cnn(target_satellite)
#         # target_satellite_emb: [batch_size, hidden_dim, 64, 64]

#         print("target_satellite_emb", target_satellite_emb.shape)

#         # Process weighted_V
#         weighted_V_emb = self.weighted_V_cnn(weighted_V)
#         # weighted_V_emb: [batch_size, hidden_dim, 64, 64]

#         print("weighted_V_emb", weighted_V_emb.shape)

#         # Combine along channel dimension
#         combined_features = torch.cat([weighted_V_emb, target_satellite_emb], dim=1)
#         # combined_features: [batch_size, 2 * hidden_dim, 64, 64]

#         print("combined_features", combined_features.shape)

#         return combined_features


# class MixedViewDiff(nn.Module):
#     def __init__(self, encoder: MixedViewEncoder, decoder: Decoder):
#         super(MixedViewDiff, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, *args, **kwargs):
#         combined_features = self.encoder(*args, **kwargs)
#         output = self.decoder(combined_features)
#         print(output)
#         #save output as jpg

        
#         return output