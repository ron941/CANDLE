## CANDLE: Color Ambient Normalization with DINO Layer Enhancement
## Core backbone for ambient light normalization with multi-stage DINO fusion.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import torch.utils.checkpoint as cp
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from einops import rearrange
pi = 3.141592653589793

################ Layer Norm ################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

################ Gated Feed-Forward Network (GFFN) ################

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        # ✅ 使用 inplace 運算與釋放中間變數，顯著降低顯存
        x1 = F.gelu(x1)
        x1.mul_(x2)        # 就地乘法，不會產生新的 tensor
        del x2             # 釋放暫存變數
        x = self.project_out(x1)
        return x


################ Transposed Self-Attention ################
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.init_dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 預設 placeholder，forward 會動態調整
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _rebuild_qkv_if_needed(self, c):
        if c != self.qkv.in_channels:
            #print(f"[Attention] rebuilding qkv: {self.qkv.in_channels} → {c}")
            self.qkv = nn.Conv2d(c, c * 3, kernel_size=1, bias=self.bias).to(self.qkv.weight.device)
            self.qkv_dwconv = nn.Conv2d(
                c * 3, c * 3, kernel_size=3, stride=1, padding=1,
                groups=c * 3, bias=self.bias
            ).to(self.qkv.weight.device)
            self.project_out = nn.Conv2d(c, c, kernel_size=1, bias=self.bias).to(self.qkv.weight.device)

    def forward(self, x):
        b, c, h, w = x.shape
        orig_h, orig_w = h, w  # ✅ 保留原始尺寸

        # --- 🔧 自動補齊空間尺寸為 8 的倍數 ---
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            #print(f"[Attention] padded spatial from ({h},{w}) → ({h+pad_h},{w+pad_w})")
            h += pad_h
            w += pad_w

        # ✅ 通道補齊
        if c % self.num_heads != 0:
            pad_c = self.num_heads - (c % self.num_heads)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_c))
            c += pad_c
            #print(f"[Attention] channel padded from {c - pad_c} → {c}")

        self._rebuild_qkv_if_needed(c)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if h * w > 128 * 128:
            window_size = 8
            q = rearrange(q, 'b (head c) (h p1) (w p2) -> (b h w) head c (p1 p2)',
                          head=self.num_heads, p1=window_size, p2=window_size)
            k = rearrange(k, 'b (head c) (h p1) (w p2) -> (b h w) head c (p1 p2)',
                          head=self.num_heads, p1=window_size, p2=window_size)
            v = rearrange(v, 'b (head c) (h p1) (w p2) -> (b h w) head c (p1 p2)',
                          head=self.num_heads, p1=window_size, p2=window_size)

            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = attn @ v

            out = rearrange(out, '(b h w) head c (p1 p2) -> b (head c) (h p1) (w p2)',
                            head=self.num_heads, h=h // window_size, w=w // window_size,
                            p1=window_size, p2=window_size)
        else:
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        # ✅ 保證輸出與輸入尺寸完全一致（防止 residual 加法錯誤）
        out = out[:, :, :orig_h, :orig_w]
        return out




################ Geometry-Guided Self-Attention ################

class DepthAwareCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DepthAwareCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        
        self.q_depth = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_depth_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, depth):
        b,c,h,w = x.shape
        _, _, h_d, w_d = depth.shape

        kv_rgb = self.kv_dwconv(self.kv(x))
        q_depth = self.q_depth_dwconv(self.q_depth(depth))
        k_rgb, v_rgb = kv_rgb.chunk(2, dim=1)
                
        q_depth = rearrange(q_depth, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        k_rgb = rearrange(k_rgb, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_rgb = rearrange(v_rgb, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_depth = torch.nn.functional.normalize(q_depth, dim=-1)
        k_rgb = torch.nn.functional.normalize(k_rgb, dim=-1)
        attn = (q_depth @ k_rgb.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_rgb)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out

################ Geometry-Aware Transformer Block ################
import torch.utils.checkpoint as cp

class GeometryAwareTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(GeometryAwareTransformerBlock, self).__init__()

        self.norm1_combined = LayerNorm(dim * 2, LayerNorm_type)
        self.attn = DepthAwareCrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.norm1_combined(x)
        rgb, depth = torch.chunk(x, 2, dim=1)

        # ✅ checkpoint attention + ffn
        rgb = rgb + self.attn(rgb, depth)
        rgb = rgb + self.ffn(self.norm2(rgb))


        # ✅ 拼回原結構
        return torch.cat([rgb, depth], dim=1)


################ Transformer Block ################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # ✅ 包住 attention 與 ffn，避免顯存堆疊
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

    
################ Downsample and Upsample ################

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        # 🔇 自動修正尺寸，但不再輸出 log
        if (h % 2 != 0) or (w % 2 != 0):
            new_h, new_w = (h // 2) * 2, (w // 2) * 2
            x = x[:, :, :new_h, :new_w]
        return self.body(x)



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

################ Overlap Patch Embedding ################

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        x = self.proj(x)

        return x


class LayerGatedDinoFusion(nn.Module):
    def __init__(self, dino_dim, query_dim, stage_feat_dim, num_candidates, gate_hidden=64):
        super(LayerGatedDinoFusion, self).__init__()
        self.num_candidates = num_candidates
        self.query_dim = query_dim
        self.projs = nn.ModuleList([
            nn.Conv2d(dino_dim, query_dim, kernel_size=1, bias=False) for _ in range(num_candidates)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(stage_feat_dim, gate_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_hidden, num_candidates, kernel_size=1, bias=True),
        )
        self.norm = nn.GroupNorm(1, query_dim)

    def forward(self, dino_candidates, stage_feat, out_hw):
        if len(dino_candidates) != self.num_candidates:
            raise ValueError(f"Expected {self.num_candidates} candidates, got {len(dino_candidates)}")

        fused = 0.0
        gate_logits = self.gate(stage_feat).view(stage_feat.shape[0], self.num_candidates)
        gate_weights = torch.softmax(gate_logits, dim=1).view(stage_feat.shape[0], self.num_candidates, 1, 1, 1)
        for idx in range(self.num_candidates):
            di = F.interpolate(dino_candidates[idx], size=out_hw, mode='bilinear', align_corners=False)
            di = self.projs[idx](di)
            fused = fused + gate_weights[:, idx] * di
        return self.norm(fused)


class DRFusionBlock(nn.Module):
    def __init__(self, feat_dim, query_dim, heads=4, dropout=0.0, alpha_init=0.0, spatial_reduction=1):
        super(DRFusionBlock, self).__init__()
        if feat_dim % heads != 0:
            raise ValueError(f"feat_dim ({feat_dim}) must be divisible by heads ({heads})")
        self.feat_dim = feat_dim
        self.heads = heads
        self.head_dim = feat_dim // heads
        self.scale = self.head_dim ** -0.5
        self.spatial_reduction = max(int(spatial_reduction), 1)

        self.dino_adapt = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=1, bias=False),
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1, groups=query_dim, bias=False),
            nn.GroupNorm(1, query_dim),
            nn.GELU(),
        )
        self.q_proj = nn.Conv2d(query_dim, feat_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _to_heads(self, x):
        return rearrange(x, 'b (head c) h w -> b head (h w) c', head=self.heads)

    def _from_heads(self, x, h, w):
        return rearrange(x, 'b head (h w) c -> b (head c) h w', head=self.heads, h=h, w=w)

    def forward(self, feat, dino_query):
        h, w = feat.shape[-2:]
        dino_query = F.interpolate(dino_query, size=(h, w), mode='bilinear', align_corners=False)
        dino_query = self.dino_adapt(dino_query)
        if self.spatial_reduction > 1:
            feat_attn = F.avg_pool2d(feat, kernel_size=self.spatial_reduction, stride=self.spatial_reduction)
            dino_attn = F.avg_pool2d(dino_query, kernel_size=self.spatial_reduction, stride=self.spatial_reduction)
        else:
            feat_attn = feat
            dino_attn = dino_query

        q = self._to_heads(self.q_proj(dino_attn))
        k = self._to_heads(self.k_proj(feat_attn))
        v = self._to_heads(self.v_proj(feat_attn))
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v.float())
        h_attn, w_attn = feat_attn.shape[-2:]
        out = self._from_heads(out, h_attn, w_attn).to(feat.dtype)
        if (h_attn != h) or (w_attn != w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = self.out_proj(out)
        return feat + self.alpha * out


class PreConcatSFFB(nn.Module):
    def __init__(self, high_ch, low_ch):
        super(PreConcatSFFB, self).__init__()
        self.high_proj = nn.Conv2d(high_ch, low_ch, kernel_size=1, bias=False)
        self.high_lp = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=3, padding=1, groups=low_ch, bias=False),
            nn.Conv2d(low_ch, low_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, low_ch),
            nn.GELU(),
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, low_ch),
            nn.GELU(),
        )
        self.dwt = HaarDWT2D(low_ch)
        self.idwt = HaarIDWT2D(low_ch)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, low_ch),
            nn.GELU(),
        )
        self.det_branch = nn.Sequential(
            nn.Conv2d(low_ch * 3, low_ch * 3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, low_ch * 3),
            nn.GELU(),
        )
        self.gate = nn.Conv2d(low_ch * 4, 3, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(low_ch, low_ch, kernel_size=1, bias=False)
        self.out_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_high, f_low):
        h, w = f_low.shape[-2:]
        f_high_raw = F.interpolate(f_high, size=(h, w), mode='bilinear', align_corners=False)
        f_high_base = self.high_proj(f_high_raw)
        f_high_lp = self.high_lp(f_high_base)

        f_low = self.low_proj(f_low)
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            f_low_pad = F.pad(f_low, (0, pad_w, 0, pad_h), mode='replicate')
        else:
            f_low_pad = f_low

        ll, lh, hl, hh = self.dwt(f_low_pad)
        ll_ref = self.reg_branch(ll)
        det_ref = self.det_branch(torch.cat([lh, hl, hh], dim=1))
        lh_ref, hl_ref, hh_ref = det_ref.chunk(3, dim=1)

        zeros = torch.zeros_like(ll_ref)
        f_reg = self.idwt(ll_ref, zeros, zeros, zeros)
        f_det = self.idwt(zeros, lh_ref, hl_ref, hh_ref)
        if pad_h or pad_w:
            f_reg = f_reg[:, :, :h, :w]
            f_det = f_det[:, :, :h, :w]

        gate_logits = self.gate(torch.cat([f_high_base, f_high_lp, f_reg, f_det], dim=1))
        gate = torch.softmax(gate_logits, dim=1)
        g_h, g_r, g_d = gate.chunk(3, dim=1)
        f_mix = g_h * f_high_lp + g_r * f_reg + g_d * f_det
        return f_high_base + self.out_scale * self.out_proj(f_mix)


class BFACG(nn.Module):
    def __init__(self, feat_ch, enc_ch, hidden=64, res_scale_init=0.0, neutral_strength=0.0):
        super(BFACG, self).__init__()
        if feat_ch % 2 != 0:
            raise ValueError(f"feat_ch must be even for BFACG split, got {feat_ch}")

        half = feat_ch // 2
        guide_in_ch = enc_ch + 3 + 2  # encoder feature + RGB + Sobel(x,y)
        hidden = max(int(hidden), 16)

        self.struct_net = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.chroma_pre = nn.Conv2d(half, half, kernel_size=1, bias=False)
        self.low_blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.low_smooth = nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1, groups=half, bias=False)
        self._init_low_smooth()
        self.chroma_post = nn.Conv2d(half, half, kernel_size=1, bias=False)

        self.guide_net = nn.Sequential(
            nn.Conv2d(guide_in_ch, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(float(res_scale_init)))
        self.neutral_strength = float(neutral_strength)

        self.register_buffer(
            "sobel_x",
            torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32).view(1, 1, 3, 3),
            persistent=False,
        )

    def _init_low_smooth(self):
        with torch.no_grad():
            self.low_smooth.weight.fill_(1.0 / 9.0)

    def _sobel(self, rgb):
        gray = rgb.mean(dim=1, keepdim=True)
        sobel_x = self.sobel_x.to(dtype=gray.dtype, device=gray.device)
        sobel_y = self.sobel_y.to(dtype=gray.dtype, device=gray.device)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x, enc_feat, rgb):
        h, w = x.shape[-2:]
        enc_feat = F.interpolate(enc_feat, size=(h, w), mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, size=(h, w), mode='bilinear', align_corners=False)

        barrier = torch.sigmoid(self.guide_net(torch.cat([enc_feat, rgb, self._sobel(rgb)], dim=1)))
        if self.neutral_strength > 0.0:
            sat = rgb.max(dim=1, keepdim=True)[0] - rgb.min(dim=1, keepdim=True)[0]
            neutral = torch.clamp(1.0 - sat, 0.0, 1.0)
            barrier = torch.clamp(barrier + self.neutral_strength * neutral, 0.0, 1.0)

        xs, xc = torch.chunk(x, 2, dim=1)
        xs_ref = self.struct_net(xs)

        xc = self.chroma_pre(xc)
        xc_low = self.low_blur(xc)
        xc_high = xc - xc_low
        xc_smooth = self.low_smooth(xc_low)
        xc_low_ref = xc_low + (1.0 - barrier) * (xc_smooth - xc_low)
        xc_ref = self.chroma_post(xc_low_ref + xc_high)

        delta = self.fuse(torch.cat([xs_ref, xc_ref], dim=1))
        return x + self.scale * delta


class ColorLineProjection(nn.Module):
    def __init__(self, feat_ch, latent_ch=16, res_scale_init=0.0, conf_bias_init=-2.0):
        super(ColorLineProjection, self).__init__()
        if feat_ch % 2 != 0:
            raise ValueError(f"feat_ch must be even for CLP split, got {feat_ch}")
        half = feat_ch // 2
        latent_ch = max(4, min(int(latent_ch), half))

        self.proj_in = nn.Sequential(
            nn.Conv2d(half, latent_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dir_net = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, stride=1, padding=1, groups=latent_ch, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_ch, latent_ch, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conf_net = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, kernel_size=3, stride=1, padding=1, groups=latent_ch, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(latent_ch, half, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(half, half, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(float(res_scale_init)))
        with torch.no_grad():
            nn.init.constant_(self.conf_net[-1].bias, float(conf_bias_init))

    def forward(self, x):
        xs, xc = torch.chunk(x, 2, dim=1)
        z = self.proj_in(xc)
        v = F.normalize(self.dir_net(z), dim=1, eps=1e-6)
        coef = (z * v).sum(dim=1, keepdim=True)
        z_proj = coef * v
        conf = torch.sigmoid(self.conf_net(z))
        z_out = (1.0 - conf) * z + conf * z_proj
        xc_out = self.proj_out(z_out)
        delta = self.fuse(torch.cat([xs, xc_out], dim=1))
        return x + self.scale * delta


class ICAPriorEncoder(nn.Module):
    def __init__(self, in_ch=3, aux_ch=32):
        super(ICAPriorEncoder, self).__init__()
        aux_ch = max(int(aux_ch), 8)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.down1 = nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.down2 = nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.down3 = nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.enc4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(aux_ch, aux_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

    def forward(self, rgb):
        f1 = self.in_proj(rgb)
        f2 = self.enc2(self.down1(f1))
        f3 = self.enc3(self.down2(f2))
        f4 = self.enc4(self.down3(f3))

        d3 = self.dec3(F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False) + f3)
        d2 = self.dec2(F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False) + f2)
        d1 = self.dec1(F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False) + f1)

        # 7-stage priors aligned to encoder/latent/decoder path.
        return [f1, f2, f3, f4, d3, d2, d1]


class ICAInjectBlock(nn.Module):
    def __init__(self, feat_ch, aux_ch=32, hidden=64, res_scale_init=0.0):
        super(ICAInjectBlock, self).__init__()
        hidden = max(int(hidden), 16)
        self.aux_proj = nn.Sequential(
            nn.Conv2d(aux_ch, feat_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
        )
        self.feat_refine = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1, groups=feat_ch, bias=False),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.scale = nn.Parameter(torch.tensor(float(res_scale_init)))

    def forward(self, feat, aux):
        if aux is None:
            return feat
        h, w = feat.shape[-2:]
        aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
        aux = self.aux_proj(aux)
        feat_ref = self.feat_refine(feat)
        gate = torch.sigmoid(self.gate_net(torch.cat([feat, aux], dim=1)))
        mix = gate * feat_ref + (1.0 - gate) * aux
        delta = self.out_proj(mix)
        return feat + self.scale * delta


class ABCAuxBranch(nn.Module):
    def __init__(
        self,
        embed_dim=16,
        hist_ckpt_path=None,
        lab_ckpt_path=None,
        token_projection="linear",
        token_mlp="TwoDCFF",
        abc_repo_root="/raid/ron/ALN_768/ABC-Former-main/ABC-Former",
    ):
        super(ABCAuxBranch, self).__init__()
        self.abc_repo_root = abc_repo_root
        if self.abc_repo_root not in sys.path:
            sys.path.insert(0, self.abc_repo_root)

        try:
            from PDFformer_hist import Hist_Histoformer
            from PDFformer_lab import Lab_Histoformer
        except Exception as exc:
            raise ImportError(
                f"Failed to import ABC auxiliary branches from '{self.abc_repo_root}'."
            ) from exc

        self.hist_branch = Hist_Histoformer(
            embed_dim=embed_dim,
            token_projection=token_projection,
            token_mlp=token_mlp,
        )
        self.lab_branch = Lab_Histoformer(
            embed_dim=embed_dim,
            token_projection=token_projection,
            token_mlp=token_mlp,
        )

        if hist_ckpt_path:
            self._load_branch_ckpt(self.hist_branch, hist_ckpt_path, "hist")
        if lab_ckpt_path:
            self._load_branch_ckpt(self.lab_branch, lab_ckpt_path, "lab")

    def _load_branch_ckpt(self, branch, ckpt_path, branch_name):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ABC {branch_name} checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = branch.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ABC-AUX][{branch_name}] missing keys: {missing}")
        if unexpected:
            print(f"[ABC-AUX][{branch_name}] unexpected keys: {unexpected}")

    def forward_hist(self, input_hist):
        return self.hist_branch(input_hist)

    def forward_lab(self, input_lab_hist):
        return self.lab_branch(input_lab_hist)

    def forward(self, input_hist, input_lab_hist):
        pred_hist, hist_weights = self.forward_hist(input_hist)
        pred_lab, lab_weights = self.forward_lab(input_lab_hist)
        return pred_hist, pred_lab, hist_weights, lab_weights

################ HVI Prompt Modulation Block ################

class HVIPromptBlock(nn.Module):
    def __init__(self, feat_dim, hidden=64, use_hv=True, hvi_eps=1e-8):
        super(HVIPromptBlock, self).__init__()
        cond_in_ch = 3 if use_hv else 1
        self.use_hv = use_hv
        self.hvi_phys = DynamicAnalyticalHVI(eps=hvi_eps)

        self.hvi_proj = nn.Sequential(
            nn.Conv2d(cond_in_ch, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.to_gamma = nn.Conv2d(hidden, feat_dim, kernel_size=1, bias=False)
        self.to_beta = nn.Conv2d(hidden, feat_dim, kernel_size=1, bias=False)
        self.to_conf = nn.Conv2d(hidden, feat_dim, kernel_size=1, bias=False)

    def forward(self, feat, rgb_img):
        rgb_small = F.interpolate(rgb_img, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        h, v, i, _ = self.hvi_phys.forward_hvit(rgb_small)
        cond = torch.cat([h, v, i], dim=1) if self.use_hv else i

        cond = self.hvi_proj(cond)
        base = self.feat_proj(feat)
        f = self.fuse(torch.cat([base, cond], dim=1))
        gamma = self.to_gamma(f)
        beta = self.to_beta(f)
        conf = torch.sigmoid(self.to_conf(f))
        return gamma, beta, conf

################ Depth Encoder ################

def depth_encoder(input_dim, output_dim, num_layers):
    layers = []
    for i in range(num_layers):
        in_channels = input_dim if i == 0 else output_dim
        layers.extend([
            nn.Conv2d(in_channels, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        ])
    return nn.Sequential(*layers)

class FeatureHVIProjector(nn.Module):
    def __init__(self, ch_in, ch_hvi):
        super(FeatureHVIProjector, self).__init__()
        self.ch_hvi = ch_hvi
        self.to_hvi = nn.Conv2d(ch_in, ch_hvi * 3, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, ch_hvi * 3)

    def forward(self, x):
        x = self.norm(self.to_hvi(x))
        h = torch.tanh(x[:, :self.ch_hvi, :, :])
        v = torch.tanh(x[:, self.ch_hvi:2 * self.ch_hvi, :, :])
        i = torch.sigmoid(x[:, 2 * self.ch_hvi:, :, :])
        return torch.cat([h, v, i], dim=1)

class DynamicAnalyticalHVI(nn.Module):
    def __init__(self, eps=1e-8):
        super(DynamicAnalyticalHVI, self).__init__()
        self.k_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.eps = max(float(eps), 1e-8)

    def forward_hvit(self, img):
        # img: [B, 3, H, W], expected in [0, 1]
        in_dtype = img.dtype
        img = torch.clamp(img.float(), 0.0, 1.0)
        value, max_idx = torch.max(img, dim=1, keepdim=True)
        pred_k = self.k_predictor(value) + 0.1  # [0.1, 1.1]
        img_min, _ = torch.min(img, dim=1, keepdim=True)
        saturation = (value - img_min) / (value + self.eps)

        r, g, b = torch.chunk(img, 3, dim=1)
        delta = value - img_min + self.eps
        hue = torch.zeros_like(value)
        mask_r = max_idx == 0
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        mask_g = max_idx == 1
        hue[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
        mask_b = max_idx == 2
        hue[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
        hue = hue / 6.0

        color_sensitive = torch.pow(torch.sin(value * 0.5 * pi) + self.eps, pred_k)
        h_comp = color_sensitive * saturation * torch.cos(2 * pi * hue)
        v_comp = color_sensitive * saturation * torch.sin(2 * pi * hue)
        return h_comp.to(in_dtype), v_comp.to(in_dtype), value.to(in_dtype), pred_k.to(in_dtype)

    def forward_phvit(self, h_comp, v_comp, value, k):
        in_dtype = h_comp.dtype
        h_comp = h_comp.float()
        v_comp = v_comp.float()
        value = torch.clamp(value.float(), 0.0, 1.0)
        k = k.float()

        hue = torch.atan2(v_comp + self.eps, h_comp + self.eps) / (2 * pi)
        hue = hue % 1.0

        color_sensitive = torch.pow(torch.sin(value * 0.5 * pi) + self.eps, k)
        chroma_mag = torch.sqrt(h_comp ** 2 + v_comp ** 2 + self.eps)
        safe_denom = torch.where(color_sensitive < 1e-4, torch.ones_like(color_sensitive), color_sensitive)
        saturation = chroma_mag / safe_denom
        saturation = torch.where(color_sensitive < 1e-4, torch.zeros_like(saturation), saturation)
        saturation = torch.clamp(saturation, 0.0, 1.0)

        c = value * saturation
        h6 = hue * 6.0
        x = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
        m = value - c
        zero = torch.zeros_like(hue)

        r_prime = torch.zeros_like(hue)
        g_prime = torch.zeros_like(hue)
        b_prime = torch.zeros_like(hue)

        mask = h6 < 1
        r_prime[mask], g_prime[mask], b_prime[mask] = c[mask], x[mask], zero[mask]
        mask = (h6 >= 1) & (h6 < 2)
        r_prime[mask], g_prime[mask], b_prime[mask] = x[mask], c[mask], zero[mask]
        mask = (h6 >= 2) & (h6 < 3)
        r_prime[mask], g_prime[mask], b_prime[mask] = zero[mask], c[mask], x[mask]
        mask = (h6 >= 3) & (h6 < 4)
        r_prime[mask], g_prime[mask], b_prime[mask] = zero[mask], x[mask], c[mask]
        mask = (h6 >= 4) & (h6 < 5)
        r_prime[mask], g_prime[mask], b_prime[mask] = x[mask], zero[mask], c[mask]
        mask = h6 >= 5
        r_prime[mask], g_prime[mask], b_prime[mask] = c[mask], zero[mask], x[mask]

        rgb = torch.cat([r_prime, g_prime, b_prime], dim=1) + m
        return torch.clamp(rgb, 0.0, 1.0).to(in_dtype)

class HaarDWT2D(nn.Module):
    def __init__(self, ch):
        super(HaarDWT2D, self).__init__()
        self.ch = ch
        base = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],   # LL
                [[-0.5, -0.5], [0.5, 0.5]], # LH
                [[-0.5, 0.5], [-0.5, 0.5]], # HL
                [[0.5, -0.5], [-0.5, 0.5]], # HH
            ],
            dtype=torch.float32
        ).unsqueeze(1)  # [4, 1, 2, 2]
        weight = base.repeat(ch, 1, 1, 1)  # [4*ch, 1, 2, 2]
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        w = self.weight.to(device=x.device, dtype=x.dtype)
        y = F.conv2d(x, w, stride=2, groups=self.ch)
        return y.chunk(4, dim=1)

class HaarIDWT2D(nn.Module):
    def __init__(self, ch):
        super(HaarIDWT2D, self).__init__()
        self.ch = ch
        base = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],   # LL
                [[-0.5, -0.5], [0.5, 0.5]], # LH
                [[-0.5, 0.5], [-0.5, 0.5]], # HL
                [[0.5, -0.5], [-0.5, 0.5]], # HH
            ],
            dtype=torch.float32
        ).unsqueeze(1)  # [4, 1, 2, 2]
        weight = base.repeat(ch, 1, 1, 1)  # [4*ch, 1, 2, 2]
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, ll, lh, hl, hh):
        x = torch.cat([ll, lh, hl, hh], dim=1)
        w = self.weight.to(device=x.device, dtype=x.dtype)
        return F.conv_transpose2d(x, w, stride=2, groups=self.ch)

class CDFFA(nn.Module):
    def __init__(self, query_dim, kv_dim, heads, dropout=0.0):
        super(CDFFA, self).__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q_proj = nn.Conv2d(query_dim, kv_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(kv_dim, kv_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(kv_dim, kv_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(kv_dim, kv_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _pad_channels(self, x):
        c = x.shape[1]
        if c % self.heads == 0:
            return x, c
        pad_c = self.heads - (c % self.heads)
        return F.pad(x, (0, 0, 0, 0, 0, pad_c)), c

    def forward(self, query_map, ll_hvi):
        q = F.interpolate(query_map, size=ll_hvi.shape[-2:], mode="bilinear", align_corners=False)
        q = self.q_proj(q)
        k = self.k_proj(ll_hvi)
        v = self.v_proj(ll_hvi)

        q, c0 = self._pad_channels(q)
        k, _ = self._pad_channels(k)
        v, _ = self._pad_channels(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.heads)

        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature.float()
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v.float()
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.heads, h=ll_hvi.shape[2], w=ll_hvi.shape[3])

        out = out[:, :c0, :, :].to(ll_hvi.dtype)
        out = self.out_proj(out)
        out = self.dropout(out)
        return ll_hvi + out

class HVI_SIR_Bottleneck(nn.Module):
    def __init__(
        self,
        dim_l4,
        attn_dim=64,
        heads=4,
        dropout=0.0,
        lambda_init=0.0,
        hvi_eps=1e-8,
        use_blur_ill=True,
    ):
        super(HVI_SIR_Bottleneck, self).__init__()
        if attn_dim % heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by heads ({heads}).")

        self.attn_dim = attn_dim
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.scale = self.head_dim ** -0.5
        self.use_blur_ill = bool(use_blur_ill)
        self.hvi_phys = DynamicAnalyticalHVI(eps=hvi_eps)

        blur_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32
        ) / 16.0
        self.register_buffer("ill_blur_kernel", blur_kernel.view(1, 1, 3, 3), persistent=False)

        self.ill_encoder = nn.Sequential(
            nn.Conv2d(1, attn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, attn_dim),
            nn.GELU(),
            nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False),
        )
        self.chr_encoder = nn.Sequential(
            nn.Conv2d(2, attn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, attn_dim),
            nn.GELU(),
            nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False),
        )
        self.sem_proj = nn.Sequential(
            nn.Conv2d(dim_l4, attn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, attn_dim),
            nn.GELU(),
        )

        self.q_proj = nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False)
        self.k_sem = nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False)
        self.v_sem = nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False)
        self.k_ill = nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False)
        self.v_hv = nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=False)

        self.hv_gate = nn.Sequential(
            nn.Conv2d(attn_dim, attn_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(attn_dim * 2, dim_l4, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.lambda_ill = nn.Parameter(torch.full((heads, 1, 1), float(lambda_init)))
        self.temperature_sem = nn.Parameter(torch.ones(heads, 1, 1))
        self.temperature_ill = nn.Parameter(torch.ones(heads, 1, 1))
        self.out_scale = nn.Parameter(torch.tensor(0.0))

    def _blur_ill(self, x):
        if not self.use_blur_ill:
            return x
        kernel = self.ill_blur_kernel.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel, stride=1, padding=1)

    def _to_heads(self, x):
        return rearrange(x, 'b (head c) h w -> b head (h w) c', head=self.heads)

    def _from_heads(self, x, h, w):
        return rearrange(x, 'b head (h w) c -> b (head c) h w', head=self.heads, h=h, w=w)

    def forward(self, latent_in, input_img_full):
        real_rgb_down = F.interpolate(
            input_img_full,
            size=latent_in.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        h, v, i, _ = self.hvi_phys.forward_hvit(real_rgb_down)
        hv = torch.cat([h, v], dim=1)

        i_lf = self._blur_ill(i)
        ill_feat = self.ill_encoder(i_lf)
        hv_feat = self.chr_encoder(hv)
        sem_feat = self.sem_proj(latent_in)

        q = self._to_heads(self.q_proj(sem_feat))
        k_sem = self._to_heads(self.k_sem(sem_feat))
        v_sem = self._to_heads(self.v_sem(sem_feat))
        k_ill = self._to_heads(self.k_ill(ill_feat))
        v_hv = self._to_heads(self.v_hv(hv_feat))

        temp_sem = F.softplus(self.temperature_sem).view(1, self.heads, 1, 1)
        temp_ill = F.softplus(self.temperature_ill).view(1, self.heads, 1, 1)
        s_sem = torch.matmul(q.float(), k_sem.float().transpose(-2, -1)) * self.scale * temp_sem
        s_ill = torch.matmul(q.float(), k_ill.float().transpose(-2, -1)) * self.scale * temp_ill

        lam = torch.sigmoid(self.lambda_ill).view(1, self.heads, 1, 1)
        a_rect = torch.softmax(s_sem - lam * s_ill, dim=-1)
        a_rect = self.dropout(a_rect)

        o_sem = torch.matmul(a_rect, v_sem.float())
        o_hv = torch.matmul(a_rect, v_hv.float())

        h0, w0 = latent_in.shape[-2:]
        o_sem = self._from_heads(o_sem, h0, w0).to(latent_in.dtype)
        o_hv = self._from_heads(o_hv, h0, w0).to(latent_in.dtype)

        g_hv = self.hv_gate(hv_feat)
        fused = torch.cat([o_sem, g_hv * o_hv], dim=1)
        delta = self.out_proj(fused)
        return latent_in + self.out_scale * delta

def _srgb_to_linear(rgb):
    return torch.where(
        rgb > 0.04045,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92,
    )


def _linear_to_srgb(rgb):
    rgb = torch.clamp(rgb, min=0.0)
    return torch.where(
        rgb > 0.0031308,
        1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055,
        12.92 * rgb,
    )


def rgb_to_lab_torch(rgb):
    rgb = torch.clamp(rgb, 0.0, 1.0)
    lin = _srgb_to_linear(rgb)
    r = lin[:, 0:1]
    g = lin[:, 1:2]
    b = lin[:, 2:3]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xr = x / 0.95047
    yr = y / 1.0
    zr = z / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    xr_safe = torch.clamp_min(xr, 1e-8)
    yr_safe = torch.clamp_min(yr, 1e-8)
    zr_safe = torch.clamp_min(zr, 1e-8)
    fx = torch.where(xr > eps, torch.pow(xr_safe, 1.0 / 3.0), (kappa * xr + 16.0) / 116.0)
    fy = torch.where(yr > eps, torch.pow(yr_safe, 1.0 / 3.0), (kappa * yr + 16.0) / 116.0)
    fz = torch.where(zr > eps, torch.pow(zr_safe, 1.0 / 3.0), (kappa * zr + 16.0) / 116.0)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([l, a, b], dim=1)


def lab_to_rgb_torch(lab):
    l = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    fx3 = fx * fx * fx
    fy3 = fy * fy * fy
    fz3 = fz * fz * fz

    xr = torch.where(fx3 > eps, fx3, (116.0 * fx - 16.0) / kappa)
    yr = torch.where(fy3 > eps, fy3, (116.0 * fy - 16.0) / kappa)
    zr = torch.where(fz3 > eps, fz3, (116.0 * fz - 16.0) / kappa)

    x = xr * 0.95047
    y = yr * 1.0
    z = zr * 1.08883

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb_lin = torch.cat([r_lin, g_lin, b_lin], dim=1)
    rgb = _linear_to_srgb(rgb_lin)
    return torch.clamp(rgb, 0.0, 1.0)


class LowFreqChromaBiasNet(nn.Module):
    def __init__(self, hidden=32, kernel_size=11, sigma=3.0):
        super().__init__()
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.hidden, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 3, 1, 1),
            nn.GELU(),
        )
        self.predict = nn.Conv2d(self.hidden, 2, 3, 1, 1)
        self.max_bias = 20.0

        k = self._build_gaussian_kernel(self.kernel_size, self.sigma)
        self.register_buffer("gauss_kernel", k, persistent=False)

    @staticmethod
    def _build_gaussian_kernel(kernel_size, sigma):
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        g = g / (g.sum() + 1e-8)
        k2d = torch.outer(g, g)
        k2d = k2d / (k2d.sum() + 1e-8)
        return k2d.view(1, 1, kernel_size, kernel_size)

    def _blur(self, x):
        c = x.shape[1]
        k = self.gauss_kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        pad = self.kernel_size // 2
        return F.conv2d(x, k, padding=pad, groups=c)

    def build_target_bias(self, inp_rgb, gt_rgb):
        with torch.cuda.amp.autocast(enabled=False):
            inp = inp_rgb.float()
            gt = gt_rgb.float()
            inp_lab = rgb_to_lab_torch(inp)
            gt_lab = rgb_to_lab_torch(gt)
            inp_ab_low = self._blur(inp_lab[:, 1:3])
            gt_ab_low = self._blur(gt_lab[:, 1:3])
            return inp_ab_low - gt_ab_low

    def forward(self, inp_rgb, return_bias=False):
        with torch.cuda.amp.autocast(enabled=False):
            inp = inp_rgb.float()
            feat = self.encoder(inp)
            bias_low = self.predict(feat)
            bias_low = F.interpolate(bias_low, size=inp.shape[-2:], mode="bilinear", align_corners=False)
            bias_low = self._blur(bias_low)
            bias_low = self.max_bias * torch.tanh(bias_low / max(self.max_bias, 1e-6))

            inp_lab = rgb_to_lab_torch(inp)
            out_lab = inp_lab.clone()
            out_lab[:, 1:3] = out_lab[:, 1:3] - bias_low
            out_rgb = lab_to_rgb_torch(out_lab)

        if return_bias:
            return out_rgb, bias_low
        return out_rgb


################ CANDLE Model ################

class CANDLE(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False,
        dino_dim = 1024,
        query_dim = 32,
        use_psf_dr = False,
        dr_heads = 4,
        dr_dropout = 0.0,
        dr_alpha_init = 0.0,
        psf_gate_hidden = 64,
        use_sffb_decoder = False,
        use_bfacg_decoder = True,
        bfacg_hidden = 64,
        bfacg_res_scale_init = 0.0,
        bfacg_variant = "v1",
        bfacg_neutral_k = 0.3,
        use_clp_decoder = False,
        clp_latent_ch = 16,
        clp_res_scale_init = 0.0,
        clp_conf_bias_init = -2.0,
        use_ica7 = False,
        use_abc_ica = False,
        ica_aux_ch = 32,
        ica_hidden = 64,
        ica_res_scale_init = 0.0,
        abc_hist_ckpt_path = "/raid/ron/ALN_768/ABC-Former-main/ABC-Former/checkpoints_CL3AN_resize512/hist/Hist_d16_last.pth",
        abc_lab_ckpt_path = "/raid/ron/ALN_768/ABC-Former-main/ABC-Former/checkpoints_CL3AN_resize512/lab/Lab_d16_last.pth",
        abc_aux_embed_dim = 16,
        abc_detach_aux_weight = True,
        use_hvi_bottleneck = True,
        sir_attn_dim = 64,
        sir_heads = 4,
        sir_dropout = 0.0,
        sir_lambda_init = 0.0,
        sir_use_blur_ill = True,
        hvi_eps = 1e-8,
    ):

        super(CANDLE, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_depth = OverlapPatchEmbed(query_dim, dim)
        # Stage-wise DINO projection: L6/L12/L18/L24 -> model level1/2/3/4.
        self.dino_to_query = nn.ModuleList([
            nn.Conv2d(dino_dim, query_dim, kernel_size=1, bias=False) for _ in range(4)
        ])
        self.dino_to_query_norm = nn.ModuleList([nn.GroupNorm(1, query_dim) for _ in range(4)])
        self.use_psf_dr = bool(use_psf_dr)
        self.psf_stage_candidates = [
            [0, 1],      # Stage 1: L6 + L12
            [0, 1, 2],   # Stage 2: L6 + L12 + L18
            [1, 2, 3],   # Stage 3: L12 + L18 + L24
            [2, 3],      # Stage 4: L18 + L24
        ]
        self.dr_stage_reduction = [8, 4, 2, 1]
        stage_dims = [dim, int(dim * 2**1), int(dim * 2**2), int(dim * 2**3)]
        self.psf_fusion = nn.ModuleList([
            LayerGatedDinoFusion(
                dino_dim=dino_dim,
                query_dim=query_dim,
                stage_feat_dim=stage_dims[i],
                num_candidates=len(self.psf_stage_candidates[i]),
                gate_hidden=psf_gate_hidden,
            )
            for i in range(4)
        ])
        self.dr_fusion = nn.ModuleList([
            DRFusionBlock(
                feat_dim=stage_dims[i],
                query_dim=query_dim,
                heads=dr_heads,
                dropout=dr_dropout,
                alpha_init=dr_alpha_init,
                spatial_reduction=self.dr_stage_reduction[i],
            )
            for i in range(4)
        ])
        self.dino_to_depth_l2 = nn.Conv2d(query_dim, int(dim * 2**1), kernel_size=1, bias=False)
        self.dino_to_depth_l3 = nn.Conv2d(query_dim, int(dim * 2**2), kernel_size=1, bias=False)
        self.dino_to_depth_l4 = nn.Conv2d(query_dim, int(dim * 2**3), kernel_size=1, bias=False)
        self.dino_to_depth_l2_norm = nn.GroupNorm(1, int(dim * 2**1))
        self.dino_to_depth_l3_norm = nn.GroupNorm(1, int(dim * 2**2))
        self.dino_to_depth_l4_norm = nn.GroupNorm(1, int(dim * 2**3))


        self.decoder = decoder
        self.use_sffb_decoder = bool(use_sffb_decoder)
        self.use_bfacg_decoder = bool(use_bfacg_decoder)
        self.use_clp_decoder = bool(use_clp_decoder)
        self.bfacg_variant = str(bfacg_variant).lower()
        if self.bfacg_variant not in ("v1", "v1_5"):
            raise ValueError(f"Unsupported bfacg_variant='{bfacg_variant}'. Expected one of ['v1', 'v1_5'].")
        self.use_ica7 = bool(use_ica7)
        self.use_abc_ica = bool(use_abc_ica)
        self.abc_detach_aux_weight = bool(abc_detach_aux_weight)
        if self.use_ica7 and self.use_abc_ica:
            raise ValueError("use_ica7 and use_abc_ica are mutually exclusive.")

        if self.decoder:
            self.prompt1 = HVIPromptBlock(feat_dim=int(dim * 2**1), hidden=64, use_hv=False, hvi_eps=hvi_eps)
            self.prompt2 = HVIPromptBlock(feat_dim=int(dim * 2**2), hidden=64, use_hv=True, hvi_eps=hvi_eps)
            self.prompt3 = HVIPromptBlock(feat_dim=int(dim * 2**2), hidden=64, use_hv=True, hvi_eps=hvi_eps)

        if self.use_ica7:
            self.ica_prior = ICAPriorEncoder(in_ch=inp_channels, aux_ch=ica_aux_ch)
            self.ica_enc1 = ICAInjectBlock(feat_ch=dim, aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_enc2 = ICAInjectBlock(feat_ch=int(dim * 2**1), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_enc3 = ICAInjectBlock(feat_ch=int(dim * 2**2), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_latent = ICAInjectBlock(feat_ch=int(dim * 2**3), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_dec3 = ICAInjectBlock(feat_ch=int(dim * 2**2), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_dec2 = ICAInjectBlock(feat_ch=int(dim * 2**1), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)
            self.ica_dec1 = ICAInjectBlock(feat_ch=int(dim * 2**1), aux_ch=ica_aux_ch, hidden=ica_hidden, res_scale_init=ica_res_scale_init)

        self.encoder_level1 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level1_depth = depth_encoder(input_dim=dim, output_dim=dim, num_layers=3)
        self.down1_2_depth = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level2_depth = depth_encoder(input_dim=int(dim*2**1), output_dim=int(dim*2**1), num_layers=3)
        self.down2_3_depth = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[GeometryAwareTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        
        self.encoder_level3_depth = depth_encoder(input_dim=int(dim*2**2), output_dim=int(dim*2**2), num_layers=3)
        self.down3_4_depth = Downsample(int(dim*2**2)) ## From Level 2 to Level 3

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.use_hvi_bottleneck = bool(use_hvi_bottleneck)
        self.hvi_bottleneck = HVI_SIR_Bottleneck(
            dim_l4=int(dim * 2**3),
            attn_dim=sir_attn_dim,
            heads=sir_heads,
            dropout=sir_dropout,
            lambda_init=sir_lambda_init,
            hvi_eps=hvi_eps,
            use_blur_ill=bool(sir_use_blur_ill),
        )
        self.latent_to_dec3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.sffb3 = PreConcatSFFB(high_ch=int(dim * 2**1), low_ch=int(dim * 2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+(dim*4), int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.sffb2 = PreConcatSFFB(high_ch=int(dim * 2**1), low_ch=int(dim * 2**1))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.bfacg2 = BFACG(
            feat_ch=int(dim * 2**1),
            enc_ch=int(dim * 2**1),
            hidden=bfacg_hidden,
            res_scale_init=bfacg_res_scale_init,
            neutral_strength=bfacg_neutral_k if self.bfacg_variant == "v1_5" else 0.0,
        )
        if self.use_clp_decoder:
            self.clp2 = ColorLineProjection(
                feat_ch=int(dim * 2**1),
                latent_ch=clp_latent_ch,
                res_scale_init=clp_res_scale_init,
                conf_bias_init=clp_conf_bias_init,
            )
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.bfacg1 = BFACG(
            feat_ch=int(dim * 2**1),
            enc_ch=dim,
            hidden=bfacg_hidden,
            res_scale_init=bfacg_res_scale_init,
            neutral_strength=bfacg_neutral_k if self.bfacg_variant == "v1_5" else 0.0,
        )
        if self.use_clp_decoder:
            self.clp1 = ColorLineProjection(
                feat_ch=int(dim * 2**1),
                latent_ch=clp_latent_ch,
                res_scale_init=clp_res_scale_init,
                conf_bias_init=clp_conf_bias_init,
            )

        if self.use_abc_ica:
            self.abc_aux_branch = ABCAuxBranch(
                embed_dim=int(abc_aux_embed_dim),
                hist_ckpt_path=abc_hist_ckpt_path,
                lab_ckpt_path=abc_lab_ckpt_path,
                token_projection="linear",
                token_mlp="TwoDCFF",
            )
            self.abc_stage_block_counts = [
                len(self.encoder_level1),
                len(self.encoder_level2),
                len(self.encoder_level3),
                len(self.latent),
                len(self.decoder_level3),
                len(self.decoder_level2),
                len(self.decoder_level1),
            ]
            self.abc_stage_aux_dims = [
                int(abc_aux_embed_dim),
                int(abc_aux_embed_dim * 2),
                int(abc_aux_embed_dim * 4),
                int(abc_aux_embed_dim * 8),
                int(abc_aux_embed_dim * 8),
                int(abc_aux_embed_dim * 4),
                int(abc_aux_embed_dim * 2),
            ]
            self.abc_stage_target_dims = [
                dim,
                int(dim * 2**1),
                int(dim * 2**2),
                int(dim * 2**3),
                int(dim * 2**2),
                int(dim * 2**1),
                int(dim * 2**1),
            ]
            self.abc_hist_proj = nn.ModuleList([
                nn.Linear(self.abc_stage_aux_dims[i], self.abc_stage_target_dims[i], bias=False)
                for i in range(7)
            ])
            self.abc_lab_proj = nn.ModuleList([
                nn.Linear(self.abc_stage_aux_dims[i], self.abc_stage_target_dims[i], bias=False)
                for i in range(7)
            ])
            self.abc_hist_scale = nn.ParameterList([
                nn.Parameter(torch.zeros(n), requires_grad=True) for n in self.abc_stage_block_counts
            ])
            self.abc_lab_scale = nn.ParameterList([
                nn.Parameter(torch.zeros(n), requires_grad=True) for n in self.abc_stage_block_counts
            ])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    # 🔧 新增這個小函數
    def _run_refinement(self, x):
        for blk in self.refinement:
            x = blk(x)
        return x

    def _norm_query(self, x):
        return x

    def _to_query_stage(self, dino_stage, stage_idx, out_hw):
        x = F.interpolate(dino_stage, size=out_hw, mode="bilinear", align_corners=False)
        x = self.dino_to_query[stage_idx](x)
        x = self.dino_to_query_norm[stage_idx](x)
        return x

    def _to_query_stage_psf(self, dino_stages, stage_feat, stage_idx, out_hw):
        cand_idx = self.psf_stage_candidates[stage_idx]
        dino_candidates = [dino_stages[i] for i in cand_idx]
        return self.psf_fusion[stage_idx](dino_candidates, stage_feat, out_hw)

    def _split_dino_stages(self, dino_tokens):
        # New format: [B, 4, C, H, W]
        if dino_tokens.ndim == 5:
            if dino_tokens.shape[1] != 4:
                raise ValueError(f"Expected 4 DINO stages, got shape {tuple(dino_tokens.shape)}")
            return [dino_tokens[:, i] for i in range(4)]
        # Backward compatibility: single-stage [B, C, H, W]
        if dino_tokens.ndim == 4:
            return [dino_tokens, dino_tokens, dino_tokens, dino_tokens]
        raise ValueError(f"Unsupported dino_tokens shape: {tuple(dino_tokens.shape)}")

    def forward_abc_aux(self, input_hist, input_lab_hist):
        if not self.use_abc_ica:
            raise RuntimeError("forward_abc_aux is only available when use_abc_ica=1")
        return self.abc_aux_branch(input_hist, input_lab_hist)

    def get_abc_aux_modules(self):
        if not self.use_abc_ica:
            return None, None
        return self.abc_aux_branch.hist_branch, self.abc_aux_branch.lab_branch

    @staticmethod
    def expand_stage_weights_2_to_n(stage_weights, out_blocks):
        if stage_weights is None:
            return None
        if not isinstance(stage_weights, (list, tuple)) or len(stage_weights) == 0:
            raise ValueError("stage_weights must be a non-empty list/tuple of tensors.")
        if len(stage_weights) == out_blocks:
            return list(stage_weights)

        src = torch.stack([w if w.ndim == 3 else w.unsqueeze(1) for w in stage_weights], dim=-1)  # [B, 1, C, K]
        b, one, c, k = src.shape
        src = src.contiguous().view(b * one * c, 1, k)
        interp = F.interpolate(src, size=out_blocks, mode="linear", align_corners=True)
        interp = interp.view(b, one, c, out_blocks)
        return [interp[..., i] for i in range(out_blocks)]

    def _prepare_abc_weights(self, hist_raw, lab_raw):
        if hist_raw is None or lab_raw is None:
            return None, None
        if len(hist_raw) != 7 or len(lab_raw) != 7:
            raise ValueError("ABC raw weights must contain 7 stages.")

        hist_stage_weights = []
        lab_stage_weights = []
        for stage_idx in range(7):
            n_blocks = self.abc_stage_block_counts[stage_idx]
            hist_expanded = self.expand_stage_weights_2_to_n(hist_raw[stage_idx], n_blocks)
            lab_expanded = self.expand_stage_weights_2_to_n(lab_raw[stage_idx], n_blocks)

            hist_stage = []
            lab_stage = []
            for blk_idx in range(n_blocks):
                h_in = hist_expanded[blk_idx]
                l_in = lab_expanded[blk_idx]
                h_w = self.abc_hist_proj[stage_idx](h_in.squeeze(1)).unsqueeze(1)
                l_w = self.abc_lab_proj[stage_idx](l_in.squeeze(1)).unsqueeze(1)
                hist_stage.append(h_w)
                lab_stage.append(l_w)
            hist_stage_weights.append(hist_stage)
            lab_stage_weights.append(lab_stage)
        return hist_stage_weights, lab_stage_weights

    def _apply_abc_injection(self, feat, hist_w, lab_w, stage_idx, block_idx):
        if hist_w is None or lab_w is None:
            return feat

        if self.abc_detach_aux_weight:
            hist_w = hist_w.detach()
            lab_w = lab_w.detach()

        if feat.ndim == 4:
            h_w = hist_w.transpose(1, 2).unsqueeze(-1)  # [B, C, 1, 1]
            l_w = lab_w.transpose(1, 2).unsqueeze(-1)   # [B, C, 1, 1]
        elif feat.ndim == 3:
            h_w = hist_w
            l_w = lab_w
        else:
            raise ValueError(f"Unsupported feat ndim for ABC injection: {feat.ndim}")

        lam_h = self.abc_hist_scale[stage_idx][block_idx]
        lam_l = self.abc_lab_scale[stage_idx][block_idx]
        return feat + lam_h * h_w * feat + lam_l * l_w * feat

    def _run_geometry_stage(self, blocks, rgb, depth, stage_idx, hist_stage=None, lab_stage=None):
        x = torch.cat([rgb, depth], dim=1)
        for blk_idx, blk in enumerate(blocks):
            if self.training and torch.is_grad_enabled():
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if hist_stage is not None and lab_stage is not None:
                rgb_i, depth_i = torch.chunk(x, 2, dim=1)
                rgb_i = self._apply_abc_injection(rgb_i, hist_stage[blk_idx], lab_stage[blk_idx], stage_idx, blk_idx)
                x = torch.cat([rgb_i, depth_i], dim=1)
        return x

    def _run_transformer_stage(self, blocks, feat, stage_idx, hist_stage=None, lab_stage=None):
        x = feat
        for blk_idx, blk in enumerate(blocks):
            if self.training and torch.is_grad_enabled():
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if hist_stage is not None and lab_stage is not None:
                x = self._apply_abc_injection(x, hist_stage[blk_idx], lab_stage[blk_idx], stage_idx, blk_idx)
        return x

    def forward(
        self,
        inp_img,
        dino_tokens,
        input_hist=None,
        input_lab_hist=None,
        hist_raw_weights=None,
        lab_raw_weights=None,
        noise_emb=None,
    ):
        _, _, h, w = inp_img.shape
        new_h, new_w = (h // 8) * 8, (w // 8) * 8
        if (h != new_h) or (w != new_w):
            inp_img = inp_img[:, :, :new_h, :new_w]
            #print(f"[AutoCrop] resized input from ({h},{w}) to ({new_h},{new_w})")

        with torch.cuda.amp.autocast():
            dino_s1, dino_s2, dino_s3, dino_s4 = self._split_dino_stages(dino_tokens)
            dino_stages = [dino_s1, dino_s2, dino_s3, dino_s4]
            ica_priors = self.ica_prior(inp_img) if self.use_ica7 else [None] * 7
            hist_stage_weights = [None] * 7
            lab_stage_weights = [None] * 7
            if self.use_abc_ica:
                if hist_raw_weights is None or lab_raw_weights is None:
                    if input_hist is None or input_lab_hist is None:
                        raise ValueError("input_hist/input_lab_hist are required when use_abc_ica=1.")
                    _, _, hist_raw_weights, lab_raw_weights = self.forward_abc_aux(input_hist, input_lab_hist)
                hist_stage_weights, lab_stage_weights = self._prepare_abc_weights(hist_raw_weights, lab_raw_weights)

            # ---- Encoder ----
            inp_enc_level1 = self.patch_embed(inp_img)
            if self.use_psf_dr:
                query_map_l1 = self._to_query_stage_psf(dino_stages, inp_enc_level1, stage_idx=0, out_hw=inp_img.shape[-2:])
                inp_enc_level1 = self.dr_fusion[0](inp_enc_level1, query_map_l1)
            else:
                query_map_l1 = self._to_query_stage(dino_s1, stage_idx=0, out_hw=inp_img.shape[-2:])
            depth_enc_level1 = self.patch_embed_depth(query_map_l1)
            out_enc_level1 = self._run_geometry_stage(
                self.encoder_level1,
                inp_enc_level1,
                depth_enc_level1,
                stage_idx=0,
                hist_stage=hist_stage_weights[0],
                lab_stage=lab_stage_weights[0],
            )
            out_enc_level1, _ = torch.chunk(out_enc_level1, 2, dim=1)
            if self.use_ica7:
                out_enc_level1 = self.ica_enc1(out_enc_level1, ica_priors[0])
            _ = self.encoder_level1_depth(depth_enc_level1)

            inp_enc_level2 = self.down1_2(out_enc_level1)
            if self.use_psf_dr:
                query_map_l2 = self._to_query_stage_psf(dino_stages, inp_enc_level2, stage_idx=1, out_hw=inp_enc_level2.shape[-2:])
                inp_enc_level2 = self.dr_fusion[1](inp_enc_level2, query_map_l2)
            else:
                query_map_l2 = self._to_query_stage(dino_s2, stage_idx=1, out_hw=inp_enc_level2.shape[-2:])
            inp_depth_level2 = self.dino_to_depth_l2_norm(self.dino_to_depth_l2(query_map_l2))
            out_enc_level2 = self._run_geometry_stage(
                self.encoder_level2,
                inp_enc_level2,
                inp_depth_level2,
                stage_idx=1,
                hist_stage=hist_stage_weights[1],
                lab_stage=lab_stage_weights[1],
            )
            out_enc_level2, _ = torch.chunk(out_enc_level2, 2, dim=1)
            if self.use_ica7:
                out_enc_level2 = self.ica_enc2(out_enc_level2, ica_priors[1])
            _ = self.encoder_level2_depth(inp_depth_level2)

            inp_enc_level3 = self.down2_3(out_enc_level2)
            if self.use_psf_dr:
                query_map_l3 = self._to_query_stage_psf(dino_stages, inp_enc_level3, stage_idx=2, out_hw=inp_enc_level3.shape[-2:])
                inp_enc_level3 = self.dr_fusion[2](inp_enc_level3, query_map_l3)
            else:
                query_map_l3 = self._to_query_stage(dino_s3, stage_idx=2, out_hw=inp_enc_level3.shape[-2:])
            inp_depth_level3 = self.dino_to_depth_l3_norm(self.dino_to_depth_l3(query_map_l3))
            out_enc_level3 = self._run_geometry_stage(
                self.encoder_level3,
                inp_enc_level3,
                inp_depth_level3,
                stage_idx=2,
                hist_stage=hist_stage_weights[2],
                lab_stage=lab_stage_weights[2],
            )
            out_enc_level3, _ = torch.chunk(out_enc_level3, 2, dim=1)
            if self.use_ica7:
                out_enc_level3 = self.ica_enc3(out_enc_level3, ica_priors[2])
            _ = self.encoder_level3_depth(inp_depth_level3)

            inp_enc_level4 = self.down3_4(out_enc_level3)
            if self.use_psf_dr:
                query_map_l4 = self._to_query_stage_psf(dino_stages, inp_enc_level4, stage_idx=3, out_hw=inp_enc_level4.shape[-2:])
                inp_enc_level4 = self.dr_fusion[3](inp_enc_level4, query_map_l4)
            else:
                query_map_l4 = self._to_query_stage(dino_s4, stage_idx=3, out_hw=inp_enc_level4.shape[-2:])
            latent_depth = self.dino_to_depth_l4_norm(self.dino_to_depth_l4(query_map_l4))
            inp_enc_level4 = inp_enc_level4 + latent_depth

            # ---- Latent ----
            latent = self._run_transformer_stage(
                self.latent,
                inp_enc_level4,
                stage_idx=3,
                hist_stage=hist_stage_weights[3],
                lab_stage=lab_stage_weights[3],
            )
            if self.use_hvi_bottleneck:
                latent = self.hvi_bottleneck(latent, inp_img)
            if self.use_ica7:
                latent = self.ica_latent(latent, ica_priors[3])

            # ---- Decoder level3 ----
            latent_dec3 = self.latent_to_dec3(latent)
            inp_dec_level3 = self.up4_3(latent_dec3)
            min_h = min(inp_dec_level3.shape[2], out_enc_level3.shape[2])
            min_w = min(inp_dec_level3.shape[3], out_enc_level3.shape[3])
            inp_dec_level3 = inp_dec_level3[:, :, :min_h, :min_w]
            out_enc_level3 = out_enc_level3[:, :, :min_h, :min_w]
            if self.use_sffb_decoder:
                out_enc_level3 = self.sffb3(inp_dec_level3, out_enc_level3)
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
            inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
            if self.decoder:
                gamma3, beta3, conf3 = self.prompt3(inp_dec_level3, inp_img)
                inp_dec_level3 = inp_dec_level3 * (1.0 + conf3 * gamma3) + conf3 * beta3
                inp_dec_level3 = self.noise_level3(inp_dec_level3)
            out_dec_level3 = self._run_transformer_stage(
                self.decoder_level3,
                inp_dec_level3,
                stage_idx=4,
                hist_stage=hist_stage_weights[4],
                lab_stage=lab_stage_weights[4],
            )
            if self.use_ica7:
                out_dec_level3 = self.ica_dec3(out_dec_level3, ica_priors[4])

            # ---- Decoder level2 ----
            if self.decoder:
                gamma2, beta2, conf2 = self.prompt2(out_dec_level3, inp_img)
                out_dec_level3 = out_dec_level3 * (1.0 + conf2 * gamma2) + conf2 * beta2
                out_dec_level3 = self.noise_level2(out_dec_level3)

            inp_dec_level2 = self.up3_2(out_dec_level3)
            min_h = min(inp_dec_level2.shape[2], out_enc_level2.shape[2])
            min_w = min(inp_dec_level2.shape[3], out_enc_level2.shape[3])
            inp_dec_level2 = inp_dec_level2[:, :, :min_h, :min_w]
            out_enc_level2 = out_enc_level2[:, :, :min_h, :min_w]
            if self.use_sffb_decoder:
                out_enc_level2 = self.sffb2(inp_dec_level2, out_enc_level2)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self._run_transformer_stage(
                self.decoder_level2,
                inp_dec_level2,
                stage_idx=5,
                hist_stage=hist_stage_weights[5],
                lab_stage=lab_stage_weights[5],
            )
            if self.use_ica7:
                out_dec_level2 = self.ica_dec2(out_dec_level2, ica_priors[5])
            if self.use_bfacg_decoder:
                out_dec_level2 = self.bfacg2(out_dec_level2, out_enc_level2, inp_img)
                if self.use_clp_decoder:
                    out_dec_level2 = self.clp2(out_dec_level2)

            # ---- Decoder level1 ----
            if self.decoder:
                gamma1, beta1, conf1 = self.prompt1(out_dec_level2, inp_img)
                out_dec_level2 = out_dec_level2 * (1.0 + conf1 * gamma1) + conf1 * beta1
                out_dec_level2 = self.noise_level1(out_dec_level2)

            inp_dec_level1 = self.up2_1(out_dec_level2)
            min_h = min(inp_dec_level1.shape[2], out_enc_level1.shape[2])
            min_w = min(inp_dec_level1.shape[3], out_enc_level1.shape[3])
            inp_dec_level1 = inp_dec_level1[:, :, :min_h, :min_w]
            out_enc_level1 = out_enc_level1[:, :, :min_h, :min_w]
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
            out_dec_level1 = self._run_transformer_stage(
                self.decoder_level1,
                inp_dec_level1,
                stage_idx=6,
                hist_stage=hist_stage_weights[6],
                lab_stage=lab_stage_weights[6],
            )
            if self.use_ica7:
                out_dec_level1 = self.ica_dec1(out_dec_level1, ica_priors[6])
            if self.use_bfacg_decoder:
                out_dec_level1 = self.bfacg1(out_dec_level1, out_enc_level1, inp_img)
                if self.use_clp_decoder:
                    out_dec_level1 = self.clp1(out_dec_level1)

            # ---- Output ----
            out_dec_level1 = self._run_refinement(out_dec_level1)
            out_dec_level1 = self.output(out_dec_level1)

            if out_dec_level1.shape[2:] != inp_img.shape[2:]:
                out_dec_level1 = F.interpolate(
                    out_dec_level1, size=inp_img.shape[2:], mode='bilinear', align_corners=False
                )
                #print(f"[AutoResize] adjusted output from {out_dec_level1.shape[2:]} to {inp_img.shape[2:]}")

            out_dec_level1 = out_dec_level1 + inp_img

            del inp_img, dino_tokens
            return out_dec_level1
