"""
lightx2v/common/ops/attn/__init__.py

H100（flash_attn3）のみ即時import。
その他は実際に使う時に遅延importされる。
"""

# H100で使用中（即時import）
from .flash_attn import FlashAttn3Weight
from .torch_sdpa import TorchSDPAWeight  # フォールバック用

# 以下は遅延import（使用時に自動的にimportされる）
# Ampere/Ada/Hopper共通
# from .flash_attn import FlashAttn2Weight

# Blackwell（RTX5090）用
# from .flash_attn import FlashAttn4Weight, SparseFlashAttn4Weight
# from .sage_attn import SageAttn3Weight, SparseSageAttn3Weight

# SageAttention（H100/A100/RTX4090等）
# from .sage_attn import SageAttn2Weight, SparseSageAttn2Weight

# マルチGPU分散アテンション
# from .ring_attn import RingAttnWeight
# from .ulysses_attn import UlyssesAttnWeight, Ulysses4090AttnWeight

# 特殊用途
# from .draft_attn import DraftAttnWeight
# from .general_sparse_attn import GeneralSparseAttnWeight
# from .nbhd_attn import NbhdAttnWeight, NbhdAttnWeightFlashInfer
# from .radial_attn import RadialAttnWeight
# from .sla_attn import SlaAttnWeight
# from .sparge_attn import SpargeAttnWeight
# from .sparse_mask_generator import NbhdMaskGenerator, SlaMaskGenerator, SpargeMaskGenerator, SvgMaskGenerator
# from .sparse_operator import FlashinferOperator, FlexBlockOperator, MagiOperator, SlaTritonOperator, SparseFlashAttentionV4Operator, SparseSageAttentionV2Operator, SparseSageAttentionV3Operator
# from .svg2_attn import Svg2AttnWeight
# from .svg_attn import SvgAttnWeight
