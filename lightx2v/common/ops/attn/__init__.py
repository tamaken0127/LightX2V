import time
_t = time.time()

from .flash_attn import FlashAttn2Weight, FlashAttn3Weight, FlashAttn4Weight, SparseFlashAttn4Weight
print(f"[Timing/attn] flash_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .general_sparse_attn import GeneralSparseAttnWeight
print(f"[Timing/attn] general_sparse_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

# ring_attn: @torch.jit.scriptで29秒かかるためコメントアウト
# from .ring_attn import RingAttnWeight
print(f"[Timing/attn] ring_attn: SKIPPED", flush=True); _t = time.time()

from .sage_attn import SageAttn2Weight, SageAttn3Weight, SparseSageAttn2Weight, SparseSageAttn3Weight
print(f"[Timing/attn] sage_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .sla_attn import SlaAttnWeight
print(f"[Timing/attn] sla_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .sparge_attn import SpargeAttnWeight
print(f"[Timing/attn] sparge_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .sparse_mask_generator import NbhdMaskGenerator, SlaMaskGenerator, SpargeMaskGenerator, SvgMaskGenerator
print(f"[Timing/attn] sparse_mask_generator: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .sparse_operator import FlashinferOperator, FlexBlockOperator, MagiOperator, SlaTritonOperator, SparseFlashAttentionV4Operator, SparseSageAttentionV2Operator, SparseSageAttentionV3Operator
print(f"[Timing/attn] sparse_operator: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .torch_sdpa import TorchSDPAWeight
print(f"[Timing/attn] torch_sdpa: {time.time()-_t:.2f}s", flush=True); _t = time.time()

# ulysses_attn: 30秒かかるためコメントアウト
# from .ulysses_attn import Ulysses4090AttnWeight, UlyssesAttnWeight
print(f"[Timing/attn] ulysses_attn: SKIPPED", flush=True); _t = time.time()

from .draft_attn import DraftAttnWeight
print(f"[Timing/attn] draft_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .nbhd_attn import NbhdAttnWeight, NbhdAttnWeightFlashInfer
print(f"[Timing/attn] nbhd_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .radial_attn import RadialAttnWeight
print(f"[Timing/attn] radial_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .svg2_attn import Svg2AttnWeight
print(f"[Timing/attn] svg2_attn: {time.time()-_t:.2f}s", flush=True); _t = time.time()

from .svg_attn import SvgAttnWeight
print(f"[Timing/attn] svg_attn: {time.time()-_t:.2f}s", flush=True)
