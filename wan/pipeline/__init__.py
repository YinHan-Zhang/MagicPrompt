from .pipeline_wan2_2 import Wan2_2Pipeline
from .pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline
from .pipeline_wan2_2_fun_inpaint import Wan2_2FunInpaintPipeline
from .pipeline_wan2_2_fun_control_ori import Wan2_2FunControlPipeline_ori
from .pipeline_wan2_2_fun_inpaint_ori import Wan2_2FunInpaintPipeline_ori
from .pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline



# Backwards-compatible aliases. Some names belong to pipeline variants that are not
# included in this checkout (e.g. Wan2.1 / VACE / Phantom); guard them so import still works.
try:
    WanFunPipeline = WanPipeline
except NameError:
    pass
try:
    WanI2VPipeline = WanFunInpaintPipeline
except NameError:
    pass

Wan2_2FunPipeline = Wan2_2Pipeline
Wan2_2I2VPipeline = Wan2_2FunInpaintPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunInpaintPipeline.__call__ = sparse_reset(WanFunInpaintPipeline.__call__)
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)
    WanI2VPipeline.__call__ = sparse_reset(WanI2VPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)
    WanVacePipeline.__call__ = sparse_reset(WanVacePipeline.__call__)

    # Phantom
    WanFunPhantomPipeline.__call__ = sparse_reset(WanFunPhantomPipeline.__call__)

    # Wan2.2
    Wan2_2FunInpaintPipeline.__call__ = sparse_reset(Wan2_2FunInpaintPipeline.__call__)
    Wan2_2FunPipeline.__call__ = sparse_reset(Wan2_2FunPipeline.__call__)
    Wan2_2FunControlPipeline.__call__ = sparse_reset(Wan2_2FunControlPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)
    Wan2_2TI2VPipeline.__call__ = sparse_reset(Wan2_2TI2VPipeline.__call__)
    Wan2_2S2VPipeline.__call__ = sparse_reset(Wan2_2S2VPipeline.__call__)
    Wan2_2VaceFunPipeline.__call__ = sparse_reset(Wan2_2VaceFunPipeline.__call__)
    Wan2_2AnimatePipeline.__call__ = sparse_reset(Wan2_2AnimatePipeline.__call__)