# Wenny add multi_apply from OpenMMLab
#-------------------------------------------------------------------------
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .misc import add_prefix, multi_apply
from .dist_utils import (DistOptimizerHook, all_reduce_dict, allreduce_grads,
                         reduce_mean)


__all__ = [
    'add_prefix', 'multi_apply', 'DistOptimizerHook', 'allreduce_grads',
    'all_reduce_dict', 'reduce_mean'
]
