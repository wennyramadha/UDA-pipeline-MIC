# Wenny adds mask2former encoder-decoder
# -------------------------------------------------------
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .hrda_encoder_decoder import HRDAEncoderDecoder
from .encoder_decoder_mask2former import EncoderDecoderMask2Former
from .encoder_decoder_mask2former_aug import EncoderDecoderMask2FormerAug

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'HRDAEncoderDecoder', 'EncoderDecoderMask2Former', 'EncoderDecoderMask2FormerAug']
