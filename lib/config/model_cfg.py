# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# Your model related params
# examples
VGG19 = CN(new_allowed=True)
VGG19.INPUT_SIZE = [224, 224, 3]
VGG19.NUM_CLASSES = 3


MODEL_EXTRAS = {
    'VGG19': VGG19,
}
