# --------------------------------------------------------
# Reference from HRNet-Human-Pose-Estimation
# refact code from old one.
# Written by Jeongmin Kim(jm.kim@dankook.ac.kr)
# ----------------------------------------------------
from .vgg19 import VGG19
from .yolo import YOLOv5

__factory = {
    'VGG19' : VGG19,
    'YOLOV5' : YOLOv5,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)