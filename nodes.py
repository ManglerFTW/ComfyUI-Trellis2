import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

from .trellis2.pipelines import Trellis2ImageTo3DPipeline

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
tensor2pil = transforms.ToPILImage()  
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array

class Trellis2LoadModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelname": ("STRING",["microsoft/TRELLIS.2-4B"]),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Trellis2Wrapper"

    def process(self, modelname):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
        
        return (pipeline,)


NODE_CLASS_MAPPINGS = {
    "Trellis2LoadModel": Trellis2LoadModel,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadModel": "Trellis2 - LoadModel",
    }
