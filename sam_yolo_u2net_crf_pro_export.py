#!/usr/bin/env python3
"""
Professional Segmentation Suite: YOLO-World + SAM + U²-Net + CRF

Comprehensive pipeline integrating:
- YOLO-World for accurate object detection
- SAM for high-quality semantic segmentation
- U²-Net for mask refinement and boundary enhancement
- CRF for precise contour refinement
- Professional SVG/DXF export with customizable quality levels

Author: FETHI
Date: 2025-03-21
Version: 9.0.0
"""

import os
import sys
import cv2
import json
import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.measure import find_contours
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
import io

from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv
from segment_anything import sam_model_registry, SamPredictor

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    SVG_VERIFICATION_AVAILABLE = True
except ImportError:
    SVG_VERIFICATION_AVAILABLE = False

# Script information
SCRIPT_VERSION = "9.0.0"
SCRIPT_DATE = "2025-03-21 13:40:07"
SCRIPT_AUTHOR = "FETHI"

print(f"Professional Segmentation Suite v{SCRIPT_VERSION}")
print(f"Date: {SCRIPT_DATE}")
print(f"Author: {SCRIPT_AUTHOR}")
print("-" * 60)

# Initialize annotators for visualization
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                text_scale=0.5,
                                text_thickness=1)


#################################
# U²-Net Model Definition
#################################

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = F.interpolate(hx6d,scale_factor=2,mode='bilinear',align_corners=True)
        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = F.interpolate(hx5d,scale_factor=2,mode='bilinear',align_corners=True)
        hx4d =  self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d,scale_factor=2,mode='bilinear',align_corners=True)
        hx3d =  self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,scale_factor=2,mode='bilinear',align_corners=True)
        hx2d =  self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,scale_factor=2,mode='bilinear',align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = F.interpolate(hx5d,scale_factor=2,mode='bilinear',align_corners=True)
        hx4d =  self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d,scale_factor=2,mode='bilinear',align_corners=True)
        hx3d =  self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,scale_factor=2,mode='bilinear',align_corners=True)
        hx2d =  self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,scale_factor=2,mode='bilinear',align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d =  self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = F.interpolate(hx4d,scale_factor=2,mode='bilinear',align_corners=True)
        hx3d =  self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,scale_factor=2,mode='bilinear',align_corners=True)
        hx2d =  self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,scale_factor=2,mode='bilinear',align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d =  self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = F.interpolate(hx3d,scale_factor=2,mode='bilinear',align_corners=True)
        hx2d =  self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,scale_factor=2,mode='bilinear',align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d =  self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d =  self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))
        return hx1d + hxin

class U2NET(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage6 = RSU4F(512,256,512)
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        hx = x
        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        #stage 6
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6,scale_factor=2,mode='bilinear',align_corners=True)
        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = F.interpolate(hx5d,scale_factor=2,mode='bilinear',align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d,scale_factor=2,mode='bilinear',align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,scale_factor=2,mode='bilinear',align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,scale_factor=2,mode='bilinear',align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        #side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2,scale_factor=2,mode='bilinear',align_corners=True)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3,scale_factor=4,mode='bilinear',align_corners=True)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4,scale_factor=8,mode='bilinear',align_corners=True)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5,scale_factor=16,mode='bilinear',align_corners=True)
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6,scale_factor=32,mode='bilinear',align_corners=True)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


#################################
# Main Classes and Functions
#################################

class U2NetPredictor:
    """Predictor class for U²-Net"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize the U²-Net model"""
        self.device = device
        self.model = self.load_model(checkpoint_path, device)
        
    def load_model(self, checkpoint_path, device):
        """Load the U²-Net model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ U²-Net checkpoint not found: {checkpoint_path}")
            return None
            
        try:
            print(f"Loading U²-Net weights from {checkpoint_path}")
            
            # Create model and load weights directly
            model = U2NET(3, 1)  # 3 input channels, 1 output channel
            
            # Load state dict based on device
            if device == 'cpu':
                state_dict = torch.load(checkpoint_path, map_location='cpu')
            else:
                state_dict = torch.load(checkpoint_path, map_location=device)
                
            # Try loading the state dict with different approaches to handle different formats
            try:
                model.load_state_dict(state_dict)
            except:
                print("⚠️ First loading attempt failed, trying modified approach...")
                # Sometimes the state dict has a different structure, try to fix it
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if k.startswith('module.'):
                        name = k[7:]  # Remove 'module.' prefix
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            
            model.to(device)
            model.eval()
            print("✓ U²-Net loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading U²-Net: {str(e)}")
            traceback.print_exc()
            return None
            
    def predict(self, image, size=320):
        """
        Predict using U²-Net
        
        Args:
            image: RGB image as numpy array
            size: Input size for U²-Net (will resize)
            
        Returns:
            Prediction mask
        """
        if self.model is None:
            print("⚠️ U²-Net model not available")
            return None
            
        try:
            # Preprocess the image
            image_resized = cv2.resize(image, (size, size))
            image_tensor = self.preprocess(image_resized)
            
            # Run inference
            with torch.no_grad():
                # Forward pass
                outputs = self.model(image_tensor)
                
                # Get the final output (first element in tuple)
                pred = outputs[0]
                
            # Convert to numpy and resize back to original size
            pred_np = pred.cpu().squeeze().numpy()
            pred_resized = cv2.resize(pred_np, (image.shape[1], image.shape[0]))
            
            return pred_resized
            
        except Exception as e:
            print(f"❌ Error during U²-Net inference: {str(e)}")
            traceback.print_exc()
            return None
            
    def preprocess(self, image):
        """Preprocess image for U²-Net"""
        try:
            tmp = image.copy()
            tmp = tmp / 255.0
            tmp = tmp.transpose((2, 0, 1))  # H,W,C -> C,H,W
            tmp = tmp.astype(np.float32)
            
            # Convert to tensor
            tensor = torch.from_numpy(tmp)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            tensor = tensor.to(self.device)
            
            return tensor
        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            traceback.print_exc()
            return None


class CRFProcessor:
    """Class for refining masks with Conditional Random Fields"""
    
    def __init__(self):
        self.default_params = {
            'bilateral_sxy': 80,       # Spatial standard deviation
            'bilateral_srgb': 13,      # Color standard deviation
            'bilateral_compat': 10,    # Compatibility parameter for bilateral potentials
            'gaussian_sxy': 3,         # Spatial standard deviation for Gaussian potentials
            'gaussian_compat': 3,      # Compatibility parameter for Gaussian potentials
            'iterations': 10,          # Number of inference iterations
        }
    
    def apply_crf(self, image, mask, params=None):
        """
        Refine mask using DenseCRF
        
        Args:
            image: RGB image as numpy array
            mask: Binary mask as numpy array
            params: CRF parameters (optional)
            
        Returns:
            Refined mask as numpy array
        """
        if params is None:
            params = self.default_params
            
        try:
            h, w = image.shape[:2]
            
            # Create CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 labels: foreground and background
            
            # Ensure image is contiguous and proper type
            image = np.ascontiguousarray(image)
            
            # Convert mask to unary potentials
            # For best results, we need to provide class probabilities
            mask_uint8 = mask.astype(np.uint8)
            
            # Create the unary potentials based on the mask
            U = np.zeros((2, h, w), dtype=np.float32)
            
            # Foreground probability
            U[0, mask_uint8 == 0] = 0.9  # Background probability where mask is 0
            U[0, mask_uint8 == 1] = 0.1
            U[1, mask_uint8 == 1] = 0.9  # Foreground probability where mask is 1
            U[1, mask_uint8 == 0] = 0.1
            
            # Set unary potentials (negative log probability)
            U = unary_from_softmax(U)
            d.setUnaryEnergy(U)
            
            # Add pairwise Gaussian potentials (smoothing)
            d.addPairwiseGaussian(
                sxy=(params['gaussian_sxy'], params['gaussian_sxy']),
                compat=params['gaussian_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )
            
            # Add pairwise bilateral potentials (edge-preserving)
            d.addPairwiseBilateral(
                sxy=(params['bilateral_sxy'], params['bilateral_sxy']),
                srgb=(params['bilateral_srgb'], params['bilateral_srgb'], params['bilateral_srgb']),
                rgbim=image,
                compat=params['bilateral_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )
            
            # Perform inference
            Q = d.inference(params['iterations'])
            
            # Get the most probable class for each pixel
            MAP = np.argmax(Q, axis=0).reshape((h, w))
            
            return MAP.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Error applying CRF: {str(e)}")
            traceback.print_exc()
            return mask


class SVGExporter:
    """
    Enhanced SVG exporter for segmentation results with unique colors and detailed metadata
    """
    def __init__(self, line_thickness=1.5, quality_level="professional"):
        """Initialize the SVG exporter"""
        self.line_thickness = line_thickness
        self.quality_level = quality_level  # standard, professional, maximum
        
        # Set quality-specific parameters
        if quality_level == "standard":
            self.use_bezier = False
            self.include_metadata = False
            self.precision = 2
            self.add_labels = True
        elif quality_level == "professional":
            self.use_bezier = True
            self.include_metadata = True
            self.precision = 4
            self.add_labels = True
        elif quality_level == "maximum":
            self.use_bezier = True
            self.include_metadata = True
            self.precision = 6
            self.add_labels = True
        else:
            # Default to professional
            self.use_bezier = True
            self.include_metadata = True
            self.precision = 4
            self.add_labels = True
    
    def create_svg_document(self, size, metadata=None):
        """Create a new SVG document with specified size"""
        width, height = size
        
        # Create SVG root
        svg = ET.Element('svg')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        svg.set('version', '1.1')
        
        # Add metadata
        if metadata:
            title = metadata.get('title', 'Segmentation with SAM')
            author = metadata.get('author', SCRIPT_AUTHOR)
            date = metadata.get('date', SCRIPT_DATE)
            version = metadata.get('version', SCRIPT_VERSION)
            
            # Create metadata section
            metadata_elem = ET.SubElement(svg, 'metadata')
            
            # Use Dublin Core elements for better compatibility
            dc = ET.SubElement(metadata_elem, 'rdf:RDF')
            dc.set('xmlns:rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
            dc.set('xmlns:dc', 'http://purl.org/dc/elements/1.1/')
            dc.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
            
            work = ET.SubElement(dc, 'rdf:Description')
            
            title_elem = ET.SubElement(work, 'dc:title')
            title_elem.text = title
            
            creator = ET.SubElement(work, 'dc:creator')
            creator.text = author
            
            date_elem = ET.SubElement(work, 'dc:date')
            date_elem.text = date
            
            description = ET.SubElement(work, 'dc:description')
            description.text = f"Segmentation generated with Professional Segmentation Suite v{version}"
            
            # Also add simple description accessible to more SVG viewers
            desc = ET.SubElement(svg, 'desc')
            desc.text = f"{title}\nCreated by: {author}\nDate: {date}\nVersion: {version}"

        # Add style definitions with enhanced styling
        defs = ET.SubElement(svg, 'defs')
        style = ET.SubElement(defs, 'style')
        style.set('type', 'text/css')
        
        # Enhanced CSS rules
        css = """
            .mask { 
                fill-opacity: 0.5; 
                stroke-width: 1.5; 
                stroke-opacity: 0.9;
                transition: fill-opacity 0.3s ease;
            }
            .mask:hover {
                fill-opacity: 0.7;
                stroke-opacity: 1.0;
            }
            .segment {
                opacity: 0.95;
            }
            .label {
                font-family: Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                dominant-baseline: middle;
                text-anchor: middle;
                pointer-events: none;
            }
            .label-bg {
                fill: white;
                fill-opacity: 0.7;
                stroke: black;
                stroke-width: 0.5;
                stroke-opacity: 0.7;
                rx: 3;
                ry: 3;
            }
        """
        style.text = css
        
        return svg
    
    def bezier_smooth_points(self, points, smoothness=0.2):
        """
        Convert a series of points to smooth Bézier curves
        
        Args:
            points: Array of points [(x1,y1), (x2,y2), ...]
            smoothness: Controls curve smoothness (0.1-0.5 works well)
            
        Returns:
            SVG path data string with curves
        """
        if len(points) < 3:
            # For very few points, just return a simple polyline
            fmt = "{:.{prec}f},{:.{prec}f}".format
            path = f"M {fmt(points[0][0], points[0][1], prec=self.precision)}"
            for point in points[1:]:
                path += f" L {fmt(point[0], point[1], prec=self.precision)}"
            return path + " Z"
        
        # Format with appropriate precision
        fmt = "{:.{prec}f},{:.{prec}f}".format
        path = [f"M {fmt(points[0][0], points[0][1], prec=self.precision)}"]
        
        n = len(points)
        
        # Use cubic Bézier curves for maximum quality
        if self.quality_level == "maximum":
            for i in range(0, n-1):
                # Current point and next point
                current = points[i]
                next_pt = points[(i+1) % n]
                
                # Previous and next points (for tangent calculation)
                prev_pt = points[(i-1) % n]
                next_next = points[(i+2) % n]
                
                # Calculate tangent vectors
                tangent1 = np.array([next_pt[0] - prev_pt[0], next_pt[1] - prev_pt[1]])
                tangent2 = np.array([next_next[0] - current[0], next_next[1] - current[1]])
                
                # Normalize tangent vectors
                length1 = np.sqrt(tangent1[0]**2 + tangent1[1]**2)
                length2 = np.sqrt(tangent2[0]**2 + tangent2[1]**2)
                
                if length1 > 0:
                    tangent1 = tangent1 / length1
                if length2 > 0:
                    tangent2 = tangent2 / length2
                    
                # Calculate distance between current and next point
                dist = np.sqrt((next_pt[0] - current[0])**2 + (next_pt[1] - current[1])**2)
                
                # Control point distances
                ctrl_dist1 = dist * smoothness
                ctrl_dist2 = dist * smoothness
                
                # Control points
                ctrl1 = [current[0] + tangent1[0] * ctrl_dist1, current[1] + tangent1[1] * ctrl_dist1]
                ctrl2 = [next_pt[0] - tangent2[0] * ctrl_dist2, next_pt[1] - tangent2[1] * ctrl_dist2]
                
                # Add cubic Bézier curve command
                path.append(
                    f"C {fmt(ctrl1[0], ctrl1[1], prec=self.precision)} "
                    f"{fmt(ctrl2[0], ctrl2[1], prec=self.precision)} "
                    f"{fmt(next_pt[0], next_pt[1], prec=self.precision)}"
                )
        else:
            # Use simpler quadratic Bézier curves for standard and professional quality
            for i in range(0, n-1):
                # Current and next points
                curr = points[i]
                next_pt = points[(i+1) % n]
                
                # Calculate control point (mid-point with slight offset)
                control_x = (curr[0] + next_pt[0]) / 2
                control_y = (curr[1] + next_pt[1]) / 2
                
                # Add quadratic Bézier
                path.append(
                    f"Q {fmt(control_x, control_y, prec=self.precision)} "
                    f"{fmt(next_pt[0], next_pt[1], prec=self.precision)}"
                )
        
        # Close the path
        path.append("Z")
        
        return " ".join(path)
    
    def add_shape_to_svg(self, svg_root, element, color=None, segment_id=None, segment_info=None):
        """Add a geometric element to the SVG document with enhanced details"""
        element_type = element['type'].lower()
        shape_elem = None
        
        # Convert color to CSS format if provided
        color_css = None
        if color:
            r, g, b = color
            color_css = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        
        # Create SVG element based on type
        if element_type == 'cercle' or element_type == 'circle':
            center = element['center']
            radius = element['radius']
            
            shape_elem = ET.SubElement(svg_root, 'circle')
            shape_elem.set('cx', str(center[0]))
            shape_elem.set('cy', str(center[1]))
            shape_elem.set('r', str(radius))
            
        elif element_type == 'ellipse':
            center = element['center']
            axes = element['axes']
            angle = element.get('angle', 0)
            
            shape_elem = ET.SubElement(svg_root, 'ellipse')
            shape_elem.set('cx', str(center[0]))
            shape_elem.set('cy', str(center[1]))
            shape_elem.set('rx', str(axes[0]/2))
            shape_elem.set('ry', str(axes[1]/2))
            
            # Apply rotation if needed
            if angle != 0:
                shape_elem.set('transform', f'rotate({angle} {center[0]} {center[1]})')
                
        elif element_type in ['rectangle', 'polygon', 'polygone', 'contour']:
            points = element['points']
            
            # Create path element
            shape_elem = ET.SubElement(svg_root, 'path')
            
            # Generate path data based on quality settings
            if self.use_bezier and len(points) > 3:
                # Use Bezier curves for smoother paths
                path_data = self.bezier_smooth_points(points, smoothness=0.2)
            else:
                # Use simple polyline for faster rendering
                fmt = "{:.{prec}f},{:.{prec}f}".format
                path_data = f"M {fmt(points[0][0], points[0][1], prec=self.precision)}"
                for x, y in points[1:]:
                    path_data += f" L {fmt(x, y, prec=self.precision)}"
                path_data += " Z"  # Close the path
            
            shape_elem.set('d', path_data)
            
        elif element_type == 'lignes' or element_type == 'lines':
            segments = element['segments']
            
            # Create separate paths for each segment
            for i, (start, end) in enumerate(segments):
                line_elem = ET.SubElement(svg_root, 'line')
                line_elem.set('x1', str(start[0]))
                line_elem.set('y1', str(start[1]))
                line_elem.set('x2', str(end[0]))
                line_elem.set('y2', str(end[1]))
                
                # Apply styling
                line_elem.set('stroke', color_css if color_css else 'black')
                line_elem.set('stroke-width', str(self.line_thickness))
                line_elem.set('fill', 'none')
                
                # Add ID if provided
                if segment_id is not None:
                    line_elem.set('id', f'segment_{segment_id}_line_{i}')
        
        # Apply common styling to shape element
        if shape_elem is not None:
            shape_elem.set('stroke', color_css if color_css else 'black')
            shape_elem.set('stroke-width', str(self.line_thickness))
            
            # Different fill behavior based on quality level
            shape_elem.set('fill', color_css if color_css else 'none')
            shape_elem.set('fill-opacity', "0.5")
            
            # Add class for CSS styling
            shape_elem.set('class', "mask")
            
            # Add ID if provided
            if segment_id is not None:
                shape_elem.set('id', f'segment_{segment_id}')
            
            # Add detailed metadata if available and requested
            if segment_info and self.include_metadata:
                # Add data attributes for interactivity and metadata
                if 'class' in segment_info:
                    shape_elem.set('data-class', segment_info['class'])
                if 'confidence' in segment_info:
                    shape_elem.set('data-confidence', f"{segment_info['confidence']:.4f}")
                if 'area' in segment_info:
                    shape_elem.set('data-area', f"{segment_info['area']:.1f}")
                if 'object_id' in segment_info:
                    shape_elem.set('data-object-id', segment_info['object_id'])
    
    def add_label_to_svg(self, svg_root, segment, color):
        """Add a text label to the SVG for this segment"""
        if not self.add_labels or not segment or 'center' not in segment:
            return
            
        # Extract label information
        class_name = segment.get('class', 'unknown')
        confidence = segment.get('confidence', 0)
        center_x, center_y = segment.get('center', (0, 0))
        
        # Create label group
        label_group = ET.SubElement(svg_root, 'g')
        label_group.set('class', 'label-group')
        
        # Create text with label
        text_elem = ET.SubElement(label_group, 'text')
        text_elem.set('x', str(center_x))
        text_elem.set('y', str(center_y))
        text_elem.set('class', 'label')
        
        # Set contrasting text color based on background color
        r, g, b = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = 'black' if luminance > 0.5 else 'white'
        text_elem.set('fill', text_color)
        
        # Label with confidence
        text_elem.text = f"{class_name}: {confidence:.2f}"
        
        # Add background rectangle for better readability
        # This will be created in the final SVG after calculating text dimensions
        # For now, add a placeholder with estimated dimensions
        bg_elem = ET.SubElement(label_group, 'rect')
        bg_elem.set('class', 'label-bg')
        bg_elem.set('x', str(center_x - 40))  # Estimated width
        bg_elem.set('y', str(center_y - 10))  # Estimated height
        bg_elem.set('width', '80')
        bg_elem.set('height', '20')
        
        # Place background behind text
        bg_elem.attrib['style'] = 'opacity: 0.7;'
        
        # Move the background before text in the DOM (so it renders behind)
        label_group.remove(bg_elem)
        label_group.insert(0, bg_elem)
    
    def export_to_svg(self, segments_data, output_path, size, metadata=None):
        """Export segments to SVG with enhanced visualization"""
        print(f"Exporting to SVG ({self.quality_level} quality): {output_path}")
        
        if not segments_data or len(segments_data) == 0:
            print("❌ Error: No segments to export")
            return False
        
        try:
            # Create SVG document
            svg_root = self.create_svg_document(size, metadata)
            
            # Count valid contours for reporting
            valid_contour_count = 0
            
            # Process segments - add objects in reverse order so overlaps work correctly
            for idx, segment in enumerate(segments_data):
                # Get segment color if available, otherwise use default
                color = segment.get('color', (0, 0, 1))  # Default to blue
                segment_id = segment.get('id', idx + 1)  # Use index+1 if no ID
                object_id = segment.get('object_id', f"object_{segment_id}")
                
                # Create group element to hold this segment's elements
                group = ET.SubElement(svg_root, 'g')
                group.set('id', f"segment_{segment_id}")
                group.set('class', "segment")
                
                # Add metadata to group element
                if "class" in segment:
                    group.set('data-class', segment["class"])
                if "confidence" in segment:
                    group.set('data-confidence', f"{segment['confidence']:.4f}")
                if "area" in segment:
                    group.set('data-area', str(segment["area"]))
                group.set('data-object-id', object_id)
                
                # Check if the segment has geometric elements
                if not segment.get('geometric_elements'):
                    print(f"  ⚠️ Segment {segment_id} has no geometric elements to export")
                    continue
                
                segment_has_valid_contours = False
                    
                # Process each geometric element
                for element in segment['geometric_elements']:
                    if not element or 'type' not in element:
                        print(f"  ⚠️ Invalid element in segment {segment_id}")
                        continue
                    
                    # Skip adding bounding box outlines (as requested) - only add contours
                    if element['type'].lower() in ['rectangle'] and 'points' in element and len(element['points']) == 4:
                        # This is likely a bounding box - skip it
                        continue
                    
                    # For contours, check if there are enough points
                    if element['type'] == 'contour' and 'points' in element:
                        if len(element['points']) < 3:
                            print(f"  ⚠️ Not enough points in contour for segment {segment_id} (needs >=3)")
                            continue
                        
                        valid_contour_count += 1
                        segment_has_valid_contours = True
                        
                    self.add_shape_to_svg(group, element, color, segment_id, segment)
                
                # Add label with class name and confidence
                self.add_label_to_svg(group, segment, color)
                
                # If no valid contours in this segment, add a simple text note
                if not segment_has_valid_contours and 'class' in segment:
                    # Create a text element
                    text_elem = ET.SubElement(group, 'text')
                    text_elem.set('x', str(10))
                    text_elem.set('y', str(20 + idx * 25))
                    text_elem.set('fill', 'black')
                    text_elem.set('font-family', 'Arial')
                    text_elem.set('font-size', '12')
                    text_elem.set('font-weight', 'bold')
                    text_elem.text = f"{segment['class']} ({segment.get('confidence', 0):.2f})"
            
            # Format the XML for readability
            rough_string = ET.tostring(svg_root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(pretty_xml)
                
            if valid_contour_count > 0:
                print(f"✅ SVG file exported successfully with {valid_contour_count} valid contours to: {output_path}")
            else:
                print(f"⚠️ SVG file exported to {output_path} but contains no valid contours!")
            
            return True
                
        except Exception as e:
            print(f"❌ Error exporting to SVG: {str(e)}")
            traceback.print_exc()
            return False

class AdvancedMaskRefinement:
    """Enhanced algorithm for SAM mask boundary refinement"""
    
    @staticmethod
    def refine_mask(image, mask, params=None):
        """
        Apply advanced refinement to improve mask boundaries
        
        Args:
            image: RGB image as numpy array
            mask: Binary mask as numpy array
            params: Dictionary of refinement parameters
            
        Returns:
            Refined binary mask
        """
        if params is None:
            params = {
                'contour_smoothing': True,
                'edge_snap': True, 
                'small_object_removal': True,
                'hole_filling': True,
                'boundary_expansion': 0,
                'morphological_cleanup': True,
                'min_object_size': 50,
                'max_hole_size': 100
            }
        
        # Convert mask to uint8 to avoid type errors
        mask_uint8 = mask.astype(np.uint8)
        
        # Step 1: Fill small holes
        if params.get('hole_filling', True):
            # Fill holes using morphological closing
            kernel_size = 5  # Adjust based on image resolution
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_filled = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        else:
            mask_filled = mask_uint8.copy()
        
        # Step 2: Remove small isolated objects
        if params.get('small_object_removal', True):
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
            
            # Find the largest component (foreground)
            sizes = stats[:, -1]
            if num_labels > 1:  # At least one component besides background
                # Skip background (label 0)
                max_label = np.argmax(sizes[1:]) + 1
                min_size = params.get('min_object_size', 50)
                
                # Keep only large enough components
                mask_cleaned = np.zeros_like(mask_filled)
                for i in range(1, num_labels):  # Skip background
                    if sizes[i] >= min_size:
                        mask_cleaned[labels == i] = 1
            else:
                mask_cleaned = mask_filled
        else:
            mask_cleaned = mask_filled
        
        # Step 3: Edge detection in original image for boundary snapping
        if params.get('edge_snap', True):
            # Convert to grayscale if image is RGB
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges slightly for better overlap
            edge_kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
            
            # Compute boundary of current mask
            mask_boundary = cv2.dilate(mask_cleaned, edge_kernel, iterations=1) - mask_cleaned
            
            # Where the mask boundary and image edges overlap, keep the original mask
            # For non-overlapping boundaries, use conservative approach
            edge_overlap = edges_dilated * mask_boundary
            
            # Apply boundary refinement where there's an overlap
            if np.sum(edge_overlap) > 0:
                # Create a distance map from edges
                edge_distance = cv2.distanceTransform(1 - edges_dilated, cv2.DIST_L2, 3)
                edge_distance = edge_distance / edge_distance.max()  # Normalize
                
                # Use watershed or active contour for precise boundary adjustment
                # For simplicity, we'll use a threshold on the distance map
                mask_refined = mask_cleaned.copy()
                boundary_region = cv2.dilate(mask_cleaned, np.ones((5, 5), np.uint8)) - \
                                 cv2.erode(mask_cleaned, np.ones((5, 5), np.uint8))
                
                # Only modify the boundary region
                boundary_indices = np.where(boundary_region > 0)
                for y, x in zip(*boundary_indices):
                    if edge_distance[y, x] < 0.2:  # Close to an edge
                        mask_refined[y, x] = 1
                    elif edge_distance[y, x] > 0.8:  # Far from any edge
                        mask_refined[y, x] = 0
            else:
                mask_refined = mask_cleaned
        else:
            mask_refined = mask_cleaned
        
        # Step 4: Contour smoothing if requested
        if params.get('contour_smoothing', True):
            # Extract contours
            contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create empty mask
            smooth_mask = np.zeros_like(mask_refined)
            
            # Process each contour
            for contour in contours:
                # Skip tiny contours
                if cv2.contourArea(contour) < params.get('min_object_size', 50):
                    continue
                
                # Smooth the contour using Gaussian or polynomial approximation
                epsilon = 0.001 * cv2.arcLength(contour, True)  # Adjust smoothing strength
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Draw the smoothed contour
                cv2.drawContours(smooth_mask, [approx_contour], 0, 1, -1)
            
            mask_refined = smooth_mask
        
        # Step 5: Final morphological cleanup
        if params.get('morphological_cleanup', True):
            # Use opening to remove small noise
            cleanup_kernel = np.ones((3, 3), np.uint8)
            mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, cleanup_kernel)
            
            # Use closing to fill small holes and smooth edges
            mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, cleanup_kernel)
        
        # Step 6: Expand or contract boundary if requested
        boundary_expansion = params.get('boundary_expansion', 0)
        if boundary_expansion != 0:
            kernel = np.ones((3, 3), np.uint8)
            if boundary_expansion > 0:
                mask_refined = cv2.dilate(mask_refined, kernel, iterations=boundary_expansion)
            else:
                mask_refined = cv2.erode(mask_refined, kernel, iterations=abs(boundary_expansion))
        
        return mask_refined


def check_gpu(select_gpu=None, force_nvidia=True):
    """
    Check GPU availability and select the optimal GPU based on memory.
    
    Args:
        select_gpu: Either None, an integer GPU index, or 'max_mem' to select the GPU with most memory
        force_nvidia: Prioritize NVIDIA GPUs over others
        
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. Using CPU.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA is available with {gpu_count} GPU{'s' if gpu_count > 1 else ''}:")
    
    # Collect GPU info
    gpu_infos = []
    nvidia_found = False
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        is_nvidia = "nvidia" in gpu_name.lower()
        if is_nvidia:
            nvidia_found = True
            
        print(f"  • GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        gpu_infos.append((i, gpu_name, gpu_mem, is_nvidia))
    
    # Select GPU based on input
    target_gpu = 0  # Default to first GPU
    
    # If force_nvidia is True and we found an NVIDIA GPU, filter for only NVIDIA GPUs
    if force_nvidia and nvidia_found:
        nvidia_gpus = [(i, name, mem, is_nvidia) for i, name, mem, is_nvidia in gpu_infos if is_nvidia]
        if nvidia_gpus:
            gpu_infos = nvidia_gpus
    
    if select_gpu == 'max_mem' or select_gpu is None:
        # Find GPU with maximum memory (typically the 16GB one)
        target_gpu = max(gpu_infos, key=lambda x: x[2])[0]
        print(f"✓ Auto-selected GPU {target_gpu} with maximum memory ({gpu_infos[target_gpu][2]:.1f} GB)")
    elif isinstance(select_gpu, int) and 0 <= select_gpu < gpu_count:
        target_gpu = select_gpu
        print(f"✓ Using specified GPU {target_gpu}: {gpu_infos[target_gpu][1]} ({gpu_infos[target_gpu][2]:.1f} GB)")
    
    # Set as default GPU
    torch.cuda.set_device(target_gpu)
    print(f"✓ Current active GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name()}")
    
    return True


def print_gpu_memory(device=None, prefix=""):
    """Print GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    
    print(f"{prefix}GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Professional Segmentation Suite with YOLO-World, SAM, U²-Net, CRF, and SVG export')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=None,
                        help='specify which GPU to use (default: auto-select NVIDIA with max memory)')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument('--sam-model',
                        default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--sam-checkpoint',
                        default='checkpoints/sam_vit_h_4b8939.pth',
                        help='SAM checkpoint file')
    parser.add_argument('--u2net-checkpoint',
                        default='checkpoints/u2net.pth',
                        help='U²-Net checkpoint file for mask refinement')
    parser.add_argument('--multiscale-sam',
                        action='store_true',
                        help='enable multi-scale SAM inference for more detailed masks')
    parser.add_argument('--export-svg',
                        action='store_true',
                        help='export segmentation masks as SVG files')
    parser.add_argument('--export-png',
                        action='store_true',
                        help='export individual masks as transparent PNG files')
    

    parser.add_argument('--contour-method',
                        choices=['cv2', 'skimage', 'both'],
                        default='skimage',
                        help='contour extraction method for SVG')
    parser.add_argument('--simplify-tolerance',
                        type=float,
                        default=0.1,
                        help='tolerance for simplifying contours (0-1, lower = more detailed)')
    parser.add_argument('--curve-smoothness',
                        type=float,
                        default=0.2,
                        help='smoothness of Bézier curves (0.1-0.5, higher = smoother)')
    parser.add_argument('--precision-level',
                        type=int,
                        default=6,
                        help='SVG decimal precision (1-6)')
    
    # CRF refinement parameters
    parser.add_argument('--use-crf',
                        action='store_true',
                        help='apply CRF refinement to masks')
    parser.add_argument('--crf-iter',
                        type=int,
                        default=10,
                        help='number of CRF iterations')
    parser.add_argument('--crf-pos-w',
                        type=float,
                        default=3.0,
                        help='CRF position weight (Gaussian kernel)')
    parser.add_argument('--crf-bi-w',
                        type=float,
                        default=10.0,
                        help='CRF bilateral weight')
    parser.add_argument('--crf-bi-xy',
                        type=float,
                        default=80.0,
                        help='CRF bilateral xy std dev')
    parser.add_argument('--crf-bi-rgb',
                        type=float,
                        default=13.0,
                        help='CRF bilateral RGB std dev')
    
    # Mask refinement options
    parser.add_argument('--smooth-mask',
                        action='store_true',
                        help='apply additional smoothing to masks before SVG conversion')
    parser.add_argument('--smooth-kernel',
                        type=int,
                        default=3,
                        help='kernel size for smoothing (must be odd number)')
    parser.add_argument('--detailed-refinement',
                        action='store_true',
                        help='enable advanced mask refinement techniques')
    parser.add_argument('--edge-refinement',
                        action='store_true',
                        help='enable edge-aware refinement for sharper boundaries')
    parser.add_argument('--expand-box',
                        type=float,
                        default=0.1,
                        help='expand detection boxes by this fraction to capture more context')
    
    # SVG export quality
    parser.add_argument('--svg-quality',
                        choices=['standard', 'professional', 'maximum'],
                        default='professional',
                        help='SVG export quality level')
    parser.add_argument('--verify-export',
                        action='store_true',
                        help='automatically verify SVG quality against original mask')
    parser.add_argument('--verification-report',
                        action='store_true',
                        help='generate detailed verification report with metrics')
    
    # Advanced options
    parser.add_argument('--gpu-stats',
                        action='store_true',
                        help='show GPU memory usage statistics')
    parser.add_argument('--pipeline-name',
                        type=str,
                        default='Professional Segmentation Suite',
                        help='name used in exports and metadata')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def initialize_sam(model_type, checkpoint, device):
    """Initialize SAM model with better error handling."""
    print(f"Initializing SAM model ({model_type}) on {device}...")
    try:
        if not os.path.exists(checkpoint):
            print(f"❌ SAM checkpoint not found: {checkpoint}")
            print(f"Please download from: https://github.com/facebookresearch/segment-anything")
            return None
            
        # Load model with explicit device placement
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        
        # Verify model is on the correct device
        print(f"SAM model device: {next(sam.parameters()).device}")
        
        # Create predictor and verify
        predictor = SamPredictor(sam)
        print("✓ SAM initialized successfully!")
        return predictor
        
    except Exception as e:
        print(f"❌ Error initializing SAM: {str(e)}")
        traceback.print_exc()
        return None

def enhance_sam_masks(sam_predictor, image_rgb, box, multimask_output=True):
    """Generate multiple SAM masks with enhanced quality control."""
    try:
        # Convert box to NumPy array format expected by SAM
        # The error happens because we're passing a tensor but SAM expects a NumPy array
        if isinstance(box, torch.Tensor):
            box_numpy = box.cpu().numpy()  # Convert tensor to numpy
        else:
            box_numpy = np.array(box)  # Ensure it's a numpy array even if passed as a list
        
        # Get masks from SAM with multiple outputs to choose the best one
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_numpy,  # Use numpy array instead of tensor
            multimask_output=True,  # Always get multiple masks
        )
        
        # If we have multiple masks, select the best one based on score and detail level
        if masks is not None and masks.shape[0] > 1:
            # Compute a detail score for each mask
            detail_scores = []
            for i, mask in enumerate(masks):
                # Count the number of boundary pixels as a measure of detail
                kernel = np.ones((3, 3), np.uint8)
                boundary = cv2.dilate(mask.astype(np.uint8), kernel) - cv2.erode(mask.astype(np.uint8), kernel)
                boundary_ratio = np.sum(boundary) / max(1, np.sum(mask))
                
                # Combine SAM's confidence score with boundary detail
                # Weight both factors in the final score
                combined_score = 0.7 * scores[i] + 0.3 * min(1.0, boundary_ratio * 10)
                detail_scores.append(combined_score)
            
            # Select the mask with highest combined score
            best_idx = np.argmax(detail_scores)
            return masks[best_idx:best_idx+1], np.array([scores[best_idx]])
        
        # Return all masks and scores for further processing
        return masks, scores
        
    except Exception as e:
        print(f"  ❌ SAM inference error: {str(e)}")
        traceback.print_exc()
        return None, None


def expand_box(box, expand_ratio, image_size):
    """
    Expand a bounding box by a given ratio, while ensuring it stays within image boundaries.
    """
    height, width = image_size
    
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # Calculate expansion amount
    dx = w * expand_ratio
    dy = h * expand_ratio
    
    # Expand the box
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(width - 1, x2 + dx)
    y2 = min(height - 1, y2 + dy)
    
    return [x1, y1, x2, y2]


def refine_mask_with_u2net(image_rgb, mask, u2net_predictor):
    """
    Refine the segmentation mask using U²-Net
    
    Args:
        image_rgb: RGB image as numpy array (H, W, 3)
        mask: Binary mask from SAM as numpy array
        u2net_predictor: Initialized U2NetPredictor instance
        
    Returns:
        Refined binary mask
    """
    # Skip U2Net refinement if predictor is None or has no model
    if u2net_predictor is None or u2net_predictor.model is None:
        print("  ⚠️ U²-Net not available for refinement, using original mask")
        return mask
        
    try:
        # Step 1: Apply the initial mask to the image to focus U²-Net
        masked_image = image_rgb.copy()
        
        # Convert mask to uint8 to avoid type errors with OpenCV
        mask_uint8 = mask.astype(np.uint8)
        
        mask_expanded = np.expand_dims(mask_uint8, axis=-1) * 255
        masked_region = cv2.bitwise_and(image_rgb, np.repeat(mask_expanded, 3, axis=2))
        
        # Create a 50-pixel border around the mask to include context
        kernel = np.ones((51, 51), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        # Create new image with black background and only the ROI preserved
        context_region = image_rgb.copy()
        context_region[dilated_mask == 0] = [0, 0, 0]
        
        # Step 2: Use U²-Net to predict a refined mask
        u2net_mask = u2net_predictor.predict(context_region, size=320)
        
        # If prediction failed, return original mask
        if u2net_mask is None:
            print("  ⚠️ U²-Net prediction failed, using original mask")
            return mask
            
        # Step 3: Combine the original mask with U²-Net mask
        # Normalize the U²-Net probability map
        u2net_mask = (u2net_mask - u2net_mask.min()) / (u2net_mask.max() - u2net_mask.min() + 1e-8)
        
        # Generate binary mask using an adaptive threshold
        # Keep only strong predictions from U²-Net within the dilated region
        binary_u2net = (u2net_mask > 0.5).astype(np.uint8)
        binary_u2net = binary_u2net & dilated_mask
        
        # Step 4: Combine masks - use U²-Net for detail but preserve the general shape from SAM
        combined_mask = binary_u2net | mask_uint8
        
        # Step 5: Clean up the mask
        # Remove small holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # Remove small disconnected regions
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        # Keep only the largest component
        if num_labels > 1:
            # Skip label 0 (background)
            largest_area = 0
            largest_label = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = i
            
            # Create a new mask with only the largest component
            refined_mask = (labels == largest_label).astype(np.uint8)
        else:
            refined_mask = combined_mask
        
        return refined_mask
        
    except Exception as e:
        print(f"  ❌ Error during U²-Net refinement: {str(e)}")
        traceback.print_exc()
        return mask  # Return the original mask if any error occurs


def apply_crf(image, mask, crf_params):
    """
    Apply Conditional Random Field refinement to a binary mask.
    
    Args:
        image: RGB image as numpy array
        mask: Binary mask as numpy array
        crf_params: Dictionary with CRF parameters
        
    Returns:
        Refined binary mask
    """
    try:
        # Create CRF processor
        crf_processor = CRFProcessor()
        
        # Apply CRF refinement
        refined_mask = crf_processor.apply_crf(image, mask, crf_params)
        
        return refined_mask
        
    except Exception as e:
        print(f"  ❌ Error in CRF processing: {str(e)}")
        return mask  # Return original mask on error


def smooth_mask(mask, kernel_size=3):
    """Apply smoothing to the binary mask."""
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological operations for smoothing
    mask_smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_OPEN, kernel)
    mask_smoothed = cv2.medianBlur(mask_smoothed, kernel_size)
    
    return mask_smoothed


def optimized_mask_refinement(image_rgb, mask, refinement_mode='high_quality'):
    """
    Apply optimized mask refinement with carefully tuned parameters.
    
    Args:
        image_rgb: RGB image as numpy array
        mask: Binary mask as numpy array
        refinement_mode: Refinement preset ('standard', 'high_quality', or 'edge_precise')
        
    Returns:
        Refined binary mask
    """
    # Define parameters based on refinement mode
    if refinement_mode == 'standard':
        detail_params = {
            'contour_smoothing': True,
            'edge_snap': False,
            'small_object_removal': True,
            'hole_filling': True,
            'boundary_expansion': 0,
            'morphological_cleanup': True,
            'min_object_size': 100,
            'max_hole_size': 100
        }
    elif refinement_mode == 'high_quality':
        detail_params = {
            'contour_smoothing': True,
            'edge_snap': True,
            'small_object_removal': True,
            'hole_filling': True,
            'boundary_expansion': 0,
            'morphological_cleanup': True,
            'min_object_size': 50,
            'max_hole_size': 200
        }
    elif refinement_mode == 'edge_precise':
        detail_params = {
            'contour_smoothing': False,  # Preserve original edges
            'edge_snap': True,
            'small_object_removal': True,
            'hole_filling': True,
            'boundary_expansion': 0,
            'morphological_cleanup': False,  # No smoothing to preserve details
            'min_object_size': 20,
            'max_hole_size': 300
        }
    
    # Apply advanced mask refinement
    refined_mask = AdvancedMaskRefinement.refine_mask(image_rgb, mask, detail_params)
    
    return refined_mask


def extract_contours_skimage(mask, simplify_tolerance=0.1):
    """
    Extract contours using scikit-image for better quality.
    """
    # Convert mask to uint8 if needed
    mask_uint8 = mask.astype(np.uint8)
    
    # Find contours at level 0.5 (binary mask boundary)
    contours = find_contours(mask_uint8.astype(float), 0.5)
    
    # No contours found
    if not contours:
        return []
    
    # Process each contour
    processed_contours = []
    for contour in contours:
        # Skip tiny contours
        if len(contour) < 3:
            continue
            
        # Simplify if requested (using Douglas-Peucker algorithm)
        if simplify_tolerance > 0:
            # Converting to the format expected by cv2.approxPolyDP
            contour_cv = np.array(contour, dtype=np.float32).reshape((-1, 1, 2))
            epsilon = simplify_tolerance * cv2.arcLength(contour_cv, True) / 100
            approx_contour = cv2.approxPolyDP(contour_cv, epsilon, True)
            contour = approx_contour.reshape(-1, 2)
        
        processed_contours.append(contour)
    
    return processed_contours

def create_segments_for_svg_export(masks, labels, image_shape, contour_method='skimage', 
                                  simplify_tolerance=0.1, svg_quality='professional'):
    """
    Convert masks to a format suitable for SVGExporter with truly unique colors for each object
    
    This function prepares segmentation data for SVG export, ensuring each object gets
    a unique color regardless of class, with detailed contours and metadata.
    
    Args:
        masks: List of binary masks
        labels: List of label strings (class name + confidence)
        image_shape: Shape of the original image (h, w)
        contour_method: Method to extract contours ('cv2', 'skimage', or 'both')
        simplify_tolerance: Tolerance for simplifying contours (lower = more detailed)
        svg_quality: Quality level for SVG export ('standard', 'professional', 'maximum')
        
    Returns:
        List of segment dictionaries compatible with SVGExporter
        
    Author: FETHl
    Last updated: 2025-03-21 14:54:38
    """
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    segments_data = []
    
    # Professional color palette with distinct colors
    colors = [
        (0.9, 0.1, 0.1),   # Red
        (0.1, 0.6, 0.9),   # Blue
        (0.2, 0.7, 0.2),   # Green
        (0.8, 0.2, 0.8),   # Purple
        (1.0, 0.6, 0.0),   # Orange
        (0.0, 0.6, 0.6),   # Teal
        (0.8, 0.8, 0.2),   # Yellow
        (0.6, 0.0, 0.6),   # Magenta
        (0.4, 0.2, 0.8),   # Indigo
        (0.8, 0.4, 0.0),   # Brown
        (0.0, 0.4, 0.8),   # Royal Blue
        (0.4, 0.8, 0.4),   # Light Green
        (0.9, 0.3, 0.3),   # Light Red
        (0.6, 0.6, 0.2),   # Olive
        (0.6, 0.2, 0.0),   # Rust
        (0.2, 0.8, 0.8),   # Cyan
        (0.7, 0.0, 0.2),   # Crimson
        (0.4, 0.7, 0.7),   # Steel Blue
        (0.8, 0.2, 0.4),   # Rose
        (0.2, 0.5, 0.2),   # Forest Green
        (0.5, 0.5, 0.8),   # Periwinkle
        (0.9, 0.6, 0.4),   # Peach
        (0.5, 0.3, 0.1),   # Sienna
        (0.6, 0.9, 0.6),   # Mint
    ]
    
    # Track used colors to avoid duplicates even for same class
    used_colors = set()
    
    # Refine simplify_tolerance based on quality setting
    effective_tolerance = simplify_tolerance
    if svg_quality == 'maximum':
        effective_tolerance = max(0.01, simplify_tolerance / 4)
    elif svg_quality == 'professional':
        effective_tolerance = max(0.05, simplify_tolerance / 2)
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Ensure mask is uint8
        mask_uint8 = mask.astype(np.uint8)
        
        # Parse label to get class name and confidence
        class_name = label.split()[0]
        confidence = float(label.split()[1]) if len(label.split()) > 1 else 1.0
        
        # Generate a truly unique color for each object
        color_assigned = False
        attempts = 0
        color = None
        
        # Try to generate a unique color that differs from previous ones
        while not color_assigned and attempts < 20:
            if i < len(colors):
                # Use predefined colors first
                color = colors[i]
            else:
                # Generate color using golden ratio conjugate for even distribution
                golden_ratio_conjugate = 0.618033988749895
                hue = ((i * golden_ratio_conjugate) + (attempts * 0.1)) % 1.0
                saturation = 0.7 + min(0.3, attempts * 0.02)  # Increase saturation slightly on retries
                value = 0.9
                r, g, b = plt.cm.hsv(hue*255)[:3]
                color = (r, g, b)
            
            # Convert to integer RGB for comparison
            rgb_int = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Check if we've used a similar color before
            if rgb_int not in used_colors:
                used_colors.add(rgb_int)
                color_assigned = True
            else:
                attempts += 1
        
        # In case we couldn't find a unique color after attempts
        if not color_assigned:
            # Generate a random color as last resort
            color = (random.random(), random.random(), random.random())
        
        # Extract contours using selected method
        if contour_method == 'skimage':
            contours = extract_contours_skimage(mask_uint8, effective_tolerance)
        elif contour_method == 'cv2':
            # Use OpenCV for contours
            cv_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = []
            for cnt in cv_contours:
                # Simplify contour if requested
                if effective_tolerance > 0:
                    epsilon = effective_tolerance * cv2.arcLength(cnt, True) / 100
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)
                # Convert to correct format
                contours.append(cnt.reshape(-1, 2))
        else:  # 'both' - use both methods and combine results
            # Get contours from both methods
            skimage_contours = extract_contours_skimage(mask_uint8, effective_tolerance)
            
            cv_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2_contours = []
            for cnt in cv_contours:
                if effective_tolerance > 0:
                    epsilon = effective_tolerance * cv2.arcLength(cnt, True) / 100
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)
                cv2_contours.append(cnt.reshape(-1, 2))
            
            # Use the method that produced more detailed contours
            total_skimage_points = sum(len(c) for c in skimage_contours) if skimage_contours else 0
            total_cv2_points = sum(len(c) for c in cv2_contours) if cv2_contours else 0
            
            contours = skimage_contours if total_skimage_points >= total_cv2_points else cv2_contours
        
        # Create geometric elements for this segment
        geometric_elements = []
        for contour in contours:
            # Convert contour coordinates to the format expected by SVGExporter
            points = []
            for point in contour:
                if contour_method == 'cv2':
                    # For cv2 contours, points are already in (x,y) format
                    points.append((float(point[0]), float(point[1])))
                else:
                    # For skimage contours, points are in (row,col) = (y,x) format, need to flip
                    points.append((float(point[1]), float(point[0])))
            
            if len(points) >= 3:  # Need at least 3 points for a polygon
                geometric_elements.append({
                    'type': 'contour',
                    'points': points
                })
        
        # Calculate additional properties for detailed metadata
        area = np.sum(mask_uint8)
        y_indices, x_indices = np.where(mask_uint8 > 0)
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # Calculate center of mass
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)
            
            # Calculate boundary complexity (perimeter to area ratio)
            boundary_pixels = cv2.dilate(mask_uint8, np.ones((3,3), np.uint8)) - cv2.erode(mask_uint8, np.ones((3,3), np.uint8))
            boundary_complexity = np.sum(boundary_pixels) / max(1, area)
            
            # Add bounding box as a separate property but not as a geometric element
            bbox_element = {
                'type': 'bbox',
                'points': [(float(bbox[0]), float(bbox[1])), 
                          (float(bbox[2]), float(bbox[1])), 
                          (float(bbox[2]), float(bbox[3])), 
                          (float(bbox[0]), float(bbox[3]))]
            }
            
            # If we don't have good contours, use simplified inner details
            if not geometric_elements or (len(contours) == 1 and len(contours[0]) <= 4):
                # Try to extract internal details
                internal_contours = []
                
                # Apply edge detection to find internal details
                internal_mask = mask_uint8.copy()
                internal_edges = cv2.Canny(internal_mask, 50, 150)
                
                # Get contours of internal edges
                int_contours, _ = cv2.findContours(internal_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in int_contours:
                    if cv2.contourArea(cnt) > 10:  # Filter tiny contours
                        # Simplify contour
                        epsilon = effective_tolerance * cv2.arcLength(cnt, True) / 100
                        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                        
                        # Add as geometric element
                        points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx_cnt]
                        if len(points) >= 3:
                            geometric_elements.append({
                                'type': 'contour',
                                'points': points
                            })
                            
                # If still no good contours, use the bounding box as fallback
                if not geometric_elements:
                    geometric_elements.append(bbox_element)
        else:
            # No mask data, use defaults
            bbox = [0, 0, 0, 0]
            bbox_width, bbox_height = 0, 0
            center_x, center_y = 0, 0
            boundary_complexity = 0
        
        # Create segment data with enhanced metadata
        segment = {
            'id': i + 1,
            'class': class_name,
            'confidence': confidence,
            'color': color,
            'geometric_elements': geometric_elements,
            'area': float(area),
            'bbox': bbox,
            'bbox_dimensions': (float(bbox_width), float(bbox_height)),
            'center': (float(center_x), float(center_y)),
            'boundary_complexity': float(boundary_complexity),
            # Ensure each object has a unique ID
            'object_id': f"{class_name}_{i+1}",
            'export_quality': svg_quality,
            'export_timestamp': '2025-03-21 14:54:38'
        }
        
        segments_data.append(segment)
    
    return segments_data

def save_mask_as_png(mask, output_path, class_name, color):
    """
    Save an individual mask as a transparent PNG with colored overlay.
    """
    # Convert to uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Convert to CSS format if provided as RGB tuple (0-1)
    if isinstance(color, tuple) and len(color) == 3:
        color_rgb = tuple(int(c * 255) for c in color)
    else:
        # Default color
        color_rgb = (0, 0, 255)
    
    # Create a colored version of the mask with transparency
    h, w = mask_uint8.shape
    colored_mask = np.zeros((h, w, 4), dtype=np.uint8)
    colored_mask[mask_uint8 == 1] = (*color_rgb, 128)  # Semi-transparent color
    
    # Save as PNG with transparency
    cv2.imwrite(output_path, colored_mask)


def verify_svg_export(svg_path, original_mask, output_path=None):
    """
    Verify the quality of an SVG export by comparing with original mask.
    
    Args:
        svg_path: Path to SVG file
        original_mask: Original binary mask (numpy array)
        output_path: Optional path to save verification image
        
    Returns:
        dict: Verification metrics
    """
    if not SVG_VERIFICATION_AVAILABLE:
        print("⚠️  SVG verification requires svglib and reportlab. Install with: pip install svglib reportlab")
        return None
    
    try:
        # Load the SVG and convert to raster
        drawing = svg2rlg(svg_path)
        
        # Create a PIL image from the SVG
        svg_image_data = renderPM.drawToString(drawing, fmt='PNG')
        svg_image = Image.open(io.BytesIO(svg_image_data))
        
        # Convert to numpy array
        svg_array = np.array(svg_image)
        
        # Extract alpha channel and convert to binary mask (any non-zero alpha = mask)
        if svg_array.shape[2] == 4:  # With alpha channel
            svg_mask = (svg_array[:, :, 3] > 0).astype(np.uint8)
        else:  # No alpha channel, use grayscale
            svg_gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)
            svg_mask = (svg_gray > 0).astype(np.uint8)
        
        # Resize to match original mask if needed
        if svg_mask.shape != original_mask.shape:
            svg_mask = cv2.resize(
                svg_mask, 
                (original_mask.shape[1], original_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Calculate verification metrics
        intersection = np.logical_and(svg_mask, original_mask)
        union = np.logical_or(svg_mask, original_mask)
        
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        pixel_accuracy = np.mean(svg_mask == original_mask)
        
        # Calculate boundary precision (how well the boundaries match)
        original_boundary = cv2.dilate(original_mask, np.ones((3,3), np.uint8)) ^ original_mask
        svg_boundary = cv2.dilate(svg_mask, np.ones((3,3), np.uint8)) ^ svg_mask
        
        boundary_intersection = np.logical_and(original_boundary, svg_boundary)
        boundary_union = np.logical_or(original_boundary, svg_boundary)
        
        boundary_iou = (np.sum(boundary_intersection) / np.sum(boundary_union) 
                        if np.sum(boundary_union) > 0 else 0)
        
        # Save verification image if requested
        if output_path:
            # Create visual comparison
            h, w = original_mask.shape
            verification_image = np.zeros((h, w*3, 3), dtype=np.uint8)
            
            # Original mask (red)
            verification_image[:, :w, 0] = original_mask * 255
            
            # SVG mask (green)
            verification_image[:, w:2*w, 1] = svg_mask * 255
            
            # Differences (blue for original only, red for SVG only, purple for both)
            diff = np.zeros((h, w, 3), dtype=np.uint8)
            diff[original_mask == 1, 0] = 255  # Red for original mask
            diff[svg_mask == 1, 2] = 255  # Blue for SVG mask
            verification_image[:, 2*w:, :] = diff
            
            cv2.imwrite(output_path, verification_image)
            print(f"✓ Verification image saved to {output_path}")
        
        # Return metrics
        metrics = {
            "iou": float(iou),
            "pixel_accuracy": float(pixel_accuracy),
            "boundary_iou": float(boundary_iou),
            "quality_score": float((iou + boundary_iou) / 2)
        }
        
        # Print summary
        print(f"SVG Export Verification:")
        print(f"  IoU: {iou:.4f}")
        print(f"  Boundary IoU: {boundary_iou:.4f}")
        print(f"  Pixel Accuracy: {pixel_accuracy:.4f}")
        print(f"  Overall Quality Score: {metrics['quality_score']:.4f}")
        
        quality_level = "Excellent" if metrics['quality_score'] > 0.9 else \
                       "Good" if metrics['quality_score'] > 0.8 else \
                       "Fair" if metrics['quality_score'] > 0.7 else \
                       "Poor"
        print(f"  Quality Assessment: {quality_level}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error verifying SVG export: {str(e)}")
        traceback.print_exc()
        return None


def save_verification_report(metrics, svg_path, output_dir):
    """Save verification metrics to a JSON file"""
    if metrics is None:
        return
    
    try:
        # Create filename based on SVG path
        basename = os.path.basename(svg_path)
        name_without_ext = os.path.splitext(basename)[0]
        report_path = os.path.join(output_dir, f"{name_without_ext}_report.json")
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["svg_file"] = basename
        
        # Write JSON file
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"✓ Verification report saved to {report_path}")
        
    except Exception as e:
        print(f"❌ Error saving verification report: {str(e)}")


def process_pipeline(model,
                    image_path,
                    texts,
                    test_pipeline,
                    sam_predictor,
                    u2net_predictor=None,
                    score_thr=0.3,
                    output_dir='./output',
                    use_amp=False,
                    use_crf=True,
                    crf_params=None,
                    detailed_refinement=True,
                    edge_refinement=True,
                    export_svg=True,
                    export_png=True,
                    contour_method='skimage',
                    svg_quality='professional',
                    **kwargs):
    """
    Process an image through the entire pipeline: detection, segmentation, refinement, and export.
    
    Args:
        model: YOLO-World model
        image_path: Path to the image file
        texts: Text prompts for YOLO-World
        test_pipeline: Test pipeline for YOLO-World
        sam_predictor: SAM predictor object
        u2net_predictor: U²-Net predictor object
        score_thr: Confidence score threshold
        output_dir: Output directory
        use_amp: Whether to use automatic mixed precision
        use_crf: Whether to use CRF refinement
        crf_params: Parameters for CRF
        detailed_refinement: Whether to apply detailed refinement
        edge_refinement: Whether to enhance edge details
        export_svg: Whether to export SVG
        export_png: Whether to export PNG masks
        contour_method: Method to extract contours
        svg_quality: SVG export quality level
        **kwargs: Additional arguments
        
    Returns:
        dict: Processing results
    """
    # Set the refinement mode based on parameters
    if detailed_refinement and edge_refinement:
        refinement_mode = 'edge_precise'
    elif detailed_refinement:
        refinement_mode = 'high_quality'
    else:
        refinement_mode = 'standard'
        
    # Set CRF parameters if not provided
    if crf_params is None:
        crf_params = {
            'bilateral_sxy': 80,
            'bilateral_srgb': 13,
            'bilateral_compat': 10,
            'gaussian_sxy': 3,
            'gaussian_compat': 3,
            'iterations': 10
        }
        
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image basename
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Create subdirectories for different outputs
    svg_dir = os.path.join(output_dir, "svg")
    png_dir = os.path.join(output_dir, "png")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n{'='*30}")
    print(f"Processing: {basename}")
    print(f"{'='*30}")
    
    # Step 1: Load the image
    print("\n1. Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return None
        
    # Convert to RGB for SAM (which expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 2: Run YOLO-World detection
    print("\n2. Running YOLO-World object detection...")
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                     data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    # Convert to supervision Detections format
    pred_instances_np = pred_instances.cpu().numpy()
    detections = sv.Detections(xyxy=pred_instances_np['bboxes'],
                              class_id=pred_instances_np['labels'],
                              confidence=pred_instances_np['confidence'] if 'confidence' in pred_instances_np else pred_instances_np['scores'])

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    print(f"  • Detected {len(detections.xyxy)} objects")
    for i, (label, conf) in enumerate(zip(labels, detections.confidence)):
        print(f"    [{i+1}] {label} (Score: {conf:.2f})")
    
    # Create bounding box visualization
    detection_vis = image.copy()
    detection_vis = BOUNDING_BOX_ANNOTATOR.annotate(detection_vis, detections)
    detection_vis = LABEL_ANNOTATOR.annotate(detection_vis, detections, labels=labels)
    
    # Save the detection visualization
    detection_path = os.path.join(vis_dir, f"{name_without_ext}_detection.jpg")
    cv2.imwrite(detection_path, detection_vis)
    
    # Step 3: Generate segmentation masks with SAM
    print("\n3. Generating segmentation masks with SAM...")
    all_masks = []
    
    if sam_predictor is None:
        print("⚠️ SAM predictor not available. Using simple box masks.")
        # Create simple box masks
        for box in detections.xyxy:
            box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            box_mask[y1:y2, x1:x2] = 1
            all_masks.append(box_mask)
    else:
        # Initialize SAM with the image
        sam_predictor.set_image(image_rgb)
        
        # Process each detection to get masks
        for i, box in enumerate(detections.xyxy):
            print(f"  • Processing object {i+1}/{len(detections.xyxy)}: {labels[i]}")
            
            # Expand box slightly to capture context
            expand_ratio = kwargs.get('expand_box', 0.1)
            if expand_ratio > 0:
                expanded_box = expand_box(box, expand_ratio, image_rgb.shape[:2])
            else:
                expanded_box = box
                
            try:
                # Generate masks with SAM
                masks, scores = enhance_sam_masks(sam_predictor, image_rgb, expanded_box, 
                                               multimask_output=kwargs.get('multiscale_sam', True))
                
                if masks is not None and masks.shape[0] > 0:
                    # Use the mask with the highest score
                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    
                    # Convert to uint8
                    mask_uint8 = mask.astype(np.uint8)
                    all_masks.append(mask_uint8)
                else:
                    # Fallback to box if SAM fails
                    print(f"  ⚠️ SAM failed for object {i+1}, using box mask")
                    box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, expanded_box)
                    box_mask[y1:y2, x1:x2] = 1
                    all_masks.append(box_mask)
            except Exception as e:
                print(f"  ❌ Error processing object {i+1}: {str(e)}")
                # Create fallback box mask
                box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = map(int, expanded_box)
                box_mask[y1:y2, x1:x2] = 1
                all_masks.append(box_mask)
    
    # Step 4: Refine masks with U²-Net and/or CRF
    print("\n4. Refining masks with advanced techniques...")
    refined_masks = []
    
    for i, mask in enumerate(all_masks):
        print(f"  • Refining mask {i+1}/{len(all_masks)}: {labels[i]}")
        processed_mask = mask.copy()
        
        # Step 4.1: Apply U²-Net refinement if available
        if u2net_predictor is not None and u2net_predictor.model is not None:
            print(f"    - Applying U²-Net refinement...")
            processed_mask = refine_mask_with_u2net(image_rgb, processed_mask, u2net_predictor)
        
        # Step 4.2: Apply CRF refinement if requested
        if use_crf:
            print(f"    - Applying CRF refinement...")
            processed_mask = apply_crf(image_rgb, processed_mask, crf_params)
        
        # Step 4.3: Apply advanced refinement based on mode
        if detailed_refinement:
            print(f"    - Applying {refinement_mode} refinement...")
            processed_mask = optimized_mask_refinement(image_rgb, processed_mask, refinement_mode)
            
        # Step 4.4: Apply smoothing if requested
        smooth_mask_flag = kwargs.get('smooth_mask', False)
        if smooth_mask_flag:
            kernel_size = kwargs.get('smooth_kernel', 3)
            print(f"    - Smoothing mask (kernel size: {kernel_size})...")
            processed_mask = smooth_mask(processed_mask, kernel_size)
            
        refined_masks.append(processed_mask)
    
    # Step 5: Create segmentation visualization
    print("\n5. Creating visualizations...")
    seg_vis = image.copy()
    
    # Add masks to the detections for visualization
    if refined_masks and len(refined_masks) == len(detections):
        # Convert to appropriate format expected by supervision
        mask_array = np.array([mask.astype(bool) for mask in refined_masks])
        
        # Create a new Detections object with masks
        detections_with_masks = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            mask=mask_array
        )
        
        seg_vis = MASK_ANNOTATOR.annotate(seg_vis, detections_with_masks)
        seg_vis = BOUNDING_BOX_ANNOTATOR.annotate(seg_vis, detections)
        seg_vis = LABEL_ANNOTATOR.annotate(seg_vis, detections, labels=labels)
    
    # Save the segmentation visualization
    segmentation_path = os.path.join(vis_dir, f"{name_without_ext}_segmentation.jpg")
    cv2.imwrite(segmentation_path, seg_vis)
    
    # Create side-by-side comparison
    combined_h, combined_w = image.shape[0], image.shape[1] * 2
    combined_vis = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined_vis[:, :image.shape[1]] = detection_vis
    combined_vis[:, image.shape[1]:] = seg_vis
    
    # Save the combined visualization
    combined_path = os.path.join(vis_dir, f"{name_without_ext}_combined.jpg")
    cv2.imwrite(combined_path, combined_vis)
    
    # Step 6: Export to SVG/PNG
    export_results = {}
    
    if export_svg and refined_masks:
        print("\n6. Exporting to SVG...")
        svg_path = os.path.join(svg_dir, f"{name_without_ext}.svg")
        
        # Prepare segments data for SVG export
        segments_data = create_segments_for_svg_export(
            refined_masks,
            labels,
            image.shape[:2],
            contour_method=contour_method,
            simplify_tolerance=kwargs.get('simplify_tolerance', 0.1),
            svg_quality=svg_quality
        )
        
        if segments_data:
            # Prepare metadata
            metadata = {
                'title': f"YOLO-World + SAM segmentation of {basename}",
                'author': SCRIPT_AUTHOR,
                'date': SCRIPT_DATE,
                'version': SCRIPT_VERSION
            }
            
            # Export using SVGExporter
            svg_exporter = SVGExporter(
                line_thickness=1.5, 
                quality_level=svg_quality
            )
            
            svg_success = svg_exporter.export_to_svg(
                segments_data,
                svg_path,
                image.shape[:2][::-1],  # Size as (width, height)
                metadata=metadata
            )
            
            export_results['svg_path'] = svg_path if svg_success else None
            
            # Verify export if requested
            verify_export = kwargs.get('verify_export', False)
            verification_report = kwargs.get('verification_report', False)
            
            if svg_success and verify_export and len(refined_masks) > 0:
                # Use first mask for verification
                verification_image_path = os.path.join(vis_dir, f"{name_without_ext}_svg_verification.png")
                metrics = verify_svg_export(svg_path, refined_masks[0], verification_image_path)
                export_results['svg_metrics'] = metrics
                
                # Save verification report if requested
                if verification_report and metrics:
                    save_verification_report(metrics, svg_path, svg_dir)
        else:
            print("⚠️ No valid segments for SVG export")
    
    if export_png and refined_masks:
        print("\n7. Exporting individual PNG masks...")
        
        # Generate colors for classes
        class_colors = {}
        for i, label in enumerate(labels):
            class_name = label.split()[0]
            if class_name not in class_colors:
                # Use color map or generate new one
                hue = (i * 0.618033988749895) % 1.0
                class_colors[class_name] = tuple(plt.cm.hsv(hue)[:3])
        
        png_paths = []
        
        # Save each mask as transparent PNG
        for i, (mask, label) in enumerate(zip(refined_masks, labels)):
            class_name = label.split()[0]
            color = class_colors[class_name]
            
            png_path = os.path.join(png_dir, f"{name_without_ext}_{class_name}_{i}.png")
            save_mask_as_png(mask, png_path, class_name, color)
            print(f"  • PNG mask exported to: {png_path}")
            png_paths.append(png_path)
            
        export_results['png_paths'] = png_paths
    
    # Prepare results summary
    results = {
        'image_path': image_path,
        'detection_count': len(detections.xyxy),
        'detection_labels': labels,
        'detection_scores': detections.confidence.tolist(),
        'masks_count': len(refined_masks),
        'visualizations': {
            'detection': detection_path,
            'segmentation': segmentation_path,
            'combined': combined_path
        },
        'exports': export_results
    }
    
    print(f"\n✅ Processing complete for {basename}")
    
    return results

def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       sam_predictor,
                       u2net_predictor=None,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./output',
                       use_amp=False,
                       show=False,
                       annotation=False,
                       export_svg=True,
                       export_png=True,
                       simplify_tolerance=0.1,
                       contour_method='both',  # Change default to 'both' for better results
                       precision_level=6,
                       curve_smoothness=0.2,
                       use_crf=True,
                       crf_params=None,
                       smooth_mask=False,
                       smooth_kernel_size=3,
                       detailed_refinement=True,
                       edge_refinement=True,
                       expand_box_ratio=0.15,  # Increase default expansion to capture more context
                       multiscale_sam=True,
                       svg_quality='professional',
                       verify_export=False,
                       verification_report=False,
                       show_gpu_stats=False):
    """
    Improved main inference function that processes a single image or directory.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Get image basename for output files
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Convert to RGB for SAM (which expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if show_gpu_stats:
        print_gpu_memory(prefix="Before YOLO-World: ")
    
    # YOLO-World detection
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                     data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                              class_id=pred_instances['labels'],
                              confidence=pred_instances['scores'])

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    
    if show_gpu_stats:
        print_gpu_memory(prefix="After YOLO-World: ")
        torch.cuda.empty_cache()
        print_gpu_memory(prefix="After cache clear: ")

    # Segmentation with SAM
    all_masks = []
    sam_success_count = 0

    if sam_predictor is None:
        print("⚠️ SAM predictor not initialized. Using bounding box masks only.")
        # Create simple box masks
        for box in detections.xyxy:
            box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            box_mask[y1:y2, x1:x2] = 1
            all_masks.append(box_mask)
    else:
        try:
            # Initialize SAM predictor with the image
            sam_predictor.set_image(image_rgb)
            
            # Process each detection
            for i, box in enumerate(detections.xyxy):
                print(f"Processing object {i+1}/{len(detections.xyxy)}: {labels[i]}")
                
                # Expand box to capture more context
                expanded_box = expand_box(box, expand_box_ratio, image_rgb.shape[:2])
                
                try:
                    # Generate improved masks with SAM - ensure box is numpy array
                    if isinstance(expanded_box, torch.Tensor):
                        expanded_box_numpy = expanded_box.cpu().numpy()
                    else:
                        expanded_box_numpy = np.array(expanded_box)
                        
                    masks, scores = enhance_sam_masks(sam_predictor, image_rgb, expanded_box_numpy, multimask_output=True)
                    # Generate improved masks with SAM - get multiple masks
                    # masks, scores = enhance_sam_masks(sam_predictor, image_rgb, expanded_box, multimask_output=True)
                    
                    if masks is not None and masks.shape[0] > 0:
                        # Use the best mask
                        best_mask = masks[0].astype(np.uint8)
                        all_masks.append(best_mask)
                        sam_success_count += 1
                    else:
                        # Generate multiple points in the box for point prompting as a fallback
                        x1, y1, x2, y2 = map(int, expanded_box)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Create grid of points
                        grid_size = 3  # 3x3 grid
                        points = []
                        for gx in range(grid_size):
                            for gy in range(grid_size):
                                px = x1 + (x2 - x1) * gx // (grid_size - 1)
                                py = y1 + (y2 - y1) * gy // (grid_size - 1)
                                points.append([px, py])
                        
                        # Convert to numpy arrays
                        point_coords = np.array(points)
                        point_labels = np.ones(len(points))  # All foreground
                        
                        # Try with point prompts
                        try:
                            masks, scores, _ = sam_predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                box=np.array(expanded_box),
                                multimask_output=True
                            )
                            
                            if masks is not None and masks.shape[0] > 0:
                                best_mask = masks[0].astype(np.uint8)
                                all_masks.append(best_mask)
                                sam_success_count += 1
                                print(f"  ✓ Used point prompting as fallback")
                            else:
                                # Fallback to box if point prompting also fails
                                print(f"  ⚠️ SAM failed for object {i+1}, using box mask")
                                box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                                box_mask[y1:y2, x1:x2] = 1
                                all_masks.append(box_mask)
                        except Exception as e:
                            # Fallback to box
                            print(f"  ⚠️ Point prompting failed: {e}. Using box mask")
                            box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                            box_mask[y1:y2, x1:x2] = 1
                            all_masks.append(box_mask)
                except Exception as e:
                    print(f"  ❌ Error processing object {i+1}: {str(e)}")
                    # Create fallback box mask
                    box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, expanded_box)
                    box_mask[y1:y2, x1:x2] = 1
                    all_masks.append(box_mask)
                    
            print(f"SAM successfully generated detailed masks for {sam_success_count}/{len(detections.xyxy)} objects")
            
        except Exception as e:
            print(f"❌ SAM segmentation error: {str(e)}")
            traceback.print_exc()
            # Create simple box masks as fallback
            all_masks = []
            for box in detections.xyxy:
                box_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                box_mask[y1:y2, x1:x2] = 1
                all_masks.append(box_mask)
    
    # Process masks with refined techniques
    refined_masks = []
    
    if all_masks:
        print(f"Refining {len(all_masks)} masks...")
        
        # Determine refinement mode based on parameters
        refinement_mode = 'standard'
        if detailed_refinement and edge_refinement:
            refinement_mode = 'edge_precise'
        elif detailed_refinement:
            refinement_mode = 'high_quality'
        
        for i, mask in enumerate(all_masks):
            print(f"Refining mask {i+1}/{len(all_masks)}: {labels[i]}")
            processed_mask = mask.copy()
            
            # Apply U²-Net refinement if available
            if u2net_predictor is not None and u2net_predictor.model is not None:
                print(f"  Applying U²-Net refinement...")
                processed_mask = refine_mask_with_u2net(image_rgb, processed_mask, u2net_predictor)
            
            # Apply CRF refinement if requested
            if use_crf:
                print(f"  Applying CRF refinement...")
                
                # Set default CRF parameters if not provided
                if crf_params is None:
                    crf_params = {
                        'bilateral_sxy': 80,
                        'bilateral_srgb': 13,
                        'bilateral_compat': 10,
                        'gaussian_sxy': 3,
                        'gaussian_compat': 3,
                        'iterations': 10
                    }
                
                # Apply CRF refinement
                processed_mask = apply_crf(image_rgb, processed_mask, crf_params)
                
            # Apply advanced refinement based on mode
            if detailed_refinement:
                print(f"  Applying {refinement_mode} refinement...")
                processed_mask = optimized_mask_refinement(image_rgb, processed_mask, refinement_mode)
                
            # Apply smoothing if requested
            if smooth_mask:
                kernel_size = smooth_kernel_size
                print(f"  Smoothing mask (kernel size: {kernel_size})...")
                processed_mask = smooth_mask(processed_mask, kernel_size)
                
            # Add the refined mask
            refined_masks.append(processed_mask)
    
    # Create visualizations
    # 1. YOLO-World detection visualization
    yolo_vis = image.copy()
    yolo_vis = BOUNDING_BOX_ANNOTATOR.annotate(yolo_vis, detections)
    yolo_vis = LABEL_ANNOTATOR.annotate(yolo_vis, detections, labels=labels)
    
    # 2. Segmentation visualization
    seg_vis = image.copy()
    
    # Add masks to the detections
    if refined_masks and len(refined_masks) == len(detections):
        # Convert to appropriate format expected by supervision
        mask_array = np.array([mask.astype(bool) for mask in refined_masks])
        
        # Create a new Detections object with masks
        detections_with_masks = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            mask=mask_array
        )
        
        seg_vis = MASK_ANNOTATOR.annotate(seg_vis, detections_with_masks)
        seg_vis = BOUNDING_BOX_ANNOTATOR.annotate(seg_vis, detections)
        seg_vis = LABEL_ANNOTATOR.annotate(seg_vis, detections, labels=labels)
    
    # 3. Side-by-side comparison
    combined_h, combined_w = image.shape[0], image.shape[1] * 2
    combined_vis = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined_vis[:, :image.shape[1]] = yolo_vis
    combined_vis[:, image.shape[1]:] = seg_vis
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(vis_dir, f"{name_without_ext}_yolo.jpg"), yolo_vis)
    cv2.imwrite(os.path.join(vis_dir, f"{name_without_ext}_segmentation.jpg"), seg_vis)
    cv2.imwrite(os.path.join(vis_dir, f"{name_without_ext}_combined.jpg"), combined_vis)
    
    # Export SVG if requested using custom SVGExporter
    if export_svg and refined_masks:
        svg_dir = os.path.join(output_dir, "svg")
        os.makedirs(svg_dir, exist_ok=True)
        svg_name = f"{name_without_ext}.svg"
        svg_path = os.path.join(svg_dir, svg_name)
        
        # Prepare segments data for SVG export - with per-object unique colors
        segments_data = create_segments_for_svg_export(
            refined_masks,
            labels,
            image.shape[:2],
            contour_method=contour_method,
            simplify_tolerance=simplify_tolerance,
            svg_quality=svg_quality
        )
        
        if segments_data:
            # Prepare metadata
            metadata = {
                'title': f"YOLO-World + SAM segmentation of {basename}",
                'author': SCRIPT_AUTHOR,
                'date': SCRIPT_DATE,
                'version': SCRIPT_VERSION
            }
            
            # Export using SVGExporter
            svg_exporter = SVGExporter(line_thickness=1.5, quality_level=svg_quality)
            success = svg_exporter.export_to_svg(
                segments_data,
                svg_path,
                image.shape[:2][::-1],  # Size as (width, height)
                metadata=metadata
            )
            
            if success:
                print(f"\n✅ SVG file exported to: {svg_path}")
            else:
                print(f"\n❌ Failed to export SVG to: {svg_path}")
                
    # Export individual PNG masks if requested
    if export_png and refined_masks:
        png_dir = os.path.join(output_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
        
        # Generate unique colors for each object
        object_colors = {}
        for i, label in enumerate(labels):
            # Use same unique color generation as in create_segments_for_svg_export
            if i < 20:  # Assuming we have 20 colors defined
                hue = i / 20.0
            else:
                hue = (i * 0.618033988749895) % 1.0  # Golden ratio conjugate
            
            r, g, b = plt.cm.hsv(hue)[:3]
            object_colors[i] = (r, g, b)
        
        # Save each mask as transparent PNG
        for i, (mask, label) in enumerate(zip(refined_masks, labels)):
            class_name = label.split()[0]
            color = object_colors[i]  # Use unique color per object
            
            png_path = os.path.join(png_dir, f"{name_without_ext}_{class_name}_{i}.png")
            save_mask_as_png(mask, png_path, class_name, color)
            print(f"PNG mask exported to: {png_path}")
    
    if show:
        # Create a matplotlib figure for visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title('YOLO-World Detection')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(seg_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title('SAM Segmentation')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the script from command line."""
    # Parse command line arguments
    args = parse_args()
    
    # Check GPU and select the optimal GPU
    gpu_id = args.gpu_id if args.gpu_id is not None else 'max_mem'
    cuda_available = check_gpu(select_gpu=gpu_id, force_nvidia=True)
    
    # Update device setting based on GPU availability
    if not cuda_available and args.device.startswith('cuda'):
        print(f"Warning: CUDA not available. Switching to CPU.")
        args.device = 'cpu'
    elif cuda_available:
        # Use the active CUDA device
        current_device = torch.cuda.current_device()
        args.device = f'cuda:{current_device}'
    
    # Show info about PyTorch and CUDA versions
    print(f"PyTorch version: {torch.__version__}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
    
    # Load configuration file
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    
    # Initialize YOLO-World model
    print(f"Loading YOLO-World model on {args.device}...")
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # Initialize SAM
    sam_predictor = initialize_sam(args.sam_model, args.sam_checkpoint, args.device)
    
    # Initialize U²-Net
    u2net_predictor = None
    if args.u2net_checkpoint:
        try:
            u2net_predictor = U2NetPredictor(args.u2net_checkpoint, args.device)
        except Exception as e:
            print(f"❌ Error initializing U²-Net: {str(e)}")
            traceback.print_exc()

    # Initialize test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline = Compose(test_pipeline_cfg)

    # Parse text prompts
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # Create output directory
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # CRF parameters
    crf_params = {
        'bilateral_sxy': args.crf_bi_xy,
        'bilateral_srgb': args.crf_bi_rgb,
        'bilateral_compat': args.crf_bi_w,
        'gaussian_sxy': args.crf_pos_w,
        'gaussian_compat': 3,
        'iterations': args.crf_iter
    }

    # Load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if not images:
            print(f"No images found in directory: {args.image}")
            exit(1)
    else:
        images = [args.image]

    # Reparameterize the model with text prompts
    model.reparameterize(texts)
    
    # Show GPU memory usage before processing
    if args.gpu_stats and torch.cuda.is_available():
        print_gpu_memory(prefix="Before processing: ")
    
    # Start the timer
    start_time = time.time()
    
    # Process each image
    from tqdm import tqdm
    results = []
    for image_path in tqdm(images, desc="Processing images"):
        result = inference_detector(model,
                           image_path,
                           texts,
                           test_pipeline,
                           sam_predictor,
                           u2net_predictor,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation,
                           export_svg=args.export_svg,
                           export_png=args.export_png,
                           simplify_tolerance=args.simplify_tolerance,
                           contour_method=args.contour_method,
                           precision_level=args.precision_level,
                           curve_smoothness=args.curve_smoothness,
                           use_crf=args.use_crf,
                           crf_params=crf_params,
                           smooth_mask=args.smooth_mask,
                           smooth_kernel_size=args.smooth_kernel,
                           detailed_refinement=args.detailed_refinement,
                           edge_refinement=args.edge_refinement,
                           expand_box_ratio=args.expand_box,
                           multiscale_sam=args.multiscale_sam,
                           svg_quality=args.svg_quality,
                           verify_export=args.verify_export,
                           verification_report=args.verification_report,
                           show_gpu_stats=args.gpu_stats)
        
        results.append(result)
        
        # Free GPU memory after each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Show final GPU memory usage
    if args.gpu_stats and torch.cuda.is_available():
        print_gpu_memory(prefix="After processing: ")
    
    # Print summary report
    num_images = len(images)
    print("\n===== PROCESSING SUMMARY =====")
    print(f"✅ Processed {num_images} image{'s' if num_images > 1 else ''} in {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/num_images:.2f} seconds")
    
    total_objects = sum(result['detection_count'] for result in results if result and 'detection_count' in result)
    print(f"Total objects detected: {total_objects}")
    
    # Statistics by class
    class_counts = {}
    for result in results:
        if result and 'detection_labels' in result:
            for label in result['detection_labels']:
                class_name = label.split()[0]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
    print("\nObjects detected by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {class_name}: {count}")
    
    # Print quality settings summary
    quality_features = []
    if args.use_crf:
        quality_features.append("CRF refinement")
    if args.detailed_refinement:
        quality_features.append("detailed refinement")
    if args.edge_refinement:
        quality_features.append("edge-aware processing")
    if args.smooth_mask:
        quality_features.append("mask smoothing")
    if args.multiscale_sam:
        quality_features.append("multi-scale SAM")
    if u2net_predictor is not None and u2net_predictor.model is not None:
        quality_features.append("U²-Net refinement")
    if args.export_svg:
        quality_features.append(f"{args.svg_quality}-quality SVG export")
    
    if quality_features:
        print(f"\nQuality enhancements: {', '.join(quality_features)}")
    
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    
    # Write processing information to a summary file
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Professional Segmentation Suite v{SCRIPT_VERSION}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Author: {SCRIPT_AUTHOR}\n\n")
        
        f.write(f"Processed {num_images} image(s) in {elapsed_time:.2f} seconds\n")
        f.write(f"Average time per image: {elapsed_time/num_images:.2f} seconds\n")
        f.write(f"Total objects detected: {total_objects}\n\n")
        
        f.write("Objects detected by class:\n")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {class_name}: {count}\n")
            
        f.write("\nQuality enhancements: " + ", ".join(quality_features) + "\n")
        
        f.write("\nCommand used:\n")
        f.write(f"python {' '.join(sys.argv)}\n")
        
    print(f"\nSummary saved to: {summary_file}")
    
    # Return command for repeated use
    print("\nRun the following command to process more images with the same settings:")
    print(f"python {sys.argv[0]} {' '.join(sys.argv[1:])}")


if __name__ == '__main__':
    main()