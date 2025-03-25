#!/usr/bin/env python3
"""
Professional Segmentation Suite: YOLO-World + Hybrid SAM + CRF
Combines YOLO-World for detection with a hybrid SAM approach (predictor + mask generator)
for superior contour detail and refined segmentation masks.

Author: FETHI
Version: 11.0.0
Date: 2025-03-25
"""

import colorsys
import os
import sys
import time
import glob
import json
import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any

# Computer vision packages
import cv2
import torch
import mmcv
from mmengine import Config
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector

# SAM and related imports
from segment_anything import sam_model_registry
from segment_anything import SamPredictor, SamAutomaticMaskGenerator

# Transformers for CLIP tokenizer
from transformers import CLIPTokenizerFast

# Import pydensecrf if available (for CRF refinement)
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: pydensecrf not found. CRF refinement will be disabled.")
    print("To enable CRF refinement, install pydensecrf: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")

# Import supervision if available (for visualization)
try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False
    print("Warning: supervision package not found. Some visualizations may be limited.")
    print("To enable all visualizations, install supervision: pip install supervision")

# Import additional packages for SVG export
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# Script information
SCRIPT_VERSION = "11.0.0"
SCRIPT_AUTHOR = "FETHI"
SCRIPT_DATE = "2025-03-25"


###########################################
# class DictAction
###########################################
class DictAction(argparse.Action):
    """
    argparse action to split an argument into key=value form
    and store it in a dictionary. This is used for setting
    config options from the command line.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        dictionary = getattr(namespace, self.dest, {})
        if dictionary is None:
            dictionary = {}
        for kv in values:
            key, value = kv.split('=', maxsplit=1)
            try:
                value = eval(value)
            except:
                value = str(value)
            dictionary[key] = value
        setattr(namespace, self.dest, dictionary)
###########################################
# UTILITY FUNCTIONS
###########################################

def format_time(seconds):
    """Format time in seconds to a readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def print_gpu_info():
    """Print information about available GPUs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA is available with {gpu_count} GPU{'s' if gpu_count > 1 else ''}:")
        
        # List all GPUs with their properties
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"  • GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✗ CUDA is not available. Using CPU mode (this will be very slow).")


def auto_select_gpu():
    """Automatically select GPU with the most available memory"""
    if not torch.cuda.is_available():
        print("No CUDA capable GPU available. Using CPU.")
        return "cpu"
    
    if torch.cuda.device_count() == 1:
        print(f"✓ Auto-selected GPU 0 with maximum memory ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
        return "cuda:0"
        
    # Find GPU with most free memory
    max_free_memory = 0
    selected_gpu = 0
    
    for i in range(torch.cuda.device_count()):
        # Get total memory
        total_memory = torch.cuda.get_device_properties(i).total_memory
        
        # Get current allocated memory
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated(i)
        
        free_memory = total_memory - allocated_memory
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            selected_gpu = i
    
    print(f"✓ Auto-selected GPU {selected_gpu} with maximum memory ({max_free_memory / (1024**3):.1f} GB)")
    print(f"✓ Current active GPU: {selected_gpu} - {torch.cuda.get_device_name(selected_gpu)}")
    
    # Return the CUDA device string
    return f"cuda:{selected_gpu}"


def log_system_info():
    """Log system information for debugging"""
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        cuda_arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
        if cuda_arch_list:
            print(f"CUDA architectures: {', '.join(cuda_arch_list)}")


class DetectionResult:
    """
    Simple container class for storing detection results
    """
    def __init__(self, xyxy: List, confidence: List, class_ids: List, class_names: List):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_ids = class_ids
        self.class_names = class_names


###########################################
# HYBRID SAM PROCESSOR CLASS
###########################################

class HybridSAMProcessor:
    """Combines SAM predictor and automatic mask generator for optimal results"""
    
    def __init__(self, sam_model, device, 
                 points_per_side=32,
                 pred_iou_thresh=0.86,
                 stability_score_thresh=0.92,
                 min_mask_region_area=100):
        """
        Initialize Hybrid SAM Processor
        
        Args:
            sam_model: SAM model instance
            device: Device to run on (cuda or cpu)
            points_per_side: Points per side for mask generator
            pred_iou_thresh: IoU threshold for mask generator predictions
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum area for mask regions
        """
        self.sam_model = sam_model
        self.device = device
        
        # Initialize both components
        self.predictor = SamPredictor(sam_model)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask"
        )
        
        print("✓ Hybrid SAM processor initialized with both predictor and mask generator")
        self.auto_masks = None
        self.current_image = None
    
    def set_image(self, image):
        """Set image for the predictor"""
        self.predictor.set_image(image)
        self.current_image = image
    
    def generate_candidate_masks(self):
        """Generate all possible masks for the current image"""
        print("Generating automatic masks as candidates...")
        self.auto_masks = self.mask_generator.generate(self.current_image)
        print(f"Generated {len(self.auto_masks)} automatic masks")
        return self.auto_masks
    
    def predict_box_hybrid(self, box, points=None, labels=None, multimask=True, 
                          expand_box=0.15):
        """
        Generate mask using box prompt and refine with automatic masks
        
        Args:
            box: Input box as [x1, y1, x2, y2]
            points: Optional point prompts
            labels: Optional point labels (1 for foreground, 0 for background)
            multimask: Whether to return multiple masks
            expand_box: Amount to expand box by for better context
        
        Returns:
            Best mask, score and auto-refined version
        """
        # Expand box if requested
        if expand_box > 0:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - expand_box * w))
            y1 = max(0, int(y1 - expand_box * h))
            x2 = min(self.current_image.shape[1], int(x2 + expand_box * w))
            y2 = min(self.current_image.shape[0], int(y2 + expand_box * h))
            box = np.array([x1, y1, x2, y2])
        
        # First, get mask from predictor using box prompt
        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask
        )
        
        # Get the best mask from predictor results
        best_idx = np.argmax(scores)
        pred_mask = masks[best_idx]
        pred_score = scores[best_idx]
        
        # If no auto masks generated yet, return predictor mask
        if self.auto_masks is None or len(self.auto_masks) == 0:
            return pred_mask, pred_score, pred_mask, 0.0
        
        # Find best matching automatic mask by IoU
        best_auto_mask = None
        best_iou = 0
        
        # Create a dilated version of the predicted mask for matching
        dilated_pred_mask = dilate_mask(pred_mask, 15)
        
        for mask_data in self.auto_masks:
            auto_mask = mask_data["segmentation"].astype(bool)
            
            # Calculate IoU between predicted mask and auto mask
            intersection = np.logical_and(dilated_pred_mask, auto_mask).sum()
            union = np.logical_or(dilated_pred_mask, auto_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou and iou > 0.25:  # Lower threshold for matching
                best_iou = iou
                best_auto_mask = auto_mask
        
        # Combine masks if a good match found
        if best_auto_mask is not None and best_iou > 0.25:
            # Refine boundary details using auto mask
            refined_mask = refine_mask_boundaries(pred_mask, best_auto_mask)
            return pred_mask, pred_score, refined_mask, best_iou
        else:
            # If no good match, return the original prediction
            return pred_mask, pred_score, pred_mask, 0.0


###########################################
# MASK PROCESSING FUNCTIONS
###########################################

def dilate_mask(mask, kernel_size=5):
    """Dilate binary mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode_mask(mask, kernel_size=5):
    """Erode binary mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def refine_mask_boundaries(base_mask, auto_mask):
    """
    Refine mask boundaries using automatic mask for better contour detail
    
    Args:
        base_mask: Base mask from predictor
        auto_mask: Automatic mask with better boundaries
        
    Returns:
        Refined mask combining base semantics with auto boundary details
    """
    # Create a boundary zone around the base mask
    dilated = dilate_mask(base_mask, 15)
    eroded = erode_mask(base_mask, 10)
    boundary_zone = np.logical_and(dilated, np.logical_not(eroded))
    
    # Extract boundary detail from auto mask
    boundary_detail = np.logical_and(auto_mask, boundary_zone)
    
    # Combine with the core of the base mask
    refined_mask = np.logical_or(eroded, boundary_detail)
    
    return refined_mask


def smooth_mask(mask, kernel_size=3):
    """
    Apply smoothing to a binary mask
    
    Args:
        mask: Binary mask
        kernel_size: Size of smoothing kernel (must be odd)
        
    Returns:
        Smoothed binary mask
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
        
    # Convert to uint8 for OpenCV operations
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), 0)
    
    # Threshold back to binary
    _, smoothed = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    
    return smoothed.astype(bool)


def apply_edge_refinement(mask, image, edge_type='precise'):
    """
    Apply edge-aware refinement for sharper boundaries
    
    Args:
        mask: Binary mask
        image: RGB image
        edge_type: Type of edge refinement ('precise', 'smooth', 'preserve')
        
    Returns:
        Edge-refined mask
    """
    # Ensure mask is binary (0/1 values only)
    mask_binary = mask.astype(bool)
    
    # Convert mask to uint8 (0 or 1 values only)
    mask_uint8 = mask_binary.astype(np.uint8)
    
    # For visualization and edge detection, we use 0-255 range
    mask_viz = mask_uint8 * 255
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply different refinement strategies based on edge_type
    if edge_type == 'precise':
        # Use Canny edge detection for precise boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate mask and find edge intersection
        dilated = cv2.dilate(mask_viz, np.ones((5, 5), np.uint8), iterations=1)
        edge_zone = cv2.bitwise_and(dilated, edges)
        
        # Combine with original mask
        refined_viz = cv2.bitwise_or(mask_viz, edge_zone)
        
        # Convert back to binary
        refined = refined_viz > 0
        
    elif edge_type == 'smooth':
        # Use bilateral filter for smooth but edge-preserving boundaries
        mask_float = mask_viz.astype(np.float32) / 255.0
        refined_float = cv2.bilateralFilter(mask_float, 9, 75, 75)
        refined = refined_float > 0.5
        
    elif edge_type == 'preserve':
        # Use grabcut algorithm for boundary refinement
        # Initialize grabcut with mask
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert mask to grabcut format (0=bg, 1=fg, 2=prob_bg, 3=prob_fg)
        grabcut_mask = np.where(mask_binary, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        
        # Run grabcut for 2 iterations
        cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        
        # Convert back to binary mask
        refined = np.where((grabcut_mask==cv2.GC_FGD) | (grabcut_mask==cv2.GC_PR_FGD), True, False)
    
    else:
        # Default: return original mask
        refined = mask_binary
    
    # Return as boolean mask
    return refined


###########################################
# CRF REFINEMENT FUNCTIONS
###########################################

def refine_mask_with_crf(image, mask, 
                        crf_iter=10, 
                        crf_pos_w=3.0, 
                        crf_bi_w=10.0, 
                        crf_bi_xy=80.0, 
                        crf_bi_rgb=13.0,
                        **kwargs):
    """
    Refine mask using CRF with direct unary potential creation to avoid index issues
    
    Args:
        image: RGB image
        mask: Binary mask to refine
        crf_iter: Number of CRF iterations
        crf_pos_w: Position weight (Gaussian kernel)
        crf_bi_w: Bilateral weight
        crf_bi_xy: Bilateral xy std dev
        crf_bi_rgb: Bilateral RGB std dev
        
    Returns:
        Refined binary mask
    """
    if not HAS_CRF:
        print("Warning: pydensecrf not available. Skipping CRF refinement.")
        return mask
    
    # Print a message
    print("  Applying CRF refinement...")
    
    # Convert mask to strict boolean and then 0/1 integers
    # This fixes potential issues with 255 values in the mask
    mask_bool = mask.astype(bool)
    mask_binary = mask_bool.astype(np.uint8)
    
    # Print unique values to debug
    unique_values = np.unique(mask_binary)
    print(f"  Mask unique values: {unique_values}")
    
    # Ensure proper shape
    h, w = image.shape[:2]
    n_pixels = h * w
    
    # Initialize CRF
    d = dcrf.DenseCRF2D(w, h, 2)  # 2 labels: fg, bg
    
    # Create unary potentials directly
    # This avoids using unary_from_labels which has index issues
    U = np.zeros((2, n_pixels), dtype=np.float32)
    
    # Set foreground/background probabilities based on mask
    # High confidence (0.9) for the class indicated by the mask
    # Low confidence (0.1) for the other class
    U[0, :] = -np.log(0.1)  # Background
    U[1, :] = -np.log(0.9)  # Foreground
    
    # Modify for mask values
    mask_flat = mask_binary.flatten()
    U[0, mask_flat > 0] = -np.log(0.9)  # Where mask is 1, background prob = 0.1
    U[1, mask_flat > 0] = -np.log(0.1)  # Where mask is 1, foreground prob = 0.9
    
    # Set unary energies
    d.setUnaryEnergy(U)
    
    # Add pairwise Gaussian potentials
    d.addPairwiseGaussian(sxy=crf_pos_w, compat=3)
    
    # Add pairwise bilateral potentials
    # This considers both spatial distance and color similarity
    d.addPairwiseBilateral(
        sxy=crf_bi_xy,
        srgb=crf_bi_rgb,
        rgbim=image,
        compat=crf_bi_w
    )
    
    # Perform inference
    Q = d.inference(crf_iter)
    
    # Get refined mask
    refined_mask = np.argmax(Q, axis=0).reshape((h, w)).astype(bool)
    
    return refined_mask


###########################################
# YOLO-WORLD DETECTION FUNCTIONS
###########################################
def get_text_embeddings(model, texts, tokenizer=None):
    """
    Get text embeddings from YOLO-World model with support for different API versions
    
    Args:
        model: YOLO-World model
        texts: List of text prompts
        tokenizer: CLIP tokenizer (optional)
        
    Returns:
        Text embeddings
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
        
    # Method 1: Try direct text embedding if available
    if hasattr(model, 'get_text_embeddings') and callable(getattr(model, 'get_text_embeddings')):
        # Some versions have a direct method
        return model.get_text_embeddings(texts)
        
    # Method 2: Try tokenize_and_embed if available
    if hasattr(model, 'tokenize_and_embed') and callable(getattr(model, 'tokenize_and_embed')):
        return model.tokenize_and_embed(texts)
    
    # Method 3: For newer YOLO-World versions with text_encoder accessor
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        # Initialize tokenizer if not provided
        if tokenizer is None:
            try:
                tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
                print("CLIP tokenizer downloaded successfully!")
            except Exception as e:
                print(f"Error loading tokenizer: {str(e)}")
                return None
                
        # Tokenize texts
        batch_tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        
        # Move tokens to model's device
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        tokens = {k: v.to(device) for k, v in batch_tokens.items()}
        
        # Extract text embeddings
        with torch.no_grad():
            text_embeddings = model.text_encoder(**tokens)
            
        return text_embeddings
    
    # Method 4: Direct inference with data format expected by YOLO-World
    print("Using direct inference for text embedding - your version of YOLO-World may handle text internally")
    return texts


def load_and_process_text(text):
    """
    Load and process text prompts
    
    Args:
        text: Text prompts as comma-separated string or path to text file
        
    Returns:
        List of text prompts
    """
    if os.path.exists(text):
        # Load from file (one prompt per line)
        with open(text, 'r') as f:
            text_prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(text_prompts)} text prompts from file: {text}")
    else:
        # Parse comma-separated text
        text_prompts = [t.strip() for t in text.split(',') if t.strip()]
        print(f"Using {len(text_prompts)} text prompts from command line")
        
    return text_prompts


def process_yolo_world(model, image, test_pipeline, text_prompts, topk=100, threshold=0.05, image_path=None):
    """
    Process image with YOLO-World model and text prompts
    
    Args:
        model: YOLO-World model
        image: RGB image
        test_pipeline: Test pipeline for image processing
        text_prompts: List of text prompts
        topk: Maximum number of detections to return
        threshold: Confidence threshold for detections
        image_path: Original image path (optional)
        
    Returns:
        DetectionResult object
    """
    # Process text prompts
    if isinstance(text_prompts, str):
        text_prompts = load_and_process_text(text_prompts)
    
    # Create tokenizer if needed
    tokenizer = None
    
    # Get text embeddings - may be actual embeddings or just the text
    text_embeddings = get_text_embeddings(model, text_prompts, tokenizer)
    
    # Prepare data for inference
    data = {}
    
    # If image_path is provided, use it
    if image_path and os.path.exists(image_path):
        data['img_path'] = image_path
        
        # Extract img_id from path
        img_id = os.path.splitext(os.path.basename(image_path))[0]
    else:
        # Create a temporary file to save the image
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        # Use timestamp as a unique identifier
        timestamp = int(time.time() * 1000)
        img_id = f"temp_img_{timestamp}"
        
        temp_path = os.path.join(temp_dir, f"{img_id}.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        data['img_path'] = temp_path
        print(f"Created temporary image file: {temp_path}")
    
    # Add img_id key - this is required by the formatter
    data['img_id'] = img_id
    
    # Add additional required keys for some pipelines
    data['ori_filename'] = os.path.basename(data['img_path'])
    data['filename'] = os.path.basename(data['img_path'])
    
    # Add text data - use 'texts' (plural) as the key name
    if isinstance(text_embeddings, list) and isinstance(text_embeddings[0], str):
        # text_embeddings is actually just the text list
        data['texts'] = text_embeddings
    else:
        # text_embeddings is actual embeddings
        data['texts'] = text_prompts
        data['embeddings'] = text_embeddings
    
    # Process image through pipeline
    try:
        data = test_pipeline(data)
        
        # Add batch dimension
        data['inputs'] = [data['inputs']]
        data['data_samples'] = [data['data_samples']]
        
        # Run inference
        with torch.no_grad():
            results = model.test_step(data)
    except Exception as e:
        print(f"Error during YOLO-World inference: {e}")
        print("Details of data passed to pipeline:")
        for k, v in data.items():
            if k == 'texts':
                print(f"- texts: {len(v)} prompts")
            elif k == 'embeddings' and hasattr(v, 'shape'):
                print(f"- embeddings: tensor of shape {v.shape}")
            elif k == 'img_path' or k == 'img_id' or k == 'filename':
                print(f"- {k}: {v}")
            else:
                print(f"- {k}: {type(v)}")
        raise
    
    # Extract predictions
    pred_instances = results[0].pred_instances
    
    # Apply confidence threshold and get top-k
    if 'scores' in pred_instances:
        scores = pred_instances.scores
        keep = scores > threshold
        
        if keep.sum() > 0:
            pred_instances = pred_instances[keep]
            
            # Get top-k predictions
            if len(pred_instances) > topk:
                indices = scores[keep].topk(topk)[1]
                pred_instances = pred_instances[indices]
    
    # Extract results
    bboxes = pred_instances.bboxes
    scores = pred_instances.scores
    labels = pred_instances.labels
    
    # Get class names (one-indexed in labels)
    class_names = [text_prompts[idx] for idx in labels.cpu().numpy()]
    
    # Create detection result object
    if len(bboxes) > 0:
        xyxy = bboxes.cpu().numpy()
        confidence = scores.cpu().numpy()
        class_ids = labels.cpu().numpy()
    else:
        # Empty results
        xyxy = []
        confidence = []
        class_ids = []
        class_names = []
    
    # Clean up temporary file if created
    if image_path is None and 'temp_path' in locals() and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"Removed temporary image file: {temp_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_path}: {e}")
    
    return DetectionResult(xyxy, confidence, class_ids, class_names)

###########################################
# VISUALIZATION FUNCTIONS
###########################################

def create_class_colors(class_names):
    """
    Create consistent colors for each class
    
    Args:
        class_names: List of class names
        
    Returns:
        Dictionary mapping class names to RGB colors
    """
    # Professional color palette
    colors_palette = [
        (0.96, 0.21, 0.32),  # Red
        (0.15, 0.68, 0.85),  # Blue
        (0.13, 0.78, 0.29),  # Green
        (0.94, 0.52, 0.14),  # Orange
        (0.68, 0.14, 0.67),  # Purple
        (0.85, 0.85, 0.15),  # Yellow
        (0.15, 0.68, 0.68),  # Teal
        (0.62, 0.31, 0.14),  # Brown
        (0.42, 0.42, 0.73),  # Periwinkle
        (1.00, 0.47, 0.66),  # Pink
    ]
    
    # Create a dictionary for class colors
    unique_classes = set(class_names)
    class_colors = {}
    
    for i, class_name in enumerate(unique_classes):
        color_idx = i % len(colors_palette)
        class_colors[class_name] = colors_palette[color_idx]
    
    return class_colors


def draw_detection_masks(image, masks, labels, alpha=0.5):
    """
    Draw masks on image with class labels
    
    Args:
        image: RGB image
        masks: List of binary masks
        labels: List of class labels
        alpha: Transparency of masks
        
    Returns:
        Image with masks
    """
    # Create a copy of the image
    result = image.copy()
    
    # Create colors for classes
    class_names = [label.split()[0] for label in labels]
    class_colors = create_class_colors(class_names)
    
    # Draw each mask
    for i, (mask, label) in enumerate(zip(masks, labels)):
        class_name = label.split()[0]
        color = class_colors[class_name]
        
        # Create RGB mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [int(c * 255) for c in color]
        
        # Blend mask with image
        cv2.addWeighted(colored_mask, alpha, result, 1 - alpha, 0, result)
        
        # Find contours for outline
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contour
        cv2.drawContours(result, contours, -1, [int(c * 255) for c in color], 2)
        
        # Add label text
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw text with a background
                text = label
                font_scale = 0.7
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(result, 
                            (cx - 5, cy - text_size[1] - 5), 
                            (cx + text_size[0] + 5, cy + 5), 
                            (255, 255, 255), 
                            -1)
                
                # Draw text
                cv2.putText(result, 
                        text, 
                        (cx, cy),
                        font, 
                        font_scale, 
                        [int(c * 255) for c in color], 
                        thickness)
    
    return result


###########################################
# SVG EXPORT FUNCTIONS
###########################################

class SVGExporter:
    """
    SVG exporter for segmentation results with color support
    """
    def __init__(self, line_thickness=1.5):
        """Initialize the SVG exporter"""
        self.line_thickness = line_thickness
    
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
            title = metadata.get('title', 'Segmentation with Hybrid SAM')
            author = metadata.get('author', SCRIPT_AUTHOR)
            date = metadata.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            version = metadata.get('version', SCRIPT_VERSION)
            
            desc = ET.SubElement(svg, 'desc')
            desc.text = f"{title}\nCreated by: {author}\nDate: {date}\nVersion: {version}"
        
        return svg
    
    def add_shape_to_svg(self, svg_root, element, color=None, segment_id=None):
        """Add a geometric element to the SVG document with color"""
        element_type = element['type'].lower()
        shape_elem = None
        
        # Convert color to CSS format if provided
        color_css = None
        if color:
            r, g, b = color
            color_css = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        
        # Create SVG element based on type
        if element_type == 'cercle':
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
                
        elif element_type == 'rectangle' or element_type == 'polygone' or element_type == 'contour':
            points = element['points']
            
            # Create path string
            path_data = f"M {points[0][0]} {points[0][1]}"
            for x, y in points[1:]:
                path_data += f" L {x} {y}"
            path_data += " Z"  # Close the path
            
            shape_elem = ET.SubElement(svg_root, 'path')
            shape_elem.set('d', path_data)
            
        elif element_type == 'lignes':
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
            shape_elem.set('fill', 'none')
            
            # Add ID if provided
            if segment_id is not None:
                shape_elem.set('id', f'segment_{segment_id}')
    
    def export_to_svg(self, segments_data, output_path, size, metadata=None):
        """Export segments to SVG with color support"""
        print(f"Exporting to SVG: {output_path}")
        
        try:
            # Create SVG document
            svg_root = self.create_svg_document(size, metadata)
            
            # Process segments
            for idx, segment in enumerate(segments_data):
                # Get segment color if available, otherwise use default
                color = segment.get('color', (0, 0, 1))  # Default to blue
                segment_id = segment.get('id', idx + 1)  # Use index+1 if no ID
                
                # Process each geometric element
                for element in segment['geometric_elements']:
                    self.add_shape_to_svg(svg_root, element, color, segment_id)
            
            # Format the XML for readability
            rough_string = ET.tostring(svg_root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(pretty_xml)
                
            print(f"SVG file exported successfully with {len(segments_data)} objects")
            return True
            
        except Exception as e:
            print(f"Error exporting to SVG: {str(e)}")
            traceback.print_exc()
            return False


def extract_contours_for_svg(mask, simplify_tolerance=0.1):
    """
    Extract contours from a binary mask for SVG export
    
    Args:
        mask: Binary mask as numpy array
        simplify_tolerance: Tolerance for contour simplification
        
    Returns:
        List of contours as points
    """
    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours
    processed_contours = []
    for contour in contours:
        # Skip tiny contours
        if cv2.contourArea(contour) < 20:  # Minimum area threshold
            continue
        
        # Simplify contour if requested
        if simplify_tolerance > 0:
            epsilon = simplify_tolerance * cv2.arcLength(contour, True) / 100
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to the format needed for SVG
        points = []
        for point in contour:
            x, y = point[0]
            points.append((float(x), float(y)))
        
        if len(points) >= 3:  # Need at least 3 points for a polygon
            processed_contours.append(points)
    
    return processed_contours


def create_segments_for_svg_export(masks, labels, image_shape):
    """
    Convert masks to a format suitable for SVG export
    
    Args:
        masks: List of binary masks
        labels: List of label strings (class name + confidence)
        image_shape: Shape of the original image (h, w)
        
    Returns:
        List of segment dictionaries for SVGExporter
    """
    segments_data = []
    
    # Professional color palette with distinct colors
    colors = [
        (0.9, 0.1, 0.1),   # Red
        (0.1, 0.6, 0.9),   # Blue
        (0.2, 0.7, 0.2),   # Green
        (0.8, 0.2, 0.8),   # Purple
        (1.0, 0.6, 0.0),   # Orange
        (0.0, 0.6, 0.6),   # Teal
        (0.8, 0.8, 0.2)    # Yellow
    ]
    
    # Track unique class names and assign colors
    class_colors = {}
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Improved parsing of labels with quotes
        if '"' in label:
            # Extract the quoted class name
            start_quote = label.find('"')
            end_quote = label.rfind('"')
            if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
                class_name = label[start_quote:end_quote+1]
                # Look for confidence after the quoted string
                remaining = label[end_quote+1:].strip()
                confidence = float(remaining) if remaining else 1.0
            else:
                # Fallback if quote parsing fails
                class_name = label.split()[0]
                confidence = 1.0
        else:
            # Standard parsing for non-quoted labels
            parts = label.split()
            class_name = parts[0]
            confidence = float(parts[1]) if len(parts) > 1 else 1.0
        
        # Assign color for this class
        if class_name not in class_colors:
            idx = len(class_colors) % len(colors)
            class_colors[class_name] = colors[idx]
        
        color = class_colors[class_name]
        
        # Extract contours
        contours = extract_contours_for_svg(mask)
        
        # Create geometric elements
        geometric_elements = []
        for contour_points in contours:
            if len(contour_points) >= 3:  # Need at least 3 points
                geometric_elements.append({
                    'type': 'contour',
                    'points': contour_points
                })
        
        # Calculate additional properties
        area = np.sum(mask)
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)
        else:
            bbox = [0, 0, 0, 0]
            center_x, center_y = 0, 0
        
        # Create segment data
        segment = {
            'id': i + 1,
            'class': class_name,
            'confidence': confidence,
            'color': color,
            'geometric_elements': geometric_elements,
            'area': float(area),
            'bbox': bbox,
            'center': (float(center_x), float(center_y)),
        }
        
        segments_data.append(segment)
    
    return segments_data


###########################################
# PNG MASK EXPORT FUNCTIONS
###########################################

def export_mask_png(mask, image_path, label, output_dir, class_colors=None):
    """
    Export a binary mask as a transparent PNG
    
    Args:
        mask: Binary mask
        image_path: Original image path for naming
        label: Label string for naming
        output_dir: Output directory
        class_colors: Optional dictionary of class colors
        
    Returns:
        Path to exported PNG file
    """
    # Ensure output directory exists
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    
    # Parse label to get class name
    if '"' in label:
        # Handle quoted class names
        start_quote = label.find('"')
        end_quote = label.rfind('"')
        if start_quote != -1 and end_quote != -1 and end_quote > start_quote:
            class_name = label[start_quote+1:end_quote]  # Remove quotes
        else:
            class_name = label.split()[0]
    else:
        class_name = label.split()[0]
    
    # Clean class name for filename
    class_name = class_name.replace(' ', '_').replace('"', '').lower()
    
    # Get base name of image
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create filename
    filename = f"{basename}_{class_name}.png"
    output_path = os.path.join(png_dir, filename)
    
    # Get color for class
    color = (255, 0, 0)  # Default red
    if class_colors and class_name in class_colors:
        r, g, b = class_colors[class_name]
        color = (int(r * 255), int(g * 255), int(b * 255))
    
    # Create RGBA image (with transparency)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Set RGB channels to the class color where mask is True
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    
    # Set alpha channel to 255 (fully opaque) where mask is True
    rgba[mask, 3] = 255
    
    # Save as PNG with transparency
    cv2.imwrite(output_path, rgba)
    
    return output_path


###########################################
# PROCESS PIPELINE AND INFERENCE FUNCTIONS
###########################################

def process_pipeline(image_path, texts, hybrid_sam, model, test_pipeline, **kwargs):
    """
    Process a single image with hybrid SAM approach
    
    Args:
        image_path: Path to image file
        texts: Text prompts
        hybrid_sam: HybridSAMProcessor instance
        model: YOLO-World model
        test_pipeline: Test pipeline for YOLO-World
        kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (refined_masks, labels, mask_scores)
    """
    # Start timing
    start_time = time.time()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return [], [], []
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get YOLO-World detections
    detections = process_yolo_world(
        model, 
        image, 
        test_pipeline, 
        texts,
        kwargs.get('topk', 100),
        kwargs.get('threshold', 0.03),  # Lower threshold to get more detections
        image_path=image_path  # Pass the original image path
        )
    
    # Set image for SAM predictor
    hybrid_sam.set_image(image)
    
    # Generate candidate masks across the whole image
    auto_masks = hybrid_sam.generate_candidate_masks()
    
    # Process each detection with hybrid approach
    refined_masks = []
    mask_scores = []
    final_labels = []
    boxes_for_vis = []  # Store boxes for visualization
    
    # Track objects by class
    class_counts = {}
    
    for i, box in enumerate(detections.xyxy):
        class_name = detections.class_names[i]
        confidence = detections.confidence[i]
        
        # For display, count objects by class
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
            
        label = f"{class_name} {confidence:.2f}"
        print(f"Processing object {i+1}/{len(detections.xyxy)}: {label}")
        
        # Get hybrid masks with box prompt
        pred_mask, pred_score, hybrid_mask, iou = hybrid_sam.predict_box_hybrid(
            box, 
            expand_box=kwargs.get('expand_box', 0.25)  # Increased expansion for better coverage
        )
        
        # Apply refinements
        refined_mask = hybrid_mask.copy()
        
        if kwargs.get('detailed_refinement', False):
            # Apply edge refinement if requested
            if kwargs.get('edge_refinement', False):
                print("  Applying edge_precise refinement...")
                refined_mask = apply_edge_refinement(refined_mask, image, edge_type='precise')
            
            # Apply smoothing if requested
            if kwargs.get('smooth_mask', False):
                kernel_size = kwargs.get('smooth_kernel', 3)
                refined_mask = smooth_mask(refined_mask, kernel_size)
                
        # Ensure mask is clean binary before CRF
        refined_mask = refined_mask.astype(bool)  # Convert to strict boolean
        print(f"  Mask shape: {refined_mask.shape}, type: {refined_mask.dtype}")
        print(f"  Mask unique values: {np.unique(refined_mask)}")

        # Apply CRF refinement if requested
        if kwargs.get('use_crf', False):
            # Apply CRF refinement to hybrid mask
            refined_mask = refine_mask_with_crf(image, refined_mask, **kwargs)
        
        # Add to results
        refined_masks.append(refined_mask)
        mask_scores.append(pred_score)
        final_labels.append(label)
        boxes_for_vis.append(box)  # Store box for visualization
        
        if iou > 0:
            print(f"  ✓ SAM mask generated (score: {pred_score:.3f}, auto-refine IoU: {iou:.3f})")
        else:
            print(f"  ✓ SAM mask generated (score: {pred_score:.3f})")
    
    print(f"SAM successfully generated detailed masks for {len(refined_masks)}/{len(detections.xyxy)} objects")
    
    # Prepare output directory
    output_dir = kwargs.get('output_dir', 'demo_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization directory with explicit error checking
    vis_dir = os.path.join(output_dir, 'visualizations')
    try:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Visualization directory: {vis_dir}")
    except Exception as e:
        print(f"Warning: Failed to create visualization directory: {e}")
    
    # Get basename for output files
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Export visualizations
    if refined_masks:
        try:
            # Create segmentation visualization
            segmentation_image = draw_detection_masks(image, refined_masks, final_labels)
            
            # Save segmentation visualization
            seg_path = os.path.join(vis_dir, f"{name_without_ext}_segmentation.jpg")
            cv2.imwrite(seg_path, cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR))
            print(f"Segmentation visualization saved to: {seg_path}")
            
            # Create detection visualization (boxes)
            detection_image = image.copy()
            colors = generate_colors(len(boxes_for_vis))
            
            for i, box in enumerate(boxes_for_vis):
                x1, y1, x2, y2 = map(int, box)
                label = final_labels[i]
                color = colors[i % len(colors)]
                
                # Draw box
                cv2.rectangle(detection_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                cv2.putText(detection_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save detection visualization
            det_path = os.path.join(vis_dir, f"{name_without_ext}_detection.jpg")
            cv2.imwrite(det_path, cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR))
            print(f"Detection visualization saved to: {det_path}")
            
            # Create combined visualization
            h, w = image.shape[:2]
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = detection_image
            combined[:, w:] = segmentation_image
            
            # Save combined visualization
            combined_path = os.path.join(vis_dir, f"{name_without_ext}_combined.jpg")
            cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Combined visualization saved to: {combined_path}")
        
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Export individual PNG masks if requested
        if kwargs.get('export_png', False):
            # Create colors for classes
            class_names = [label.split()[0] for label in final_labels]
            class_colors = create_class_colors(class_names)
            
            png_paths = []
            for i, (mask, label) in enumerate(zip(refined_masks, final_labels)):
                png_path = export_mask_png(mask, image_path, label, output_dir, class_colors)
                png_paths.append(png_path)
            
            print(f"Exported {len(png_paths)} PNG masks to {os.path.join(output_dir, 'png')}")
        
        # Export SVG if requested
        if kwargs.get('export_svg', False):
            svg_dir = os.path.join(output_dir, "svg")
            os.makedirs(svg_dir, exist_ok=True)
            svg_name = f"{name_without_ext}.svg"
            svg_path = os.path.join(svg_dir, svg_name)
            
            # Prepare segments data for SVG export
            segments_data = create_segments_for_svg_export(
                refined_masks,
                final_labels,
                image.shape[:2]
            )
            
            if segments_data:
                # Prepare metadata
                metadata = {
                    'title': f"YOLO-World + Hybrid SAM segmentation of {basename}",
                    'author': SCRIPT_AUTHOR,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'version': SCRIPT_VERSION
                }
                
                # Export using SVGExporter
                svg_exporter = SVGExporter(line_thickness=1.5)
                success = svg_exporter.export_to_svg(
                    segments_data,
                    svg_path,
                    image.shape[:2][::-1],  # Size as (width, height)
                    metadata=metadata
                )
                
                if success:
                    print(f"✅ SVG file exported to: {svg_path}")
                else:
                    print(f"❌ Failed to export SVG to: {svg_path}")
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processed {image_path} in {format_time(processing_time)}")
    
    return refined_masks, final_labels, mask_scores

# Helper function for colors
def generate_colors(n):
    """Generate n distinct colors"""
    colors = []
    for i in range(n):
        # Use HSV color space for better distinction
        h = i / n
        s = 0.8
        v = 0.9
        
        # Convert to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Scale to 0-255
        colors.append((int(r*255), int(g*255), int(b*255)))
    
    return colors

# Make sure you have this function or implement it
def draw_detection_masks(image, masks, labels):
    """Draw masks on the image with colors by class"""
    # Create a copy of the input image
    vis_img = image.copy()
    
    # Generate colors for each mask
    colors = generate_colors(len(masks))
    
    # Create mask overlay
    mask_overlay = np.zeros_like(image, dtype=np.float32)
    mask_count = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Apply each mask
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = np.array(colors[i % len(colors)]) / 255.0
        
        # Apply mask with color
        for c in range(3):
            mask_overlay[:,:,c] += mask.astype(np.float32) * color[c]
        
        # Count overlapping masks
        mask_count += mask.astype(np.float32)
    
    # Normalize mask overlay
    mask_count = np.maximum(mask_count, 1)
    mask_count = np.stack([mask_count] * 3, axis=-1)
    mask_overlay = mask_overlay / mask_count
    
    # Apply alpha blending
    alpha = 0.6
    vis_img = cv2.addWeighted(
        vis_img.astype(np.float32) / 255.0, 
        1 - alpha,
        mask_overlay, 
        alpha, 
        0
    ) * 255.0
    
    # Add labels
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Find mask centroid
        if np.any(mask):
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # Draw label
                color = colors[i % len(colors)]
                cv2.putText(vis_img, label, (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_img.astype(np.uint8)



def visualize_boxes(image, boxes, labels, scores):
    """Visualize bounding boxes on the image"""
    vis_img = image.copy()
    
    # Create a color map for different classes
    num_classes = max(labels) + 1 if len(labels) > 0 else 0
    colors = generate_colors(num_classes)
    
    for box, label, score in zip(boxes, labels, scores):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color for this class
        color = colors[label]
        
        # Draw box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # Create label text with score
        if isinstance(label, np.ndarray):
            label = label.item()
        label_txt = f"{label} {score:.2f}"
        
        # Put label text
        cv2.putText(vis_img, label_txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_img

def visualize_masks(image, masks, labels):
    """Visualize segmentation masks on the image"""
    vis_img = image.copy()
    
    # Create a color map for different classes
    num_classes = max(labels) + 1 if len(labels) > 0 else 0
    colors = generate_colors(num_classes)
    
    # Draw masks first (so they are behind annotations)
    mask_overlay = np.zeros_like(image)
    mask_count = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for mask, label in zip(masks, labels):
        # Get color for this class
        color = np.array(colors[label])
        
        # Create RGB mask
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[mask] = color
        
        # Add to overlay with alpha blending
        mask_overlay[mask] += rgb_mask[mask]
        mask_count[mask] += 1
    
    # Average colors where masks overlap
    mask_count = np.maximum(mask_count, 1)
    mask_count = np.stack([mask_count] * 3, axis=-1)
    mask_overlay = mask_overlay / mask_count
    
    # Blend with original image
    alpha = 0.5
    vis_img = cv2.addWeighted(vis_img, 1 - alpha, mask_overlay.astype(np.uint8), alpha, 0)
    
    return vis_img

def visualize_boxes_masks(image, boxes, masks, labels, scores):
    """Visualize both bounding boxes and masks on the image"""
    # First add the masks
    vis_img = visualize_masks(image.copy(), masks, labels)
    
    # Then add the boxes
    vis_img = visualize_boxes(vis_img, boxes, labels, scores)
    
    return vis_img


def inference_detector(model, image_path, texts, test_pipeline, hybrid_sam, **kwargs):
    """
    Run inference pipeline for detector and generate output visualizations
    """
    # Handle directories vs individual files
    if os.path.isdir(image_path):
        # Process all images in directory
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_files.extend(glob.glob(os.path.join(image_path, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(image_path, f'*.{ext.upper()}')))
        
        print(f"Found {len(image_files)} images in {image_path}")
        
        # Process each image
        stats = {
            'total_time': 0,
            'image_count': len(image_files),
            'object_count': 0,
            'class_counts': {}
        }
        
        # Create progress bar
        progress_bar = ProgressBar(len(image_files))
        
        for img_file in image_files:
            # Process image
            start_time = time.time()
            masks, labels, _ = process_pipeline(
                img_file, texts, hybrid_sam, model, test_pipeline, **kwargs
            )
            end_time = time.time()
            
            # Update stats
            stats['total_time'] += (end_time - start_time)
            stats['object_count'] += len(masks)
            
            # Update class counts
            for label in labels:
                class_name = label.split()[0]
                if class_name in stats['class_counts']:
                    stats['class_counts'][class_name] += 1
                else:
                    stats['class_counts'][class_name] = 1
            
            # Update progress bar
            progress_bar.update()
        
        # Print summary
        print("\n===== PROCESSING SUMMARY =====")
        print(f"✅ Processed {stats['image_count']} images in {format_time(stats['total_time'])}")
        print(f"Average time per image: {format_time(stats['total_time'] / max(1, stats['image_count']))}")
        print(f"Total objects detected: {stats['object_count']}")
        
        print("\nObjects detected by class:")
        for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {class_name}: {count}")
        
        # Save summary to file
        output_dir = kwargs.get('output_dir', 'demo_outputs')
        summary_path = os.path.join(output_dir, 'processing_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write(f"Processing Summary\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Version: {SCRIPT_VERSION}\n\n")
            
            f.write(f"Images processed: {stats['image_count']}\n")
            f.write(f"Total processing time: {format_time(stats['total_time'])}\n")
            f.write(f"Average time per image: {format_time(stats['total_time'] / max(1, stats['image_count']))}\n")
            f.write(f"Total objects detected: {stats['object_count']}\n\n")
            
            f.write("Objects detected by class:\n")
            for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {class_name}: {count}\n")
        
        print(f"\nSummary saved to: {summary_path}")
        
        # Return stats
        return stats
    else:
        # Process single image
        masks, labels, scores = process_pipeline(
            image_path, texts, hybrid_sam, model, test_pipeline, **kwargs
        )
        
        # Return simple stats
        return {
            'masks': masks,
            'labels': labels,
            'scores': scores,
        }


###########################################
# ARGUMENT PARSING AND MAIN FUNCTION
###########################################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Professional Segmentation Suite with YOLO-World, Hybrid SAM, and CRF')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument('text',help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.')
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
    parser.add_argument('--multiscale-sam',
                        action='store_true',
                        help='enable multi-scale SAM inference for more detailed masks')
    parser.add_argument('--export-png',
                        action='store_true',
                        help='export individual masks as transparent PNG files')
    parser.add_argument('--export-svg', 
                        action='store_true',  
                        help='export segmentation masks as SVG files')

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
                        help='apply additional smoothing to masks before conversion')
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
                        default=0.15,
                        help='expand detection boxes by this fraction to capture more context')
    
    # Enhanced SAM options
    parser.add_argument('--sam-points',
                        type=int,
                        default=32,
                        help='points per side for automatic mask generator')
    parser.add_argument('--sam-stability-score-thresh',
                        type=float,
                        default=0.92, 
                        help='stability score threshold for SAM masks')
    parser.add_argument('--sam-pred-iou-thresh',
                        type=float,
                        default=0.88,
                        help='predicted IoU threshold for SAM')
    parser.add_argument('--sam-min-mask-region-area',
                        type=int,
                        default=100,
                        help='minimum mask region area')
    
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

def main():
    # Parse arguments
    args = parse_args()
    
    # Print script header
    print(f"Professional Segmentation Suite v{SCRIPT_VERSION}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: {SCRIPT_AUTHOR}")
    print("-" * 60)
    
    # Check for CRF support
    if args.use_crf and not HAS_CRF:
        print("Warning: CRF refinement was requested but pydensecrf is not installed.")
        print("CRF refinement will be disabled.")
        args.use_crf = False
    
    # Print GPU information
    print_gpu_info()
    
    # Handle GPU selection
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"
    elif args.device == 'cuda:0' and torch.cuda.is_available():
        args.device = auto_select_gpu()
    
    # Log system info
    log_system_info()
    
    # Load YOLO-World model
    print(f"Loading YOLO-World model on {args.device}...")
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
    # Build test pipeline
    test_pipeline = cfg.get('test_pipeline', {})
    if test_pipeline:
        test_pipeline = mmcv.Compose(test_pipeline)
    
    # Process text prompts
    texts = load_and_process_text(args.text)
    
    # Initialize SAM model
    print(f"Initializing SAM model ({args.sam_model}) on {args.device}...")
    
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    print(f"SAM model device: {next(sam.parameters()).device}")
    
    # Initialize hybrid SAM processor
    hybrid_sam = HybridSAMProcessor(
        sam, 
        args.device,
        points_per_side=args.sam_points,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        min_mask_region_area=args.sam_min_mask_region_area
    )
    
    # Run inference
    result = inference_detector(model,
                               args.image,
                               texts,
                               test_pipeline,
                               hybrid_sam,
                               threshold=args.threshold,
                               topk=args.topk,
                               output_dir=args.output_dir,
                               use_crf=args.use_crf,
                               crf_iter=args.crf_iter,
                               crf_pos_w=args.crf_pos_w,
                               crf_bi_w=args.crf_bi_w,
                               crf_bi_xy=args.crf_bi_xy,
                               crf_bi_rgb=args.crf_bi_rgb,
                               smooth_mask=args.smooth_mask,
                               smooth_kernel=args.smooth_kernel,
                               detailed_refinement=args.detailed_refinement,
                               edge_refinement=args.edge_refinement,
                               expand_box=args.expand_box,
                               export_png=args.export_png,
                               export_svg=args.export_svg,
                               multiscale_sam=args.multiscale_sam)
    
    # Show GPU stats if requested
    if args.gpu_stats and torch.cuda.is_available():
        print("\nGPU Memory Statistics:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    print("\n✅ Processing complete!")
    
    # Return result if needed by caller
    return result


if __name__ == '__main__':
    main()