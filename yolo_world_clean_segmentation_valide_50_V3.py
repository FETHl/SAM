#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YOLO-World + SAM Clean Segmentation
# Author: FETHl
# Date: 2025-03-27 09:02:29

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import colorsys
import traceback
import glob
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from concurrent.futures import ProcessPoolExecutor
import functools

# For CRF post-processing
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
    from pydensecrf.utils import create_pairwise_gaussian
    HAS_CRF = True
except ImportError:
    print("Warning: pydensecrf not available. Install with: pip install pydensecrf")
    HAS_CRF = False

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Logger:
    """Simple logging utility with colored output"""
    INFO = '\033[94m'  # Blue
    SUCCESS = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    ERROR = '\033[91m'  # Red
    ENDC = '\033[0m'  # Reset
    
    @staticmethod
    def info(msg):
        print(f"{Logger.INFO}[INFO] {msg}{Logger.ENDC}")
    
    @staticmethod
    def success(msg):
        print(f"{Logger.SUCCESS}[SUCCESS] {msg}{Logger.ENDC}")
    
    @staticmethod
    def warning(msg):
        print(f"{Logger.WARNING}[WARNING] {msg}{Logger.ENDC}")
    
    @staticmethod
    def error(msg):
        print(f"{Logger.ERROR}[ERROR] {msg}{Logger.ENDC}")


class CRFPostProcessor:
    """CRF-based mask refinement with custom unary potential calculation"""
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def apply_crf(self, image, mask, params=None):
        """Apply CRF to refine the mask with custom unary potentials"""
        if not HAS_CRF:
            Logger.warning("CRF package not available. Skipping refinement.")
            return mask
        
        # Default parameters
        default_params = {
            'gaussian_sxy': 3,
            'gaussian_compat': 3,
            'bilateral_sxy': 80,
            'bilateral_srgb': 13,
            'bilateral_compat': 10,
            'iterations': 5
        }
        
        # Use provided parameters or defaults
        if params is None:
            params = default_params
        else:
            # Fill missing with defaults
            for key in default_params:
                if key not in params:
                    params[key] = default_params[key]
        
        # Ensure 3-channel RGB image
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # CRITICAL: Create fresh binary mask with guaranteed 0/1 values as boolean
        h, w = mask.shape[:2]
        binary_mask = (mask > 0).astype(bool)
        
        # Debug info
        if self.debug:
            Logger.info(f"Binary mask shape: {binary_mask.shape}, dtype: {binary_mask.dtype}, unique values: {np.unique(binary_mask)}")
        
        try:
            # Create the CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
            
            # CUSTOM: Create unary potentials directly without using unary_from_labels
            # This is a safer approach than using the built-in unary_from_labels
            foreground_prob = 0.9  # High probability for foreground
            background_prob = 0.9  # High probability for background
            
            # Initialize unary potentials - negative log probabilities
            U = np.zeros((2, h*w), dtype=np.float32)
            
            # Flatten the mask for processing
            flat_mask = binary_mask.flatten()
            
            # Set foreground probabilities (class 1)
            U[1, flat_mask] = -np.log(foreground_prob)  # High probability where mask=True
            U[1, ~flat_mask] = -np.log(1-foreground_prob)  # Low probability where mask=False
            
            # Set background probabilities (class 0)
            U[0, flat_mask] = -np.log(1-background_prob)  # Low probability where mask=True
            U[0, ~flat_mask] = -np.log(background_prob)  # High probability where mask=False
            
            # Set the unary energies
            d.setUnaryEnergy(U)
            
            # Add pairwise potentials
            d.addPairwiseGaussian(
                sxy=params['gaussian_sxy'],
                compat=params['gaussian_compat']
            )
            
            d.addPairwiseBilateral(
                sxy=params['bilateral_sxy'],
                srgb=params['bilateral_srgb'],
                rgbim=image,
                compat=params['bilateral_compat']
            )
            
            # Perform inference
            q = d.inference(params['iterations'])
            
            # Create refined mask - class 1 probability at each pixel
            result = np.array(q).reshape((2, h, w)).transpose(1, 2, 0)
            refined_mask = result[:, :, 1]  # Take foreground channel
            refined_mask = (refined_mask > 0.5).astype(np.uint8)
            
            return refined_mask
        except Exception as e:
            Logger.error(f"Error in CRF processing: {e}")
            traceback.print_exc()
            # Return the original binary mask as fallback
            return binary_mask.astype(np.uint8)


class SVGExporter:
    """SVG exporter for segmentation results with color support"""
    
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
            title = metadata.get('title', 'Segmentation with SAM')
            author = metadata.get('author', 'FETHl')
            date = metadata.get('date', '2025-03-27 09:02:29')
            version = metadata.get('version', '5.0.1')
            
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


class CleanSegmentationProcessor:

    """Main processor for clean segmentation with YOLO-World and SAM"""
    
    def __init__(self, sam_predictor=None, min_contour_area=20, debug=False):
        self.debug = debug
        self.min_contour_area = min_contour_area
        
        # Initialize helpers
        self.crf_processor = CRFPostProcessor(debug=debug)
        
        # Store SAM predictor directly
        self.sam_predictor = sam_predictor
        self.sam_available = self.sam_predictor is not None
        
        if self.sam_available:
            Logger.info("SAM predictor is available and will be used for segmentation")
        else:
            Logger.warning("SAM predictor is not available - only bounding boxes will be drawn")
    
    def load_image(self, filepath):
        """Load image with basic preprocessing"""
        Logger.info(f"Loading image: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load image
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image: {filepath}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    
    def preprocess_image(self, image):
        """Basic image preprocessing for better segmentation"""
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply CLAHE on value channel for better contrast
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hsv[:,:,2] = clahe.apply(img_hsv[:,:,2])
        enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Light denoising that preserves edges
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 7, 7, 7, 21)
        
        return denoised
    
    def expand_box(self, box, expand_ratio, image_shape):
        """Expand box with boundary check"""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box
        
        # Calculate expansion
        width = x2 - x1
        height = y2 - y1
        dx = width * expand_ratio
        dy = height * expand_ratio
        
        # Expand box
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w - 1, x2 + dx)
        y2 = min(h - 1, y2 + dy)
        
        return [x1, y1, x2, y2]
    
    def predict_mask_from_box(self, box):
        """Get mask prediction from a bounding box using SAM"""
        if not self.sam_available or self.sam_predictor is None:
            return None, 0
        
        try:
            # Format box for SAM
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box),
                multimask_output=True
            )
            
            # Use best mask
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                # CRITICAL: Ensure binary mask using boolean conversion
                mask = masks[best_idx].astype(bool).astype(np.uint8)
                score = scores[best_idx]
                return mask, score
            else:
                return None, 0
        except Exception as e:
            Logger.error(f"Error in SAM prediction: {e}")
            traceback.print_exc()
            return None, 0
    
    def _apply_sam_segmentation(self, image_color):
        """Apply SAM segmentation with optimized parameters"""
        segments = []
        method = "SAM"
        
        # Check if we have SAM helper available
        if not hasattr(self, 'sam_predictor') or self.sam_predictor is None:
            Logger.warning("SAM predictor not available, falling back to traditional methods")
            # Fall back to basic contour detection
            gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Skip contours that are too small
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue
                
                # Simplify the contour
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                
                # Create segment
                segment = {
                    'id': i + 1,
                    'source': 'threshold',
                    'geometric_elements': [
                        {
                            'type': 'contour',
                            'points': points
                        }
                    ]
                }
                
                # Assign a random color
                r, g, b = np.random.random(3)
                segment['color'] = (r, g, b)
                
                segments.append(segment)
                
            method = "threshold"
            return segments, method
            
        try:
            # Apply SAM directly if available
            Logger.info("Applying SAM segmentation...")
            
            # Use SAM's automatic mask generation
            if hasattr(self.sam_predictor, 'generate_masks'):
                masks = self.sam_predictor.generate_masks(image_color)
            else:
                # Fall back to using SAM on the whole image
                h, w = image_color.shape[:2]
                box = [0, 0, w, h]
                self.sam_predictor.set_image(image_color)
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(box),
                    multimask_output=True
                )
            
            if not masks:
                Logger.warning("SAM did not generate any masks, falling back to traditional methods")
                gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process contours (same as in fallback)
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) < self.min_contour_area:
                        continue
                    
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = approx.reshape(-1, 2).tolist()
                    
                    segment = {
                        'id': i + 1,
                        'source': 'threshold',
                        'geometric_elements': [
                            {
                                'type': 'contour',
                                'points': points
                            }
                        ]
                    }
                    
                    r, g, b = np.random.random(3)
                    segment['color'] = (r, g, b)
                    
                    segments.append(segment)
                
                method = "threshold"
                return segments, method
                
            Logger.info(f"SAM found {len(masks)} potential segments")
            
            # Convert SAM masks to our segment format
            for i, mask in enumerate(masks):
                # Skip masks that are too small
                if np.sum(mask) < self.min_contour_area:
                    continue
                    
                # Find contours in the mask
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    continue
                    
                # Use largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Skip contours that are too small
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue
                    
                # Simplify the contour (Douglas-Peucker algorithm)
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                
                # Create a new segment
                segment = {
                    'id': i + 1,
                    'source': 'sam',
                    'geometric_elements': [
                        {
                            'type': 'contour',
                            'points': points
                        }
                    ]
                }
                
                # Assign a random color
                r, g, b = np.random.random(3)
                segment['color'] = (r, g, b)
                
                segments.append(segment)
                
            Logger.info(f"Converted {len(segments)} valid segments from SAM masks")
            
        except Exception as e:
            Logger.error(f"Error in SAM segmentation: {e}")
            if self.debug:
                traceback.print_exc()
                
            Logger.warning("Falling back to traditional segmentation methods")
            gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours (same as before)
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue
                
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
                
                segment = {
                    'id': i + 1,
                    'source': 'threshold',
                    'geometric_elements': [
                        {
                            'type': 'contour',
                            'points': points
                        }
                    ]
                }
                
                r, g, b = np.random.random(3)
                segment['color'] = (r, g, b)
                
                segments.append(segment)
            
            method = "threshold"
            
        return segments, method
    
    def extract_contours_from_mask(self, mask, min_contour_area=10):
        """Extract all internal and external contours from a binary mask"""
        # Ensure binary mask
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find all contours, both external and internal
        # RETR_CCOMP retrieves all the contours and organizes them into a two-level hierarchy
        # where the top level are external boundaries, and the second level are holes
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_CCOMP,  # Use CCOMP to get hierarchy of contours
            cv2.CHAIN_APPROX_TC89_KCOS  # Better approximation for natural shapes
        )
        
        # Filter contours by area and process them
        filtered_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                # Apply light smoothing
                epsilon = 0.0005 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                
                # Add to filtered contours
                filtered_contours.append({
                    'contour': smoothed,
                    'area': area,
                    'hierarchy': hierarchy[0][i] if hierarchy is not None else None
                })
        
        return filtered_contours
    
    def extract_natural_contours(self, image, mask, box=None):
        """Extract natural contours using Canny edge detection combined with SAM mask"""
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # If box is provided, create a cropped region for more focused processing
        if box is not None:
            x1, y1, x2, y2 = map(int, box)
            # Add padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1] - 1, x2 + pad)
            y2 = min(image.shape[0] - 1, y2 + pad)
            
            # Create ROI mask - only process within the box
            roi_mask = np.zeros_like(binary_mask)
            roi_mask[y1:y2, x1:x2] = 1
            binary_mask = binary_mask * roi_mask
        
        # Apply Canny edge detection to the image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adaptive parameters based on image statistics
        sigma = 0.33
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, lower, upper)
        
        # Combine edges with the SAM mask - only keep edges within the mask
        edge_mask = edges * binary_mask
        
        # Find contours in the edge mask
        contours, _ = cv2.findContours(
            edge_mask,
            cv2.RETR_LIST,  # Retrieve all contours without hierarchy
            cv2.CHAIN_APPROX_TC89_KCOS  # Better approximation for natural shapes
        )
        
        # If no edge contours found, fall back to the SAM mask contours
        if not contours:
            return self.extract_contours_from_mask(binary_mask)
        
        # Process edge contours
        processed_contours = []
        for contour in contours:
            # Check minimum length and area
            if len(contour) >= 5 and cv2.contourArea(contour) >= 20:
                # Light smoothing
                epsilon = 0.0005 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                
                # Add to processed contours
                processed_contours.append({
                    'contour': smoothed,
                    'area': cv2.contourArea(smoothed),
                    'is_edge': True
                })
        
        # Get SAM contours as well (for areas that might not have strong edges)
        sam_contours = self.extract_contours_from_mask(binary_mask)
        
        # Combine both sets of contours
        all_contours = processed_contours + sam_contours
        
        # Sort by area (largest first)
        all_contours.sort(key=lambda x: x['area'], reverse=True)
        
        return all_contours
    
    def convert_contours_to_geometric_elements(self, contours):
        """Convert OpenCV contours to geometric elements for SVG export"""
        geometric_elements = []
        
        for contour_info in contours:
            contour = contour_info['contour']
            
            # Convert to points list
            points = contour.reshape(-1, 2).tolist()
            
            # Create geometric element
            element = {
                'type': 'contour',
                'points': points
            }
            
            geometric_elements.append(element)
        
        return geometric_elements
    
    def process_image(self, image_path, model, test_pipeline, prompts, 
                      threshold=0.05, expand_ratio=0.15, use_crf=False, 
                      output_dir='outputs', save_masks=False):
        """Process image with YOLO-World and SAM for clean segmentation"""
        Logger.info(f"Processing image: {image_path}")
        
        # Get basename for output paths
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Load and preprocess image
        original_image = self.load_image(image_path)
        processed_image = self.preprocess_image(original_image)
        
        # Run YOLO-World detection
        Logger.info("Running YOLO-World detection...")
        
        try:
            # Prepare input data
            data_info = dict(img_id=0, img_path=image_path, texts=prompts)
            data_info = test_pipeline(data_info)
            data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                             data_samples=[data_info['data_samples']])
            
            # Run model
            with torch.no_grad():
                output = model.test_step(data_batch)[0]
                pred_instances = output.pred_instances
                pred_instances = pred_instances[pred_instances.scores.float() > threshold]
        except Exception as e:
            Logger.error(f"Error in YOLO-World inference: {e}")
            traceback.print_exc()
            return None
        
        # Get detection results
        boxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        
        # Check if we found any objects
        if len(boxes) == 0:
            Logger.warning("No objects detected")
            return None
        
        # Create text labels
        text_labels = []
        for label_id, score in zip(labels, scores):
            class_name = prompts[label_id][0]
            text_labels.append(f"{class_name} {score:.2f}")
        
        Logger.success(f"Found {len(boxes)} objects")
        
        # Generate colors for visualization
        colors = []
        for i in range(len(boxes)):
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good color distribution
            color = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
            colors.append(color)
        
        # Visualization image
        vis_image = original_image.copy()
        
        # Prepare data for SVG export
        svg_segments = []
        
        # Process each detection with SAM
        all_masks = []
        all_contours = []
        all_labels = []
        
        if self.sam_available and self.sam_predictor is not None:
            Logger.info("Running SAM segmentation for each detection...")
            
            # Set image for SAM predictor
            self.sam_predictor.set_image(processed_image)
            
            # Process each box
            for i, box in enumerate(boxes):
                Logger.info(f"Processing {text_labels[i]}...")
                
                # Expand box for better context
                expanded_box = self.expand_box(box, expand_ratio, processed_image.shape)
                
                # Get mask from SAM
                mask, score_sam = self.predict_mask_from_box(expanded_box)
                
                if mask is not None:
                    # CRITICAL: Explicitly create a binary mask with only 0 and 1 values
                    binary_mask = np.zeros_like(mask, dtype=np.uint8)
                    binary_mask[mask > 0] = 1
                    
                    # Apply CRF refinement if requested
                    if use_crf and HAS_CRF:
                        Logger.info(f"Applying CRF refinement...")
                        try:
                            binary_mask = self.crf_processor.apply_crf(processed_image, binary_mask)
                        except Exception as e:
                            Logger.error(f"CRF refinement failed: {e}")
                    
                    # Extract all natural contours from the mask
                    natural_contours = self.extract_natural_contours(
                        processed_image, binary_mask, expanded_box
                    )
                    
                    if natural_contours:
                        # Use all contours for visualization
                        all_contours_combined = [c['contour'] for c in natural_contours]
                        
                        # Convert to geometric elements for SVG export
                        geometric_elements = self.convert_contours_to_geometric_elements(natural_contours)
                        
                        # Create SVG segment
                        svg_segment = {
                            'id': i + 1,
                            'source': 'sam',
                            'color': colors[i],
                            'geometric_elements': geometric_elements
                        }
                        svg_segments.append(svg_segment)
                        
                        # Save mask and contours
                        all_masks.append(binary_mask)
                        all_contours.append(all_contours_combined)
                        all_labels.append(text_labels[i])
                        
 
                        # Draw on visualization
                        color = colors[i]
                        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                        
                        # Draw mask overlay with transparency
                        mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
                        mask_overlay[binary_mask > 0] = [int(c*255) for c in color]
                        alpha = 0.4
                        vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, alpha, 0)
                        
                        # Draw all contours with different styles
                        main_contours = []
                        detail_contours = []
                        
                        # Categorize contours by area
                        for c_info in natural_contours:
                            if c_info['area'] > 200:  # Larger contours as main outlines
                                main_contours.append(c_info['contour'])
                            else:  # Smaller contours as details
                                detail_contours.append(c_info['contour'])
                        
                        # Draw main contours with thicker lines
                        cv2.drawContours(vis_image, main_contours, -1, color_bgr, 2)
                        
                        # Draw detail contours with thinner lines
                        detail_color = (int(color[2]*220), int(color[1]*220), int(color[0]*220))
                        cv2.drawContours(vis_image, detail_contours, -1, detail_color, 1)
                        
                        # Add label with original box coordinates for consistency
                        x1, y1, x2, y2 = map(int, box)
                        label_bg = np.zeros((30, len(text_labels[i])*11 + 10, 3), dtype=np.uint8)
                        label_bg[:] = (*color_bgr, )
                        
                        # Create label with background
                        label_y = max(y1 - 20, 20)  # Keep label visible
                        blend_region = vis_image[label_y-15:label_y+15, x1:x1+len(text_labels[i])*11+10]
                        
                        # Check if blend region is valid
                        if blend_region.shape[0] > 0 and blend_region.shape[1] > 0:
                            # Create semi-transparent label background
                            label_bg_resized = cv2.resize(label_bg, (blend_region.shape[1], blend_region.shape[0]))
                            vis_image[label_y-15:label_y+15, x1:x1+blend_region.shape[1]] = \
                                cv2.addWeighted(blend_region, 0.5, label_bg_resized, 0.5, 0)
                            
                            # Add text
                            cv2.putText(vis_image, text_labels[i], (x1+5, label_y+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        else:
                            # Simple label fallback
                            cv2.putText(vis_image, text_labels[i], (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                        
                        # Generate a unique timestamp for debug information
                        timestamp = f"UTC: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        
                        # Add debug data to SVG metadata as needed
                        debug_info = {
                            'object': text_labels[i],
                            'contours': len(natural_contours),
                            'main_contours': len(main_contours),
                            'detail_contours': len(detail_contours),
                            'box': f"[{x1},{y1},{x2},{y2}]",
                            'timestamp': timestamp,
                            'user': 'FETHl'
                        }
                        
                        # Add debug info to segment for SVG export
                        svg_segment = {
                            'id': i + 1,
                            'source': 'sam+natural',
                            'color': colors[i],
                            'label': text_labels[i],
                            'debug': debug_info,
                            'geometric_elements': self.convert_contours_to_geometric_elements(natural_contours)
                        }
                        svg_segments.append(svg_segment)
                    else:
                        Logger.warning(f"Could not extract natural contours for {text_labels[i]}")
                        
                        # Just draw the bounding box if no contour
                        x1, y1, x2, y2 = map(int, box)
                        color = colors[i]
                        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                        cv2.putText(vis_image, text_labels[i], (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                        
                        # Create a fallback mask from the box
                        fallback_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                        cv2.rectangle(fallback_mask, (x1, y1), (x2, y2), 1, -1)  # -1 means filled
                        all_masks.append(fallback_mask)
                        
                        # Create a simple rectangular contour
                        rect_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
                        all_contours.append([rect_contour])
                        all_labels.append(text_labels[i])
                        
                        # Add fallback segment for SVG
                        svg_segment = {
                            'id': i + 1,
                            'source': 'fallback',
                            'color': colors[i],
                            'label': text_labels[i],
                            'geometric_elements': [{
                                'type': 'contour',
                                'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            }]
                        }
                        svg_segments.append(svg_segment)
                else:
                    Logger.warning(f"SAM did not generate a mask for {text_labels[i]}")
                    
                    # Draw bounding box if no mask
                    color = colors[i]
                    color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                    cv2.putText(vis_image, text_labels[i], (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                    
                    # For fallback mode, create simple rectangular masks for consistency
                    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)  # -1 means filled
                    all_masks.append(mask)
                    
                    # Create a simple rectangular contour
                    rect_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
                    all_contours.append([rect_contour])
                    all_labels.append(text_labels[i])
                    
                    # Add fallback segment for SVG
                    svg_segment = {
                        'id': i + 1,
                        'source': 'fallback',
                        'color': colors[i],
                        'label': text_labels[i],
                        'geometric_elements': [{
                            'type': 'contour',
                            'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        }]
                    }
                    svg_segments.append(svg_segment)
        else:
            # Fallback to simple visualization without SAM
            Logger.warning("SAM not available. Using simple box visualization.")
            
            for i, box in enumerate(boxes):
                color = colors[i]
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                
                # Draw box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                
                # Add label
                cv2.putText(vis_image, text_labels[i], (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                
                # Create a simple rectangular mask for consistency
                mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)  # -1 means filled
                all_masks.append(mask)
                
                # Create a simple rectangular contour
                rect_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
                all_contours.append([rect_contour])
                all_labels.append(text_labels[i])
                
                # Add fallback segment for SVG
                svg_segment = {
                    'id': i + 1,
                    'source': 'fallback',
                    'color': colors[i],
                    'label': text_labels[i],
                    'geometric_elements': [{
                        'type': 'contour',
                        'points': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    }]
                }
                svg_segments.append(svg_segment)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{name_without_ext}_viz_{timestamp}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        Logger.success(f"Visualization saved to: {vis_path}")
        
        # Export SVG if we have segments
        svg_path = None
        if svg_segments:
            svg_dir = os.path.join(output_dir, "svg")
            os.makedirs(svg_dir, exist_ok=True)
            svg_path = os.path.join(svg_dir, f"{name_without_ext}_{timestamp}.svg")
            
            # Use the SVGExporter
            svg_exporter = SVGExporter(line_thickness=1.5)
            metadata = {
                'title': f"YOLO-World + SAM Segmentation for {basename}",
                'author': 'FETHl',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '5.0.1'
            }
            svg_exporter.export_to_svg(svg_segments, svg_path, size=original_image.shape[:2], metadata=metadata)
            Logger.success(f"SVG exported to: {svg_path}")
        
        # Save individual mask images if requested
        if save_masks and all_masks:
            mask_dir = os.path.join(output_dir, "masks", name_without_ext)
            os.makedirs(mask_dir, exist_ok=True)
            
            for i, mask in enumerate(all_masks):
                mask_path = os.path.join(mask_dir, f"{i:02d}_{all_labels[i].split()[0]}_{timestamp}.png")
                cv2.imwrite(mask_path, mask * 255)  # Convert binary mask to 0-255 range
            
            Logger.success(f"Individual masks saved to: {mask_dir}")
        
        # Create enhanced composite visualization with all contours
        composite_path = None
        if all_masks:
            # Create colorful overlay of all masks
            overlay = np.zeros_like(original_image)
            for i, mask in enumerate(all_masks):
                color = colors[i]
                color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                
                # Add this mask to overlay with its color
                for c in range(3):  # RGB channels
                    overlay[:,:,c] = np.where(mask > 0, 
                                            overlay[:,:,c] * 0.5 + color_rgb[c] * 0.5,  # Blend if overlapping
                                            overlay[:,:,c])
            
            # Blend with original image
            composite = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
            
            # Add contours on top with varying thickness for different importance
            for i, contours_list in enumerate(all_contours):
                color = colors[i]
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                
                # Categorize contours by size if possible
                if len(contours_list) > 1:
                    # Sort contours by area
                    contours_with_area = [(c, cv2.contourArea(c)) for c in contours_list]
                    contours_with_area.sort(key=lambda x: x[1], reverse=True)
                    
                    # Draw main contours (largest) with thicker lines
                    for contour, area in contours_with_area[:3]:  # Top 3 largest
                        cv2.drawContours(composite, [contour], -1, color_bgr, 2)
                    
                    # Draw remaining contours with thinner lines
                    for contour, area in contours_with_area[3:]:
                        cv2.drawContours(composite, [contour], -1, color_bgr, 1)
                else:
                    # Just draw all contours
                    for contour in contours_list:
                        cv2.drawContours(composite, [contour], -1, color_bgr, 2)
            
            # Add labels for each object
            for i, label in enumerate(all_labels):
                # Use box coordinates for label placement
                x1, y1, _, _ = map(int, boxes[i])
                
                # Create better looking labels with background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_w, text_h = text_size
                
                # Draw text background
                cv2.rectangle(composite, (x1, y1-text_h-10), (x1+text_w+10, y1), colors[i], -1)
                cv2.putText(composite, label, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Add timestamp and metadata to image
            info_text = f"Date: {timestamp} | User: FETHl | Objects: {len(all_labels)}"
            cv2.putText(composite, info_text, (10, composite.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)  # Black outline
            cv2.putText(composite, info_text, (10, composite.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)  # White text
            
            # Save composite image
            composite_path = os.path.join(output_dir, f"{name_without_ext}_composite_{timestamp}.jpg")
            cv2.imwrite(composite_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            Logger.success(f"Enhanced composite visualization saved to: {composite_path}")
            
            # Create a contour-only visualization
            contour_vis = np.zeros_like(original_image)
            contour_vis[:] = (255, 255, 255)  # White background
            
            # Draw all contours
            for i, contours_list in enumerate(all_contours):
                color = colors[i]
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                
                # Draw contours with varying thickness
                for contour in contours_list:
                    cv2.drawContours(contour_vis, [contour], -1, color_bgr, 1)
            
            # Save contour-only visualization
            contour_path = os.path.join(output_dir, f"{name_without_ext}_contours_{timestamp}.jpg")
            cv2.imwrite(contour_path, cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))
            Logger.success(f"Contour-only visualization saved to: {contour_path}")
        
        # Return results
        return {
            'masks': all_masks,
            'contours': all_contours,
            'labels': all_labels,
            'boxes': boxes,
            'visualization_path': vis_path,
            'svg_path': svg_path,
            'composite_path': composite_path,
            'contour_path': contour_path if all_masks else None,
            'timestamp': timestamp,
            'user': 'FETHl'
        }
    



def load_prompts(prompt_file):
    """Load prompts from a text file"""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r') as f:
        # Each line is a separate prompt
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Format prompts for YOLO-World
    # Each prompt should be a list of strings
    formatted_prompts = [[p] for p in prompts]
    
    return formatted_prompts


def initialize_yolo_world(config_file, checkpoint_file):
    """Initialize YOLO-World model and pipeline"""
    try:
        from mmdet.apis import init_detector
        from mmyolo.utils import register_all_modules
        from mmcv.transforms import Compose
        
        # Register modules
        register_all_modules()
        
        # Initialize detector
        Logger.info(f"Loading YOLO-World model from {checkpoint_file}")
        model = init_detector(config_file, checkpoint_file, device=DEVICE)
        
        # Create test pipeline as a Compose object
        test_pipeline_cfg = model.cfg.test_dataloader.dataset.pipeline
        test_pipeline = Compose(test_pipeline_cfg)
        
        Logger.success("YOLO-World model initialized successfully")
        return model, test_pipeline
    except Exception as e:
        Logger.error(f"Failed to initialize YOLO-World: {e}")
        traceback.print_exc()
        return None, None


def initialize_sam(checkpoint_file):
    """Initialize SAM model"""
    try:
        from segment_anything import SamPredictor, sam_model_registry
        
        # Determine model type from checkpoint filename
        if "vit_h" in checkpoint_file.lower():
            model_type = "vit_h"
        elif "vit_l" in checkpoint_file.lower():
            model_type = "vit_l"
        elif "vit_b" in checkpoint_file.lower():
            model_type = "vit_b"
        else:
            model_type = "vit_h"  # Default to ViT-H
        
        # Load SAM model
        Logger.info(f"Loading SAM {model_type} from {checkpoint_file}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_file)
        sam.to(device=DEVICE)
        
        # Create predictor
        predictor = SamPredictor(sam)
        
        Logger.success("SAM initialized successfully")
        return predictor
    except Exception as e:
        Logger.error(f"Failed to initialize SAM: {e}")
        traceback.print_exc()
        return None


def main():
    """Main function"""
    # Print header with version info
    print(f"YOLO-World + SAM Clean Segmentation")
    print(f"Author: FETHl")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print("-" * 50)
    
    parser = argparse.ArgumentParser(description="YOLO-World + SAM Clean Segmentation")
    parser.add_argument("config", help="YOLO-World config file")
    parser.add_argument("checkpoint", help="YOLO-World checkpoint file")
    parser.add_argument("image_path", help="Path to input image or directory of images")
    parser.add_argument("prompt_file", help="Path to text file with prompts")
    parser.add_argument("--sam-checkpoint", help="Path to SAM checkpoint file", default=None)
    parser.add_argument("--output-dir", help="Output directory", default="outputs")
    parser.add_argument("--threshold", help="Detection threshold", type=float, default=0.05)
    parser.add_argument("--expand-ratio", help="Box expansion ratio for SAM", type=float, default=0.15)
    parser.add_argument("--use-crf", help="Apply CRF refinement", action="store_true")
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.config):
        Logger.error(f"Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        Logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.prompt_file):
        Logger.error(f"Prompt file not found: {args.prompt_file}")
        return
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    Logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    
    # Initialize YOLO-World
    model, test_pipeline = initialize_yolo_world(args.config, args.checkpoint)
    if model is None or test_pipeline is None:
        return
    
    # Initialize SAM if checkpoint provided
    sam_predictor = None
    if args.sam_checkpoint:
        if not os.path.exists(args.sam_checkpoint):
            Logger.warning(f"SAM checkpoint not found: {args.sam_checkpoint}")
        else:
            sam_predictor = initialize_sam(args.sam_checkpoint)
    
    # Check if CRF is available if requested
    if args.use_crf and not HAS_CRF:
        Logger.warning("CRF requested but pydensecrf not available. Will skip CRF refinement.")
    
    # Initialize processor with direct SAM predictor
    processor = CleanSegmentationProcessor(
        sam_predictor=sam_predictor,
        debug=args.debug
    )
    
    # Process images
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext.upper()}")))
        
        if not image_files:
            Logger.error(f"No image files found in {args.image_path}")
            return
        
        Logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                processor.process_image(
                    image_file, model, test_pipeline, prompts,
                    threshold=args.threshold,
                    expand_ratio=args.expand_ratio,
                    use_crf=args.use_crf and HAS_CRF,
                    output_dir=args.output_dir
                )
            except Exception as e:
                Logger.error(f"Error processing {image_file}: {e}")
                traceback.print_exc()
    else:
        # Process single image
        if not os.path.exists(args.image_path):
            Logger.error(f"Image file not found: {args.image_path}")
            return
        
        try:
            processor.process_image(
                args.image_path, model, test_pipeline, prompts,
                threshold=args.threshold,
                expand_ratio=args.expand_ratio,
                use_crf=args.use_crf and HAS_CRF,
                output_dir=args.output_dir
            )
        except Exception as e:
            Logger.error(f"Error processing {args.image_path}: {e}")
            traceback.print_exc()
    
    Logger.success("Processing complete!")


if __name__ == "__main__":
    main()