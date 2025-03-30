#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YOLO-World + SAM Clean Segmentation
# Author: FETHl
# Date: 2025-03-26 14:40:12

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
    """Simple logging utility"""
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

class InternalContourDetector:
    """Advanced detection of internal contours specifically for clothing and object parts"""
    
    def __init__(self, min_internal_area=20, max_internal_area_ratio=0.9, debug=False):
        self.min_internal_area = min_internal_area
        self.max_internal_area_ratio = max_internal_area_ratio
        self.debug = debug



    
    def detect_internal_contours(self, mask, original_image=None):
        """Enhanced internal contour detection using aggressive methods for clothing"""
        # If no original image is provided, fall back to basic detection
        if original_image is None:
            contours = self._basic_internal_contours(mask)
            Logger.info(f"Basic detection found {len(contours)} internal contours")
            return contours
            
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Extract the masked region from the original image
        masked_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
        
        # Set up debugging if enabled
        if self.debug:
            debug_dir = "debug_outputs"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/masked_image.jpg", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
        
        # ====================== MULTI-METHOD DETECTION ======================
        # We'll use multiple methods and combine the results
        all_contours = []
        
        # Method 1: Basic Canny edge detection
        gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        edges_canny = cv2.Canny(gray, 10, 50)  # Very low thresholds to catch more edges
        
        # Method 2: Color-based segmentation using clustering
        color_edges = self._color_clustering_edges(masked_image)
        
        # Method 3: LAB color space gradients (good for color transitions)
        lab_edges = self._lab_color_edges(masked_image)
        
        # Method 4: Adaptive thresholding on each channel
        adaptive_edges = self._adaptive_threshold_edges(masked_image)
        
        # Combine all edge detection methods
        combined_edges = cv2.bitwise_or(edges_canny, color_edges)
        combined_edges = cv2.bitwise_or(combined_edges, lab_edges)
        combined_edges = cv2.bitwise_or(combined_edges, adaptive_edges)
        
        # Apply morphological operations to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Make sure edges are only within the mask area
        combined_edges = cv2.bitwise_and(combined_edges, combined_edges, mask=binary_mask)
        
        # Save debug images if debug is enabled
        if self.debug:
            cv2.imwrite(f"{debug_dir}/edges_canny.jpg", edges_canny)
            cv2.imwrite(f"{debug_dir}/color_edges.jpg", color_edges)
            cv2.imwrite(f"{debug_dir}/lab_edges.jpg", lab_edges)
            cv2.imwrite(f"{debug_dir}/adaptive_edges.jpg", adaptive_edges)
            cv2.imwrite(f"{debug_dir}/combined_edges.jpg", combined_edges)
        
        # Find contours from the combined edges
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and clean the contours
        filtered_contours = self._filter_contours(contours, binary_mask, masked_image)
        
        Logger.info(f"Advanced detection found {len(filtered_contours)} internal contours")
        
        # If we didn't find any contours, try a more aggressive approach
        if len(filtered_contours) == 0:
            Logger.info("No contours found, trying aggressive detection...")
            filtered_contours = self._aggressive_contour_detection(masked_image, binary_mask)
            
        # Debug visualization with contours
        if self.debug and len(filtered_contours) > 0:
            debug_contour_img = masked_image.copy()
            for i, c in enumerate(filtered_contours):
                cv2.drawContours(debug_contour_img, [c], 0, (0,255,0), 1)
            cv2.imwrite(f"{debug_dir}/detected_contours.jpg", 
                       cv2.cvtColor(debug_contour_img, cv2.COLOR_RGB2BGR))
            
        return filtered_contours
   
    def _color_clustering_edges(self, image):
        """Enhanced color-based edge detection optimized for clothing boundaries"""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Reshape image for clustering
        pixels = filtered.reshape((-1, 3)).astype(np.float32)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 5  # Increased from 4 to 5 to better capture clothing details
        
        # Handle empty images
        if pixels.size == 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Run k-means
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Convert back to original shape
        segmented_img = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
        
        # Create a mask for edges using morphological gradient
        kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Apply threshold to get final edges
        _, edges = cv2.threshold(gradient, 15, 255, cv2.THRESH_BINARY)
        
        return edges
    


    def _lab_color_edges(self, image):
        """Find edges using LAB color space gradients"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        
        # Find gradients in a and b channels (where color changes happen)
        sobelx_a = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
        sobely_a = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        sobelx_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
        sobely_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        grad_a = cv2.magnitude(sobelx_a, sobely_a)
        grad_b = cv2.magnitude(sobelx_b, sobely_b)
        
        combined_grad = cv2.addWeighted(grad_a, 0.5, grad_b, 0.5, 0)
        
        # Normalize and threshold
        combined_grad = cv2.normalize(combined_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(combined_grad, 20, 255, cv2.THRESH_BINARY)
        
        return edges
    
    def _adaptive_threshold_edges(self, image):
        """Apply adaptive thresholding to each channel"""
        # Split channels
        b, g, r = cv2.split(image)
        
        # Apply adaptive thresholding to each
        thresh_b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine thresholds
        combined = cv2.bitwise_or(thresh_b, thresh_g)
        combined = cv2.bitwise_or(combined, thresh_r)
        
        # Apply gradient to find edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(combined, cv2.MORPH_GRADIENT, kernel)
        
        return edges
    
    def _aggressive_contour_detection(self, image, mask):
        """Improved aggressive contour detection without artificial straight lines"""
        h, w = image.shape[:2]
        
        # Method 1: Try different color spaces and very low thresholds
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # Threshold the saturation channel - often differentiates clothing
        _, s_thresh = cv2.threshold(s_channel, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours in the saturation channel
        contours, _ = cv2.findContours(s_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 2: If that fails, use improved gradient-based segmentation
        if len(contours) < 2:
            # Prepare markers for watershed
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply bilateral filter to preserve edges while reducing noise
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply morphological gradient to find edges
            kernel = np.ones((3,3), np.uint8)
            morpho_grad = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold the gradient
            _, thresh = cv2.threshold(morpho_grad, 15, 255, cv2.THRESH_BINARY)
            
            # Find contours in the gradient image
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply mask to filter out contours outside our object
        filtered_contours = []
        for contour in contours:
            # Skip tiny contours
            if len(contour) < 3 or cv2.contourArea(contour) < 20:
                continue
                
            # Create temp mask for this contour
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp_mask, [contour], 0, 255, 1)
            
            # Check overlap with original mask
            overlap = cv2.bitwise_and(temp_mask, mask)
            if np.count_nonzero(overlap) > 0:
                # Check if this is not just the outer contour
                if cv2.contourArea(contour) < cv2.countNonZero(mask) * 0.8:
                    filtered_contours.append(contour)
        
        # Filter by minimum area and other criteria
        return self._filter_contours(filtered_contours, mask, image, aggressive=True)
    
    def _basic_internal_contours(self, mask):
        """Basic internal contour detection when no image is available"""
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Find external contour first to get reference area
        external_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not external_contours:
            return []
            
        # Get largest external contour
        external_contour = max(external_contours, key=cv2.contourArea)
        external_area = cv2.contourArea(external_contour)
        
        # Find all contours including internal ones
        all_contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter internal contours
        internal_contours = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, (contour, h) in enumerate(zip(all_contours, hierarchy)):
                # h[3] > -1 means this contour has a parent (it's an internal contour)
                if h[3] > -1:
                    area = cv2.contourArea(contour)
                    # Filter by minimum area and maximum ratio to external contour
                    if (area >= self.min_internal_area and 
                        area <= external_area * self.max_internal_area_ratio):
                        internal_contours.append(contour)
        
        return internal_contours
        
    def _filter_contours(self, contours, mask, image, aggressive=False):
        """Enhanced filtering optimized for natural-looking clothing contours"""
        h, w = mask.shape[:2]
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get the external contour for area reference
        external_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not external_contours:
            return []
            
        # Get largest external contour area
        external_area = cv2.contourArea(max(external_contours, key=cv2.contourArea))
        
        # Filter contours
        filtered_contours = []
        
        # Sort contours by area (largest first)
        valid_contours = []
        for contour in contours:
            if len(contour) < 3:  # Skip contours with too few points
                continue
            
            # Check if within mask
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp_mask, [contour], 0, 255, 1)
            overlap = cv2.bitwise_and(temp_mask, mask)
            if np.count_nonzero(overlap) > 0:
                valid_contours.append(contour)
                
        # If in aggressive mode, relax filtering criteria
        min_area = self.min_internal_area // 2 if aggressive else self.min_internal_area
        overlap_threshold = 0.9 if aggressive else 0.8
        
        # Calculate contour "naturalness" - natural contours typically have:
        # 1. Lower convexity defect values
        # 2. Smoother curvature changes
        # 3. More points relative to their perimeter
        
        scored_contours = []
        for contour in valid_contours:
            area = cv2.contourArea(contour)
            
            # Skip too small contours
            if area < min_area:
                continue
                
            # Skip too large contours (relative to the external contour)
            if area > external_area * self.max_internal_area_ratio:
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate naturalness score factors
            
            # 1. Convexity - how close is this to a convex shape?
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # 2. Contour complexity (perimeter^2/area) - clothing has moderate complexity
            complexity = (perimeter**2) / area if area > 0 else float('inf')
            
            # 3. Number of points relative to perimeter
            points_density = len(contour) / perimeter if perimeter > 0 else 0
            
            # Calculate overall naturalness score (higher is better)
            # These weights prioritize contours that look like clothing boundaries
            naturalness = (0.4 * convexity) + (0.3 * (1.0 - min(1.0, complexity/100))) + (0.3 * min(1.0, points_density))
            
            # Store contour with its score
            scored_contours.append((contour, area, naturalness))
        
        # Sort by naturalness score (highest first) then by area (largest first)
        sorted_contours = sorted(scored_contours, key=lambda x: (x[2], x[1]), reverse=True)
        
        # Take the best contours, avoiding too much overlap
        for contour_info in sorted_contours:
            contour = contour_info[0]
            
            # Check if this contour is too similar to existing ones
            temp_mask = np.zeros_like(contour_mask)
            cv2.drawContours(temp_mask, [contour], 0, 255, 1)
            
            # Skip if this contour is largely already covered by previous contours
            overlap = cv2.bitwise_and(temp_mask, contour_mask)
            overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(temp_mask) if np.count_nonzero(temp_mask) > 0 else 0
            
            if overlap_ratio > overlap_threshold:
                continue
            
            # Add contour to filtered set
            filtered_contours.append(contour)
            
            # Update the mask with this contour
            cv2.drawContours(contour_mask, [contour], 0, 255, 1)
            
            # Limit to top 10 most natural contours
            if len(filtered_contours) >= 10:
                break
        
        return filtered_contours
    
    def smooth_internal_contours(self, contours, smoothing_factor=0.0025):
        """
        Apply advanced smoothing to internal contours for natural, smooth appearance
        using cubic spline interpolation
        """
        import scipy.interpolate as interp
        
        smoothed = []
        for contour in contours:
            # Skip if not enough points for proper smoothing
            if len(contour) < 4:
                continue
                
            # Extract contour points
            points = contour.squeeze()
            x = points[:, 0]
            y = points[:, 1]
            
            # If we have very few points, use a different approach
            if len(points) < 10:
                # For small contours, use Douglas-Peucker simplification with gentler settings
                perimeter = cv2.arcLength(contour, True)
                epsilon = smoothing_factor * 1.5 * perimeter
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                smoothed.append(smoothed_contour)
                continue
            
            # Get contour length and create a parameterization variable t
            t = np.zeros(len(points))
            for i in range(1, len(points)):
                # Cumulative chord length parameterization
                t[i] = t[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
                
            if t[-1] == 0:  # Avoid division by zero
                smoothed.append(contour)
                continue
                
            # Normalize t to [0, 1]
            t = t / t[-1]
            
            # Create a periodic spline for closed contours
            # Generate more points for a smooth appearance
            num_points = min(max(int(len(points) * 1.5), 20), 100)  # Min 20, max 100 points
            try:
                # For closed contours, wrap values for periodic interpolation
                if np.all(points[0] == points[-1]):
                    # Create periodic spline
                    x_periodic = np.append(x[:-1], x[0])
                    y_periodic = np.append(y[:-1], y[0])
                    t_periodic = np.append(t[:-1], 1.0)
                    
                    # Create periodic parameter space
                    t_new = np.linspace(0, 1, num_points+1)[:-1]
                    
                    # Fit periodic spline
                    cs_x = interp.CubicSpline(t_periodic, x_periodic, bc_type='periodic')
                    cs_y = interp.CubicSpline(t_periodic, y_periodic, bc_type='periodic')
                    
                    # Generate new smooth contour
                    x_new = cs_x(t_new)
                    y_new = cs_y(t_new)
                else:
                    # For open contours, use standard spline
                    t_new = np.linspace(0, 1, num_points)
                    cs_x = interp.CubicSpline(t, x)
                    cs_y = interp.CubicSpline(t, y)
                    x_new = cs_x(t_new)
                    y_new = cs_y(t_new)
                    
                # Create array of new contour points
                new_points = np.column_stack((x_new, y_new)).astype(np.int32)
                smoothed_contour = new_points.reshape((-1, 1, 2))
                
                # For clothing boundaries, we want to preserve key features while removing noise
                # Apply light poly approximation after spline to remove noisy oscillations
                if len(smoothed_contour) > 10:
                    perimeter = cv2.arcLength(smoothed_contour, True)
                    epsilon = smoothing_factor * 0.4 * perimeter  # Gentler approximation to preserve curve
                    smoothed_contour = cv2.approxPolyDP(smoothed_contour, epsilon, True)
                
                smoothed.append(smoothed_contour)
                
            except Exception:
                # Fall back to basic smoothing if spline fails
                perimeter = cv2.arcLength(contour, True)
                epsilon = smoothing_factor * perimeter
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                smoothed.append(smoothed_contour)
        
        return smoothed

class ContourCleaner:
    """Clean and simplify contours for better visualization"""
    
    def __init__(self, min_contour_area=100, min_internal_contour_area=20):
        self.min_contour_area = min_contour_area
        self.min_internal_contour_area = min_internal_contour_area
    
    def clean_mask(self, mask):
        """Clean a binary mask by removing noise and filling holes"""
        # CRITICAL: Strictly enforce binary mask (0 or 1)
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Morphological operations to clean mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(cleaned)
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                cv2.drawContours(filled_mask, [contour], 0, 1, -1)  # -1 means fill
        
        return filled_mask
    
    def get_largest_contour(self, mask):
        """Extract the largest contour from a mask"""
        # CRITICAL: Strictly enforce binary mask (0 or 1)
        mask_binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Skip tiny contours
        if cv2.contourArea(largest) < self.min_contour_area:
            return None
            
        return largest
    
    def smooth_contour(self, contour, smoothing_factor=0.002):
        """Smooth a contour for better visualization"""
        if contour is None or len(contour) < 3:
            return contour
            
        # Calculate adaptive epsilon based on contour length
        perimeter = cv2.arcLength(contour, True)
        epsilon = smoothing_factor * perimeter
        
        # Apply Douglas-Peucker algorithm to simplify contour
        smoothed = cv2.approxPolyDP(contour, epsilon, True)
        
        return smoothed
    
    def process_mask(self, mask):
        """Process a mask to get a single clean contour and internal edges"""
        # CRITICAL: Strictly enforce binary mask (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Clean the mask
        cleaned_mask = self.clean_mask(binary_mask)
        
        # Get the largest contour (outer edge)
        outer_contour = self.get_largest_contour(cleaned_mask)
        
        # Smooth outer contour
        smoothed_outer_contour = None
        if outer_contour is not None:
            smoothed_outer_contour = self.smooth_contour(outer_contour)
        
        return cleaned_mask, smoothed_outer_contour

class CleanSVGExporter:
    """Export clean contours to SVG format"""
    
    def export_to_svg(self, contours_data, shape, output_path):
        """
        Export contours to SVG file with clean representation
        
        Args:
            contours_data: List of dictionaries with structure:
                {
                    'outer_contour': outer_contour,
                    'internal_contours': [internal_contours],
                    'label': label,
                    'color': color
                }
            shape: Image shape (h, w)
            output_path: Path to save SVG file
        """
        # Extract image dimensions
        h, w = shape[:2]
        
        # SVG header with metadata
        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <metadata>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
      <rdf:Description>
        <dc:creator>FETHl</dc:creator>
        <dc:date>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</dc:date>
        <dc:description>Generated by YOLO-World + SAM Segmentation</dc:description>
      </rdf:Description>
    </rdf:RDF>
  </metadata>

  <!-- Background (transparent) -->
  <rect width="{w}" height="{h}" fill="none" />

'''
        
        # Process each contour
        for i, data in enumerate(contours_data):
            outer_contour = data['outer_contour']
            internal_contours = data.get('internal_contours', [])
            label = data['label']
            color = data['color']
            
            if outer_contour is None or len(outer_contour) < 3:
                continue
                
            # Convert color to hex
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
            
            # Create SVG path from outer contour
            outer_path_data = "M "
            for j, point in enumerate(outer_contour):
                x, y = point[0]
                outer_path_data += f"{x},{y} "
                if j < len(outer_contour) - 1:
                    outer_path_data += "L "
            outer_path_data += "Z"  # Close the path
            
            # Start group for this object
            svg += f'  <g id="object-{i+1}">\n'
            
            # Add outer contour to SVG with good styling
            svg += f'    <path d="{outer_path_data}" fill="{color_hex}" fill-opacity="0.3" stroke="{color_hex}" stroke-width="2" stroke-opacity="0.8" />\n'
            
            # Add internal contours
            for j, internal_contour in enumerate(internal_contours):
                if len(internal_contour) < 2:
                    continue
                
                # Create path for internal contour
                internal_path_data = "M "
                for k, point in enumerate(internal_contour):
                    x, y = point[0]
                    internal_path_data += f"{x},{y} "
                    if k < len(internal_contour) - 1:
                        internal_path_data += "L "
                
                # For open contours (internal edges), don't close the path
                # Add internal contour with different styling
                svg += f'    <path d="{internal_path_data}" fill="none" stroke="{color_hex}" stroke-width="1.5" stroke-opacity="0.7" stroke-dasharray="0" />\n'
            
            # Add label
            # Find a good position for the label (near the top of the contour)
            x_values = [p[0][0] for p in outer_contour]
            y_values = [p[0][1] for p in outer_contour]
            
            if x_values and y_values:
                # Use centroid for label position
                label_x = sum(x_values) / len(x_values)
                label_y = min(y_values) - 10
                
                # Keep within bounds
                label_y = max(15, label_y)
                
                # Add label text
                svg += f'    <text x="{label_x}" y="{label_y}" font-family="Arial" font-size="14" fill="{color_hex}" text-anchor="middle" font-weight="bold">{label}</text>\n'
            
            # Close group
            svg += '  </g>\n'
        
        # Close SVG
        svg += '</svg>'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(svg)
        
        return True


class CleanSegmentationProcessor:
    """Main processor for clean segmentation using dual SAM approach"""
    
    def __init__(self, sam_models=None, debug=False):
        self.debug = debug
        
        # Initialize helpers
        self.crf_processor = CRFPostProcessor(debug=debug)
        self.contour_cleaner = ContourCleaner(min_contour_area=100, min_internal_contour_area=20)
        self.svg_exporter = CleanSVGExporter()
        
        # Store both SAM models - predictor and mask generator
        if sam_models is not None:
            self.sam_predictor = sam_models.get("predictor")
            self.sam_mask_generator = sam_models.get("mask_generator")
            self.sam_available = (self.sam_predictor is not None)
        else:
            self.sam_predictor = None
            self.sam_mask_generator = None
            self.sam_available = False
        
        if self.sam_available:
            if self.sam_mask_generator is not None:
                Logger.info("SAM dual mode is available (predictor + automatic mask generator)")
            else:
                Logger.info("SAM predictor is available and will be used for segmentation")
        else:
            Logger.warning("SAM is not available - only bounding boxes will be drawn")

    
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
        """Get mask prediction from a bounding box using SAM automatic mask generator"""
        if not self.sam_available or self.sam_predictor is None:
            return None, 0
        
        try:
            # Format box to get region of interest (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)
            
            # Get the region of interest (ROI) from the image
            # Note: We're assuming processed_image is available as an instance variable or from a previous call
            roi = self.current_image[y1:y2, x1:x2]
            
            if roi.size == 0:  # Check if ROI is empty
                return None, 0
            
            # Generate masks for the ROI
            masks = self.sam_predictor.generate(roi)
            
            if not masks:
                return None, 0
            
            # Sort masks by area (largest first)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # Get the largest mask (most likely the main object)
            best_mask = masks[0]
            
            # Convert the segmentation to a binary mask
            segmentation = best_mask['segmentation'].astype(np.uint8)
            
            # Create full image mask
            full_mask = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)
            
            # Place the mask in the correct position
            full_mask[y1:y2, x1:x2] = segmentation
            
            # Use the confidence score from SAM
            score = best_mask.get('predicted_iou', 0.0)
            
            return full_mask, score
        except Exception as e:
            Logger.error(f"Error in SAM automatic mask generation: {e}")
            traceback.print_exc()
            return None, 0
    
    
    def detect_internal_contours_with_sam(self, mask, original_image):
        """Detect internal contours using SAM automatic mask generator"""
        if not self.sam_available or not hasattr(self, 'sam_mask_generator') or self.sam_mask_generator is None:
            # Fall back to traditional method if SAM mask generator not available
            Logger.info("SAM mask generator not available, using traditional method")
            if hasattr(self, 'internal_detector'):
                return self.internal_detector.detect_internal_contours(mask, original_image)
            return []
        
        try:
            # Create binary mask
            binary_mask = (mask > 0).astype(np.uint8) * 255
            
            # Extract the masked region from the original image
            masked_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
            
            # Get bounding box of the mask to crop the region
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand slightly for context (5% padding)
            padding_x = int(w * 0.05)
            padding_y = int(h * 0.05)
            
            # Make sure we're within bounds
            x_start = max(0, x - padding_x)
            y_start = max(0, y - padding_y)
            x_end = min(original_image.shape[1], x + w + padding_x)
            y_end = min(original_image.shape[0], y + h + padding_y)
            
            # Crop the masked region
            cropped_mask = binary_mask[y_start:y_end, x_start:x_end]
            cropped_image = original_image[y_start:y_end, x_start:x_end].copy()
            
            # Apply mask to cropped image (focus SAM on the object)
            cropped_masked_image = cv2.bitwise_and(cropped_image, cropped_image, 
                                                mask=cropped_mask)
            
            # Debug output
            if self.debug:
                debug_dir = "debug_outputs"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/cropped_masked_image.jpg", 
                        cv2.cvtColor(cropped_masked_image, cv2.COLOR_RGB2BGR))
            
            # Generate masks for the cropped region
            sam_masks = self.sam_mask_generator.generate(cropped_image)
            
            # Debug info
            Logger.info(f"SAM generated {len(sam_masks)} internal masks")
            
            # Filter masks to keep only those that:
            # 1. Are fully inside our object (high overlap with main mask)
            # 2. Are not too small or too large
            # 3. Have high confidence
            internal_contours = []
            
            # Get area of the main mask for reference
            main_mask_area = np.count_nonzero(cropped_mask)
            
            for sam_mask in sam_masks:
                # Get the mask
                mask_data = sam_mask['segmentation'].astype(np.uint8)
                
                # Calculate overlap with main mask
                overlap = cv2.bitwise_and(mask_data, cropped_mask)
                overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(mask_data) if np.count_nonzero(mask_data) > 0 else 0
                
                # Calculate area ratio compared to main mask
                area_ratio = np.count_nonzero(mask_data) / main_mask_area if main_mask_area > 0 else 0
                
                # Skip masks that:
                # - Have low overlap with the main mask
                # - Are too large (likely duplicates of main mask)
                # - Are too small (likely noise)
                if (overlap_ratio < 0.95 or  # Must be mostly inside main mask
                    area_ratio > 0.8 or      # Not too large
                    area_ratio < 0.01 or     # Not too small
                    sam_mask.get('predicted_iou', 0) < 0.7):  # High confidence
                    continue
                
                # Get the contour of this mask
                mask_contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in mask_contours:
                    # Adjust contour coordinates back to original image space
                    adjusted_contour = contour.copy()
                    adjusted_contour[:,:,0] += x_start
                    adjusted_contour[:,:,1] += y_start
                    
                    # Add to our internal contours list
                    if len(adjusted_contour) >= 4:  # Ensure it has enough points
                        internal_contours.append(adjusted_contour)
            
            # Debug visualization
            if self.debug and len(internal_contours) > 0:
                debug_img = original_image.copy()
                cv2.drawContours(debug_img, internal_contours, -1, (0,255,0), 1)
                cv2.imwrite(f"{debug_dir}/sam_internal_contours.jpg", 
                        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            
            # Return internal contours found by SAM
            return internal_contours
            
        except Exception as e:
            Logger.error(f"Error in SAM internal contour detection: {e}")
            traceback.print_exc()
            
            # Fall back to traditional method
            if hasattr(self, 'internal_detector'):
                return self.internal_detector.detect_internal_contours(mask, original_image)
            return []

    def process_image(self, image_path, model, test_pipeline, prompts, 
                    threshold=0.05, expand_ratio=0.15, use_crf=False, 
                    detect_internal=True, output_dir='outputs'):
        """Process image with YOLO-World and dual SAM approach for clean segmentation with internal contours"""
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
        
        # Create text labels and extract class names
        text_labels = []
        class_names = []
        for label_id, score in zip(labels, scores):
            class_name = prompts[label_id][0]
            class_names.append(class_name)
            text_labels.append(f"{class_name} {score:.2f}")
        
        Logger.success(f"Found {len(boxes)} objects")
        
        # Get unique class names and count instances of each class
        unique_classes = sorted(set(class_names))
        class_counts = {cls: class_names.count(cls) for cls in unique_classes}
        
        # Generate a color for each unique class
        class_colors = {}
        for i, class_name in enumerate(unique_classes):
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good color distribution
            class_colors[class_name] = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        
        # Map detection to its color based on class name
        colors = [class_colors[class_names[i]] for i in range(len(boxes))]
        
        # Visualization image
        vis_image = original_image.copy()
        
        # Process each detection with dual SAM approach
        all_masks = []
        all_contours = []
        all_internal_contours = []
        all_labels = []
        all_class_names = []
        
        if self.sam_available and self.sam_predictor is not None:
            Logger.info("Running SAM segmentation for each detection...")
            
            # Set image for SAM predictor (for outer contours)
            self.sam_predictor.set_image(processed_image)
            
            # Process each box
            for i, box in enumerate(boxes):
                Logger.info(f"Processing {text_labels[i]}...")
                
                # Expand box for better context
                expanded_box = self.expand_box(box, expand_ratio, processed_image.shape)
                
                # Get mask from SAM predictor (for outer contour)
                mask = None
                score_sam = 0.0
                
                try:
                    # Get outer mask from SAM predictor
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(expanded_box),
                        multimask_output=True
                    )
                    
                    # Use best mask
                    if len(masks) > 0:
                        best_idx = np.argmax(scores)
                        # CRITICAL: Ensure binary mask using boolean conversion
                        mask = masks[best_idx].astype(bool).astype(np.uint8)
                        score_sam = scores[best_idx]
                except Exception as e:
                    Logger.error(f"Error in SAM prediction: {e}")
                    traceback.print_exc()
                    mask = None
                
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
                            # Continue without CRF if it fails
                    
                    # Clean the mask and extract a single clean contour
                    cleaned_mask, clean_contour = self.contour_cleaner.process_mask(binary_mask)
                    
                    # Enhanced internal contours detection using dual approach
                    internal_contours = []
                    if detect_internal:
                        try:
                            # First try with SAM automatic mask generator if available
                            if hasattr(self, 'sam_mask_generator') and self.sam_mask_generator is not None:
                                Logger.info(f"Detecting internal contours with SAM for {text_labels[i]}...")
                                internal_contours = self.detect_internal_contours_with_sam(
                                    cleaned_mask,
                                    processed_image
                                )
                                Logger.info(f"SAM found {len(internal_contours)} internal contours")
                        except Exception as e:
                            Logger.error(f"Error in SAM internal contour detection: {e}")
                            traceback.print_exc()
                            internal_contours = []
                        
                        # Fall back to traditional method if SAM didn't find any or failed
                        if len(internal_contours) == 0 and hasattr(self, 'internal_detector'):
                            try:
                                Logger.info(f"Falling back to traditional internal contour detection...")
                                internal_contours = self.internal_detector.detect_internal_contours(
                                    cleaned_mask,
                                    processed_image
                                )
                                Logger.info(f"Traditional method found {len(internal_contours)} internal contours")
                            except Exception as e:
                                Logger.error(f"Error in traditional internal contour detection: {e}")
                                traceback.print_exc()
                                internal_contours = []
                        
                        # Smooth internal contours if we have any and the method exists
                        if len(internal_contours) > 0 and hasattr(self.internal_detector, 'smooth_internal_contours'):
                            try:
                                internal_contours = self.internal_detector.smooth_internal_contours(
                                    internal_contours, 
                                    smoothing_factor=0.0025
                                )
                            except Exception as e:
                                Logger.error(f"Error smoothing internal contours: {e}")
                                # Continue with unsmoothed contours
                    
                    if clean_contour is not None:
                        # Add to results
                        all_masks.append(cleaned_mask)
                        all_contours.append(clean_contour)
                        all_internal_contours.append(internal_contours)
                        all_labels.append(text_labels[i])
                        all_class_names.append(class_names[i])
                        
                        # Draw on visualization
                        color = colors[i]
                        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                        
                        # Draw mask overlay with transparency
                        mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
                        mask_overlay[cleaned_mask > 0] = [int(c*255) for c in color]
                        alpha = 0.3  # Lower alpha for less obtrusive overlays
                        vis_image = cv2.addWeighted(vis_image, 1, mask_overlay, alpha, 0)
                        
                        # Draw external contour
                        cv2.drawContours(vis_image, [clean_contour], -1, color_bgr, 2)
                        
                        # Draw internal contours if available
                        if detect_internal and internal_contours:
                            # Create a brighter color for internal contours
                            internal_color_bgr = (
                                min(int(color[2]*255) + 40, 255),
                                min(int(color[1]*255) + 40, 255),
                                min(int(color[0]*255) + 40, 255)
                            )
                            
                            for internal_contour in internal_contours:
                                cv2.drawContours(vis_image, [internal_contour], -1, internal_color_bgr, 1)
                    else:
                        Logger.warning(f"Could not extract clean contour for {text_labels[i]}")
                        
                        # Just draw the bounding box if no contour
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
                else:
                    Logger.warning(f"SAM did not generate a mask for {text_labels[i]}")
                    
                    # Draw bounding box if no mask
                    color = colors[i]
                    color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
        else:
            # Fallback to simple visualization without SAM
            Logger.warning("SAM not available. Using simple box visualization.")
            
            for i, box in enumerate(boxes):
                color = colors[i]
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
                
                # Draw box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Add legend to the visualization
        vis_image = self._add_legend_to_image(vis_image, class_colors, class_counts)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{name_without_ext}_visualization.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        Logger.success(f"Visualization saved to: {vis_path}")
        
        # Export SVG if we have contours
        svg_path = None
        if all_contours:
            svg_dir = os.path.join(output_dir, "svg")
            os.makedirs(svg_dir, exist_ok=True)
            svg_path = os.path.join(svg_dir, f"{name_without_ext}.svg")
            
            # Export SVG with internal contours
            if detect_internal and all_internal_contours and any(len(ic) > 0 for ic in all_internal_contours):
                try:
                    self.export_svg_with_internal_contours(
                        all_contours, 
                        all_internal_contours,
                        all_labels,
                        all_class_names,
                        class_colors,
                        class_counts,
                        original_image.shape, 
                        svg_path
                    )
                except Exception as e:
                    Logger.error(f"Error exporting SVG with internal contours: {e}")
                    traceback.print_exc()
                    
                    # Fall back to standard SVG export
                    try:
                        self.svg_exporter.export_to_svg(
                            [{'outer_contour': contour, 
                            'internal_contours': [], 
                            'label': label, 
                            'color': class_colors[class_name]} 
                            for contour, label, class_name in zip(all_contours, all_labels, all_class_names)],
                            original_image.shape,
                            svg_path
                        )
                    except Exception as e:
                        Logger.error(f"Error exporting standard SVG: {e}")
                        svg_path = None
            else:
                # Fall back to standard SVG export if no internal contours
                try:
                    self.svg_exporter.export_to_svg(
                        [{'outer_contour': contour, 
                        'internal_contours': [], 
                        'label': label, 
                        'color': class_colors[class_name]} 
                        for contour, label, class_name in zip(all_contours, all_labels, all_class_names)],
                        original_image.shape,
                        svg_path
                    )
                except Exception as e:
                    Logger.error(f"Error exporting standard SVG: {e}")
                    svg_path = None
                    
            if svg_path:
                Logger.success(f"SVG exported to: {svg_path}")
        
        # Return results
        return {
            'masks': all_masks,
            'contours': all_contours,
            'internal_contours': all_internal_contours if detect_internal else [],
            'labels': all_labels,
            'class_names': all_class_names,
            'class_colors': class_colors,
            'class_counts': class_counts,
            'boxes': boxes,
            'visualization_path': vis_path,
            'svg_path': svg_path,
            'metadata': {
                'created_by': 'FETHl',
                'created_at': '2025-03-30 22:21:34',
                'source_image': image_path
            }
        }


    def _add_legend_to_image(self, image, class_colors, class_counts):
        """Add a clean legend to the visualization image"""
        h, w = image.shape[:2]
        
        # Determine legend dimensions
        num_classes = len(class_colors)
        
        # Skip legend if no classes
        if num_classes == 0:
            return image
        
        # Padding around legend
        padding = 10
        box_size = 20
        line_height = 25
        font_scale = 0.6
        font_thickness = 1
        
        # Calculate legend width based on longest class name
        max_text_width = 0
        for class_name in class_colors.keys():
            # Include count in width calculation
            text = f"{class_name} ({class_counts[class_name]})"
            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            max_text_width = max(max_text_width, text_width)
        
        legend_width = box_size + 10 + max_text_width + 2 * padding
        legend_height = padding * 2 + num_classes * line_height
        
        # Determine legend position (top right)
        legend_x = w - legend_width - 20
        legend_y = 20
        
        # Create semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay, 
            (legend_x, legend_y), 
            (legend_x + legend_width, legend_y + legend_height), 
            (245, 245, 245), 
            -1
        )
        
        # Add border
        cv2.rectangle(
            overlay, 
            (legend_x, legend_y), 
            (legend_x + legend_width, legend_y + legend_height), 
            (128, 128, 128), 
            1
        )
        
        # Add title
        cv2.putText(
            overlay, 
            "Legend", 
            (legend_x + padding, legend_y + padding + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 0), 
            2
        )
        
        # Add class entries
        y = legend_y + padding + line_height + 10
        
        # Sort classes alphabetically
        sorted_classes = sorted(class_colors.keys())
        
        for class_name in sorted_classes:
            color = class_colors[class_name]
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            
            # Draw color box
            cv2.rectangle(
                overlay, 
                (legend_x + padding, y - box_size + 5), 
                (legend_x + padding + box_size, y + 5), 
                color_bgr, 
                -1
            )
            
            # Draw class name with count
            count = class_counts[class_name]
            text = f"{class_name} ({count})"
            cv2.putText(
                overlay, 
                text, 
                (legend_x + padding + box_size + 10, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0), 
                font_thickness
            )
            
            y += line_height
        
        # Blend overlay with original image
        alpha = 0.9
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return image    



    def export_svg_with_internal_contours(self, contours, internal_contours, labels, 
                                        class_names, class_colors, class_counts, shape, output_path):
        """Export contours to SVG file with internal contours and legend"""
        # Extract image dimensions
        h, w = shape[:2]
        
        # SVG header with metadata
        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <metadata>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:dc="http://purl.org/dc/elements/1.1/">
        <rdf:Description>
            <dc:creator>FETHl</dc:creator>
            <dc:date>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</dc:date>
            <dc:description>Generated by YOLO-World + SAM Segmentation with internal contours</dc:description>
        </rdf:Description>
        </rdf:RDF>
    </metadata>

    <!-- CSS Styles -->
    <style>
        .legend-box {{ fill: #f5f5f5; stroke: #808080; stroke-width: 1px; }}
        .legend-title {{ font-family: Arial; font-size: 14px; font-weight: bold; }}
        .legend-text {{ font-family: Arial; font-size: 12px; }}
    </style>

    <!-- Background (transparent) -->
    <rect width="{w}" height="{h}" fill="none" />

    '''
        
        # Process each contour
        for i, (contour, internal_contours_list, label, class_name) in enumerate(zip(contours, internal_contours, labels, class_names)):
            if contour is None or len(contour) < 3:
                continue
                
            # Get color based on class name
            color = class_colors[class_name]
            
            # Convert color to hex
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
            
            # Create a slightly brighter color for internal contours
            internal_color = (
                min(color[0] * 1.2, 1.0),
                min(color[1] * 1.2, 1.0),
                min(color[2] * 1.2, 1.0)
            )
            
            internal_color_hex = "#{:02x}{:02x}{:02x}".format(
                int(internal_color[0] * 255),
                int(internal_color[1] * 255),
                int(internal_color[2] * 255)
            )
            
            # Create SVG path from main contour
            path_data = "M "
            for j, point in enumerate(contour):
                x, y = point[0]
                path_data += f"{x},{y} "
                if j < len(contour) - 1:
                    path_data += "L "
            path_data += "Z"  # Close the path
            
            # Start group for this object
            svg += f'  <g id="object-{i+1}" class="{class_name}">\n'
            
            # Add main contour to SVG with good styling
            svg += f'    <path d="{path_data}" fill="{color_hex}" fill-opacity="0.3" stroke="{color_hex}" stroke-width="2" stroke-opacity="0.8" />\n'
            
            # Add internal contours if any
            for j, internal_contour in enumerate(internal_contours_list):
                if len(internal_contour) < 4:  # Need at least 4 points for curves
                    continue
                    
                # Use SVG path with cubic bezier curves
                points = internal_contour.squeeze()
                
                # Start the path
                smooth_path = f"M {points[0][0]},{points[0][1]} "
                
                # For each segment, calculate a Bezier curve
                for i in range(1, len(points)-2):
                    # Current point
                    x1, y1 = points[i]
                    
                    # Next point
                    x2, y2 = points[i+1]
                    
                    # Calculate control points (1/3 of the way to next points)
                    cp1x = x1 + (x2-x1)/3
                    cp1y = y1 + (y2-y1)/3
                    
                    cp2x = x1 + 2*(x2-x1)/3
                    cp2y = y1 + 2*(y2-y1)/3
                    
                    # Add cubic bezier segment
                    smooth_path += f"C {cp1x},{cp1y} {cp2x},{cp2y} {x2},{y2} "
                
                # Close the path
                smooth_path += "Z"
                
                # Add internal contour with smoother styling
                svg += f'    <path d="{smooth_path}" fill="none" stroke="{internal_color_hex}" stroke-width="1.2" stroke-opacity="0.9" stroke-linejoin="round" stroke-linecap="round" />\n'
            # Close group
            svg += '  </g>\n'
        
        # Add legend
        legend_width = 180
        legend_height = 30 + len(class_colors) * 25
        legend_x = w - legend_width - 20
        legend_y = 20
        
        # Legend background
        svg += f'  <g id="legend">\n'
        svg += f'    <rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" class="legend-box" />\n'
        svg += f'    <text x="{legend_x + 10}" y="{legend_y + 20}" class="legend-title">Legend</text>\n'
        
        # Legend entries
        y = legend_y + 45
        sorted_classes = sorted(class_colors.keys())
        
        for class_name in sorted_classes:
            color = class_colors[class_name]
            count = class_counts[class_name]
            
            # Convert color to hex
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
            
            # Color box
            svg += f'    <rect x="{legend_x + 10}" y="{y - 15}" width="20" height="20" fill="{color_hex}" stroke="black" stroke-width="0.5" />\n'
            
            # Class name with count
            svg += f'    <text x="{legend_x + 40}" y="{y}" class="legend-text">{class_name} ({count})</text>\n'
            
            y += 25
        
        svg += '  </g>\n'
        
        # Close SVG
        svg += '</svg>'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(svg)
        
        return True



    def export_svg_with_class_colors(self, contours, labels, class_names, class_colors, shape, output_path):
        """Export contours to SVG file with class-based colors"""
        # Extract image dimensions
        h, w = shape[:2]
        
        # SVG header with metadata
        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <metadata>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:dc="http://purl.org/dc/elements/1.1/">
        <rdf:Description>
            <dc:creator>FETHl</dc:creator>
            <dc:date>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</dc:date>
            <dc:description>Generated by YOLO-World + SAM Segmentation</dc:description>
        </rdf:Description>
        </rdf:RDF>
    </metadata>

    <!-- Background (transparent) -->
    <rect width="{w}" height="{h}" fill="none" />

    '''
        
        # Process each contour
        for i, (contour, label, class_name) in enumerate(zip(contours, labels, class_names)):
            if contour is None or len(contour) < 3:
                continue
                
            # Get color based on class name
            color = class_colors[class_name]
            
            # Convert color to hex
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
            
            # Create SVG path from contour
            path_data = "M "
            for j, point in enumerate(contour):
                x, y = point[0]
                path_data += f"{x},{y} "
                if j < len(contour) - 1:
                    path_data += "L "
            path_data += "Z"  # Close the path
            
            # Start group for this object
            svg += f'  <g id="object-{i+1}" class="{class_name}">\n'
            
            # Add contour to SVG with good styling
            svg += f'    <path d="{path_data}" fill="{color_hex}" fill-opacity="0.3" stroke="{color_hex}" stroke-width="2" stroke-opacity="0.8" />\n'
            
            # Add label
            # Find a good position for the label (near the top of the contour)
            x_values = [p[0][0] for p in contour]
            y_values = [p[0][1] for p in contour]
            
            if x_values and y_values:
                # Use centroid for label position
                label_x = sum(x_values) / len(x_values)
                label_y = min(y_values) - 10
                
                # Keep within bounds
                label_y = max(15, label_y)
                
                # Add label text
                svg += f'    <text x="{label_x}" y="{label_y}" font-family="Arial" font-size="14" fill="{color_hex}" text-anchor="middle" font-weight="bold">{label}</text>\n'
            
            # Close group
            svg += '  </g>\n'
        
        # Close SVG
        svg += '</svg>'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(svg)
        
        return True

  



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
    """Initialize both SAM predictor and automatic mask generator"""
    try:
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
        
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
        
        # Create predictor for outer contours
        predictor = SamPredictor(sam)
        
        # Create automatic mask generator for internal contours
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,           # Default is 32, reduce for speed
            pred_iou_thresh=0.86,         # Default is 0.88, slightly lower for more masks
            stability_score_thresh=0.92,  # Default is 0.95
            crop_n_layers=0,              # No need for crops in small regions
            min_mask_region_area=20,      # Lower minimum area to catch internal features
            output_mode='binary_mask'
        )
        
        Logger.success(f"SAM initialized successfully with {model_type} model (dual mode)")
        return {
            "predictor": predictor,
            "mask_generator": mask_generator
        }
    except ImportError:
        Logger.error("Failed to import segment_anything. Please install it with: pip install segment-anything")
        return None
    except FileNotFoundError:
        Logger.error(f"SAM checkpoint file not found: {checkpoint_file}")
        return None
    except Exception as e:
        Logger.error(f"Failed to initialize SAM: {str(e)}")
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
    parser.add_argument("--detect-internal", help="Detect internal contours", action="store_true", default=True)
    parser.add_argument("--min-internal-area", help="Minimum area for internal contours", type=int, default=50)
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    parser.add_argument("--debug-internal", help="Enable debugging for internal contour detection", action="store_true")
    parser.add_argument("--smooth-factor", help="Smoothing factor for internal contours (lower is smoother)", 
                    type=float, default=0.001)
    parser.add_argument("--no-artificial-contours", help="Disable generation of artificial contours when none are found", 
                    action="store_true")



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
   ############################################################################
   # 
   # Initialize SAM if checkpoint provided
    sam_models = None
    if args.sam_checkpoint:
        if not os.path.exists(args.sam_checkpoint):
            Logger.warning(f"SAM checkpoint not found: {args.sam_checkpoint}")
        else:
            sam_models = initialize_sam(args.sam_checkpoint)

    # Initialize processor with dual SAM models
    processor = CleanSegmentationProcessor(
        sam_models=sam_models,
        debug=args.debug
    )

    # Make sure internal_detector is properly initialized
    if args.detect_internal:
        # Create internal detector instance
        internal_detector = InternalContourDetector(
            min_internal_area=args.min_internal_area,
            max_internal_area_ratio=0.9,
            debug=args.debug_internal if hasattr(args, 'debug_internal') else args.debug
        )
        # Attach it to the processor
        processor.internal_detector = internal_detector
        Logger.info(f"Internal contour detection enabled (min area: {args.min_internal_area})")
   
    #######################################################################################
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
                    detect_internal=args.detect_internal,
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
                detect_internal=args.detect_internal,
                output_dir=args.output_dir
            )
        except Exception as e:
            Logger.error(f"Error processing {args.image_path}: {e}")
            traceback.print_exc()
    
    Logger.success("Processing complete!")

if __name__ == "__main__":
    main()
    
