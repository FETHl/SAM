#!/usr/bin/env python3
"""
Professional Interactive Segmentation Pipeline:

Combines SAM (Segment Anything Model) with CRF refinement and interactive editing tools
for professional-grade image segmentation with DXF export.

Features:
- Interactive region and point selection
- Automatic segmentation with SAM
- Contour refinement with CRF
- Interactive contour editing and manipulation
- Custom color assignment for segments
- Advanced shape detection
- DXF export with precise line thickness
- Support for large images through automatic tiling
"""

import logging
import os
import sys
import time
import datetime
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, LassoSelector, Button
from matplotlib.path import Path
import torch
import ezdxf
from tqdm import tqdm
from scipy import ndimage
from skimage.feature import peak_local_max, canny
from skimage.segmentation import watershed
from skimage.morphology import disk, closing, opening, dilation, skeletonize
from skimage.filters import threshold_otsu, threshold_local
from scipy.spatial import ConvexHull
from PIL import Image
import tkinter as tk
from tkinter import filedialog, colorchooser

# Constants
CURRENT_USER = "FETHl"
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
VERSION = "5.0.0"  # Combined version

# SAM Model Configuration
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image splitting configuration
TILE_SIZE = 512
MAX_SIZE_WITHOUT_SPLITTING = 1024

def custom_peak_local_max(image, min_distance=1, labels=None):
    """
    Compatible version of peak_local_max that works with newer scikit-image versions
    """
    from skimage.feature import peak_local_max as skimage_peak_local_max
    
    # Use the function with the new parameters
    coordinates = skimage_peak_local_max(image, min_distance=min_distance, 
                                        exclude_border=False)
    
    # Create a mask from coordinates
    mask = np.zeros_like(image, dtype=bool)
    if len(coordinates) > 0:
        mask[tuple(coordinates.T)] = True
    
    # If labels is provided, limit to regions in labels
    if labels is not None:
        mask = mask & (labels > 0)
    
    return mask

def custom_anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, option=1):
    """
    Anisotropic diffusion filter to preserve edges while reducing noise
    
    Parameters:
    -----------
    img : ndarray
        Input image
    niter : int
        Number of iterations
    kappa : float
        Conduction coefficient, controls the sensitivity to edges
    gamma : float
        Rate of diffusion (0 <= gamma <= 1)
    option : int
        1 for Perona-Malik diffusion equation No. 1, 
        2 for Perona-Malik diffusion equation No. 2
    
    Returns:
    --------
    filtered : ndarray
        Filtered image
    """
    # Convert to float
    img = img.astype(np.float32)
    
    # Initialize output
    filtered = img.copy()
    
    # Center pixel distance
    dx = 1
    dy = 1
    dd = np.sqrt(2)
    
    # 2D finite difference scheme [N, S, E, W, NE, SE, SW, NW]
    flow_weight = np.array([
        [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
    ])
    
    flow_dist = np.array([dy, dy, dx, dx, dd, dd, dd, dd])
    
    # Run iterations
    for _ in range(niter):
        # Calculate 8 different flows
        flows = np.zeros((8, *filtered.shape))
        
        for i in range(8):
            # Convolution with discrete 2D flow kernel
            flows[i] = ndimage.convolve(filtered, flow_weight[i])
            
            # Calculate flow gradients
            if option == 1:
                # Perona-Malik equation 1
                flows[i] = np.exp(-(flows[i]/kappa)**2) * flows[i]
            elif option == 2:
                # Perona-Malik equation 2
                flows[i] = flows[i] / (1 + (flows[i]/kappa)**2)
            
            # Scale based on distance
            flows[i] /= flow_dist[i]
        
        # Update image with all flows
        filtered += gamma * np.sum(flows, axis=0)
    
    return filtered

def point_to_line_distance(x, y, x1, y1, x2, y2):
    """Calculate the distance from a point (x, y) to a line defined by two points (x1, y1) and (x2, y2)"""
    # Length of the line
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if line_length == 0:
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        
    # Calculate the normalized distance from the point to the line
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length ** 2)
    
    # Limit t to [0, 1] to stay on the segment
    t = max(0, min(1, t))
    
    # Find the projection point
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    # Calculate the distance to the projection point
    return np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

class InteractiveRegionSelector:
    """
    Interactive interface for selecting regions of interest in an image
    """
    def __init__(self):
        self.selections = []
        self.current_mask = None
        self.selection_mode = 'rectangle'  # 'rectangle', 'lasso', or 'point'
        self.fig = None
        self.ax = None
        self.image = None
        self.mask = None
        self.canvas = None
        self.selector = None
        self.point_size = 20  # Size of points for point selection
        
    def select_regions(self, image, title="Select Regions of Interest"):
        """
        Open an interactive window for region selection
        
        Args:
            image: Input image (RGB or grayscale)
            title: Window title
        
        Returns:
            mask: Binary mask of selected regions
        """
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_img = image.copy()
        
        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle(title, fontsize=16)
        
        # Store image dimensions
        self.height, self.width = image.shape[:2]
        self.image = display_img
        
        # Initialize mask
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Display the image
        self.ax.imshow(display_img)
        
        # Create selector based on mode
        self._setup_rectangle_selector()
        
        # Add buttons for different selection modes and actions
        ax_rect = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_lasso = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_clear = plt.axes([0.59, 0.05, 0.1, 0.075])
        ax_done = plt.axes([0.92, 0.05, 0.07, 0.075])
        
        btn_rect = Button(ax_rect, 'Rectangle')
        btn_lasso = Button(ax_lasso, 'Lasso')
        btn_clear = Button(ax_clear, 'Clear')
        btn_done = Button(ax_done, 'Done')
        
        btn_rect.on_clicked(self._switch_to_rectangle)
        btn_lasso.on_clicked(self._switch_to_lasso)
        btn_clear.on_clicked(self._clear_selection)
        btn_done.on_clicked(self._finish_selection)
        
        # Instructions text
        plt.figtext(0.5, 0.01, "Use Rectangle or Lasso tools to select regions. Press 'Done' when finished.",
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Show the plot in non-blocking mode
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for buttons
        self.canvas = self.fig.canvas
        plt.show(block=True)  # Block until window is closed
        
        return self.mask
        
    def _setup_rectangle_selector(self):
        """Set up rectangle selection tool"""
        if self.selector:
            self.selector.disconnect_events()
        
        self.selector = RectangleSelector(
            self.ax, 
            self._on_rectangle_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, 
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.selection_mode = 'rectangle'
        
    def _setup_lasso_selector(self):
        """Set up lasso selection tool"""
        if self.selector:
            self.selector.disconnect_events()
            
        self.selector = LassoSelector(
            self.ax,
            self._on_lasso_select,
            button=[1]  # Left mouse button
        )
        self.selection_mode = 'lasso'
        
    def _on_rectangle_select(self, eclick, erelease):
        """Handle rectangle selection event"""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        # Create mask for this selection
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_mask[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = 1
        
        # Add to overall mask
        self.mask = cv2.bitwise_or(self.mask, self.current_mask)
        
        # Show the current selection
        self._update_display()
        
    def _on_lasso_select(self, vertices):
        """Handle lasso selection event"""
        # Convert vertices to numpy array
        vertices = np.array(vertices)
        if len(vertices) < 3:  # Need at least 3 points for a polygon
            return
            
        # Create a mask for this selection
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create a grid of points
        y, x = np.mgrid[:self.height, :self.width]
        points = np.vstack((x.flatten(), y.flatten())).T
        
        # Create a Path from vertices
        path = Path(vertices)
        
        # Find points inside the path
        mask = path.contains_points(points)
        mask = mask.reshape(self.height, self.width)
        
        # Convert to uint8
        self.current_mask = mask.astype(np.uint8)
        
        # Add to overall mask
        self.mask = cv2.bitwise_or(self.mask, self.current_mask)
        
        # Show the current selection
        self._update_display()
        
    def _update_display(self):
        """Update the display with current selection mask"""
        # Create an overlay showing selection
        overlay = self.image.copy()
        
        # Show mask areas in semi-transparent highlight
        highlight = np.zeros_like(overlay)
        highlight[self.mask == 1] = [0, 255, 0]  # Green for selected areas
        
        # Blend with original image
        cv2.addWeighted(highlight, 0.4, overlay, 1, 0, overlay)
        
        # Update display
        self.ax.clear()
        self.ax.imshow(overlay)
        self.canvas.draw_idle()
        
    def _switch_to_rectangle(self, event):
        """Switch to rectangle selection mode"""
        self._setup_rectangle_selector()
        
    def _switch_to_lasso(self, event):
        """Switch to lasso selection mode"""
        self._setup_lasso_selector()
        
    def _clear_selection(self, event):
        """Clear current selection"""
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.ax.clear()
        self.ax.imshow(self.image)
        self.canvas.draw_idle()
        
    def _finish_selection(self, event):
        """Finish selection process"""
        plt.close(self.fig)


class InteractiveFileSelector:
    """
    Class for interactive file selection via file dialog
    """
    def __init__(self):
        self.selected_file = None
        self.file_types = [
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("Bitmap files", "*.bmp"),
            ("TIFF files", "*.tif;*.tiff"),
            ("All files", "*.*")
        ]
    
    def select_file(self, title="Select an Image File"):
        """
        Open a file dialog to select an image file
        
        Args:
            title: Dialog window title
            
        Returns:
            Selected file path or None if canceled
        """
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=self.file_types
        )
        
        # Store the selected file path
        self.selected_file = file_path if file_path else None
        
        # Destroy the root window
        root.destroy()
        
        return self.selected_file


class ImageTileSplitter:
    """
    Class to split large images into tiles for processing
    """
    def __init__(self, tile_size=512, max_size=1024, debug=False):
        """
        Initialize the tile splitter
        
        Args:
            tile_size: Size of tiles (width and height)
            max_size: Maximum size before splitting is required
            debug: Whether to print debug information
        """
        self.tile_size = tile_size
        self.max_size = max_size
        self.debug = debug
        
    def should_split(self, image):
        """Check if image should be split into tiles"""
        h, w = image.shape[:2]
        return h > self.max_size or w > self.max_size
    
    def split_image(self, image):
        """
        Split image into tiles of tile_size x tile_size
        
        Args:
            image: Image to split (numpy array)
            
        Returns:
            List of tuples (tile, (x_offset, y_offset))
        """
        h, w = image.shape[:2]
        
        if not self.should_split(image):
            return [(image, (0, 0))]
            
        if self.debug:
            print(f"Splitting image of size {w}x{h} into tiles of {self.tile_size}x{self.tile_size}")
            
        tiles = []
        
        # Calculate number of tiles in each dimension
        n_tiles_x = (w + self.tile_size - 1) // self.tile_size
        n_tiles_y = (h + self.tile_size - 1) // self.tile_size
        
        if self.debug:
            print(f"Creating {n_tiles_x * n_tiles_y} tiles ({n_tiles_x}x{n_tiles_y})")
            
        # Split the image into tiles
        for y in range(n_tiles_y):
            for x in range(n_tiles_x):
                # Calculate tile coordinates
                x_start = x * self.tile_size
                y_start = y * self.tile_size
                x_end = min(x_start + self.tile_size, w)
                y_end = min(y_start + self.tile_size, h)
                
                # Extract the tile
                tile = image[y_start:y_end, x_start:x_end]
                
                # Store the tile along with its position
                tiles.append((tile, (x_start, y_start)))
                
        return tiles
    
    def reassemble_segments(self, tile_segments, original_image_shape):
        """
        Reassemble segmented tiles into a full image
        
        Args:
            tile_segments: List of tuples (segments, (x_offset, y_offset))
            original_image_shape: Shape of the original image
            
        Returns:
            Reassembled list of segments
        """
        if self.debug:
            print("Reassembling segments from tiles...")
            
        h, w = original_image_shape[:2]
        all_segments = []
        segment_id_counter = 1
        
        for segments, (x_offset, y_offset) in tile_segments:
            if not segments:
                continue
                
            # Adjust segment data to account for tile position
            for segment in segments:
                # Adjust ID to avoid duplicates
                segment['id'] = segment_id_counter
                segment_id_counter += 1
                
                # Adjust contour coordinates
                if 'contour' in segment:
                    segment['contour'][:,:,0] += x_offset
                    segment['contour'][:,:,1] += y_offset
                    
                # Adjust smoothed contour coordinates
                if 'smoothed_contour' in segment:
                    segment['smoothed_contour'][:,:,0] += x_offset
                    segment['smoothed_contour'][:,:,1] += y_offset
                    
                # Adjust position
                if 'position' in segment:
                    x, y = segment['position']
                    segment['position'] = (x + x_offset, y + y_offset)
                    
                # Adjust center
                if 'center' in segment:
                    cx, cy = segment['center']
                    segment['center'] = (cx + x_offset, cy + y_offset)
                    
                # Adjust mask
                if 'mask' in segment:
                    # Create full-size mask
                    full_mask = np.zeros((h, w), dtype=segment['mask'].dtype)
                    
                    # Calculate where to place this tile's mask
                    tile_h, tile_w = segment['mask'].shape
                    full_mask[y_offset:y_offset+tile_h, x_offset:x_offset+tile_w] = segment['mask']
                    segment['mask'] = full_mask
                    
                # Add the adjusted segment
                all_segments.append(segment)
                
        if self.debug:
            print(f"Reassembled {len(all_segments)} segments from all tiles.")
            
        return all_segments


class CRFPostProcessor:
    """
    Class for post-processing masks with CRF (Conditional Random Fields)
    Enhanced: Pre-filtering with Gaussian for better artifact reduction
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.crf_available = self._check_crf()
        
    def _check_crf(self):
        """Check if PyDenseCRF is installed"""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
            return True
        except ImportError:
            print("WARNING: PyDenseCRF not installed. Using alternative.")
            print("To install: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")
            return False
    
    def apply_crf(self, image, mask, crf_params=None):
        """
        Refine mask boundaries using DenseCRF
        Enhanced version with Gaussian pre-filtering and optimized parameters
        """
        if not self.crf_available:
            return self.apply_crf_alternative(image, mask)
            
        if crf_params is None:
            # Optimized parameters for better segmentation
            crf_params = {
                'gaussian_sxy': 3,
                'gaussian_compat': 4,  # Increased for stronger smoothing
                'bilateral_sxy': 60,   # Reduced for better respect of edges
                'bilateral_srgb': 10,  # Reduced for more sensitivity to color changes
                'bilateral_compat': 12, # Increased to strengthen the importance of colors
                'iterations': 10
            }
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
            
            # Gaussian pre-filtering to reduce noise (ENHANCEMENT)
            smoothed_image = cv2.GaussianBlur(image, (3, 3), 0.5)
            
            # Convert inputs to proper format and ensure contiguity
            smoothed_image = np.ascontiguousarray(smoothed_image)
            if mask.dtype not in [np.bool_, np.uint8, np.int32, np.int64]:
                # Convertir en entier
                mask = mask.astype(np.uint8)
            
            h, w = image.shape[:2]
            
            # Initialize CRF
            if self.debug:
                print("Applying DenseCRF with pre-filtering to refine edges...")
            
            # Create proper unary potential with safeguards
            # Create label array with proper binary encoding (0 for background, 1 for foreground)
            # This is critical to avoid the index out-of-bounds issue
            labels = np.zeros((2, h, w), dtype=np.float32)
            
            # Safety check for mask values
            safe_mask = np.clip(mask, 0, 1)  # Ensure mask has only 0 and 1 values
            
            # Set probabilities for background and foreground
            labels[0, safe_mask == 0] = 0.9  # Background probability where mask is 0 (increased)
            labels[1, safe_mask == 1] = 0.9  # Foreground probability where mask is 1 (increased)
            
            # Create CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes (bg, fg)
            
            # Create unary potentials from labels
            unary = unary_from_labels(labels, 2, gt_prob=0.8)  # Increased for more confidence
            d.setUnaryEnergy(unary)

            # Add pairwise Gaussian potentials (spatial constraint)
            d.addPairwiseGaussian(
                sxy=(crf_params['gaussian_sxy'], crf_params['gaussian_sxy']),
                compat=crf_params['gaussian_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )

            # Add pairwise bilateral potentials (color-dependent)
            # Ensure image is in the correct format for CRF
            if smoothed_image.dtype != np.uint8:
                proc_image = np.clip(smoothed_image * 255, 0, 255).astype(np.uint8)
            else:
                proc_image = smoothed_image
                
            d.addPairwiseBilateral(
                sxy=(crf_params['bilateral_sxy'], crf_params['bilateral_sxy']),
                srgb=(crf_params['bilateral_srgb'],) * 3,
                rgbim=proc_image,
                compat=crf_params['bilateral_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )

            # Perform inference
            Q = d.inference(crf_params['iterations'])
            
            # Get MAP (maximum a posteriori) estimate
            map_soln = np.argmax(Q, axis=0).reshape((h, w))
            
            # Post-processing to remove small artifacts (ENHANCEMENT)
            result = map_soln.astype(np.uint8)
            
            # Remove isolated small regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=8)
            min_size = 20  # Minimum size of regions to keep
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    result[labels == i] = 0
            
            return result

        except Exception as e:
            print(f"Error applying CRF: {e}")
            # Fallback to alternative method in case of error
            return self.apply_crf_alternative(image, mask)

    def apply_crf_alternative(self, image, mask, sigma=1.0, radius=2):
        """
        Enhanced alternative to CRF using advanced image processing techniques
        Incorporates anisotropic filtering for better edge preservation
        """
        if self.debug:
            print("Using enhanced alternative to CRF for refining edges...")
        
        try:
            # Convert to grayscale for gradient
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
            
            # ENHANCEMENT: Anisotropic filtering to preserve edges
            # This filter reduces noise while preserving important edges
            aniso_filtered = custom_anisotropic_diffusion(gray, niter=10, kappa=20, gamma=0.1, option=1)
            aniso_filtered = (aniso_filtered * 255).astype(np.uint8)
            
            # Adaptive Gaussian smoothing to reduce noise
            smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
            
            # ENHANCEMENT: Adaptive thresholding with optimized parameters
            binary = (smoothed > 0.5).astype(np.uint8)
            
            # Improved morphological operations
            # Using opening/closing operations with refined parameters
            disk_kernel = disk(radius)
            refined = closing(binary, disk_kernel)  # Close small holes
            refined = opening(refined, disk(1))     # Remove small artifacts
            
            # Convert to appropriate format for OpenCV
            refined = (refined * 255).astype(np.uint8)
            
            # Remove isolated small regions with finer analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
            
            # Statistical analysis to automatically determine minimum size
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                if len(areas) > 0:
                    median_area = np.median(areas)
                    min_size = max(50, int(median_area * 0.1))  # Dynamic, based on median
                else:
                    min_size = 50
            else:
                min_size = 50
                
            refined_mask = np.zeros_like(refined)
            
            # Consider component 0 as background
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    refined_mask[labels == i] = 255
            
            # Convert to binary (0-1)
            refined_mask = (refined_mask > 0).astype(np.uint8)
                    
            if self.debug:
                print("Enhanced alternative edge refinement completed")
                
            return refined_mask
        
        except Exception as e:
            print(f"Error applying alternative to CRF: {e}")
            return mask  # Return original mask in case of error


class SegmentAnythingHelper:
    """
    Helper class to integrate Meta's Segment Anything Model (SAM)
    Enhanced: Better point-of-interest selection and mask filtering
    """
    def __init__(self, checkpoint_path=None, device=None):
        """Initialize the SAM model"""
        self.model = None
        self.predictor = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Only load if checkpoint is provided and exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.initialize_model(checkpoint_path)
        else:
            print("SAM checkpoint not found. Using alternative segmentation methods.")
    
    def initialize_model(self, checkpoint_path):
        """Load the SAM model"""
        try:
            print("Initializing SAM model...")
            # Import SAM modules here to avoid dependency issues if not available
            from segment_anything import sam_model_registry, SamPredictor
            
            # Determine model type from filename
            if "vit_b" in checkpoint_path:
                model_type = "vit_b"
            elif "vit_l" in checkpoint_path:
                model_type = "vit_l"
            else:
                model_type = "vit_h"  # Default to highest quality
                
            # Load SAM model
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            print(f"SAM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error initializing SAM model: {str(e)}")
            self.model = None
            self.predictor = None
            return False
    
    def is_initialized(self):
        """Check if the model is initialized"""
        return self.predictor is not None
    
    def generate_masks_from_points(self, image, points, labels=None):
        """
        Generate masks using SAM with user-provided points
        
        Args:
            image: Input RGB image
            points: List of (x, y) point coordinates
            labels: List of labels (1 for foreground, 0 for background)
                   If None, all points are treated as foreground
        
        Returns:
            List of masks, scores, and logits
        """
        if not self.is_initialized():
            return None
            
        try:
            # Set the image for embedding
            self.predictor.set_image(image)
            
            # Convert points to numpy arrays
            input_points = np.array(points)
            
            # Default to foreground points if no labels provided
            if labels is None:
                input_labels = np.ones(len(points))
            else:
                input_labels = np.array(labels)
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            return masks, scores, logits
        
        except Exception as e:
            print(f"Error generating SAM masks from points: {str(e)}")
            return None
    
    def generate_masks(self, image, num_points=10):
        """Generate masks using the SAM model with intelligent automatic points"""
        if not self.is_initialized():
            return None
            
        try:
            # Set the image for embedding
            self.predictor.set_image(image)
            
            # ENHANCEMENT: More intelligent point-of-interest generation
            height, width = image.shape[:2]
            points = []
            
            # Convert to grayscale for point-of-interest detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Use Harris corners to find interesting points
            corners = cv2.cornerHarris(np.float32(gray), 5, 3, 0.04)
            corners = cv2.dilate(corners, None)
            # Threshold to find the best corners
            threshold = 0.01 * corners.max()
            corner_coords = np.where(corners > threshold)
            
            # Limit the number of corners if needed
            max_corners = min(20, len(corner_coords[0]))
            if len(corner_coords[0]) > 0:
                # Sample points from detected corners
                indices = np.random.choice(len(corner_coords[0]), size=min(max_corners, len(corner_coords[0])), replace=False)
                for idx in indices:
                    y, x = corner_coords[0][idx], corner_coords[1][idx]
                    points.append([x, y])
            
            # Complete with regular grid sampling for full coverage
            if len(points) < num_points:
                grid_points = []
                rows, cols = 4, 4  # 4x4 grid for better coverage
                for i in range(rows):
                    for j in range(cols):
                        x = int(width * (j + 0.5) / cols)
                        y = int(height * (i + 0.5) / rows)
                        grid_points.append([x, y])
                
                # Add grid points until reaching num_points
                remaining = num_points - len(points)
                if remaining > 0:
                    # Randomly select from grid points
                    grid_indices = np.random.choice(len(grid_points), size=min(remaining, len(grid_points)), replace=False)
                    for idx in grid_indices:
                        points.append(grid_points[idx])
                        
            # Convert points to numpy arrays
            input_points = np.array(points)
            input_labels = np.ones(len(points))  # Assume all foreground
            
            # Generate masks with multiple returns for better selection
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
                return_logits=True
            )
            
            # ENHANCEMENT: More sophisticated filtering of generated masks
            good_masks = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Enhanced quality criteria
                if score < 0.8:  # Confidence threshold
                    continue
                    
                # Evaluate mask regularity
                contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:  # No contour found
                    continue
                    
                # Analyze main contour
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate circularity - indicator of shape regularity
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Filter out masks that are too small or too irregular
                min_area = 100  # Minimum area in pixels
                min_circularity = 0.2  # Not too irregular
                
                if area < min_area or circularity < min_circularity:
                    continue
                
                # Add to list of good masks
                good_masks.append(mask)
            
            return good_masks
        
        except Exception as e:
            print(f"Error generating SAM masks: {str(e)}")
            return None
    
    def automatic_segmentation(self, image):
        """
        Perform fully automatic segmentation with SAM
        Enhanced: optimized parameters for better quality
        """
        if not self.is_initialized():
            return None
            
        try:
            # Import for automatic mask generation
            from segment_anything import SamAutomaticMaskGenerator
            
            # Get the SAM model from predictor
            sam = self.predictor.model
            
            # ENHANCEMENT: Optimized parameters for better segmentation
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,      # Increased for better coverage
                pred_iou_thresh=0.88,    # Higher IoU threshold for better quality
                stability_score_thresh=0.94, # Higher stability threshold
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Minimum area to reduce noise
                output_mode="coco_rle",    # More compact and efficient format
            )
            
            # Generate masks
            masks = mask_generator.generate(image)
            
            # ENHANCEMENT: Filter masks with more advanced criteria
            filtered_masks = []
            for mask_data in masks:
                # Check quality criteria
                if (
                    mask_data["area"] >= 100 and  # Minimum area
                    mask_data["stability_score"] >= 0.85 and  # Good stability
                    mask_data["predicted_iou"] >= 0.7  # Good predicted IoU
                ):
                    filtered_masks.append(mask_data)
            
            # Sort by quality (stability * predicted IoU)
            filtered_masks.sort(
                key=lambda x: x["stability_score"] * x["predicted_iou"],
                reverse=True
            )
            
            return filtered_masks
            
        except Exception as e:
            print(f"Error in automatic SAM segmentation: {str(e)}")
            return None


class InteractiveSegmentationTool:
    """
    Interactive tool for point selection and contour manipulation
    """
    def __init__(self, sam_helper=None, crf_processor=None, debug=False):
        self.sam_helper = sam_helper
        self.crf_processor = crf_processor
        self.debug = debug
        
        # Interactive session state
        self.image = None
        self.points = []
        self.labels = []
        self.working_mask = None
        self.current_contours = []
        self.selected_contour_idx = -1
        self.fig = None
        self.ax = None
        self.last_visualization = None
        self.interactive_mode = False
        
        # Colors for contours (RGB color codes)
        self.default_contour_colors = [
            'cyan',       # 1
            'yellow',     # 2
            'magenta',    # 3
            'lime',       # 4
            'red',        # 5
            'blue',       # 6
            'orange',     # 7
            'purple',     # 8
            'pink',       # 9
            'brown',      # 0
        ]
        
        # Contours and their associated colors
        self.contour_colors = {}  # {contour_idx: color}
        self.polygon_editing_mode = False
        self.edited_polygon = None
        self.edited_polygon_idx = -1
        self.dragged_vertex_idx = -1
        self.vertex_drag_threshold = 10  # pixels
        
    def set_image(self, image):
        """
        Set the image to segment
        """
        self.image = image.copy()
        self.reset_session()
        
    def reset_session(self):
        """
        Reset the interactive session
        """
        self.points = []
        self.labels = []
        self.working_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.current_contours = []
        self.selected_contour_idx = -1
        
    def add_point(self, x, y, is_object=True):
        """
        Add a point of interest (foreground or background)
        """
        self.points.append([x, y])
        self.labels.append(1 if is_object else 0)
        return len(self.points)
        
    def remove_last_point(self):
        """
        Remove the last added point
        """
        if self.points:
            self.points.pop()
            self.labels.pop()
        return len(self.points)
        
    def generate_mask_from_points(self):
        """
        Generate a mask from selected points
        """
        if not self.sam_helper or not self.sam_helper.is_initialized():
            print("SAM not initialized. Cannot generate mask.")
            return None
            
        if not self.points:
            print("No points selected.")
            return None
            
        # Generate masks with SAM
        masks, scores, _ = self.sam_helper.generate_masks_from_points(
            self.image, self.points, self.labels
        )
        
        if masks is None or len(masks) == 0:
            print("Failed to generate mask.")
            return None
            
        # Take the mask with the best score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(np.uint8)
        
        # Apply CRF to refine edges
        refined_mask = self.crf_processor.apply_crf(self.image, mask)
        self.working_mask = refined_mask.astype(np.uint8)
        
        # Update contours
        self._update_contours()
        
        return self.working_mask
    
    def _update_contours(self):
        """
        Update the list of contours from the current mask
        """
        # Find contours
        contours, _ = cv2.findContours(
            self.working_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        # Filter tiny contours
        self.current_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        self.selected_contour_idx = -1
        
        return len(self.current_contours)
    
    def select_contour_at_point(self, x, y):
        """
        Select a contour by clicking near or on it
        """
        min_dist = float('inf')
        selected_idx = -1
        
        for i, contour in enumerate(self.current_contours):
            # Calculate minimum distance from point to contour
            dist = cv2.pointPolygonTest(contour, (x, y), True)
            dist = abs(dist)  # Signed distance (negative = outside)
            
            # If the point is on contour or within 10 pixels
            if dist < min_dist and dist < 10:
                min_dist = dist
                selected_idx = i
                
        self.selected_contour_idx = selected_idx
        return selected_idx
    
    def remove_selected_contour(self):
        """
        Remove the selected contour from the mask
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Create a mask for this specific contour
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Remove this contour from the global mask
        self.working_mask[contour_mask > 0] = 0
        
        # Update contours
        self._update_contours()
        
        return True
    
    def remove_contours_outside_roi(self, roi_x, roi_y, roi_width, roi_height):
        """
        Remove all contours outside a region of interest
        """
        if not self.current_contours:
            return False
            
        # Create ROI mask
        roi_mask = np.zeros_like(self.working_mask)
        roi_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = 1
        
        # Filter contours
        kept_mask = np.zeros_like(self.working_mask)
        for contour in self.current_contours:
            # Calculate contour center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                
                # If center is in ROI, keep this contour
                if roi_mask[center_y, center_x] > 0:
                    cv2.drawContours(kept_mask, [contour], 0, 1, -1)
        
        # Update working mask
        self.working_mask = kept_mask
        
        # Update contours
        self._update_contours()
        
        return True
    
    def split_contour(self, x1, y1, x2, y2, line_thickness=3):
        """
        Split a contour by drawing a line between two points
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Create mask for selected contour
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Draw line to split contour
        split_mask = contour_mask.copy()
        cv2.line(split_mask, (x1, y1), (x2, y2), 0, thickness=line_thickness)
        
        # Update working mask
        self.working_mask = np.logical_and(self.working_mask, split_mask).astype(np.uint8)
        
        # Update contours
        self._update_contours()
        
        return True
    
    def smooth_selected_contour(self, epsilon_factor=0.002):
        """
        Smooth the selected contour
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Get contour
        contour = self.current_contours[self.selected_contour_idx]
        
        # Smooth contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Create mask for this contour
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Create mask for smoothed contour
        smooth_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(smooth_mask, [smoothed_contour], 0, 1, -1)
        
        # Replace original contour with smoothed contour in mask
        self.working_mask[contour_mask > 0] = 0  # Remove old contour
        self.working_mask[smooth_mask > 0] = 1   # Add new contour
        
        # Update contours
        self._update_contours()
        
        return True
    
    def set_contour_color(self, contour_idx, color):
        """Change the color of a specific contour"""
        if contour_idx >= 0 and contour_idx < len(self.current_contours):
            self.contour_colors[contour_idx] = color
            print(f"Color of contour {contour_idx} changed to {color}")
            return True
        return False
    
    def visualize_interactive_state(self, title="Interactive Selection", figsize=(12, 10)):
        """
        Visualize the current state of interactive selection
        """
        if self.image is None:
            return None
        
        plt.figure(figsize=figsize)
        
        # Original image with points and contours
        plt.subplot(221)
        plt.imshow(self.image)
        plt.title("Image with selection points")
        
        # Show points
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = 'g' if label == 1 else 'r'
            plt.plot(point[0], point[1], color+'o', markersize=8)
            plt.text(point[0]+5, point[1]+5, str(i), color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.7))
        
        # Current mask
        plt.subplot(222)
        plt.imshow(self.working_mask * 255, cmap='gray')
        plt.title("Segmentation mask")
        
        # Contours on original image
        plt.subplot(223)
        overlay = self.image.copy()
        
        # Draw all contours
        for i, contour in enumerate(self.current_contours):
            color = (0, 255, 0)  # Green for normal contours
            thickness = 2
            
            # Selected contour in red and thicker
            if i == self.selected_contour_idx:
                color = (255, 0, 0)  # Red
                thickness = 3
                
            cv2.drawContours(overlay, [contour], 0, color, thickness)
            
            # Add contour number
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(i), (cX, cY),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Detected contours")
        
        # Combined view with transparency
        plt.subplot(224)
        combined = self.image.copy()
        
        # Create colored mask
        mask_colored = np.zeros_like(combined)
        mask_colored[self.working_mask > 0] = [0, 255, 0]  # Semi-transparent green
        
        # Merge image and mask
        alpha = 0.5
        combined = cv2.addWeighted(combined, 1.0, mask_colored, alpha, 0)
        
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Segmentation result")
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return plt.gcf()
    
    def start_interactive_mode(self):
        """Start the fully interactive GUI mode"""
        if self.image is None:
            print("No image loaded")
            return False
            
        self.interactive_mode = True
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.image)
        self.ax.set_title("Interactive Segmentation Mode")
        
        # Instructions text
        instructions = (
            "Left click: Add object point\n"
            "Right click: Add background point\n"
            "Middle click: Select/deselect contour\n"
            "'p': Enter/exit polygon editing mode\n"
            "'g': Generate mask from points\n"
            "'s': Smooth selected contour\n"
            "'r': Remove selected contour\n"
            "'c': Clear all points\n"
            "'1-9,0': Change color of selected contour\n"
            "'q': Quit and save results\n\n"
            "In polygon edit mode:\n"
            "- Left click: drag vertices\n"
            "- Double-click: add vertex\n"
            "- Right click: delete vertex"
        )
        # Add instructions text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes, 
                    fontsize=9, verticalalignment='top', bbox=props)
        
        # Show user and date info at bottom of interface
        info_text = f"User: {CURRENT_USER} | Date: {CURRENT_DATE} | Version: {VERSION}"
        self.ax.text(0.02, 0.01, info_text, transform=self.ax.transAxes, 
                    fontsize=8, color='white', alpha=0.7,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        # Setup event handlers
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        plt.tight_layout()
        plt.show()
        
        return True

    def _on_click(self, event):
        """Mouse click event handler"""
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Polygon editing mode
        if self.polygon_editing_mode:
            # Left button: move a vertex
            if event.button == 1:
                # Check for double-click to add vertex
                if event.dblclick:
                    # Find closest edge
                    edge1, edge2 = self.find_nearest_edge(x, y)
                    if edge1 >= 0:
                        # Add vertex on this edge
                        self.add_vertex_at_edge(edge1, edge2)
                        print(f"Vertex added between vertices {edge1} and {edge2}")
                else:
                    # Find closest vertex
                    vertex_idx = self.find_nearest_vertex(x, y)
                    if vertex_idx >= 0:
                        # Start dragging this vertex
                        self.dragged_vertex_idx = vertex_idx
            
            # Right button: delete vertex
            elif event.button == 3:
                vertex_idx = self.find_nearest_vertex(x, y)
                if vertex_idx >= 0:
                    if self.delete_vertex(vertex_idx):
                        print(f"Vertex {vertex_idx} deleted")
        else:
            # Normal behavior
            if event.button == 1:
                self.add_point(x, y, is_object=True)
                
            elif event.button == 3:
                self.add_point(x, y, is_object=False)
                
            elif event.button == 2:
                if self.working_mask is not None:
                    self.select_contour_at_point(x, y)
                    
        self._update_interactive_view()

    def _on_key(self, event):
        """Keyboard event handler"""
        # 'p': Start/stop polygon editing
        if event.key == 'p':
            if self.polygon_editing_mode:
                self.stop_polygon_editing(apply_changes=True)
            elif self.selected_contour_idx >= 0:
                self.start_polygon_editing(self.selected_contour_idx)
        
        # 'escape': Cancel polygon editing
        elif event.key == 'escape':
            if self.polygon_editing_mode:
                self.stop_polygon_editing(apply_changes=False)
        
        # ... other existing keys ...
        elif event.key == 'g':
            self.generate_mask_from_points()
            
        elif event.key == 's':
            if self.selected_contour_idx >= 0:
                self.smooth_selected_contour()
                
        elif event.key == 'r':
            if self.selected_contour_idx >= 0:
                self.remove_selected_contour()
                
        elif event.key == 'c':
            self.reset_session()
                
        # Numeric keys 1-9,0 to change color
        elif event.key in '1234567890' and self.selected_contour_idx >= 0:
            color_idx = int(event.key) if event.key != '0' else 9
            if color_idx < len(self.default_contour_colors):
                self.set_contour_color(self.selected_contour_idx, self.default_contour_colors[color_idx])
                
        # 'q': Quit and save results
        elif event.key == 'q':
            plt.close(self.fig)
            return
            
        self._update_interactive_view()

    def _on_mouse_move(self, event):
        """Mouse movement event handler"""
        if not self.polygon_editing_mode or self.dragged_vertex_idx < 0:
            return
            
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Move the selected vertex
        if self.move_vertex(self.dragged_vertex_idx, x, y):
            self._update_interactive_view()

    def _on_mouse_release(self, event):
        """Mouse release event handler"""
        if self.dragged_vertex_idx >= 0:
            self.dragged_vertex_idx = -1

   

    def start_polygon_editing(self, contour_idx):
        """Start editing a polygon (contour)"""
        if contour_idx < 0 or contour_idx >= len(self.current_contours):
            print(f"Invalid contour {contour_idx} for editing")
            return False
            
        self.polygon_editing_mode = True
        self.edited_polygon_idx = contour_idx
        self.edited_polygon = self.current_contours[contour_idx].copy()
        self.dragged_vertex_idx = -1
        
        print(f"Polygon editing mode activated for contour {contour_idx}")
        print("Click on a vertex to move it, double-click to add/delete")
        return True
        
    def stop_polygon_editing(self, apply_changes=True):
        """Stop polygon editing"""
        if not self.polygon_editing_mode:
            return
            
        if apply_changes and self.edited_polygon_idx >= 0:
            # Apply changes to original contour
            self.current_contours[self.edited_polygon_idx] = self.edited_polygon.copy()
            # Update mask from modified contours
            self._update_mask_from_contours()
            print(f"Changes to polygon {self.edited_polygon_idx} applied")
        else:
            print("Editing canceled, no changes applied")
            
        # Reset editing state
        self.polygon_editing_mode = False
        self.edited_polygon = None
        self.edited_polygon_idx = -1
        self.dragged_vertex_idx = -1
    
    def _update_mask_from_contours(self):
        """Update the working mask based on current contours"""
        # Create a new mask
        new_mask = np.zeros_like(self.working_mask)
        
        # Draw all contours on the mask
        for contour in self.current_contours:
            cv2.drawContours(new_mask, [contour], 0, 1, -1)
            
        # Update the working mask
        self.working_mask = new_mask
        
        return True
        
    def find_nearest_vertex(self, x, y, max_dist=10):
        """Find the vertex closest to point (x, y)"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1
            
        # Get polygon points
        points = self.edited_polygon.reshape(-1, 2)
        
        # Calculate distances to point (x, y)
        distances = np.sqrt(np.sum((points - np.array([x, y])) ** 2, axis=1))
        
        # Find index of closest point
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # Check if distance is less than threshold
        if min_dist <= max_dist:
            return min_idx
        else:
            return -1
            
    def find_nearest_edge(self, x, y, max_dist=10):
        """Find the edge closest to point (x, y)"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1, -1
            
        # Get polygon points
        points = self.edited_polygon.reshape(-1, 2)
        n_points = len(points)
        
        min_dist = float('inf')
        nearest_edge = (-1, -1)  # (idx1, idx2)
        
        # Check each segment
        for i in range(n_points):
            j = (i + 1) % n_points
            
            # Calculate distance from point to segment
            p1 = points[i]
            p2 = points[j]
            
            # Point-line distance
            dist = point_to_line_distance(x, y, p1[0], p1[1], p2[0], p2[1])
            
            if dist < min_dist and dist <= max_dist:
                min_dist = dist
                nearest_edge = (i, j)
                
        return nearest_edge
        
    def add_vertex_at_edge(self, edge_idx1, edge_idx2):
        """Add a vertex in the middle of an edge"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1
            
        # Get polygon points
        points = self.edited_polygon.reshape(-1, 2)
        
        # Calculate midpoint
        p1 = points[edge_idx1]
        p2 = points[edge_idx2]
        new_point = (p1 + p2) / 2
        
        # Create a new array of points with the new point inserted
        new_points = np.insert(points, edge_idx2, new_point, axis=0)
        
        # Update the edited polygon
        self.edited_polygon = new_points.reshape(-1, 1, 2)
        
        return edge_idx2

    def delete_vertex(self, vertex_idx):
        """Delete a vertex from the polygon"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return False
            
        # Get polygon points
        points = self.edited_polygon.reshape(-1, 2)
        
        # Check that we have enough points
        if len(points) <= 3:  # A triangle is the minimum shape
            print("Cannot delete this vertex: polygon must have at least 3 vertices")
            return False
            
        # Delete the vertex
        new_points = np.delete(points, vertex_idx, axis=0)
        
        # Update the edited polygon
        self.edited_polygon = new_points.reshape(-1, 1, 2)
        
        return True

    def move_vertex(self, vertex_idx, new_x, new_y):
        """Move a vertex to a new position"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return False
            
        # Get polygon points
        points = self.edited_polygon.reshape(-1, 2)
        
        # Check index
        if vertex_idx < 0 or vertex_idx >= len(points):
            return False
            
        # Update position
        points[vertex_idx] = [new_x, new_y]
        
        # Update the edited polygon
        self.edited_polygon = points.reshape(-1, 1, 2)
        
        return True

    def choose_contour_color(self):
        """Open a color picker to choose a color for the selected contour"""
        if self.selected_contour_idx < 0:
            print("No contour selected")
            return False
            
        # Create root window
        root = tk.Tk()
        root.withdraw()
        
        # Open color chooser dialog
        color = colorchooser.askcolor(
            title="Choose Color for Contour",
            initialcolor="#00FFFF"  # Default to cyan
        )
        
        # Destroy root window
        root.destroy()
        
        if color[1]:  # If a color was chosen (not canceled)
            self.set_contour_color(self.selected_contour_idx, color[1])
            return True
            
        return False





    def _update_interactive_view(self):
        """Update the interactive view"""
        self.ax.clear()
        self.ax.imshow(self.image)
        
        # Show points
        fg_points = [p for p, l in zip(self.points, self.labels) if l == 1]
        bg_points = [p for p, l in zip(self.points, self.labels) if l == 0]
        
        if fg_points:
            fg_points = np.array(fg_points)
            self.ax.scatter(fg_points[:, 0], fg_points[:, 1], color='green', 
                        s=80, marker='o', label='Object')
                        
        if bg_points:
            bg_points = np.array(bg_points)
            self.ax.scatter(bg_points[:, 0], bg_points[:, 1], color='red',
                        s=80, marker='x', label='Background')
        
        # Show mask if available
        if self.working_mask is not None and not self.polygon_editing_mode:
            mask_display = np.zeros((*self.working_mask.shape, 4), dtype=np.float32)
            mask_display[self.working_mask > 0] = [0, 1, 0, 0.3]  # Semi-transparent green
            self.ax.imshow(mask_display)
        
        # Polygon editing mode
        if self.polygon_editing_mode:
            # Display only the polygon being edited
            points = self.edited_polygon.reshape(-1, 2)
            
            # Display edges
            self.ax.plot(points[:, 0], points[:, 1], 'o-', 
                        color='cyan', linewidth=2, markersize=6)
            
            # Close the polygon
            self.ax.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 
                    '-', color='cyan', linewidth=2)
                    
            # Highlight vertices
            self.ax.scatter(points[:, 0], points[:, 1], color='yellow', 
                        s=100, marker='o', alpha=0.7)
                        
            # Number the vertices
            for i, (x, y) in enumerate(points):
                self.ax.text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')
                
            # Display editing mode message
            self.ax.set_title(f"Polygon Editing Mode - Contour {self.edited_polygon_idx}")
            
            # Specific editing mode instructions
            edit_instructions = "Editing Mode: Drag=move, Double-click=add, Right-click=delete, 'p'/'Esc'=exit"
            self.ax.text(0.5, 0.02, edit_instructions, transform=self.ax.transAxes, 
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(facecolor='yellow', alpha=0.5, pad=2))
        else:
            # Normal display of contours
            for i, contour in enumerate(self.current_contours):
                # Set default color if not set before
                if i not in self.contour_colors:
                    if i == self.selected_contour_idx:
                        self.contour_colors[i] = 'yellow'  # Selected contour in yellow
                    else:
                        color_idx = i % len(self.default_contour_colors)
                        self.contour_colors[i] = self.default_contour_colors[color_idx]
                        
                color = self.contour_colors[i]
                linewidth = 2 if i == self.selected_contour_idx else 1
                contour_reshaped = contour.reshape(-1, 2)
                self.ax.plot(contour_reshaped[:, 0], contour_reshaped[:, 1], 
                            color=color, linewidth=linewidth)
                
                # Add contour index for easier reference
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.ax.text(cx, cy, str(i), color='white', 
                            fontsize=8, ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        # Instructions text
        if not self.polygon_editing_mode:
            instructions = (
                "Left click: Add object point\n"
                "Right click: Add background point\n"
                "Middle click: Select/deselect contour\n"
                "'p': Enter polygon editing mode\n"
                "'g': Generate mask from points\n"
                "'s': Smooth selected contour\n"
                "'r': Remove selected contour\n"
                "'c': Clear all points\n"
                "'1-9,0': Change color of selected contour\n"
                "'q': Quit and save results"
            )
            # Add instructions text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes, 
                        fontsize=9, verticalalignment='top', bbox=props)
        
        # Display user and date info at bottom of interface
        info_text = f"User: {CURRENT_USER} | Date: {CURRENT_DATE} | Version: {VERSION}"
        self.ax.text(0.02, 0.01, info_text, transform=self.ax.transAxes, 
                    fontsize=8, color='white', alpha=0.7,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
                    
        self.ax.set_title("Interactive Segmentation Mode" if not self.polygon_editing_mode else f"Editing Polygon {self.edited_polygon_idx}")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()


    def segment_with_multiple_layers(self, image_path):
        """
        Segmente l'image en plusieurs couches interactives
        """
        # Charger l'image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialiser les variables
        segments_by_layer = {}
        current_layer = 0
        all_segments = []
        
        while True:
            # Afficher image avec segments existants
            display_image = image_rgb.copy()
            
            # Dessiner les segments existants
            for layer_id, segments in segments_by_layer.items():
                color_map = {
                    0: (255, 0, 0),    # Rouge
                    1: (0, 0, 255),    # Bleu
                    2: (0, 255, 0),    # Vert
                    3: (255, 0, 255),  # Magenta
                    4: (255, 255, 0),  # Jaune
                }
                layer_color = color_map.get(layer_id, (255, 255, 255))
                
                # Dessiner tous les segments de cette couche
                for segment in segments:
                    cv2.drawContours(display_image, [segment['contour']], 0, layer_color, 2)
            
            # Afficher l'image pour la sélection de points
            plt.figure(figsize=(12, 10))
            plt.imshow(display_image)
            plt.title(f"Couche {current_layer+1}: Cliquez pour sélectionner des points de foreground (clic gauche) et background (clic droit)")
            plt.axis('off')
            
            # Collecte de points interactifs
            fg_points = []
            bg_points = []
            
            # Attendre les clics
            def onclick(event):
                if event.inaxes:
                    x, y = int(event.xdata), int(event.ydata)
                    if event.button == 1:  # Clic gauche = foreground
                        fg_points.append((x, y))
                        plt.plot(x, y, 'ro', markersize=8)
                    elif event.button == 3:  # Clic droit = background
                        bg_points.append((x, y))
                        plt.plot(x, y, 'bx', markersize=8)
                    plt.draw()
            
            # Connecter l'événement de clic
            cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            
            # Si aucun point, proposer de terminer
            if not fg_points and not bg_points:
                if len(segments_by_layer) > 0:
                    choice = input("Aucun point sélectionné. Terminer la segmentation? (o/n): ")
                    if choice.lower() == 'o':
                        break
                    else:
                        continue
                else:
                    print("Aucun point sélectionné et aucune couche créée. Veuillez sélectionner des points.")
                    continue
            
            # Effectuer la segmentation avec SAM pour cette couche
            print(f"Segmentation de la couche {current_layer+1} avec {len(fg_points)} points foreground et {len(bg_points)} points background...")
            
            # Préparer les points pour SAM
            input_points = np.array(fg_points) if fg_points else None
            input_labels = np.ones(len(fg_points)) if fg_points else None
            
            if bg_points:
                if input_points is None:
                    input_points = np.array(bg_points)
                    input_labels = np.zeros(len(bg_points))
                else:
                    input_points = np.vstack([input_points, np.array(bg_points)])
                    input_labels = np.concatenate([input_labels, np.zeros(len(bg_points))])
            
            # Appeler SAM pour segmenter
            masks, scores, logits = self.sam_helper.generate_masks(
                image_rgb,
                input_points,
                input_labels,
                multimask_output=True
            )
            
            # Traiter les résultats
            layer_segments = []
            for i, mask in enumerate(masks):
                mask_binary = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Ignorer les petits contours
                        continue
                        
                    # Créer un segment
                    segment = {
                        'contour': contour,
                        'mask': mask,
                        'area': area,
                        'layer': current_layer,
                        'score': scores[i]
                    }
                    
                    # Ajouter aux segments de cette couche
                    layer_segments.append(segment)
                    
                    # Ajouter à tous les segments
                    all_segments.append(segment)
            
            # Stocker les segments de cette couche
            segments_by_layer[current_layer] = layer_segments
            
            print(f"Couche {current_layer+1}: {len(layer_segments)} segments détectés.")
            
            # Demander s'il faut ajouter une nouvelle couche
            choice = input("Ajouter une nouvelle couche? (o/n): ")
            if choice.lower() != 'o':
                break
                
            # Passer à la couche suivante
            current_layer += 1
        
        return all_segments, segments_by_layer
    
class EnhancedDXFExporter:

    """
    Class to handle enhanced DXF export capabilities
    Enhanced: Line thicknesses adapted to AutoCAD standards and improved metadata
    """
    def __init__(self, line_thickness=0.35):
        """
        Initialize the DXF exporter with specified line thickness
        """
        self.line_thickness = line_thickness  # mm
        
        # ENHANCEMENT: Line thickness map adapted to AutoCAD standards
        self.thickness_map = {
            "CIRCLES": 0.5,        # Thicker for circles
            "ELLIPSES": 0.5,       # Thicker for ellipses
            "RECTANGLES": 0.35,    # Medium for rectangles
            "TRIANGLES": 0.35,     # Medium for triangles
            "POLYGONS": 0.35,      # Medium for polygons
            "LINES": 0.25,        # Thinner for lines
            "CONTOURS": 0.18,      # Thinnest for contours
            "COMPLEX_SHAPES": 0.3,  # Medium for complex shapes
            "ANNOTATIONS": 0.18,   # Thin for annotations
            "METADATA": 0.18       # Thin for metadata
        }
        
        # Standardized lineweights in AutoCAD - ENHANCEMENT: Complete list of standard thicknesses
        self.std_lineweights = {
            0.05: 5,    # 0.05mm
            0.09: 9,    # 0.09mm
            0.13: 13,   # 0.13mm
            0.15: 15,   # 0.15mm
            0.18: 18,   # 0.18mm
            0.20: 20,   # 0.20mm
            0.25: 25,   # 0.25mm
            0.30: 30,   # 0.30mm
            0.35: 35,   # 0.35mm
            0.40: 40,   # 0.40mm
            0.50: 50,   # 0.50mm
            0.53: 53,   # 0.53mm
            0.60: 60,   # 0.60mm
            0.70: 70,   # 0.70mm
            0.80: 80,   # 0.80mm
            0.90: 90,   # 0.90mm
            1.00: 100,  # 1.00mm
            1.06: 106,  # 1.06mm
            1.20: 120,  # 1.20mm
            1.40: 140,  # 1.40mm
            1.58: 158,  # 1.58mm
            2.00: 200,  # 2.00mm
            2.11: 211   # 2.11mm
        }
    
    def get_closest_std_lineweight(self, thickness):
        """Get the closest standard lineweight from a float value"""
        closest = min(self.std_lineweights.keys(), key=lambda x: abs(x - thickness))
        return self.std_lineweights[closest]
    
    def setup_layers(self, doc):
        """
        Set up layers with enhanced properties and line thicknesses
        ENHANCEMENT: More detailed descriptions and optimized layer properties
        """
        # Enhanced layer definitions with descriptions and lineweights
        layer_definitions = {
            "CIRCLES": {"color": 1, "desc": "Circular shapes - Exact geometries", 
                       "linetype": "CONTINUOUS", "lineweight": self.thickness_map["CIRCLES"]},
            "ELLIPSES": {"color": 2, "desc": "Elliptical shapes - Parametric geometries", 
                        "linetype": "CONTINUOUS", "lineweight": self.thickness_map["ELLIPSES"]},
            "RECTANGLES": {"color": 3, "desc": "Rectangular shapes - Right angles", 
                          "linetype": "CONTINUOUS", "lineweight": self.thickness_map["RECTANGLES"]},
            "TRIANGLES": {"color": 4, "desc": "Triangular shapes - 3 vertices", 
                         "linetype": "CONTINUOUS", "lineweight": self.thickness_map["TRIANGLES"]},
            "POLYGONS": {"color": 5, "desc": "Polygonal shapes - Multi-vertex", 
                         "linetype": "CONTINUOUS", "lineweight": self.thickness_map["POLYGONS"]},
            "LINES": {"color": 6, "desc": "Lines and segments - Linear elements", 
                      "linetype": "CONTINUOUS", "lineweight": self.thickness_map["LINES"]},
            "CONTOURS": {"color": 7, "desc": "Detected contours - Precise perimeters", 
                        "linetype": "CONTINUOUS", "lineweight": self.thickness_map["CONTOURS"]},
            "COMPLEX_SHAPES": {"color": 8, "desc": "Irregular shapes - Complex geometries", 
                               "linetype": "CONTINUOUS", "lineweight": self.thickness_map["COMPLEX_SHAPES"]},
            "ANNOTATIONS": {"color": 9, "desc": "Text and annotations - Explanatory information", 
                          "linetype": "CONTINUOUS", "lineweight": self.thickness_map["ANNOTATIONS"]},
            "METADATA": {"color": 250, "desc": "Document information - Metadata", 
                       "linetype": "CONTINUOUS", "lineweight": self.thickness_map["METADATA"]}
        }
        
        # Create the layers with enhanced properties
        for name, props in layer_definitions.items():
            layer = doc.layers.new(name=name)
            layer.color = props["color"]
            layer.description = props["desc"]
            
            # Add linetype if available
            if props["linetype"] in doc.linetypes:
                layer.linetype = props["linetype"]
                
            # Set lineweight - need to convert to standard AutoCAD lineweight
            std_lw = self.get_closest_std_lineweight(props["lineweight"])
            layer.lineweight = std_lw
            
        return layer_definitions
    
    def create_dxf_document(self):
        """
        Create a new DXF document with optimized settings
        ENHANCEMENT: Improved configuration for better CAD compatibility
        """
        # Create new document with R2018 format for better compatibility
        doc = ezdxf.new('R2018')
        
        # Set up layers
        layer_definitions = self.setup_layers(doc)
        
        # ENHANCEMENT: Extended configuration for better CAD compatibility
        doc.header['$LUNITS'] = 2       # Set units to decimal
        doc.header['$INSUNITS'] = 4     # Set insertion units to mm
        doc.header['$MEASUREMENT'] = 1  # Set measurement to metric
        doc.header['$LIMMIN'] = (-1000, -1000)  # Minimum drawing limits
        doc.header['$LIMMAX'] = (1000, 1000)    # Maximum drawing limits
        doc.header['$EXTMIN'] = (-1000, -1000, 0)  # Minimum extension
        doc.header['$EXTMAX'] = (1000, 1000, 0)    # Maximum extension
        doc.header['$LTSCALE'] = 1.0    # Line type scale
        doc.header['$TEXTSTYLE'] = 'STANDARD'  # Default text style
        
        # Set up layout settings
        msp = doc.modelspace()
        
        return doc, msp, layer_definitions

    def _add_dxf_element(self, msp, element, layer_defs, segment_id):
        """
        Add a geometric element to the DXF document with enhanced thickness
        ENHANCEMENT: More precise handling of geometric elements and attributes
        """
        element_type = element['type']
        
        # Map element type to layer
        if element_type == 'cercle':
            layer_name = "CIRCLES"
            # Add circle with enhanced thickness
            center = element['center']
            radius = element['radius']
            
            # Create the circle entity with specified lineweight
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.get_closest_std_lineweight(
                    self.thickness_map[layer_name])
            }
            
            # Add circle entity - ensure we're using floats
            msp.add_circle(
                (float(center[0]), float(center[1]), 0),
                float(radius),
                dxfattribs=dxf_attribs
            )
            
            # ENHANCEMENT: Add center as reference point with attributes
            msp.add_point(
                (float(center[0]), float(center[1]), 0),
                dxfattribs={"layer": layer_name, "color": 1}  # Red point
            )
            
        elif element_type == 'ellipse':
            layer_name = "ELLIPSES"
            center = element['center']
            axes = element['axes']
            angle = element['angle']
            
            # Convert to ezdxf ellipse format
            major_axis = max(axes[0], axes[1]) / 2
            ratio = min(axes[0], axes[1]) / max(axes[0], axes[1])
            
            # Calculate major axis vector
            angle_rad = angle * np.pi / 180.0
            major_axis_vector = (
                float(major_axis * np.cos(angle_rad)),
                float(major_axis * np.sin(angle_rad)),
                0
            )
            
            # Create ellipse with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.get_closest_std_lineweight(
                    self.thickness_map[layer_name])
            }
            
            # Add ellipse entity
            msp.add_ellipse(
                (float(center[0]), float(center[1]), 0),
                major_axis_vector,
                float(ratio),
                0, 2*np.pi,
                dxfattribs=dxf_attribs
            )
            
            # ENHANCEMENT: Add reference axes for the ellipse
            minor_length = major_axis * ratio
            minor_axis_vector = (
                float(-minor_length * np.sin(angle_rad)),
                float(minor_length * np.cos(angle_rad)),
                0
            )
            
            # Add axes as thin lines
            axis_attribs = {
                "layer": layer_name,
                "lineweight": 13,  # 0.13mm - thinner
                "linetype": "DASHED" if "DASHED" in msp.doc.linetypes else "CONTINUOUS",
                "color": 8  # Gray
            }
            
            # Major axis
            start = (center[0] - major_axis_vector[0], center[1] - major_axis_vector[1], 0)
            end = (center[0] + major_axis_vector[0], center[1] + major_axis_vector[1], 0)
            msp.add_line(start, end, dxfattribs=axis_attribs)
            
            # Minor axis
            start = (center[0] - minor_axis_vector[0], center[1] - minor_axis_vector[1], 0)
            end = (center[0] + minor_axis_vector[0], center[1] + minor_axis_vector[1], 0)
            msp.add_line(start, end, dxfattribs=axis_attribs)
            
        elif element_type == 'rectangle':
            layer_name = "RECTANGLES"
            points = element['points']
            
            # Safely extract 2D points and ensure they're floats
            try:
                # Handle different possible formats of points
                points_2d = []
                for p in points:
                    if isinstance(p, (list, tuple)):
                        if len(p) >= 2:
                            points_2d.append((float(p[0]), float(p[1])))
                    elif isinstance(p, np.ndarray):
                        points_2d.append((float(p[0]), float(p[1])))
                
                # Ensure we have enough points for a rectangle
                if len(points_2d) < 3:
                    print(f"Warning: Not enough points for rectangle: {len(points_2d)}")
                    return
                
                # Create rectangle with enhanced thickness
                dxf_attribs = {
                    "layer": layer_name,
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map[layer_name])
                }
                
                # ENHANCEMENT: Ensure rectangle is closed
                if len(points_2d) > 0 and points_2d[0] != points_2d[-1]:
                    points_2d.append(points_2d[0])  # Close the polygon
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
                
                # ENHANCEMENT: Add light hatching to identify the rectangle
                # Create a hatch pattern
                if hasattr(msp.doc, "hatch") and len(points_2d) >= 4:
                    hatch_attribs = {
                        "layer": layer_name,
                        "color": 8,  # Gray
                        "pattern_name": "SOLID",
                        "pattern_scale": 0.5,
                        "transparency": 0.8  # 80% transparent
                    }
                    
                    # Add very light hatch
                    boundary_path = [(0, points_2d)]
                    try:
                        msp.add_hatch(
                            color=8,
                            dxfattribs=hatch_attribs,
                            paths=boundary_path
                        )
                    except Exception:
                        # Ignore if hatching fails (optional feature)
                        pass
                    
            except Exception as e:
                print(f"Error creating rectangle in DXF: {e}")
                
        elif element_type == 'polygone':
            # Determine if it's a triangle or other polygon
            points = element['points']
            if len(points) == 3:
                layer_name = "TRIANGLES"
            else:
                layer_name = "POLYGONS"
                
            try:
                # Safely extract 2D points
                points_2d = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        points_2d.append((float(p[0]), float(p[1])))
                    elif isinstance(p, np.ndarray):
                        points_2d.append((float(p[0]), float(p[1])))
                
                # Ensure we have enough points
                if len(points_2d) < 3:
                    print(f"Warning: Not enough points for polygon: {len(points_2d)}")
                    return
                
                # ENHANCEMENT: Close the polygon if needed
                if len(points_2d) > 0 and points_2d[0] != points_2d[-1]:
                    points_2d.append(points_2d[0])
                
                # Create polygon with enhanced thickness
                dxf_attribs = {
                    "layer": layer_name,
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map[layer_name])
                }
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
            except Exception as e:
                print(f"Error creating polygon in DXF: {e}")
            
        elif element_type == 'contour':
            layer_name = "CONTOURS"
            points = element['points']
            
            try:
                # Create contour with enhanced thickness
                dxf_attribs = {
                    "layer": layer_name,
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map[layer_name])
                }
                
                # Safely extract 2D points
                points_2d = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        points_2d.append((float(p[0]), float(p[1])))
                    elif isinstance(p, np.ndarray):
                        points_2d.append((float(p[0]), float(p[1])))
                
                # Ensure we have enough points
                if len(points_2d) < 2:
                    print(f"Warning: Not enough points for contour: {len(points_2d)}")
                    return
                
                # ENHANCEMENT: Simplify contour to reduce file size
                # if number of points is very high
                if len(points_2d) > 100:
                    # Convert to numpy format for Douglas-Peucker
                    points_np = np.array(points_2d)
                    epsilon = 0.5  # Tolerance parameter - adjust as needed
                    # Manual implementation of simplification
                    points_2d = self._simplify_contour(points_np, epsilon)
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
            except Exception as e:
                print(f"Error creating contour in DXF: {e}")
            
        elif element_type == 'lignes':
            layer_name = "LINES"
            segments = element['segments']
            
            try:
                # Create lines with enhanced thickness
                dxf_attribs = {
                    "layer": layer_name,
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map[layer_name])
                }
                
                # Add individual line segments
                for segment in segments:
                    if len(segment) == 2:
                        start, end = segment
                        # Ensure coordinates are floats
                                        
                        start_point = (float(start[0]), float(start[1]), 0)
                        end_point = (float(end[0]), float(end[1]), 0)
                        
                        msp.add_line(
                            start_point,
                            end_point,
                            dxfattribs=dxf_attribs
                        )
            except Exception as e:
                print(f"Error creating lines in DXF: {e}")
                
        else:  # Unknown type - use complex shape layer
            layer_name = "COMPLEX_SHAPES"
            
            try:
                # Try to extract points or segments
                if 'points' in element:
                    points = element['points']
                    
                    # Extract 2D points safely
                    points_2d = []
                    for p in points:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            points_2d.append((float(p[0]), float(p[1])))
                        elif isinstance(p, np.ndarray):
                            points_2d.append((float(p[0]), float(p[1])))
                    
                    if len(points_2d) >= 2:
                        # Create complex shape with enhanced thickness
                        dxf_attribs = {
                            "layer": layer_name,
                            "lineweight": self.get_closest_std_lineweight(
                                self.thickness_map[layer_name])
                        }
                        
                        # Add polyline entity
                        msp.add_lwpolyline(
                            points_2d,
                            close=True,
                            dxfattribs=dxf_attribs
                        )
                    
                elif 'segments' in element:
                    segments = element['segments']
                    
                    # Create lines with enhanced thickness
                    dxf_attribs = {
                        "layer": layer_name,
                        "lineweight": self.get_closest_std_lineweight(
                            self.thickness_map[layer_name])
                    }
                    
                    # Add line segments
                    for segment in segments:
                        if len(segment) == 2:
                            start, end = segment
                            # Ensure coordinates are floats
                            start_point = (float(start[0]), float(start[1]), 0)
                            end_point = (float(end[0]), float(end[1]), 0)
                            
                            msp.add_line(
                                start_point,
                                end_point,
                                dxfattribs=dxf_attribs
                            )
            except Exception as e:
                print(f"Error creating complex shape in DXF: {e}")
        
        # Add identifier text if segment_id is provided
        if segment_id is not None:
            try:
                # Find position for label - use center of element
                if element_type == 'cercle' or element_type == 'ellipse':
                    center_pos = (float(element['center'][0]), float(element['center'][1]), 0)
                elif 'points' in element and element['points']:
                    points = element['points']
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    center_pos = (float(center_x), float(center_y), 0)
                elif 'segments' in element and element['segments']:
                    start = element['segments'][0][0]
                    center_pos = (float(start[0]), float(start[1]), 0)
                else:
                    center_pos = (0, 0, 0)
                
                # ENHANCEMENT: Label with background for better readability
                # Create a background circle
                bg_attribs = {
                    "layer": "ANNOTATIONS",
                    "color": 7,  # White
                    "lineweight": 0  # No outline
                }
                
                # Background circle radius
                bg_radius = 8.0
                
                # Add background circle
                msp.add_circle(
                    center_pos,
                    bg_radius,
                    dxfattribs=bg_attribs
                )
                
              
                # Create label with segment ID - enhanced style
                text_attribs = {
                    "layer": "ANNOTATIONS",
                    "color": 2,  # Yellow for better visibility
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map["ANNOTATIONS"]),
                    "style": "STANDARD",  # Standard text style
                    "height": 5.0,        # Text size
                    "rotation": 0,        # No rotation
                    "halign": 0,          # Horizontal alignment center
                    "valign": 0           # Vertical alignment center
                }
                
                # Add text
                msp.add_text(
                    str(segment_id),
                    dxfattribs=text_attribs,
                    position=center_pos
                )
            except Exception as e:
                print(f"Error adding text annotation in DXF: {e}")
    
    def _simplify_contour(self, points, epsilon):
        """
        ENHANCEMENT: Simplify a contour using the Douglas-Peucker algorithm
        Reduces the number of points while preserving the general shape
        """
        if len(points) <= 2:
            return points.tolist()
        
        # Recursive function for Douglas-Peucker
        def douglasPeucker(points, epsilon):
            # Find the point furthest from the line between the first and last point
            dmax = 0
            index = 0
            end = len(points) - 1
            
            # Calculate perpendicular distance of each point to the line
            for i in range(1, end):
                d = self._point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # If max distance is greater than epsilon, recursively divide
            result = []
            if dmax > epsilon:
                # Recursively divide
                rec1 = douglasPeucker(points[:index+1], epsilon)
                rec2 = douglasPeucker(points[index:], epsilon)
                
                # Build result
                result = rec1[:-1] + rec2
            else:
                result = [points[0], points[-1]]
            
            return result
        
        # Apply algorithm
        simplified = douglasPeucker(points, epsilon)
        
        # Convert to list of (x, y) tuples
        return [(float(p[0]), float(p[1])) for p in simplified]
    
    def _point_line_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from a point to a line"""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        # Vectors
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # Line length
        line_len = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_len if line_len > 0 else line_vec
        
        # Point projection on line
        projection_length = np.dot(point_vec, line_unit_vec)
        
        # Projected point
        projection = line_start + projection_length * line_unit_vec
        
        # Perpendicular distance
        return np.linalg.norm(point - projection)
            
    def _add_dxf_metadata(self, msp, doc, layer_defs):
        """
        Add metadata to the DXF file
        ENHANCEMENT: Enriched metadata for better CAD integration
        """
        # Set document variables
        doc.header['$ACADVER'] = "AC1027"  # AutoCAD 2018
        
        # ENHANCEMENT: Extended document variables for better compatibility
        for var, val in [
            ("$LUNITS", 2),         # Set units to decimal
            ("$INSUNITS", 4),       # Set insertion units to mm
            ("$MEASUREMENT", 1),     # Set measurement to metric
            ("$LIMCHECK", 1),        # Enable limits checking
            ("$FILLMODE", 1),        # Enable fill mode
            ("$MIRRTEXT", 1),        # Mirror text with parent
            ("$LTSCALE", 1.0),       # Set global linetype scale
            ("$TEXTSIZE", 5.0),      # Set default text size
            ("$CHAMFERA", 0.0),      # Chamfer distance A
            ("$CHAMFERB", 0.0),      # Chamfer distance B
            ("$CHAMFERC", 0.0),      # Chamfer angle C
            ("$CHAMFERD", 0.0),      # Chamfer angle D
            ("$FILLETRAD", 0.0),     # Fillet radius
            ("$AUNITS", 0),          # Angular units format
            ("$AUPREC", 0),          # Angular precision
            ("$CPSNTYPE", 0),        # Current plot style name type
            ("$PDMODE", 35),         # Point display mode
            ("$PDSIZE", 5),          # Point display size
            ("$DIMFIT", 3),          # Dimension fit type
            ("$DIMUNIT", 2),         # Dimension unit
            ("$SPLINESEGS", 8)       # Number of segments in a spline
        ]:
            if var in doc.header:
                doc.header[var] = val
        
        # ENHANCEMENT: Improved title block with project information
        # Create frame for title block
        title_block_x = 0
        title_block_y = -15
        title_block_width = 210  # mm (A4 format)
        title_block_height = 60  # mm
        
        # Add rectangle for title block
        title_block_points = [
            (title_block_x, title_block_y),
            (title_block_x + title_block_width, title_block_y),
            (title_block_x + title_block_width, title_block_y - title_block_height),
            (title_block_x, title_block_y - title_block_height),
            (title_block_x, title_block_y)  # Close rectangle
        ]
        
        # Title block attributes
        title_block_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(0.35),  # Slightly thicker
            "color": 7  # White
        }
        
        # Add title block rectangle
        msp.add_lwpolyline(title_block_points, dxfattribs=title_block_attribs)
        
        # Add interior lines to structure the block
        # Horizontal line under main title
        msp.add_line(
            (title_block_x, title_block_y - 20),
            (title_block_x + title_block_width, title_block_y - 20),
            dxfattribs=title_block_attribs
        )
        
        # Vertical line to separate date
        msp.add_line(
            (title_block_x + 150, title_block_y - 20),
            (title_block_x + 150, title_block_y - title_block_height),
            dxfattribs=title_block_attribs
        )
        
        # Add title block information with enhanced styling
        dxf_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(
                self.thickness_map["METADATA"]),
            "color": 7,  # White
            "height": 12.0,
            "style": "STANDARD"
        }
        
        # Main title
        msp.add_text(
            "Professional Segmentation with SAM & CRF",
            height=12.0,
            dxfattribs=dxf_attribs,
            position=(title_block_x + 5, title_block_y - 10)
        )
        
        # ENHANCEMENT: Add more structured metadata
        # Date and user with title
        msp.add_text(
            "DATE:",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 155, title_block_y - 25)
        )
        
        msp.add_text(
            CURRENT_DATE,
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 155, title_block_y - 35)
        )
        
        # User
        msp.add_text(
            "AUTHOR:",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 155, title_block_y - 45)
        )
        
        msp.add_text(
            CURRENT_USER,
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 155, title_block_y - 55)
        )
        
        # Version and software information
        msp.add_text(
            f"Enhanced SAM+CRF Segmentation v{VERSION}",
            height=6.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 6.0},
            position=(title_block_x + 5, title_block_y - 35)
        )
        
        # Project description
        msp.add_text(
            "AI-Based Professional Segmentation",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 5, title_block_y - 50)
        )
        
        # Scale
        msp.add_text(
            "SCALE: 1:1",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 5, title_block_y - 65)
        )
        
        # Coordinate system
        msp.add_text(
            "REFERENCE: Pixels (0,0) at top left",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 65, title_block_y - 65)
        )
        
    def export_to_dxf(self, segments_data, output_path, line_thickness=0.35):
        """
        Export segments to DXF with enhanced line thickness
        ENHANCEMENT: More robust export with intelligent filtering and extended metadata
        """
        print(f"Exporting to DXF with enhanced line thickness: {output_path}")
        
        # Update line thickness if provided
        self.line_thickness = line_thickness
        
        # Create a new DXF document
        doc, msp, layer_defs = self.create_dxf_document()
        
        # Process segments by quality
        high_quality = []
        medium_quality = []
        low_quality = []
        
        # ENHANCEMENT: More intelligent segment filtering
        for segment in segments_data:
            # Calculate composite quality score
            size_factor = min(1.0, segment['area'] / 1000)  # Size factor
            shape_regularity = segment['shape']['regularity'] if 'regularity' in segment['shape'] else 0.5
            
            # Composite quality score
            composite_score = segment['quality'] * 0.6 + size_factor * 0.2 + shape_regularity * 0.2
            
            # Update quality score
            segment['quality'] = composite_score
            
            # Classification based on composite score
            if composite_score > 0.75:
                high_quality.append(segment)
            elif composite_score > 0.5:
                medium_quality.append(segment)
            else:
                low_quality.append(segment)
                
        # Export high and medium quality segments
        export_segments = high_quality + medium_quality
        
        print(f"Segments to export: {len(export_segments)} "
              f"(high: {len(high_quality)}, medium: {len(medium_quality)}, "
              f"filtered: {len(low_quality)})")
        
        # ENHANCEMENT: Sort segments for organized export (large objects first)
        export_segments.sort(key=lambda s: s['area'], reverse=True)
        
        # Process each segment
        for segment in tqdm(export_segments, desc="Exporting objects to DXF"):
            for element in segment['geometric_elements']:
                self._add_dxf_element(msp, element, layer_defs, segment['id'])
        
        # Add metadata and annotations
        self._add_dxf_metadata(msp, doc, layer_defs)
        
        # ENHANCEMENT: Add a legend for shape types
        self._add_shape_legend(msp, layer_defs)
        
        # ENHANCEMENT: Add a frame and grid information
        self._add_drawing_frame(msp)
        
        # Save the document with optimized settings
        doc.saveas(output_path)
        print(f"DXF file exported successfully with {len(export_segments)} objects.")
        return doc
    
    def _add_shape_legend(self, msp, layer_defs):
        """
        ENHANCEMENT: Add a legend of shapes for easier plan reading
        """
        # Legend position
        legend_x = -100
        legend_y = 0
        legend_width = 90
        legend_height = 120
        
        # Legend frame
        legend_points = [
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y),
            (legend_x + legend_width, legend_y - legend_height),
            (legend_x, legend_y - legend_height),
            (legend_x, legend_y)
        ]
        
        # Frame attributes
        frame_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(0.25),
            "color": 7  # White
        }
        
        # Add frame
        msp.add_lwpolyline(legend_points, dxfattribs=frame_attribs)
        
        # Legend title
        msp.add_text(
            "LEGEND",
            height=7.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 7.0},
            position=(legend_x + 5, legend_y - 10)
        )
        
        # Line under title
        msp.add_line(
            (legend_x, legend_y - 15),
            (legend_x + legend_width, legend_y - 15),
            dxfattribs=frame_attribs
        )
        
        # List of shape types with colors and examples
        shape_types = [
            {"name": "Circles", "layer": "CIRCLES", "y_offset": 25},
            {"name": "Ellipses", "layer": "ELLIPSES", "y_offset": 35},
            {"name": "Rectangles", "layer": "RECTANGLES", "y_offset": 45},
            {"name": "Triangles", "layer": "TRIANGLES", "y_offset": 55},
            {"name": "Polygons", "layer": "POLYGONS", "y_offset": 65},
            {"name": "Contours", "layer": "CONTOURS", "y_offset": 75},
            {"name": "Lines", "layer": "LINES", "y_offset": 85},
            {"name": "Complex shapes", "layer": "COMPLEX_SHAPES", "y_offset": 95},
        ]
        
        # Add each legend item
        for shape in shape_types:
            # Layer color for this item
            layer_color = layer_defs.get(shape["layer"], {}).get("color", 7)
            
            # Item text
            msp.add_text(
                shape["name"],
                height=5.0,
                dxfattribs={"layer": "METADATA", "color": layer_color, "height": 5.0},
                position=(legend_x + 25, legend_y - shape["y_offset"])
            )
            
            # Small visual example
            example_attribs = {
                "layer": shape["layer"],
                "lineweight": self.get_closest_std_lineweight(
                    self.thickness_map[shape["layer"]])
            }
            
            # Draw example based on type
            if shape["layer"] == "CIRCLES":
                msp.add_circle(
                    (legend_x + 10, legend_y - shape["y_offset"] + 2), 
                    3.0,
                    dxfattribs=example_attribs
                )
            elif shape["layer"] == "ELLIPSES":
                msp.add_ellipse(
                    (legend_x + 10, legend_y - shape["y_offset"] + 2),
                    (4.0, 0.0, 0.0),
                    0.5,
                    0, 2*np.pi,
                    dxfattribs=example_attribs
                )
            elif shape["layer"] == "RECTANGLES":
                rect_points = [
                    (legend_x + 7, legend_y - shape["y_offset"]),
                    (legend_x + 13, legend_y - shape["y_offset"]),
                    (legend_x + 13, legend_y - shape["y_offset"] + 4),
                    (legend_x + 7, legend_y - shape["y_offset"] + 4),
                    (legend_x + 7, legend_y - shape["y_offset"])
                ]
                msp.add_lwpolyline(rect_points, dxfattribs=example_attribs)
            elif shape["layer"] == "TRIANGLES":
                tri_points = [
                    (legend_x + 10, legend_y - shape["y_offset"] + 4),
                    (legend_x + 7, legend_y - shape["y_offset"]),
                    (legend_x + 13, legend_y - shape["y_offset"]),
                    (legend_x + 10, legend_y - shape["y_offset"] + 4)
                ]
                msp.add_lwpolyline(tri_points, dxfattribs=example_attribs)
            elif shape["layer"] == "POLYGONS":
                poly_points = [
                    (legend_x + 10, legend_y - shape["y_offset"] + 4),
                    (legend_x + 7, legend_y - shape["y_offset"] + 2),
                    (legend_x + 7, legend_y - shape["y_offset"]),
                    (legend_x + 10, legend_y - shape["y_offset"] - 2),
                    (legend_x + 13, legend_y - shape["y_offset"]),
                    (legend_x + 13, legend_y - shape["y_offset"] + 2),
                    (legend_x + 10, legend_y - shape["y_offset"] + 4)
                ]
                msp.add_lwpolyline(poly_points, dxfattribs=example_attribs)
            elif shape["layer"] == "CONTOURS":
                # Wavy contour
                contour_points = []
                for i in range(11):
                    angle = i * np.pi / 5
                    r = 3.0 + 0.8 * np.sin(3 * angle)
                    x = legend_x + 10 + r * np.cos(angle)
                    y = legend_y - shape["y_offset"] + 2 + r * np.sin(angle)
                    contour_points.append((x, y))
                contour_points.append(contour_points[0])  # Close contour
                msp.add_lwpolyline(contour_points, dxfattribs=example_attribs)
            elif shape["layer"] == "LINES":
                # A few lines
                msp.add_line(
                    (legend_x + 7, legend_y - shape["y_offset"]),
                    (legend_x + 13, legend_y - shape["y_offset"] + 4),
                    dxfattribs=example_attribs
                )
                msp.add_line(
                    (legend_x + 7, legend_y - shape["y_offset"] + 4),
                    (legend_x + 13, legend_y - shape["y_offset"]),
                    dxfattribs=example_attribs
                )
            elif shape["layer"] == "COMPLEX_SHAPES":
                # Complex shape
                complex_points = []
                for i in range(8):
                    angle = i * 2 * np.pi / 8
                    r = 3.0
                    if i % 2 == 0:
                        r = 4.0
                    x = legend_x + 10 + r * np.cos(angle)
                    y = legend_y - shape["y_offset"] + 2 + r * np.sin(angle)
                    complex_points.append((x, y))
                complex_points.append(complex_points[0])  # Close shape
                msp.add_lwpolyline(complex_points, dxfattribs=example_attribs)
                
        # Explanatory text
        msp.add_text(
            "Elements are colored by type",
            height=4.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 4.0},
            position=(legend_x + 5, legend_y - 105)
        )
        
        msp.add_text(
            "and have adapted thicknesses",
            height=4.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 4.0},
            position=(legend_x + 5, legend_y - 112)
        )
    
    def _add_drawing_frame(self, msp):
        """
        ENHANCEMENT: Add a drawing frame with coordinate indicators
        """
        # Not implemented to avoid overloading the file
        # This method can be developed to add a frame with graduations
        pass


class EnhancedImageProcessor:
    """
    Enhanced image processor combining SAM segmentation, CRF refinement, and DXF export
    Enhancements: anisotropic filtering, optimized adaptive thresholds, and more robust CRF
    """
    def __init__(self, sam_checkpoint=None, debug=False, line_thickness=0.35):
        """Initialize the advanced image processor"""
        self.debug = debug
        
        # Initialize helpers
        self.crf_processor = CRFPostProcessor(debug=debug)
        self.sam_helper = SegmentAnythingHelper(checkpoint_path=sam_checkpoint, device=SAM_DEVICE)
        self.dxf_exporter = EnhancedDXFExporter(line_thickness=line_thickness)
        self.tile_splitter = ImageTileSplitter(tile_size=TILE_SIZE, max_size=MAX_SIZE_WITHOUT_SPLITTING, debug=debug)
        self.interactive_tool = InteractiveSegmentationTool(
            sam_helper=self.sam_helper,
            crf_processor=self.crf_processor,
            debug=debug
        )
        
        # Mark whether SAM is available
        self.sam_available = self.sam_helper.is_initialized()
        
    def load_image(self, filepath):
        """Load an image with optimized settings"""
        print(f"Loading image: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load image with best quality
        image_color = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image_color is None:
            raise ValueError(f"Unable to load image: {filepath}")

        # Convert to RGB
        if len(image_color.shape) == 3:
            image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
        else:
            # Image is already grayscale
            image_color = cv2.cvtColor(image_color, cv2.COLOR_GRAY2RGB)
            image_gray = image_color[:,:,0]

        # Normalize if needed
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if self.debug:
            print(f"Image loaded: {image_color.shape[1]}x{image_color.shape[0]} pixels")
            print(f"Format: {image_color.dtype}, Range: [{image_gray.min()}-{image_gray.max()}]")
            
        # Initialize the interactive tool with this image
        self.interactive_tool.set_image(image_color)
            
        return image_color, image_gray




    def preprocess_image(self, image, denoise_strength=9):
        """
        Enhanced preprocessing to improve quality before segmentation
        ENHANCEMENT: Anisotropic filtering to preserve edges
        """
        # Create a copy to avoid modifying the original
        processed = image.copy()
        
        # ENHANCEMENT: Use anisotropic filter to preserve edges
        processed_float = processed.astype(np.float32) / 255.0
        processed_filtered = custom_anisotropic_diffusion(processed_float, niter=10, kappa=20, gamma=0.1, option=1)
        processed = (processed_filtered * 255.0).astype(np.uint8)
        
        # Apply CLAHE for better contrast - optimized parameters
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(16, 16))
        processed = clahe.apply(processed)
        
        # ENHANCEMENT: Selective application of noise reduction
        # Detect edges to preserve their sharpness
        edges = cv2.Canny(processed, 50, 150)
        
        # Create a dilated edge mask to avoid blurring them
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        # Apply bilateral blur with optimized parameters
        blurred = cv2.bilateralFilter(processed, denoise_strength, 50, 50)
        
        # Combine blurred image and original using edge mask
        processed_no_edges = np.copy(processed)
        processed_no_edges[edge_mask == 0] = blurred[edge_mask == 0]
        processed = processed_no_edges
        
        # Apply adaptive gamma correction to enhance details in dark areas
        # Calculate adaptive gamma based on histogram
        hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # Calculate adaptive gamma based on pixel distribution
        dark_ratio = np.sum(hist_norm[:64])
        light_ratio = np.sum(hist_norm[192:])
        
        if dark_ratio > 0.6:  # Many dark areas
            gamma = 0.85  # Lighten
        elif light_ratio > 0.6:  # Many light areas
            gamma = 1.2   # Darken
        else:
            gamma = 1.0   # Neutral
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
        
        # ENHANCEMENT: Final edge enhancement
        # Enhance edges with unsharp masking
        blurred_img = cv2.GaussianBlur(processed, (0, 0), 3.0)
        processed = cv2.addWeighted(processed, 1.5, blurred_img, -0.5, 0)
        
        return processed

    def segment_with_sam(self, image_color):
        """
        Segment image using Segment Anything Model
        ENHANCEMENT: Optimized SAM usage with automatic mask filtering
        """
        print("Segmenting with SAM (Segment Anything Model)...")
        
        # Use automatic mask generation for best results
        masks = self.sam_helper.automatic_segmentation(image_color)
        
        if not masks:
            print("SAM segmentation failed or produced no masks.")
            return None
            
        print(f"SAM generated {len(masks)} masks.")
        
        # Convert masks to our segment format
        segments_data = []
        
        for i, mask_data in enumerate(masks):
            try:
                # Mask extraction with different possible formats
                if isinstance(mask_data, dict):
                    if "segmentation" in mask_data:
                        # ENHANCEMENT: More robust handling of different segmentation formats
                        if isinstance(mask_data["segmentation"], np.ndarray):
                            # Simple format: numpy array
                            mask = mask_data["segmentation"].astype(np.uint8) * 255
                            area = mask_data.get("area", np.sum(mask_data["segmentation"]))
                        elif isinstance(mask_data["segmentation"], dict):
                            # COCO RLE format: {"size": [h, w], "counts": "..."}
                            print("Detected COCO RLE format")
                            try:
                                h, w = image_color.shape[:2]
                                rle = mask_data["segmentation"]
                                # Create binary mask manually from RLE
                                if "counts" in rle and "size" in rle:
                                    # Try simplified RLE decoding
                                    h, w = rle["size"]
                                    mask = np.zeros((h, w), dtype=np.uint8)
                                    mask.fill(255)  # Temporary workaround
                                else:
                                    # Create default mask based on bounding box if available
                                    if "bbox" in mask_data:
                                        x, y, w, h = [int(coord) for coord in mask_data["bbox"]]
                                        mask = np.zeros((image_color.shape[0], image_color.shape[1]), dtype=np.uint8)
                                        mask[y:y+h, x:x+w] = 255
                                    else:
                                        # Create a rectangle in the center as last resort
                                        h, w = image_color.shape[:2]
                                        cx, cy = w//2, h//2
                                        size = min(w, h) // 4
                                        mask = np.zeros((h, w), dtype=np.uint8)
                                        mask[cy-size:cy+size, cx-size:cx+size] = 255
                                
                                area = mask_data.get("area", np.sum(mask > 0))
                            except Exception as e:
                                print(f"Error decoding RLE: {e}")
                                continue
                        else:
                            # Unsupported format
                            print(f"Unsupported segmentation format: {type(mask_data['segmentation'])}")
                            # Try to create default mask based on other information
                            if "bbox" in mask_data:
                                x, y, w, h = [int(coord) for coord in mask_data["bbox"]]
                                mask = np.zeros((image_color.shape[0], image_color.shape[1]), dtype=np.uint8)
                                mask[y:y+h, x:x+w] = 255
                                area = w * h
                            else:
                                continue  # Cannot create mask
                                # Skip tiny segments (likely noise)
                                if area < 100:
                                    continue
                                    
                # ENHANCEMENT: Preliminary segment quality analysis
                # Calculate aspect ratio to identify too elongated segments (potential artifacts)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    continue
                    
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Enhanced shape analysis
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)  # avoid division by zero
                
                # Filter out too elongated and small shapes (often artifacts)
                if aspect_ratio > 20 and area < 500:
                    continue
                
                # Apply smoothing to contour with adaptive epsilon
                # ENHANCEMENT: adaptive epsilon based on perimeter
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.0015 * perimeter  # Optimized parameter for more precise smoothing
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Calculate contour properties
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate circularity with robustness enhancement
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0
                
                # Calculate center with handling for degenerate cases
                M = cv2.moments(contour)
                if M["m00"] > 1e-5:  # Threshold to avoid numerical instability
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w//2, y + h//2
                    
                # ENHANCEMENT: More precise shape detection with convexity analysis
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull) if len(hull) > 2 else area + 1e-6
                convexity = area / hull_area if hull_area > 0 else 0
                
                # Detect shape type with in-depth analysis
                shape_info = self.detect_shape(contour, circularity, convexity)
                
                # Convert mask to binary for CRF
                mask_binary = (mask > 0).astype(np.uint8)
                
                # Extract geometric elements for DXF export
                geometric_elements = self.extract_geometric_elements(smoothed_contour, shape_info["type"])
                
                # ENHANCEMENT: Composite quality calculation
                # Based on stability, circularity, and convexity
                if isinstance(mask_data, dict) and "stability_score" in mask_data and "predicted_iou" in mask_data:
                    # Use SAM scores if available
                    composite_quality = (
                        mask_data["stability_score"] * 0.5 +
                        mask_data["predicted_iou"] * 0.3 +
                        min(1.0, circularity + 0.2) * 0.2  # favor regular shapes
                    )
                else:
                    # Calculation based on geometric properties
                    shape_regularity = (circularity + convexity) / 2
                    size_factor = min(1.0, area / 2000)  # 2000px² considered optimal size
                    composite_quality = 0.7 * shape_regularity + 0.3 * size_factor
                
                # Create segment data
                segment = {
                    'id': i + 1,
                    'contour': contour,
                    'smoothed_contour': smoothed_contour,
                    'area': area,
                    'perimeter': perimeter,
                    'dimensions': (w, h),
                    'position': (x, y),
                    'center': (cX, cY),
                    'mask': mask_binary,  # Store binary mask for later refinement
                    'shape': shape_info,
                    'circularity': circularity,
                    'convexity': convexity,  # ENHANCEMENT: new useful attribute
                    'quality': min(1.0, max(0.1, composite_quality)),  # Bound between 0.1 and 1.0
                    'geometric_elements': geometric_elements
                }
                
                segments_data.append(segment)
                
            except Exception as e:
                print(f"Error processing mask {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return segments_data
    
    #####################################################################################################################

    def detect_shape(self, contour, circularity, convexity=None):
        """
        Enhanced shape detection with more precise classifications
        ENHANCEMENT: Added convexity factor and more precise detection
        """
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Get bounding rect for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1
        
        # ENHANCEMENT: Use convexity if available
        if convexity is None:
            # Calculate convexity if not provided
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if len(hull) > 2 else cv2.contourArea(contour) + 1e-6
            convexity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
        
        # Basic shape properties
        shape_info = {
            'type': 'complex_shape',
            'regularity': 0.5,
            'vertices': num_vertices,
            'convexity': convexity,  # ENHANCEMENT: Added convexity
            'approx_contour': approx
        }
        
        # ENHANCEMENT: More precise shape detection
        if circularity > 0.9 and convexity > 0.9:
            shape_info['type'] = 'circle'
            shape_info['regularity'] = circularity * convexity
        
        elif circularity > 0.8 and num_vertices > 8:
            # Check if it's an ellipse by analyzing axes
            # Calculate ellipse fitted axes
            if len(contour) >= 5:  # Minimum 5 points for fitEllipse
                try:
                    (center, axes, angle) = cv2.fitEllipse(contour)
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    axis_ratio = minor_axis / (major_axis + 1e-6)
                    
                    if axis_ratio < 0.95:  # Not too close to a circle
                        shape_info['type'] = 'ellipse'
                        shape_info['regularity'] = 0.9 * convexity * axis_ratio
                        # Store ellipse information
                        shape_info['ellipse'] = {
                            'center': center,
                            'axes': axes,
                            'angle': angle
                        }
                    else:
                        shape_info['type'] = 'circle'
                        shape_info['regularity'] = 0.95
                except:
                    # In case of error, keep classification based on circularity
                    shape_info['type'] = 'complex_shape' if convexity < 0.8 else 'ellipse'
                    shape_info['regularity'] = 0.8 * convexity
            else:
                shape_info['type'] = 'complex_shape'
                
        elif num_vertices == 3 or (3 <= num_vertices <= 4 and circularity < 0.6):
            # ENHANCEMENT: Stricter verification for triangles
            if convexity > 0.9 and circularity < 0.7:
                shape_info['type'] = 'triangle'
                shape_info['regularity'] = calculate_regularity(approx, 3)
            else:
                shape_info['type'] = 'polygon'
                shape_info['regularity'] = convexity * 0.8
            
        elif num_vertices == 4 or (4 <= num_vertices <= 6 and circularity > 0.7):
            # Check if it's a square or rectangle
            if aspect_ratio > 0.9 and convexity > 0.9:
                shape_info['type'] = 'square'
                shape_info['regularity'] = 0.95 * aspect_ratio
            else:
                shape_info['type'] = 'rectangle'
                shape_info['regularity'] = 0.9 * convexity * aspect_ratio
                
        elif 5 <= num_vertices <= 10:
            # Check if it's a regular polygon
            shape_info['type'] = f'polygon{num_vertices}'
            shape_info['regularity'] = calculate_regularity(approx, num_vertices) * convexity
            
        else:
            # Complex shape - improve the contour for DXF export
            shape_info['type'] = 'complex_shape'
            # Calculate how smooth/regular the contour is
            shape_info['regularity'] = min(0.7, 0.5 + (circularity * convexity) / 2)
            
        return shape_info

    def calculate_regularity(points, n_sides):
        """
        Calculate how regular/symmetric a polygon is
        ENHANCEMENT: More precise algorithm for evaluating regularity
        """
        # Extract the points
        pts = np.array([pt[0] for pt in points])
        
        # Calculate side lengths
        sides = []
        for i in range(len(pts)):
            next_i = (i + 1) % len(pts)
            side_len = np.sqrt(np.sum((pts[i] - pts[next_i]) ** 2))
            sides.append(side_len)
            
        # Calculate variance normalized by mean
        if np.mean(sides) > 0:
            variance = np.std(sides) / np.mean(sides)
            regularity = max(0, min(1, 1.0 - variance))
        else:
            regularity = 0
        
        # ENHANCEMENT: Check angles too
        angles = []
        if len(pts) >= 3:
            for i in range(len(pts)):
                prev_i = (i - 1) % len(pts)
                next_i = (i + 1) % len(pts)
                
                # Side vectors
                v1 = pts[prev_i] - pts[i]
                v2 = pts[next_i] - pts[i]
                
                # Normalize vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
                
                # Dot product for angle
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # Evaluate angle regularity
        angle_regularity = 0.5
        if angles:
            expected_angle = (n_sides - 2) * np.pi / n_sides  # Theoretical angle
            angle_errors = [abs(a - expected_angle) for a in angles]
            mean_angle_error = np.mean(angle_errors) / np.pi  # Normalize by pi
            angle_regularity = max(0, min(1, 1.0 - 2 * mean_angle_error))
        
        # Combine both regularity measures
        return 0.7 * regularity + 0.3 * angle_regularity

    def extract_geometric_elements(self, contour, shape_type):
        """
        Extract geometric elements for DXF export
        ENHANCEMENT: More precise extraction with better primitive detection
        """
        elements = []
        
        # Process based on shape type
        if shape_type == 'circle':
            # ENHANCEMENT: More precise circle fitting
            # Use equidistant points from contour for more stable fitting
            if len(contour) > 5:
                # Resample contour for better fitting
                perimeter = cv2.arcLength(contour, True)
                num_points = min(20, len(contour))  # Limit to 20 points max
                epsilon = perimeter / num_points
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # Fit circle with simplified points
                (x, y), radius = cv2.minEnclosingCircle(simplified)
            else:
                # Use contour directly if few points
                (x, y), radius = cv2.minEnclosingCircle(contour)
            
            elements.append({
                'type': 'cercle',
                'center': (float(x), float(y)),
                'radius': float(radius)
            })
            
        elif shape_type == 'ellipse':
            if len(contour) >= 5:
                # ENHANCEMENT: More stable ellipse fitting
                # Filter outliers for better fitting
                try:
                    # Use fitEllipse from OpenCV
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    elements.append({
                        'type': 'ellipse',
                        'center': center,
                        'axes': axes,
                        'angle': angle
                    })
                except Exception:
                    # If fitting fails, fallback to polygon
                    points = [pt[0].tolist() for pt in contour]
                    elements.append({
                        'type': 'polygone',
                        'points': points
                    })
            else:
                # Not enough points for ellipse, fallback to polygon
                points = [pt[0].tolist() for pt in contour]
                elements.append({
                    'type': 'polygone',
                    'points': points
                })
                
        elif shape_type in ['square', 'rectangle']:
            # ENHANCEMENT: Use rotatedRect for better oriented rectangle detection
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            elements.append({
                'type': 'rectangle',
                'points': box.tolist(),
                'angle': rect[2]
            })
            
        elif shape_type.startswith('polygon') or shape_type == 'triangle':
            # ENHANCEMENT: For regular polygons, optimize the vertices
            # Extract polygon points
            points = [pt[0].tolist() for pt in contour]
            
            # For regular polygons, optimize representation
            if shape_type.startswith('polygon') and len(contour) > 6:
                # Simplify polygon if necessary
                perimeter = cv2.arcLength(contour, True)
                # Stricter epsilon for regular polygons
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [pt[0].tolist() for pt in approx]
            
            elements.append({
                'type': 'polygone',
                'points': points
            })
            
        elif shape_type == 'complex_shape':
            # For complex shapes, provide two representations: 
            # - A main contour for the overall shape
            # - A decomposition into line segments for more details
            
            # ENHANCEMENT: Optimized decomposition of complex shapes
            # Adaptively simplify contour based on its complexity
            perimeter = cv2.arcLength(contour, True)
            num_points = len(contour)
            
            # Adaptive epsilon factor: stricter for small contours
            adaptive_factor = 0.001 if num_points > 100 else (0.005 if num_points > 50 else 0.01)
            epsilon = adaptive_factor * perimeter
            
            # Simplify contour
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            points = [pt[0].tolist() for pt in simplified]
            
            # Create contour for main shape
            elements.append({
                'type': 'contour',
                'points': points
            })
            
            # Add line segments for more detailed representation
            if len(points) >= 2:
                lines = []
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i+1) % len(points)]
                    lines.append((start, end))
                    
                elements.append({
                    'type': 'lignes',
                    'segments': lines
                })
        
        return elements


    ##################################################################################################################

    def segment_with_traditional(self, image, min_area=100):
        """
        Traditional segmentation when SAM is not available
        ENHANCEMENT: More robust algorithm with anisotropic filtering and improved watershed
        """
        print("Using traditional segmentation methods...")
        
        # Preprocess the image with anisotropic filtering
        enhanced = self.preprocess_image(image)
        
        # ENHANCEMENT: Additional anisotropic filtering to preserve edges better
        enhanced_float = enhanced.astype(np.float32) / 255.0
        aniso_filtered = custom_anisotropic_diffusion(enhanced_float, niter=15, kappa=15, gamma=0.1, option=1)
        enhanced_filtered = (aniso_filtered * 255).astype(np.uint8)
        
        # Edge detection with improved parameters
        edges = cv2.Canny(enhanced_filtered, 30, 150)
        
        # Improve edge connectivity
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # ENHANCEMENT: More precise adaptive thresholding
        block_size = 35  # neighborhood size for threshold calculation (must be odd)
        C = 5  # constant to subtract
        binary_adaptive = cv2.adaptiveThreshold(enhanced_filtered, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 
                                               block_size, C)
        
        # Combine with Otsu thresholding for robustness
        _, binary_otsu = cv2.threshold(enhanced_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine both approaches with morphological operations
        binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
        
        # ENHANCEMENT: Remove small isolated regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disk(2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk(3))
        
        # Apply watershed algorithm with enhanced markers
        # First, distance transform with improved parameters
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # ENHANCEMENT: Adaptive threshold for marker detection
        dist_max = dist_transform.max()
        if dist_max > 0:
            threshold_ratio = 0.7  # Higher to avoid over-segmentation
            _, sure_fg = cv2.threshold(dist_transform, threshold_ratio*dist_max, 255, 0)
        else:
            # Fallback if transform fails
            sure_fg = binary
        
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling with robustness enhancement
        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels == 1:  # No markers found, try another approach
            # Use local maxima as markers
            local_max = custom_peak_local_max(dist_transform, min_distance=20, labels=binary)
            markers = ndimage.label(local_max)[0]
            num_labels = markers.max()
        
        # Add 1 to all labels to ensure background != 0
        markers = markers + 1
        
        # Mark the unknown region with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        image_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_color, markers)
        
        # Process the segments
        segments_data = []
        for marker_id in range(2, markers.max() + 1):
            # Create mask for this segment
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[markers == marker_id] = 255
            
            # ENHANCEMENT: Additional mask cleanup
            # Remove very small holes and isolated regions
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(1))
            
            # Extract contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                continue
                
            # Take the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Calculate area and filter by minimum size
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Basic measurements
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 1e-5:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w//2, y + h//2
                
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0
            
            # ENHANCEMENT: Calculate convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if len(hull) > 2 else area + 1
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Smooth the contour with adaptive epsilon
            epsilon = 0.002 * perimeter
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Detect shape with improved analysis
            shape_info = self.detect_shape(contour, circularity, convexity)
            
            # Store binary mask for later CRF refinement
            mask_binary = mask // 255
            
            # Extract geometric elements with improved precision
            geometric_elements = self.extract_geometric_elements(
                smoothed_contour, shape_info["type"])
            
            # ENHANCEMENT: Quality score calculation integrating multiple factors
            shape_regularity = (circularity + convexity) / 2
            size_factor = min(1.0, area / 2000)  # normalize relative to a reference size
            complexity_factor = min(1.0, (20.0 / (len(smoothed_contour) + 1)) * shape_regularity)
            
            quality_score = 0.4 * shape_regularity + 0.3 * size_factor + 0.3 * complexity_factor
            
            # Create segment
            segment = {
                'id': marker_id,
                'contour': contour,
                'smoothed_contour': smoothed_contour,
                'area': area,
                'perimeter': perimeter,
                'dimensions': (w, h),
                'position': (x, y),
                'center': (cX, cY),
                'mask': mask_binary,
                'shape': shape_info,
                'circularity': circularity,
                'convexity': convexity,  # New attribute
                'quality': min(1.0, max(0.5, quality_score)),  # Minimum quality of 0.5 for traditional segmentation
                'geometric_elements': geometric_elements
            }
            
            segments_data.append(segment)
            
        return segments_data

    def segment_image(self, image_color, image_gray, min_area=100):
        """
        Segment the image using SAM if available, fall back to traditional methods
        For large images, segment by tiles and reassemble
        """
        # Check if we need to split the image into tiles
        should_split = self.tile_splitter.should_split(image_color)
        
        if should_split:
            print(f"Image size ({image_color.shape[1]}x{image_color.shape[0]}) exceeds threshold.")
            print(f"Splitting image into tiles of {self.tile_splitter.tile_size}x{self.tile_splitter.tile_size} pixels...")
            
            # Split the image into tiles
            tiles = self.tile_splitter.split_image(image_color)
            print(f"Created {len(tiles)} tiles.")
            
            # Process each tile
            tile_segments = []
            method = None
            
            for i, (tile, (x_offset, y_offset)) in enumerate(tiles):
                print(f"Processing tile {i+1}/{len(tiles)} at position ({x_offset}, {y_offset})...")
                
                # Convert tile to grayscale
                if len(tile.shape) == 3:
                    tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                else:
                    tile_gray = tile
                
                # Process this tile
                if self.sam_available:
                    segments = self.segment_with_sam(tile)
                    if segments and len(segments) > 0:
                        tile_segments.append((segments, (x_offset, y_offset)))
                        method = "sam"
                        continue
                
                # Fallback to traditional method if SAM fails or is not available
                segments = self.segment_with_traditional(tile_gray, min_area)
                if segments and len(segments) > 0:
                    tile_segments.append((segments, (x_offset, y_offset)))
                    if method is None:
                        method = "traditional"
            
            # Reassemble segments from all tiles
            segments_data = self.tile_splitter.reassemble_segments(tile_segments, image_color.shape)
            
            # Apply CRF refinement to all segments at once
            self._refine_all_segments_with_crf(segments_data, image_color)
            
            return segments_data, method or "traditional"
        else:
            # Process the whole image at once
            if self.sam_available:
                # Try SAM segmentation first
                segments_data = self.segment_with_sam(image_color)
                
                if segments_data and len(segments_data) > 0:
                    print(f"SAM segmentation successful, found {len(segments_data)} objects.")
                    # Apply CRF refinement
                    self._refine_all_segments_with_crf(segments_data, image_color)
                    return segments_data, "sam"
                    
            # Fallback to traditional methods
            print("Using traditional segmentation methods.")
            segments_data = self.segment_with_traditional(image_gray, min_area)
            if segments_data and len(segments_data) > 0:
                # Apply CRF refinement
                self._refine_all_segments_with_crf(segments_data, image_color)
                print(f"Traditional segmentation found {len(segments_data)} objects.")
                
            return segments_data, "traditional"
    
    def _refine_all_segments_with_crf(self, segments_data, image_color):
        """
        Correction du problème "cannot access local variable 'mask'"
        """
        print("Applying CRF refinement to all segments...")
        
        if not segments_data:
            print("No segments to refine")
            return
            
        # Pre-filter image
        filtered_image = cv2.GaussianBlur(image_color, (3, 3), 0.5)
        
        for i, segment in enumerate(segments_data):
            try:
                # Vérifier explicitement l'existence de la clé 'mask'
                if 'mask' not in segment:
                    print(f"Warning: Segment {i} has no mask key")
                    continue
                    
                # Assigner la variable mask DANS le bloc try
                segment_mask = segment['mask']
                
                # Vérifier que le masque existe et n'est pas vide
                if segment_mask is None or segment_mask.size == 0:
                    print(f"Warning: Segment {i} has empty mask")
                    continue
                
                # Convertir explicitement le masque en uint8
                segment_mask = segment_mask.astype(np.uint8)
                segment['mask'] = segment_mask  # Mettre à jour dans le segment
                
                # Skip small objects
                if segment.get('area', 0) < 200:
                    continue
                
                # Adapter les paramètres CRF selon la forme
                if segment.get('circularity', 0) > 0.85:
                    crf_params = {
                        'gaussian_sxy': 2,
                        'gaussian_compat': 5,
                        'bilateral_sxy': 50,
                        'bilateral_srgb': 8,
                        'bilateral_compat': 15,
                        'iterations': 8
                    }
                else:
                    crf_params = {
                        'gaussian_sxy': 3,
                        'gaussian_compat': 4,
                        'bilateral_sxy': 70,
                        'bilateral_srgb': 13,
                        'bilateral_compat': 10,
                        'iterations': 10
                    }
                
                # Appliquer CRF
                try:
                    refined_mask = self.crf_processor.apply_crf(filtered_image, segment_mask, crf_params)
                    
                    # Assurer que le masque raffiné est uint8
                    refined_mask = refined_mask.astype(np.uint8)
                    
                    # Vérifier que le raffinement n'a pas détruit l'objet
                    if np.sum(refined_mask) > np.sum(segment_mask) * 0.25:
                        segment['mask'] = refined_mask
                    
                except Exception as e:
                    print(f"Error applying CRF: {e}")
                    
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                
        print("CRF refinement completed")
##############################################################################################################################

    def visualize_segmentation(self, image, segments_data, method="custom"):
        """
        Create a visualization of segmentation results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show the original image
        ax.imshow(image)
        
        # Overlay segmentation results
        segment_colors = plt.cm.tab20(np.linspace(0, 1, 20))  # Color map for segments
        
        # Draw each segment with a unique color
        for i, segment in enumerate(segments_data):
            # Get color from segment if available, otherwise use color cycle
            if 'color' in segment and segment['color'] is not None:
                color = segment['color']
            else:
                color = segment_colors[i % 20][:3]  # Get RGB only
            
            # Draw contour
            contour = segment['contour']
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=2)
            
            # Add segment number at center
            if 'center' in segment:
                cx, cy = segment['center']
                ax.text(cx, cy, str(i+1), color='white', fontsize=8, 
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle', color=color, alpha=0.7))
        
        # Add title with method and segment count
        ax.set_title(f"Segmentation Results: {method.title()} Method - {len(segments_data)} Objects")
        ax.axis('off')
        
        # Add legend with shape information
        shapes = {}
        for segment in segments_data:
            if 'shape' in segment and 'type' in segment['shape']:
                shape_type = segment['shape']['type']
                if shape_type not in shapes:
                    shapes[shape_type] = 0
                shapes[shape_type] += 1
        
        if shapes:
            legend_text = "Detected Shapes:\n" + "\n".join([f"- {shape}: {count}" for shape, count in shapes.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        return fig

    def export_to_dxf(self, segments_data, output_file, line_thickness=0.35):
        """
        Export segments to DXF - Version compatible avec toutes les versions d'ezdxf
        """
        try:
            import ezdxf
            from ezdxf.math import Vec2
            import datetime
            
            # Create new DXF drawing
            doc = ezdxf.new('R2010')
            
            # Add layers for different shape types
            doc.layers.new(name='CIRCLES', dxfattribs={'color': 1})  # Red
            doc.layers.new(name='RECTANGLES', dxfattribs={'color': 3})  # Green
            doc.layers.new(name='POLYGONS', dxfattribs={'color': 5})  # Blue
            doc.layers.new(name='ELLIPSES', dxfattribs={'color': 6})  # Magenta
            doc.layers.new(name='COMPLEX', dxfattribs={'color': 7})  # White
            doc.layers.new(name='TEXT', dxfattribs={'color': 2})  # Yellow
            doc.layers.new(name='TITLE', dxfattribs={'color': 4})  # Cyan
            
            # Get model space
            msp = doc.modelspace()
            
            # Add title block - Version compatible
            title = msp.add_text("SAM Professional Segmentation Export")
            title.dxf.layer = 'TITLE'
            title.dxf.height = 5
            title.dxf.insert = (0, -10)
            
            # Add metadata - Version compatible
            date_text = msp.add_text(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            date_text.dxf.layer = 'TITLE'
            date_text.dxf.height = 2
            date_text.dxf.insert = (0, -15)
            
            obj_text = msp.add_text(f"Objects: {len(segments_data)}")
            obj_text.dxf.layer = 'TITLE'
            obj_text.dxf.height = 2
            obj_text.dxf.insert = (0, -18)
            
            # Add each segment to the DXF
            for i, segment in enumerate(segments_data):
                # Skip segments with no contour or very small area
                if 'contour' not in segment or segment.get('area', 0) < 5:
                    continue
                    
                # Get shape type and contour
                shape_type = segment.get('shape', {}).get('type', 'complex_shape')
                contour = segment['contour']
                
                # Set layer based on shape type
                if 'circle' in shape_type:
                    layer = 'CIRCLES'
                elif 'rectangle' in shape_type or 'square' in shape_type:
                    layer = 'RECTANGLES'
                elif 'polygon' in shape_type or 'triangle' in shape_type:
                    layer = 'POLYGONS'
                elif 'ellipse' in shape_type:
                    layer = 'ELLIPSES'
                else:
                    layer = 'COMPLEX'
                
                # Convert OpenCV contour to DXF format
                dxf_points = [(pt[0][0], pt[0][1]) for pt in contour]
                # Close the polyline
                if len(dxf_points) > 2 and dxf_points[0] != dxf_points[-1]:
                    dxf_points.append(dxf_points[0])
                    
                # Add the polyline to model space
                msp.add_lwpolyline(
                    points=dxf_points,
                    dxfattribs={
                        'layer': layer,
                        'lineweight': int(line_thickness * 100),
                        'closed': True
                    }
                )
                
                # Add object number at center
                if 'center' in segment:
                    cx, cy = segment['center']
                    num_text = msp.add_text(str(i + 1))
                    num_text.dxf.layer = 'TEXT'
                    num_text.dxf.height = 2
                    num_text.dxf.insert = (cx, cy)
                    # Utilisation d'une méthode universelle pour le centrage du texte
                    num_text.dxf.halign = 1  # Center X
                    num_text.dxf.valign = 1  # Center Y
            
            # Save the DXF file
            doc.saveas(output_file)
            print(f"DXF file saved: {output_file}")
            
        except Exception as e:
            print(f"Error during DXF export: {e}")
            import traceback
            traceback.print_exc()




########################################################################################################################################
class ProfessionalSegmentationPipeline:
    """
    Advanced segmentation pipeline with:
    - Interactive file selection
    - Image tiling for large images
    - SAM segmentation with CRF refinement
    - Interactive contour editing
    - Part-by-part selection with color assignment
    - High-quality DXF export
    """
    def __init__(self, sam_checkpoint=None, debug=False, line_thickness=0.35):
        self.debug = debug
        self.sam_checkpoint = sam_checkpoint
        self.line_thickness = line_thickness
        self.image_processor = EnhancedImageProcessor(
            sam_checkpoint=sam_checkpoint, 
            debug=debug, 
            line_thickness=line_thickness
        )
        self.region_selector = InteractiveRegionSelector()
        self.file_selector = InteractiveFileSelector()
        
    def select_input_file(self):
        """Let the user select an input file interactively"""
        print("\nSelect an image to process:")
        input_path = self.file_selector.select_file("Select an image to process")
        
        if not input_path:
            print("No file selected. Operation canceled.")
            return None
            
        print(f"Selected file: {input_path}")
        return input_path
        
    def process_image(self, input_image=None, output_dir="./exports", interactive=True):
        """Complete pipeline with interactive file selection, tiling, SAM + CRF, and DXF export"""
        print("\n=== STARTING PIPELINE ===\n")
        
        # Interactive file selection if no input provided
        if input_image is None:
            input_image = self.select_input_file()
            if not input_image:
                return False
        
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image not found: {input_image}")
            
        print(f"Processing image: {input_image}")
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths
        basename = os.path.splitext(os.path.basename(input_image))[0]
        output_dxf = os.path.join(output_dir, f"{basename}_segmentation.dxf")
        output_viz = os.path.join(output_dir, f"{basename}_visualization.png")
        
        # 1. Load image
        print("\n1. Loading image...")
        try:
            image_color, image_gray = self.image_processor.load_image(input_image)
            print(f"  - Image loaded: {image_color.shape[1]}x{image_color.shape[0]} pixels")
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
        
        # 2. Interactive selection of regions (if enabled)
        roi_mask = None
        if interactive:
            print("\n2. Interactive selection of regions of interest...")
            roi_mask = self.region_selector.select_regions(image_color, 
                        title="Select regions to segment (then click 'Done')")
            
            # If no selections were made, use the whole image
            if roi_mask is None or np.sum(roi_mask) == 0:
                print("  - No region selected, using the entire image.")
                roi_mask = None
            else:
                # Show selected regions count
                num_regions = cv2.connectedComponents(roi_mask)[0] - 1
                print(f"  - {num_regions} region(s) selected")
                
                # Save the mask for reference
                mask_path = os.path.join(output_dir, f"{basename}_regions.png")
                cv2.imwrite(mask_path, roi_mask * 255)
                print(f"  - Selection mask saved: {mask_path}")
        
        # 3. Advanced segmentation (with ROI and tiling support)
        print(f"\n{'3' if interactive else '2'}. Advanced segmentation with SAM and CRF...")
        try:
            # If we have an ROI mask, apply it to the image
            if roi_mask is not None:
                # Only process the selected regions
                # Create a masked version of the image
                masked_color = image_color.copy()
                for c in range(3):  # Apply mask to each channel
                    masked_color[:,:,c] = masked_color[:,:,c] * roi_mask
                
                # Process the masked image with tiling
                segments_data, method = self.image_processor.segment_image(masked_color, image_gray)
            else:
                # Process the whole image with tiling
                segments_data, method = self.image_processor.segment_image(image_color, image_gray)
            
            if not segments_data or len(segments_data) == 0:
                print("No segments were detected!")
                return False
                
            print(f"  - {len(segments_data)} objects detected with method: {method}")
        except Exception as e:
            print(f"Error during segmentation: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
            
        # 4. Interactive contour editing (if enabled)
        if interactive:
            print("\n4. Interactive contour editing...")
            try:
                # Initialize interactive tool with the image and detected segments
                tool = self.image_processor.interactive_tool
                
                # Update working mask from segments
                combined_mask = np.zeros(image_gray.shape, dtype=np.uint8)
                for segment in segments_data:
                    combined_mask = np.logical_or(combined_mask, segment['mask']).astype(np.uint8)
                
                tool.working_mask = combined_mask
                tool._update_contours()
                
                # Start interactive mode
                print("  - Starting interactive editing mode...")
                print("  - Use mouse and keyboard controls to edit contours")
                tool.start_interactive_mode()
                
                # Update segments_data based on edited contours
                if tool.working_mask is not None:
                    # Extract new contours from the edited mask
                    contours, _ = cv2.findContours(
                        tool.working_mask.copy(), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_NONE
                    )
                    
                    # If contours were edited, reconstruct segments_data
                    if contours:
                        # We'll rebuild the segments data from the edited contours
                        updated_segments = []
                        
                        for i, contour in enumerate(contours):
                            if cv2.contourArea(contour) < 50:  # Skip tiny contours
                                continue
                                
                                                       # Get contour mask
                            mask = np.zeros_like(tool.working_mask)
                            cv2.drawContours(mask, [contour], 0, 1, -1)
                            
                            # Basic measurements
                            area = cv2.contourArea(contour)
                            perimeter = cv2.arcLength(contour, True)
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Calculate center
                            M = cv2.moments(contour)
                            if M["m00"] > 1e-5:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                            else:
                                cX, cY = x + w//2, y + h//2
                                
                            # Calculate circularity
                            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0
                            
                            # Calculate convexity
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull) if len(hull) > 2 else area + 1
                            convexity = area / hull_area if hull_area > 0 else 0
                            
                            # Smooth contour
                            epsilon = 0.002 * perimeter
                            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                            
                            # Detect shape
                            shape_info = self.image_processor.detect_shape(contour, circularity, convexity)
                            
                            # Extract geometric elements
                            geometric_elements = self.image_processor.extract_geometric_elements(
                                smoothed_contour, shape_info["type"])
                            
                            # Use original segment color if available
                            color = None
                            if i in tool.contour_colors:
                                color = tool.contour_colors[i]
                            elif i < len(segments_data) and 'color' in segments_data[i]:
                                color = segments_data[i]['color']
                                
                            # Create segment with new data
                            segment = {
                                'id': i + 1,
                                'contour': contour,
                                'smoothed_contour': smoothed_contour,
                                'area': area,
                                'perimeter': perimeter,
                                'dimensions': (w, h),
                                'position': (x, y),
                                'center': (cX, cY),
                                'mask': mask,
                                'shape': shape_info,
                                'circularity': circularity,
                                'convexity': convexity,
                                'quality': 0.8,  # Default quality for user-edited contours
                                'geometric_elements': geometric_elements,
                                'color': color
                            }
                            
                            updated_segments.append(segment)
                            
                        # Replace the segments data with the updated version
                        segments_data = updated_segments
                        
                        print(f"  - Updated segments: {len(segments_data)} objects")
            except Exception as e:
                print(f"Error during interactive editing: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # 5. Visualization of results
        print(f"\n{'5' if interactive else '3'}. Generating visualizations...")
        try:
            fig = self.image_processor.visualize_segmentation(image_color, segments_data, method)
            fig.savefig(output_viz, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Visualization saved: {output_viz}")
        except Exception as e:
            print(f"Error during visualization: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
        # 6. Export to DXF format
        print(f"\n{'6' if interactive else '4'}. DXF export with enhanced line thickness...")
        try:
            self.image_processor.export_to_dxf(segments_data, output_dxf, self.line_thickness)
            print(f"  - DXF file created: {output_dxf}")
        except Exception as e:
            print(f"Error during DXF export: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
                
        print("\n=== PROCESSING COMPLETE ===\n")
        
        # Results summary
        print("Summary:")
        print(f"  - Input image: {input_image}")
        print(f"  - DXF file: {output_dxf}")
        print(f"  - Visualization: {output_viz}")
        print(f"  - Objects detected: {len(segments_data)}")
        
        return True


def main():
    """
    Fonction principale du pipeline de segmentation professionnelle
    """
    # Configurer le parser d'arguments
    parser = argparse.ArgumentParser(description='Professional SAM Segmentation Pipeline')
    parser.add_argument('--image', help='Path to input image file')
    parser.add_argument('--checkpoint', default='./models/sam_vit_h_4b8939.pth', 
                        help='Path to SAM checkpoint file')
    parser.add_argument('--interactive', action='store_true', default=True,
                        help='Enable interactive mode for guided segmentation')
    parser.add_argument('--export-dir', default='./exports',
                        help='Directory to save exports')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional outputs')
    parser.add_argument('--line-thickness', type=float, default=0.35,
                        help='Line thickness for DXF export (mm)')
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Créer le dossier d'export s'il n'existe pas
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    
    # Initialiser le pipeline de segmentation
    print("\n=== PROFESSIONAL SEGMENTATION PIPELINE ===\n")
    segmentation_pipeline = ProfessionalSegmentationPipeline(
        sam_checkpoint=args.checkpoint,
        debug=args.debug,
        line_thickness=args.line_thickness
    )
    
    # Exécuter le pipeline complet en utilisant la méthode process_image
    if args.image:
        print(f"Input image: {args.image}")
    else:
        print("No image specified, will use interactive selection")
        
    # Appeler process_image qui gère tout le pipeline
    segmentation_pipeline.process_image(
        input_image=args.image,  # None déclenchera la sélection interactive
        output_dir=args.export_dir,
        interactive=args.interactive
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())