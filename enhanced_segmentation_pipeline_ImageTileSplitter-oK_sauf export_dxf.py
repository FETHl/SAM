#!/usr/bin/env python3
"""
Pipeline avancé de segmentation d'image:
- Interface interactive pour sélection des images
- Division automatique des grandes images en sections de 512x512
- Segmentation avancée avec SAM (Segment Anything Model)
- Affinage des contours avec CRF
- Export DXF haute précision avec gestion des épaisseurs de ligne
- Visualisation détaillée des résultats
"""

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
from tkinter import filedialog

# Constants
CURRENT_USER = "FETHl"
CURRENT_DATE = "2025-03-10 12:45:21"
VERSION = "4.2.0"  # Updated version with image splitting and interactive file selection

# SAM Model Configuration
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image splitting configuration
TILE_SIZE = 512
MAX_SIZE_WITHOUT_SPLITTING = 1024


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
                    
                # Adjust geometric elements
                if 'geometric_elements' in segment:
                    for element in segment['geometric_elements']:
                        if element['type'] == 'cercle':
                            cx, cy = element['center']
                            element['center'] = (cx + x_offset, cy + y_offset)
                        elif element['type'] in ['rectangle', 'polygone', 'contour']:
                            if 'points' in element:
                                for i, point in enumerate(element['points']):
                                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                                        element['points'][i] = [point[0] + x_offset, point[1] + y_offset]
                        elif element['type'] == 'lignes':
                            if 'segments' in element:
                                for i, (start, end) in enumerate(element['segments']):
                                    element['segments'][i] = (
                                        [start[0] + x_offset, start[1] + y_offset],
                                        [end[0] + x_offset, end[1] + y_offset]
                                    )
                        elif element['type'] == 'ellipse':
                            cx, cy = element['center']
                            element['center'] = (cx + x_offset, cy + y_offset)
                
                # Add the adjusted segment
                all_segments.append(segment)
                
        if self.debug:
            print(f"Reassembled {len(all_segments)} segments from all tiles.")
            
        return all_segments


class CRFPostProcessor:
    """
    Classe pour le post-traitement des masques avec CRF (Conditional Random Fields)
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.crf_available = self._check_crf()
        
    def _check_crf(self):
        """Vérifier si PyDenseCRF est installé"""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
            return True
        except ImportError:
            print("AVERTISSEMENT: PyDenseCRF n'est pas installé. Utilisation de l'alternative.")
            print("Pour installer: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")
            return False
    
    def apply_crf(self, image, mask, crf_params=None):
        """
        Affiner les contours d'un masque en utilisant DenseCRF
        Version corrigée pour éviter l'erreur d'indice hors limites
        """
        if not self.crf_available:
            return self.apply_crf_alternative(image, mask)
            
        if crf_params is None:
            crf_params = {
                'gaussian_sxy': 3,
                'gaussian_compat': 3,
                'bilateral_sxy': 80,
                'bilateral_srgb': 13,
                'bilateral_compat': 10,
                'iterations': 10
            }
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
            
            # Convert inputs to proper format and ensure contiguity
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask.astype(np.uint32))
            
            h, w = image.shape[:2]
            
            # Initialize CRF
            if self.debug:
                print("Application de DenseCRF pour affiner les contours...")
            
            # Create proper unary potential with safeguards
            # Create label array with proper binary encoding (0 for background, 1 for foreground)
            # This is critical to avoid the index out-of-bounds issue
            labels = np.zeros((2, h, w), dtype=np.float32)
            
            # Safety check for mask values
            safe_mask = np.clip(mask, 0, 1)  # Ensure mask has only 0 and 1 values
            
            # Set probabilities for background and foreground
            labels[0, safe_mask == 0] = 1.0  # Background probability where mask is 0
            labels[1, safe_mask == 1] = 1.0  # Foreground probability where mask is 1
            
            # Create CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes (bg, fg)
            
            # Create unary potentials from labels
            unary = unary_from_labels(labels, 2, gt_prob=0.7)
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
            if image.dtype != np.uint8:
                proc_image = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                proc_image = image
                
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
            
            return map_soln.astype(np.uint8)

        except Exception as e:
            print(f"Erreur lors de l'application de CRF: {e}")
            # Fallback to alternative method in case of error
            return self.apply_crf_alternative(image, mask)

    def apply_crf_alternative(self, image, mask, sigma=1.0, radius=2):
        """Alternative à CRF utilisant des techniques de traitement d'images classiques"""
        if self.debug:
            print("Utilisation de l'alternative à CRF pour affiner les contours...")
        
        try:
            # Conversion en niveau de gris pour le gradient
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
            
            # Lissage gaussien adaptatif pour réduire le bruit
            smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
            
            # Seuillage adaptatif pour récupérer un masque binaire
            binary = (smoothed > 0.5).astype(np.uint8)
            
            # Opérations morphologiques pour améliorer les contours
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            refined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Supprimer les petites régions isolées
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
            
            # Garder seulement les composantes de taille suffisante
            sizes = stats[:, -1]
            refined_mask = np.zeros_like(refined)
            
            # Considérer la composante 0 comme l'arrière-plan
            min_size = 50  # Taille minimale des objets à conserver
            for i in range(1, num_labels):
                if sizes[i] >= min_size:
                    refined_mask[labels == i] = 1
                    
            if self.debug:
                print("Affinage alternatif des contours terminé")
            return refined_mask
        
        except Exception as e:
            print(f"Erreur lors de l'application de l'alternative à CRF: {e}")
            return mask  # Retourner le masque original en cas d'erreur


class SegmentAnythingHelper:
    """
    Helper class to integrate Meta's Segment Anything Model (SAM)
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
    
    def generate_masks(self, image, num_points=10):
        """Generate masks using the SAM model with automatic points"""
        if not self.is_initialized():
            return None
            
        try:
            # Set the image for embedding
            self.predictor.set_image(image)
            
            # Generate automatic points for segmentation
            height, width = image.shape[:2]
            points = []
            
            # Grid-based sampling of points
            rows, cols = 3, 3
            for i in range(rows):
                for j in range(cols):
                    x = int(width * (j + 0.5) / cols)
                    y = int(height * (i + 0.5) / rows)
                    points.append([x, y])
                    
            # Add some random points
            for _ in range(num_points - len(points)):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                points.append([x, y])
                
            # Convert points to numpy arrays
            input_points = np.array(points)
            input_labels = np.ones(len(points))  # Assume all foreground
            
            # Generate masks
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Filter masks by score
            good_masks = []
            for mask, score in zip(masks, scores):
                if score > 0.8:  # Only include high-confidence masks
                    good_masks.append(mask)
            
            return good_masks
        
        except Exception as e:
            print(f"Error generating SAM masks: {str(e)}")
            return None
    
    def automatic_segmentation(self, image):
        """Perform fully automatic segmentation with SAM"""
        if not self.is_initialized():
            return None
            
        try:
            # Import for automatic mask generation
            from segment_anything import SamAutomaticMaskGenerator
            
            # Get the SAM model from predictor
            sam = self.predictor.model
            
            # Create automatic mask generator with fine-tuned parameters
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Minimum area to reduce noise
            )
            
            # Generate masks
            masks = mask_generator.generate(image)
            
            return masks
            
        except Exception as e:
            print(f"Error in automatic SAM segmentation: {str(e)}")
            return None


class EnhancedDXFExporter:
    """
    Class to handle enhanced DXF export capabilities
    """
    def __init__(self, line_thickness=0.35):
        """
        Initialize the DXF exporter with specified line thickness
        """
        self.line_thickness = line_thickness  # mm
        
        # Line thickness map for different entity types
        self.thickness_map = {
            "CERCLES": 0.5,        # Thicker for circles
            "ELLIPSES": 0.5,       # Thicker for ellipses
            "RECTANGLES": 0.35,    # Medium for rectangles
            "TRIANGLES": 0.35,     # Medium for triangles
            "POLYGONES": 0.35,     # Medium for polygons
            "LIGNES": 0.25,        # Thinner for lines
            "CONTOURS": 0.18,      # Thinnest for contours
            "FORMES_COMPLEXES": 0.3,  # Medium for complex shapes
            "ANNOTATIONS": 0.18,   # Thin for annotations
            "METADATA": 0.18       # Thin for metadata
        }
        
        # Standardized lineweights in AutoCAD
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
        """
        # Enhanced layer definitions with descriptions and lineweights
        layer_definitions = {
            "CERCLES": {"color": 1, "desc": "Formes circulaires", "linetype": "CONTINUOUS", 
                       "lineweight": self.thickness_map["CERCLES"]},
            "ELLIPSES": {"color": 2, "desc": "Formes elliptiques", "linetype": "CONTINUOUS",
                        "lineweight": self.thickness_map["ELLIPSES"]},
            "RECTANGLES": {"color": 3, "desc": "Formes rectangulaires", "linetype": "CONTINUOUS",
                          "lineweight": self.thickness_map["RECTANGLES"]},
            "TRIANGLES": {"color": 4, "desc": "Formes triangulaires", "linetype": "CONTINUOUS",
                         "lineweight": self.thickness_map["TRIANGLES"]},
            "POLYGONES": {"color": 5, "desc": "Formes polygonales", "linetype": "CONTINUOUS",
                         "lineweight": self.thickness_map["POLYGONES"]},
            "LIGNES": {"color": 6, "desc": "Lignes et segments", "linetype": "CONTINUOUS",
                      "lineweight": self.thickness_map["LIGNES"]},
            "CONTOURS": {"color": 7, "desc": "Contours détectés", "linetype": "CONTINUOUS",
                        "lineweight": self.thickness_map["CONTOURS"]},
            "FORMES_COMPLEXES": {"color": 8, "desc": "Formes irrégulières", "linetype": "CONTINUOUS",
                               "lineweight": self.thickness_map["FORMES_COMPLEXES"]},
            "ANNOTATIONS": {"color": 9, "desc": "Textes et annotations", "linetype": "CONTINUOUS",
                          "lineweight": self.thickness_map["ANNOTATIONS"]},
            "METADATA": {"color": 9, "desc": "Informations sur le document", "linetype": "CONTINUOUS",
                       "lineweight": self.thickness_map["METADATA"]}
        }
        
        # Create the layers
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
        """Create a new DXF document with optimized settings"""
        # Create new document with R2018 format for better compatibility
        doc = ezdxf.new('R2018')
        
        # Set up layers
        layer_definitions = self.setup_layers(doc)
        
        # Set up additional document settings
        doc.header['$LUNITS'] = 2  # Set units to decimal
        doc.header['$INSUNITS'] = 4  # Set insertion units to mm
        doc.header['$MEASUREMENT'] = 1  # Set measurement to metric
        
        # Set up layout settings
        msp = doc.modelspace()
        
        return doc, msp, layer_definitions

    def _add_dxf_element(self, msp, element, layer_defs, segment_id):
        """Add a geometric element to the DXF document with enhanced thickness"""
        element_type = element['type']
        
        # Map element type to layer
        if element_type == 'cercle':
            layer_name = "CERCLES"
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
            
            # Add center point for reference
            msp.add_point(
                (float(center[0]), float(center[1]), 0),
                dxfattribs={"layer": layer_name}
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
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
            except Exception as e:
                print(f"Error creating rectangle in DXF: {e}")
                
        elif element_type == 'polygone':
            # Determine if it's a triangle or other polygon
            points = element['points']
            if len(points) == 3:
                layer_name = "TRIANGLES"
            else:
                layer_name = "POLYGONES"
                
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
                
                # Extract 2D points safely
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
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
            except Exception as e:
                print(f"Error creating contour in DXF: {e}")
            
        elif element_type == 'lignes':
            layer_name = "LIGNES"
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
            layer_name = "FORMES_COMPLEXES"
            
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
                
                # Create label with segment ID
                msp.add_text(
                    str(segment_id),
                    height=5.0,
                    dxfattribs={
                        "layer": "ANNOTATIONS",
                        "lineweight": self.get_closest_std_lineweight(
                            self.thickness_map["ANNOTATIONS"])
                    },
                    position=center_pos
                )
            except Exception as e:
                print(f"Error adding text annotation in DXF: {e}")
            
    def _add_dxf_metadata(self, msp, doc, layer_defs):
        """Add metadata to the DXF file"""
        # Set document variables
        doc.header['$ACADVER'] = "AC1027"  # AutoCAD 2018
        
        # Add useful variables
        for var, val in [
            ("$LUNITS", 2),         # Set units to decimal
            ("$INSUNITS", 4),       # Set insertion units to mm
            ("$MEASUREMENT", 1),     # Set measurement to metric
            ("$LIMCHECK", 1),        # Enable limits checking
            ("$FILLMODE", 1),        # Enable fill mode
            ("$MIRRTEXT", 1),        # Mirror text with parent
            ("$LTSCALE", 1.0),       # Set global linetype scale
            ("$TEXTSIZE", 5.0)       # Set default text size
        ]:
            if var in doc.header:
                doc.header[var] = val
        
        # Add title block information
        dxf_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(
                self.thickness_map["METADATA"])
        }
        
        # Main title
        msp.add_text(
            "Plan de segmentation avancée avec SAM",
            height=12.0,
            dxfattribs=dxf_attribs,
            position=(0, -20, 0)
        )
        
        # Creator and date
        msp.add_text(
            f"Créé par: {CURRENT_USER} • Date: {CURRENT_DATE}",
            height=8.0,
            dxfattribs=dxf_attribs,
            position=(0, -35, 0)
        )
        
        # Version information
        msp.add_text(
            f"Enhanced SAM Segmentation v{VERSION}",
            height=6.0,
            dxfattribs=dxf_attribs,
            position=(0, -50, 0)
        )
        
    def export_to_dxf(self, segments_data, output_path, line_thickness=0.35):
        """Export segments to DXF with enhanced line thickness"""
        print(f"Exporting to DXF with enhanced line thickness: {output_path}")
        
        # Update line thickness if provided
        self.line_thickness = line_thickness
        
        # Create a new DXF document
        doc, msp, layer_defs = self.create_dxf_document()
        
        # Process segments by quality
        high_quality = []
        medium_quality = []
        low_quality = []
        
        for segment in segments_data:
            if segment['quality'] > 0.75:
                high_quality.append(segment)
            elif segment['quality'] > 0.5:
                medium_quality.append(segment)
            else:
                low_quality.append(segment)
                
        # Export high and medium quality segments
        export_segments = high_quality + medium_quality
        
        print(f"Segments to export: {len(export_segments)} "
              f"(high: {len(high_quality)}, medium: {len(medium_quality)}, "
              f"filtered: {len(low_quality)})")
        
        # Process each segment
        for segment in tqdm(export_segments, desc="Exporting objects to DXF"):
            for element in segment['geometric_elements']:
                self._add_dxf_element(msp, element, layer_defs, segment['id'])
        
        # Add metadata and annotations
        self._add_dxf_metadata(msp, doc, layer_defs)
        
        # Save the document
        doc.saveas(output_path)
        print(f"DXF file exported successfully with {len(export_segments)} objects.")
        return doc


class EnhancedImageProcessor:
    """
    Enhanced image processor combining SAM segmentation, CRF refinement, and DXF export
    """
    def __init__(self, sam_checkpoint=None, debug=False, line_thickness=0.35):
        """Initialize the advanced image processor"""
        self.debug = debug
        
        # Initialize helpers
        self.crf_processor = CRFPostProcessor(debug=debug)
        self.sam_helper = SegmentAnythingHelper(checkpoint_path=sam_checkpoint, device=SAM_DEVICE)
        self.dxf_exporter = EnhancedDXFExporter(line_thickness=line_thickness)
        self.tile_splitter = ImageTileSplitter(tile_size=TILE_SIZE, max_size=MAX_SIZE_WITHOUT_SPLITTING, debug=debug)
        
        # Mark whether SAM is available
        self.sam_available = self.sam_helper.is_initialized()
        
    def load_image(self, filepath):
        """Load an image with optimized settings"""
        print(f"Chargement de l'image : {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        # Load image with best quality
        image_color = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image_color is None:
            raise ValueError(f"Impossible de charger l'image : {filepath}")

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
            print(f"Image chargée: {image_color.shape[1]}x{image_color.shape[0]} pixels")
            print(f"Format: {image_color.dtype}, Range: [{image_gray.min()}-{image_gray.max()}]")
            
        return image_color, image_gray

    def preprocess_image(self, image, denoise_strength=9):
        """Enhanced preprocessing to improve quality before segmentation"""
        # Create a copy to avoid modifying the original
        processed = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        
        # Apply bilateral filtering for edge-preserving noise reduction
        processed = cv2.bilateralFilter(processed, denoise_strength, 75, 75)
        
        # Apply gamma correction to enhance details in dark areas
        gamma = 1.1
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
        
        return processed

    def segment_with_sam(self, image_color):
        """Segment image using Segment Anything Model"""
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
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            area = mask_data["area"]
            
            # Skip tiny segments (likely noise)
            if area < 100:
                continue
                
            # Get contours from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                continue
                
            # Get the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Apply smoothing to contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w//2, y + h//2
                
            # Detect shape type
            shape_info = self.detect_shape(contour, circularity)
            
            # Raffinement CRF du masque - this will be applied to the entire reassembled image later
            mask_binary = mask // 255
            
            # Extract geometric elements for DXF export
            geometric_elements = self.extract_geometric_elements(smoothed_contour, shape_info["type"])
            
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
                'quality': mask_data["stability_score"] if "stability_score" in mask_data else 0.8,
                'geometric_elements': geometric_elements
            }
            
            segments_data.append(segment)
        
        return segments_data

   
    def segment_with_traditional(self, image, min_area=100):
        """Traditional segmentation when SAM is not available"""
        print("Using traditional segmentation methods...")
        
        # Preprocess the image
        enhanced = self.preprocess_image(image)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 30, 150)
        
        # Improve edge connectivity
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Thresholding
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply watershed algorithm
        # First, distance transform
        dist_transform = cv2.distanceTransform(binary_otsu, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        sure_bg = cv2.dilate(binary_otsu, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
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
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = x + w//2, y + h//2
                
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Smooth the contour
            epsilon = 0.002 * perimeter
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Detect shape
            shape_info = self.detect_shape(contour, circularity)
            
            # Store binary mask for later CRF refinement
            mask_binary = mask // 255
            
            # Extract geometric elements
            geometric_elements = self.extract_geometric_elements(
                smoothed_contour, shape_info["type"])
            
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
                'quality': 0.9 if circularity > 0.7 else 0.7,  # Estimate quality
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
        """Apply CRF refinement to all segments"""
        print("Applying CRF refinement to all segments...")
        for segment in segments_data:
            mask = segment['mask']
            refined_mask = self.crf_processor.apply_crf(image_color, mask)
            segment['mask'] = refined_mask  # Update with refined mask
        

    def detect_shape(self, contour, circularity):
        """Enhanced shape detection with more precise classifications"""
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Get bounding rect for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1
        
        # Basic shape properties
        shape_info = {
            'type': 'forme_complexe',
            'regularity': 0.5,
            'vertices': num_vertices,
            'approx_contour': approx
        }
        
        # Detailed shape classification
        if circularity > 0.9:
            shape_info['type'] = 'cercle'
            shape_info['regularity'] = circularity
        
        elif circularity > 0.8 and num_vertices > 8:
            shape_info['type'] = 'ellipse'
            shape_info['regularity'] = circularity
            
        elif num_vertices == 3:
            shape_info['type'] = 'triangle'
            shape_info['regularity'] = self.calculate_regularity(approx, 3)
            
        elif num_vertices == 4:
            # Check if it's a square or rectangle
            if aspect_ratio > 0.9:
                shape_info['type'] = 'carre'
                shape_info['regularity'] = 0.9
            else:
                shape_info['type'] = 'rectangle'
                shape_info['regularity'] = 0.8
                
        elif 5 <= num_vertices <= 10:
            # Check if it's a regular polygon
            shape_info['type'] = f'polygone{num_vertices}'
            shape_info['regularity'] = self.calculate_regularity(approx, num_vertices)
            
        else:
            # Complex shape - improve the contour for DXF export
            shape_info['type'] = 'forme_complexe'
            # Calculate how smooth/regular the contour is
            shape_info['regularity'] = min(0.7, 0.5 + circularity/2)
            
        return shape_info
    
    def calculate_regularity(self, points, n_sides):
        """Calculate how regular/symmetric a polygon is"""
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
            
        return regularity
    
    def extract_geometric_elements(self, contour, shape_type):
        """Extract geometric elements for DXF export"""
        elements = []
        
        # Process based on shape type
        if shape_type == 'cercle':
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            elements.append({
                'type': 'cercle',
                'center': (float(x), float(y)),
                'radius': float(radius)
            })
            
        elif shape_type == 'ellipse':
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                elements.append({
                    'type': 'ellipse',
                    'center': ellipse[0],
                    'axes': ellipse[1],
                    'angle': ellipse[2]
                })
            else:
                # Not enough points for ellipse, fallback to polygon
                points = [pt[0].tolist() for pt in contour]
                elements.append({
                    'type': 'polygone',
                    'points': points
                })
                
        elif shape_type in ['carre', 'rectangle']:
            # Use minimum area rectangle for better precision
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            # Use np.int32 instead of np.int0 for better compatibility
            box = box.astype(np.int32)
            
            elements.append({
                'type': 'rectangle',
                'points': box.tolist(),
                'angle': rect[2]
            })
            
        elif shape_type.startswith('polygone') or shape_type == 'triangle':
            # Extract polygon points
            points = [pt[0].tolist() for pt in contour]
            elements.append({
                'type': 'polygone',
                'points': points
            })
            
        elif shape_type == 'forme_complexe':
            # For complex shapes, we provide both polyline and individual line segments
            points = [pt[0].tolist() for pt in contour]
            
            # Create a polyline for the contour
            elements.append({
                'type': 'contour',
                'points': points
            })
            
            # Also provide line segments for detailed control
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
            
    def export_to_dxf(self, segments_data, output_path, line_thickness=0.35):
        """Export segments to DXF with enhanced line thickness"""
        print(f"Exporting to DXF with enhanced line thickness: {output_path}")
        
        try:
            # Use our enhanced DXF exporter with line thickness
            return self.dxf_exporter.export_to_dxf(segments_data, output_path, line_thickness)
        except Exception as e:
            print(f"Erreur lors de l'export DXF: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    def visualize_segmentation(self, original_image, segments_data, method="sam"):
        """Visualize the segmentation results"""
        plt.figure(figsize=(20, 12))
        
        # Image with contours
        plt.subplot(231)
        plt.imshow(original_image)
        plt.title("Image originale avec contours détectés")
        
        # Draw contours with color-coded quality
        for segment in segments_data:
            contour = segment['contour']
            quality = segment['quality']
            
            if quality > 0.75:
                color = 'g'  # High quality - green
            elif quality > 0.5:
                color = 'y'  # Medium quality - yellow
            else:
                color = 'r'  # Low quality - red
                
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], color, linewidth=2)
            
            # Add ID label
            cx, cy = segment['center']
            plt.text(cx, cy, str(segment['id']), color='white', 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='blue', alpha=0.7, pad=1))
        
        # Masks overview
        plt.subplot(232)
        mask_img = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # Combine all masks with different grayscale values
        for i, segment in enumerate(segments_data):
            intensity = int(200 * (i + 1) / len(segments_data)) + 50
            mask = segment['mask']
            mask_img[mask > 0] = intensity
            
        plt.imshow(mask_img, cmap='viridis')
        plt.title("Masques de segmentation")
        
        # Shape classification
        plt.subplot(233)
        shape_img = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # Group segments by shape type
        shape_types = {}
        for segment in segments_data:
            shape_type = segment['shape']['type']
            if shape_type not in shape_types:
                shape_types[shape_type] = []
            shape_types[shape_type].append(segment)
            
        # Assign different intensities for different shape types
        for i, (shape_type, segments) in enumerate(shape_types.items()):
            intensity = int(200 * (i + 1) / len(shape_types)) + 50
            for segment in segments:
                mask = segment['mask']
                shape_img[mask > 0] = intensity
                
        plt.imshow(shape_img, cmap='jet')
        plt.title("Classification des formes")
        
        # Shape detection details
        plt.subplot(234)
        detail_img = np.zeros_like(original_image)
        
        # Draw the geometric primitives that were detected
        for segment in segments_data:
            # Use smoothed contours for better visualization
            contour = segment['smoothed_contour']
            shape_type = segment['shape']['type']
            
            if 'cercle' in shape_type:
                cv2.drawContours(detail_img, [contour], 0, (0, 255, 0), 2)
            elif 'rectangle' in shape_type or 'carre' in shape_type:
                cv2.drawContours(detail_img, [contour], 0, (255, 0, 0), 2)
            elif 'triangle' in shape_type:
                cv2.drawContours(detail_img, [contour], 0, (0, 0, 255), 2)
            elif 'ellipse' in shape_type:
                cv2.drawContours(detail_img, [contour], 0, (255, 255, 0), 2)
            elif 'polygone' in shape_type:
                cv2.drawContours(detail_img, [contour], 0, (0, 255, 255), 2)
            else:
                cv2.drawContours(detail_img, [contour], 0, (255, 0, 255), 2)
                
            # Add center points
            cx, cy = segment['center']
            cv2.circle(detail_img, (cx, cy), 3, (255, 255, 255), -1)
            
        plt.imshow(detail_img)
        plt.title("Détection des formes géométriques")
        
        # Quality heatmap
        plt.subplot(235)
        quality_img = np.zeros(original_image.shape[:2], dtype=np.float32)
        
        # Create a quality heatmap
        for segment in segments_data:
            mask = segment['mask']
            quality = segment['quality']
            quality_img[mask > 0] = quality
            
        plt.imshow(quality_img, cmap='plasma', vmin=0, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Qualité de la segmentation")
        
        # Method information
        plt.subplot(236)
        plt.axis('off')
        info_text = [
            f"Méthode: {'SAM (Segment Anything)' if method=='sam' else 'Segmentation traditionnelle'}",
            f"Objets détectés: {len(segments_data)}",
            "",
            "Types de formes:",
        ]
        
        # Count shapes by type
        shape_counts = {}
        for segment in segments_data:
            shape_type = segment['shape']['type']
            if shape_type not in shape_counts:
                shape_counts[shape_type] = 0
            shape_counts[shape_type] += 1
            
        # Add shape counts to info text
        for shape_type, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True):
            info_text.append(f"  - {shape_type}: {count}")
            
        # Add quality information
        high_q = len([s for s in segments_data if s['quality'] > 0.75])
        med_q = len([s for s in segments_data if 0.5 < s['quality'] <= 0.75])
        low_q = len([s for s in segments_data if s['quality'] <= 0.5])
        
        info_text.extend([
            "",
            f"Qualité haute: {high_q}",
            f"Qualité moyenne: {med_q}",
            f"Qualité basse: {low_q}",
            "",
            f"Utilisateur: {CURRENT_USER}",
            f"Date: {CURRENT_DATE}"
        ])
        
        plt.text(0.1, 0.5, "\n".join(info_text), fontsize=10, va='center')
        plt.title("Informations de traitement")
        
        plt.tight_layout()
        return plt.gcf()


class AdvancedSegmentationPipeline:
    """
    Advanced segmentation pipeline with:
    - Interactive file selection
    - Image tiling for large images
    - SAM segmentation with CRF refinement
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
        print("\nSelectionnez une image à traiter:")
        input_path = self.file_selector.select_file("Sélectionnez une image à traiter")
        
        if not input_path:
            print("Aucun fichier sélectionné. Opération annulée.")
            return None
            
        print(f"Fichier sélectionné: {input_path}")
        return input_path
        
    def process_image(self, input_image=None, output_dir="./exports", interactive=True):
        """Complete pipeline with interactive file selection, tiling, SAM + CRF, and DXF export"""
        print("\n=== DÉMARRAGE DU PIPELINE ===\n")
        
        # Interactive file selection if no input provided
        if input_image is None:
            input_image = self.select_input_file()
            if not input_image:
                return False
        
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Image d'entrée non trouvée: {input_image}")
            
        print(f"Traitement de l'image: {input_image}")
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Définir les chemins de sortie
        basename = os.path.splitext(os.path.basename(input_image))[0]
        output_dxf = os.path.join(output_dir, f"{basename}_segmentation.dxf")
        output_viz = os.path.join(output_dir, f"{basename}_visualisation.png")
        
        # 1. Chargement de l'image
        print("\n1. Chargement de l'image...")
        try:
            image_color, image_gray = self.image_processor.load_image(input_image)
            print(f"  - Image chargée: {image_color.shape[1]}x{image_color.shape[0]} pixels")
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
            return False
        
        # 2. Interactive selection of regions (if enabled)
        roi_mask = None
        if interactive:
            print("\n2. Sélection interactive des régions d'intérêt...")
            roi_mask = self.region_selector.select_regions(image_color, 
                        title="Sélectionnez les régions à segmenter (puis cliquez 'Done')")
            
            # If no selections were made, use the whole image
            if roi_mask is None or np.sum(roi_mask) == 0:
                print("  - Aucune région sélectionnée, utilisation de l'image entière.")
                roi_mask = None
            else:
                # Show selected regions count
                num_regions = cv2.connectedComponents(roi_mask)[0] - 1
                print(f"  - {num_regions} région(s) sélectionnée(s)")
                
                # Save the mask for reference
                mask_path = os.path.join(output_dir, f"{basename}_regions.png")
                cv2.imwrite(mask_path, roi_mask * 255)
                print(f"  - Masque de sélection sauvegardé: {mask_path}")
        
        # 3. Segmentation avancée (with ROI and tiling support)
        print(f"\n{'3' if interactive else '2'}. Segmentation avancée avec SAM et CRF...")
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
                print("Aucun segment n'a été détecté!")
                return False
                
            print(f"  - {len(segments_data)} objets détectés avec la méthode: {method}")
        except Exception as e:
            print(f"Erreur lors de la segmentation: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
            
        # 4. Visualisation des résultats
        print(f"\n{'4' if interactive else '3'}. Génération des visualisations...")
        try:
            fig = self.image_processor.visualize_segmentation(image_color, segments_data, method)
            fig.savefig(output_viz, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Visualisation sauvegardée: {output_viz}")
        except Exception as e:
            print(f"Erreur lors de la visualisation: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
        # 5. Export au format DXF
        print(f"\n{'5' if interactive else '4'}. Export DXF avec lignes épaisses...")
        try:
            self.image_processor.export_to_dxf(segments_data, output_dxf, self.line_thickness)
            print(f"  - Fichier DXF créé: {output_dxf}")
        except Exception as e:
            print(f"Erreur lors de l'export DXF: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
                
        print("\n=== TRAITEMENT TERMINÉ ===\n")
        
        # Résumé des résultats
        print("Récapitulatif:")
        print(f"  - Image d'entrée: {input_image}")
        print(f"  - Fichier DXF: {output_dxf}")
        print(f"  - Visualisation: {output_viz}")
        print(f"  - Objets détectés: {len(segments_data)}")
        
        return True




def main():
    """Main function for command line execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Segmentation avancée avec export DXF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", "-i", help="Chemin vers l'image à traiter (facultatif, sinon sélection interactive)")
    parser.add_argument("--output", "-o", help="Dossier de sortie pour les fichiers générés", default="./exports")
    parser.add_argument("--non-interactive", "-n", action="store_true", 
                        help="Désactiver la sélection interactive des régions")
    
    # Advanced options
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, 
                        help="Taille des tuiles pour la segmentation des grandes images")
    parser.add_argument("--min-area", type=int, default=100, 
                        help="Surface minimale (en pixels²) pour les objets à détecter")
    parser.add_argument("--line-thickness", type=float, default=0.35, 
                        help="Épaisseur des lignes dans le fichier DXF (mm)")
    parser.add_argument("--sam-checkpoint", type=str, default=SAM_CHECKPOINT, 
                        help="Chemin vers les poids du modèle SAM")
    
    # Display options
    parser.add_argument("--show", action="store_true", 
                        help="Afficher les visualisations")
    parser.add_argument("--debug", action="store_true", 
                        help="Activer le mode debug")
    
    args = parser.parse_args()
    
    # Use a local variable instead of modifying the global TILE_SIZE
    tile_size = args.tile_size
    
    # Afficher la bannière de démarrage
    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║  Segmentation Avancée avec Tiling v{VERSION:<24} ║")
    print(f"║  Démarré par: {CURRENT_USER:<41} ║")
    print(f"║  Date: {CURRENT_DATE:<46} ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    
    # Initialiser le pipeline
    pipeline = AdvancedSegmentationPipeline(
        sam_checkpoint=args.sam_checkpoint,
        debug=args.debug,
        line_thickness=args.line_thickness
    )
    
    # Update the tile size in the pipeline
    pipeline.image_processor.tile_splitter.tile_size = tile_size
    
    # Exécuter le pipeline
    try:
        success = pipeline.process_image(
            input_image=args.image,  # None will trigger interactive selection
            output_dir=args.output,
            interactive=not args.non_interactive
        )
        
        if success and args.show:
            # Montrer la visualisation
            basename = os.path.splitext(os.path.basename(args.image if args.image else pipeline.file_selector.selected_file))[0]
            viz_path = os.path.join(args.output, f"{basename}_visualisation.png")
            if os.path.exists(viz_path):
                from PIL import Image
                img = Image.open(viz_path)
                img.show()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nErreur fatale lors du traitement: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())