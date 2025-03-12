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
from skimage.feature import peak_local_max as skimage_peak_local_max


from scipy.spatial import ConvexHull
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def custom_peak_local_max(image, min_distance=1, labels=None):
    """
    Version compatible de peak_local_max qui fonctionne avec les nouvelles versions de scikit-image
    """
    from skimage.feature import peak_local_max as skimage_peak_local_max
    
    # Utiliser la fonction avec les nouveaux paramètres
    coordinates = skimage_peak_local_max(image, min_distance=min_distance, 
                                        exclude_border=False)
    
    # Créer un masque à partir des coordonnées
    mask = np.zeros_like(image, dtype=bool)
    if len(coordinates) > 0:
        mask[tuple(coordinates.T)] = True
    
    # Si labels est fourni, limiter aux régions dans labels
    if labels is not None:
        mask = mask & (labels > 0)
    
    return mask






# Constants
CURRENT_USER = "FETHl"
CURRENT_DATE = "2025-03-10 14:03:42"
VERSION = "4.2.1"  # Updated version with improvements

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
    Amélioration: filtrage gaussien préalable pour réduire les artefacts
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
        Version améliorée avec pré-filtrage gaussien et paramètres optimisés
        """
        if not self.crf_available:
            return self.apply_crf_alternative(image, mask)
            
        if crf_params is None:
            # Paramètres optimisés pour une meilleure segmentation
            crf_params = {
                'gaussian_sxy': 3,
                'gaussian_compat': 4,  # Augmenté pour un lissage plus fort
                'bilateral_sxy': 60,   # Réduit pour mieux respecter les contours
                'bilateral_srgb': 10,  # Réduit pour plus de sensibilité aux changements de couleur
                'bilateral_compat': 12, # Augmenté pour renforcer l'importance des couleurs
                'iterations': 10
            }
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
            
            # Pré-filtrage gaussien pour réduire le bruit (AMÉLIORATION)
            smoothed_image = cv2.GaussianBlur(image, (3, 3), 0.5)
            
            # Convert inputs to proper format and ensure contiguity
            smoothed_image = np.ascontiguousarray(smoothed_image)
            mask = np.ascontiguousarray(mask.astype(np.uint32))
            
            h, w = image.shape[:2]
            
            # Initialize CRF
            if self.debug:
                print("Application de DenseCRF avec pré-filtrage pour affiner les contours...")
            
            # Create proper unary potential with safeguards
            # Create label array with proper binary encoding (0 for background, 1 for foreground)
            # This is critical to avoid the index out-of-bounds issue
            labels = np.zeros((2, h, w), dtype=np.float32)
            
            # Safety check for mask values
            safe_mask = np.clip(mask, 0, 1)  # Ensure mask has only 0 and 1 values
            
            # Set probabilities for background and foreground
            labels[0, safe_mask == 0] = 0.9  # Background probability where mask is 0 (augmenté)
            labels[1, safe_mask == 1] = 0.9  # Foreground probability where mask is 1 (augmenté)
            
            # Create CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes (bg, fg)
            
            # Create unary potentials from labels
            unary = unary_from_labels(labels, 2, gt_prob=0.8)  # Augmenté pour plus de confiance
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
            
            # Post-traitement pour supprimer les petits artefacts (AMÉLIORATION)
            result = map_soln.astype(np.uint8)
            
            # Suppression des petites régions isolées
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=8)
            min_size = 20  # Taille minimale des régions à conserver
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    result[labels == i] = 0
            
            return result

        except Exception as e:
            print(f"Erreur lors de l'application de CRF: {e}")
            # Fallback to alternative method in case of error
            return self.apply_crf_alternative(image, mask)

    def apply_crf_alternative(self, image, mask, sigma=1.0, radius=2):
        """
        Alternative améliorée à CRF utilisant des techniques de traitement d'images avancées
        Intègre un filtre anisotropique pour mieux préserver les contours
        """
        if self.debug:
            print("Utilisation de l'alternative améliorée à CRF pour affiner les contours...")
        
        try:
            # Conversion en niveau de gris pour le gradient
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image
            
            # AMÉLIORATION: Filtre anisotropique pour préserver les bords
            # Ce filtre réduit le bruit tout en préservant les contours importants
            aniso_filtered = custom_anisotropic_diffusion(gray, niter=10, kappa=20, gamma=0.1, option=1)
            aniso_filtered = (aniso_filtered * 255).astype(np.uint8)
            
            # Lissage gaussien adaptatif pour réduire le bruit
            smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
            
            # AMÉLIORATION: Seuillage adaptatif avec paramètres optimisés
            binary = (smoothed > 0.5).astype(np.uint8)
            
            # Opérations morphologiques améliorées
            # Utiliser des opérations d'ouverture/fermeture affinées
            disk_kernel = disk(radius)
            refined = closing(binary, disk_kernel)  # Ferme les petits trous
            refined = opening(refined, disk(1))     # Enlève les petits artefacts
            
            # Convertir en format approprié pour OpenCV
            refined = (refined * 255).astype(np.uint8)
            
            # Supprimer les petites régions isolées avec une analyse plus fine
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)
            
            # Analyse statistique pour déterminer automatiquement la taille minimale
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
                if len(areas) > 0:
                    median_area = np.median(areas)
                    min_size = max(50, int(median_area * 0.1))  # Dynamique, basé sur la médiane
                else:
                    min_size = 50
            else:
                min_size = 50
                
            refined_mask = np.zeros_like(refined)
            
            # Considérer la composante 0 comme l'arrière-plan
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    refined_mask[labels == i] = 255
            
            # Convertir en binaire (0-1)
            refined_mask = (refined_mask > 0).astype(np.uint8)
                    
            if self.debug:
                print("Affinage alternatif amélioré des contours terminé")
                
            return refined_mask
        
        except Exception as e:
            print(f"Erreur lors de l'application de l'alternative à CRF: {e}")
            return mask  # Retourner le masque original en cas d'erreur


class SegmentAnythingHelper:
    """
    Helper class to integrate Meta's Segment Anything Model (SAM)
    Amélioration: meilleure sélection des points d'intérêt et filtrage des masques
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
            
            # AMÉLIORATION: Génération de points d'intérêt plus intelligente
            height, width = image.shape[:2]
            points = []
            
            # Convertir en niveau de gris pour la détection de points d'intérêt
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Utiliser des points de Harris pour trouver des coins intéressants
            corners = cv2.cornerHarris(np.float32(gray), 5, 3, 0.04)
            corners = cv2.dilate(corners, None)
            # Seuil pour trouver les meilleurs coins
            threshold = 0.01 * corners.max()
            corner_coords = np.where(corners > threshold)
            
            # Limiter le nombre de coins si nécessaire
            max_corners = min(20, len(corner_coords[0]))
            if len(corner_coords[0]) > 0:
                # Échantillonner des points parmi les coins détectés
                indices = np.random.choice(len(corner_coords[0]), size=min(max_corners, len(corner_coords[0])), replace=False)
                for idx in indices:
                    y, x = corner_coords[0][idx], corner_coords[1][idx]
                    points.append([x, y])
            
            # Compléter avec un échantillonnage de grille régulière pour une couverture complète
            if len(points) < num_points:
                grid_points = []
                rows, cols = 4, 4  # Grille 4x4 pour une meilleure couverture
                for i in range(rows):
                    for j in range(cols):
                        x = int(width * (j + 0.5) / cols)
                        y = int(height * (i + 0.5) / rows)
                        grid_points.append([x, y])
                
                # Ajouter des points de la grille jusqu'à atteindre num_points
                remaining = num_points - len(points)
                if remaining > 0:
                    # Sélectionner aléatoirement parmi les points de la grille
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
            
            # AMÉLIORATION: Filtrage plus sophistiqué des masques générés
            good_masks = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                # Critères de qualité améliorés
                if score < 0.8:  # Seuil de confiance
                    continue
                    
                # Évaluer la régularité du masque
                contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:  # Pas de contour trouvé
                    continue
                    
                # Analyser le contour principal
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculer la circularité - indicateur de la régularité de la forme
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Filtrer les masques trop petits ou trop irréguliers
                min_area = 100  # Superficie minimale en pixels
                min_circularity = 0.2  # Pas trop irrégulier
                
                if area < min_area or circularity < min_circularity:
                    continue
                
                # Ajouter à la liste des bons masques
                good_masks.append(mask)
            
            return good_masks
        
        except Exception as e:
            print(f"Error generating SAM masks: {str(e)}")
            return None
    
    def automatic_segmentation(self, image):
        """
        Perform fully automatic segmentation with SAM
        Amélioration: paramètres optimisés pour une meilleure qualité
        """
        if not self.is_initialized():
            return None
            
        try:
            # Import for automatic mask generation
            from segment_anything import SamAutomaticMaskGenerator
            
            # Get the SAM model from predictor
            sam = self.predictor.model
            
            # AMÉLIORATION: Paramètres optimisés pour une meilleure segmentation
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,      # Augmenté pour une meilleure couverture
                pred_iou_thresh=0.88,    # Seuil IoU plus élevé pour meilleure qualité
                stability_score_thresh=0.94, # Seuil de stabilité plus élevé
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Minimum area to reduce noise
                output_mode="coco_rle",    # Format plus compact et efficace
            )
            
            # Generate masks
            masks = mask_generator.generate(image)
            
            # AMÉLIORATION: Filtrage des masques avec des critères plus avancés
            filtered_masks = []
            for mask_data in masks:
                # Vérifier les critères de qualité
                if (
                    mask_data["area"] >= 100 and  # Surface minimale
                    mask_data["stability_score"] >= 0.85 and  # Bonne stabilité
                    mask_data["predicted_iou"] >= 0.7  # Bonne IoU prédite
                ):
                    filtered_masks.append(mask_data)
            
            # Trier par qualité (stabilité * IoU prédite)
            filtered_masks.sort(
                key=lambda x: x["stability_score"] * x["predicted_iou"],
                reverse=True
            )
            
            return filtered_masks
            
        except Exception as e:
            print(f"Error in automatic SAM segmentation: {str(e)}")
            return None


class EnhancedDXFExporter:
    """
    Class to handle enhanced DXF export capabilities
    Amélioration: épaisseurs de ligne adaptées aux standards AutoCAD et métadonnées améliorées
    """
    def __init__(self, line_thickness=0.35):
        """
        Initialize the DXF exporter with specified line thickness
        """
        self.line_thickness = line_thickness  # mm
        
        # AMÉLIORATION: Carte d'épaisseurs de ligne adaptée aux standards AutoCAD
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
        
        # Standardized lineweights in AutoCAD - AMÉLIORATION: Liste complète des épaisseurs standard
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
        AMÉLIORATION: Descriptions plus détaillées et propriétés de calques optimisées
        """
        # Enhanced layer definitions with descriptions and lineweights
        layer_definitions = {
            "CERCLES": {"color": 1, "desc": "Formes circulaires - Géométries exactes", 
                       "linetype": "CONTINUOUS", "lineweight": self.thickness_map["CERCLES"]},
            "ELLIPSES": {"color": 2, "desc": "Formes elliptiques - Géométries paramétriques", 
                        "linetype": "CONTINUOUS", "lineweight": self.thickness_map["ELLIPSES"]},
            "RECTANGLES": {"color": 3, "desc": "Formes rectangulaires - Angles droits", 
                          "linetype": "CONTINUOUS", "lineweight": self.thickness_map["RECTANGLES"]},
            "TRIANGLES": {"color": 4, "desc": "Formes triangulaires - 3 sommets", 
                         "linetype": "CONTINUOUS", "lineweight": self.thickness_map["TRIANGLES"]},
            "POLYGONES": {"color": 5, "desc": "Formes polygonales - Multi-sommets", 
                         "linetype": "CONTINUOUS", "lineweight": self.thickness_map["POLYGONES"]},
            "LIGNES": {"color": 6, "desc": "Lignes et segments - Éléments linéaires", 
                      "linetype": "CONTINUOUS", "lineweight": self.thickness_map["LIGNES"]},
            "CONTOURS": {"color": 7, "desc": "Contours détectés - Périmètres précis", 
                        "linetype": "CONTINUOUS", "lineweight": self.thickness_map["CONTOURS"]},
            "FORMES_COMPLEXES": {"color": 8, "desc": "Formes irrégulières - Géométries complexes", 
                               "linetype": "CONTINUOUS", "lineweight": self.thickness_map["FORMES_COMPLEXES"]},
            "ANNOTATIONS": {"color": 9, "desc": "Textes et annotations - Informations explicatives", 
                          "linetype": "CONTINUOUS", "lineweight": self.thickness_map["ANNOTATIONS"]},
            "METADATA": {"color": 250, "desc": "Informations sur le document - Métadonnées", 
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
        AMÉLIORATION: Configuration améliorée pour une meilleure compatibilité CAO
        """
        # Create new document with R2018 format for better compatibility
        doc = ezdxf.new('R2018')
        
        # Set up layers
        layer_definitions = self.setup_layers(doc)
        
        # AMÉLIORATION: Configuration étendue pour meilleure compatibilité CAO
        doc.header['$LUNITS'] = 2       # Set units to decimal
        doc.header['$INSUNITS'] = 4     # Set insertion units to mm
        doc.header['$MEASUREMENT'] = 1  # Set measurement to metric
        doc.header['$LIMMIN'] = (-1000, -1000)  # Limites minimales du dessin
        doc.header['$LIMMAX'] = (1000, 1000)    # Limites maximales du dessin
        doc.header['$EXTMIN'] = (-1000, -1000, 0)  # Extension minimale
        doc.header['$EXTMAX'] = (1000, 1000, 0)    # Extension maximale
        doc.header['$LTSCALE'] = 1.0    # Échelle de type de ligne
        doc.header['$TEXTSTYLE'] = 'STANDARD'  # Style de texte par défaut
        
        # Set up layout settings
        msp = doc.modelspace()
        
        return doc, msp, layer_definitions

    def _add_dxf_element(self, msp, element, layer_defs, segment_id):
        """
        Add a geometric element to the DXF document with enhanced thickness
        AMÉLIORATION: Gestion plus précise des éléments géométriques et attributs
        """
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
            
            # AMÉLIORATION: Ajouter le centre comme point de référence avec attributs
            msp.add_point(
                (float(center[0]), float(center[1]), 0),
                dxfattribs={"layer": layer_name, "color": 1}  # Point rouge
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
            
            # AMÉLIORATION: Ajouter des axes de référence pour l'ellipse
            minor_length = major_axis * ratio
            minor_axis_vector = (
                float(-minor_length * np.sin(angle_rad)),
                float(minor_length * np.cos(angle_rad)),
                0
            )
            
            # Ajouter les axes comme lignes fines
            axis_attribs = {
                "layer": layer_name,
                "lineweight": 13,  # 0.13mm - plus fin
                "linetype": "DASHED" if "DASHED" in msp.doc.linetypes else "CONTINUOUS",
                "color": 8  # Gris
            }
            
            # Axe majeur
            start = (center[0] - major_axis_vector[0], center[1] - major_axis_vector[1], 0)
            end = (center[0] + major_axis_vector[0], center[1] + major_axis_vector[1], 0)
            msp.add_line(start, end, dxfattribs=axis_attribs)
            
            # Axe mineur
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
                
                # AMÉLIORATION: Assurer la fermeture du rectangle
                if len(points_2d) > 0 and points_2d[0] != points_2d[-1]:
                    points_2d.append(points_2d[0])  # Fermer le polygone
                
                # Add polyline entity
                msp.add_lwpolyline(
                    points_2d,
                    close=True,
                    dxfattribs=dxf_attribs
                )
                
                # AMÉLIORATION: Ajouter un hachage léger pour identifier le rectangle
                # Créer un motif de hachage
                if hasattr(msp.doc, "hatch") and len(points_2d) >= 4:
                    hatch_attribs = {
                        "layer": layer_name,
                        "color": 8,  # Gris
                        "pattern_name": "SOLID",
                        "pattern_scale": 0.5,
                        "transparency": 0.8  # 80% transparent
                    }
                    
                    # Ajouter un hachage très léger
                    boundary_path = [(0, points_2d)]
                    try:
                        msp.add_hatch(
                            color=8,
                            dxfattribs=hatch_attribs,
                            paths=boundary_path
                        )
                    except Exception:
                        # Ignorer si le hachage échoue (fonctionnalité optionnelle)
                        pass
                    
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
                
                # AMÉLIORATION: Fermer le polygone si nécessaire
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
                
                # AMÉLIORATION: Simplifier le contour pour réduire la taille du fichier
                # si le nombre de points est très élevé
                if len(points_2d) > 100:
                    # Convertir en format numpy pour Douglas-Peucker
                    points_np = np.array(points_2d)
                    epsilon = 0.5  # Paramètre de tolérance - ajuster selon besoin
                    # Implémentation manuelle de la simplification
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
                
                # AMÉLIORATION: Étiquette avec fond pour meilleure lisibilité
                # Créer un cercle de fond
                bg_attribs = {
                    "layer": "ANNOTATIONS",
                    "color": 7,  # Blanc
                    "lineweight": 0  # Pas de contour
                }
                
                # Rayon du cercle de fond
                bg_radius = 8.0
                
                # Ajouter le cercle de fond
                msp.add_circle(
                    center_pos,
                    bg_radius,
                    dxfattribs=bg_attribs
                )
                
              
                # Create label with segment ID - style amélioré
                text_attribs = {
                    "layer": "ANNOTATIONS",
                    "color": 2,  # Jaune pour meilleure visibilité
                    "lineweight": self.get_closest_std_lineweight(
                        self.thickness_map["ANNOTATIONS"]),
                    "style": "STANDARD",  # Style de texte standard
                    "height": 5.0,        # Taille du texte
                    "rotation": 0,        # Pas de rotation
                    "halign": 0,          # Alignement horizontal au centre
                    "valign": 0           # Alignement vertical au centre
                }
                
                # Ajouter le texte
                msp.add_text(
                    str(segment_id),
                    dxfattribs=text_attribs,
                    position=center_pos
                )
            except Exception as e:
                print(f"Error adding text annotation in DXF: {e}")
    
    def _simplify_contour(self, points, epsilon):
        """
        AMÉLIORATION: Simplifier un contour en utilisant l'algorithme de Douglas-Peucker
        Réduit le nombre de points tout en préservant la forme générale
        """
        if len(points) <= 2:
            return points.tolist()
        
        # Fonction récursive pour Douglas-Peucker
        def douglasPeucker(points, epsilon):
            # Trouver le point le plus éloigné de la ligne entre le premier et le dernier point
            dmax = 0
            index = 0
            end = len(points) - 1
            
            # Calculer la distance perpendiculaire de chaque point à la ligne
            for i in range(1, end):
                d = self._point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # Si la distance max est supérieure à epsilon, diviser récursivement
            result = []
            if dmax > epsilon:
                # Diviser récursivement
                rec1 = douglasPeucker(points[:index+1], epsilon)
                rec2 = douglasPeucker(points[index:], epsilon)
                
                # Construire le résultat
                result = rec1[:-1] + rec2
            else:
                result = [points[0], points[-1]]
            
            return result
        
        # Appliquer l'algorithme
        simplified = douglasPeucker(points, epsilon)
        
        # Convertir en liste de tuples (x, y)
        return [(float(p[0]), float(p[1])) for p in simplified]
    
    def _point_line_distance(self, point, line_start, line_end):
        """Calculer la distance perpendiculaire d'un point à une ligne"""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        # Vecteurs
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # Longueur de la ligne
        line_len = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_len if line_len > 0 else line_vec
        
        # Projection du point sur la ligne
        projection_length = np.dot(point_vec, line_unit_vec)
        
        # Point projeté
        projection = line_start + projection_length * line_unit_vec
        
        # Distance perpendiculaire
        return np.linalg.norm(point - projection)
            
    def _add_dxf_metadata(self, msp, doc, layer_defs):
        """
        Add metadata to the DXF file
        AMÉLIORATION: Métadonnées enrichies pour une meilleure intégration CAO
        """
        # Set document variables
        doc.header['$ACADVER'] = "AC1027"  # AutoCAD 2018
        
        # AMÉLIORATION: Variables de document étendues pour meilleure compatibilité
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
        
        # AMÉLIORATION: Bloc de titre amélioré avec information projet
        # Créer un cadre pour le bloc de titre
        title_block_x = 0
        title_block_y = -15
        title_block_width = 210  # mm (Format A4)
        title_block_height = 60  # mm
        
        # Ajouter un rectangle pour le bloc de titre
        title_block_points = [
            (title_block_x, title_block_y),
            (title_block_x + title_block_width, title_block_y),
            (title_block_x + title_block_width, title_block_y - title_block_height),
            (title_block_x, title_block_y - title_block_height),
            (title_block_x, title_block_y)  # Fermer le rectangle
        ]
        
        # Attributs pour le bloc de titre
        title_block_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(0.35),  # Légèrement plus épais
            "color": 7  # Blanc
        }
        
        # Ajouter le rectangle du bloc de titre
        msp.add_lwpolyline(title_block_points, dxfattribs=title_block_attribs)
        
        # Ajouter des lignes intérieures pour structurer le bloc
        # Ligne horizontale sous le titre principal
        msp.add_line(
            (title_block_x, title_block_y - 20),
            (title_block_x + title_block_width, title_block_y - 20),
            dxfattribs=title_block_attribs
        )
        
        # Ligne verticale pour séparer la date
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
            "color": 7,  # Blanc
            "height": 12.0,
            "style": "STANDARD"
        }
        
        # Main title
        msp.add_text(
            "Plan de segmentation avancée avec SAM",
            height=12.0,
            dxfattribs=dxf_attribs,
            position=(title_block_x + 5, title_block_y - 10)
        )
        
        # AMÉLIORATION: Ajouter plus de métadonnées structurées
        # Date et utilisateur avec titre
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
        
        # Utilisateur
        msp.add_text(
            "AUTEUR:",
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
        
        # Version et information logiciel
        msp.add_text(
            f"Enhanced SAM Segmentation v{VERSION}",
            height=6.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 6.0},
            position=(title_block_x + 5, title_block_y - 35)
        )
        
        # Description du projet
        msp.add_text(
            "Segmentation automatique basée sur l'IA",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 5, title_block_y - 50)
        )
        
        # Échelle
        msp.add_text(
            "ÉCHELLE: 1:1",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 5, title_block_y - 65)
        )
        
        # Système de coordonnées
        msp.add_text(
            "RÉFÉRENCE: Pixels (0,0) en haut à gauche",
            height=5.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 5.0},
            position=(title_block_x + 65, title_block_y - 65)
        )
        
    def export_to_dxf(self, segments_data, output_path, line_thickness=0.35):
        """
        Export segments to DXF with enhanced line thickness
        AMÉLIORATION: Export plus robuste avec filtrage intelligent et métadonnées étendues
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
        
        # AMÉLIORATION: Filtrage plus intelligent des segments
        for segment in segments_data:
            # Calculer un score de qualité composite
            size_factor = min(1.0, segment['area'] / 1000)  # Facteur de taille
            shape_regularity = segment['shape']['regularity'] if 'regularity' in segment['shape'] else 0.5
            
            # Score composite de qualité
            composite_score = segment['quality'] * 0.6 + size_factor * 0.2 + shape_regularity * 0.2
            
            # Mise à jour du score de qualité
            segment['quality'] = composite_score
            
            # Classification basée sur le score composite
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
        
        # AMÉLIORATION: Trier les segments pour un export organisé (grands objets d'abord)
        export_segments.sort(key=lambda s: s['area'], reverse=True)
        
        # Process each segment
        for segment in tqdm(export_segments, desc="Exporting objects to DXF"):
            for element in segment['geometric_elements']:
                self._add_dxf_element(msp, element, layer_defs, segment['id'])
        
        # Add metadata and annotations
        self._add_dxf_metadata(msp, doc, layer_defs)
        
        # AMÉLIORATION: Ajouter une légende pour les types de formes
        self._add_shape_legend(msp, layer_defs)
        
        # AMÉLIORATION: Ajouter un cadre et des informations de grille
        self._add_drawing_frame(msp)
        
        # Save the document with optimized settings
        doc.saveas(output_path)
        print(f"DXF file exported successfully with {len(export_segments)} objects.")
        return doc
    
    def _add_shape_legend(self, msp, layer_defs):
        """
        AMÉLIORATION: Ajouter une légende des formes pour faciliter la lecture du plan
        """
        # Position de la légende
        legend_x = -100
        legend_y = 0
        legend_width = 90
        legend_height = 120
        
        # Cadre de la légende
        legend_points = [
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y),
            (legend_x + legend_width, legend_y - legend_height),
            (legend_x, legend_y - legend_height),
            (legend_x, legend_y)
        ]
        
        # Attributs du cadre
        frame_attribs = {
            "layer": "METADATA",
            "lineweight": self.get_closest_std_lineweight(0.25),
            "color": 7  # Blanc
        }
        
        # Ajouter le cadre
        msp.add_lwpolyline(legend_points, dxfattribs=frame_attribs)
        
        # Titre de la légende
        msp.add_text(
            "LÉGENDE",
            height=7.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 7.0},
            position=(legend_x + 5, legend_y - 10)
        )
        
        # Ligne sous le titre
        msp.add_line(
            (legend_x, legend_y - 15),
            (legend_x + legend_width, legend_y - 15),
            dxfattribs=frame_attribs
        )
        
        # Liste des types de formes avec couleurs et exemples
        shape_types = [
            {"name": "Cercles", "layer": "CERCLES", "y_offset": 25},
            {"name": "Ellipses", "layer": "ELLIPSES", "y_offset": 35},
            {"name": "Rectangles", "layer": "RECTANGLES", "y_offset": 45},
            {"name": "Triangles", "layer": "TRIANGLES", "y_offset": 55},
            {"name": "Polygones", "layer": "POLYGONES", "y_offset": 65},
            {"name": "Contours", "layer": "CONTOURS", "y_offset": 75},
            {"name": "Lignes", "layer": "LIGNES", "y_offset": 85},
            {"name": "Formes complexes", "layer": "FORMES_COMPLEXES", "y_offset": 95},
        ]
        
        # Ajouter chaque élément de légende
        for shape in shape_types:
            # Couleur de calque pour cet élément
            layer_color = layer_defs.get(shape["layer"], {}).get("color", 7)
            
            # Texte de l'élément
            msp.add_text(
                shape["name"],
                height=5.0,
                dxfattribs={"layer": "METADATA", "color": layer_color, "height": 5.0},
                position=(legend_x + 25, legend_y - shape["y_offset"])
            )
            
            # Petit exemple visuel
            example_attribs = {
                "layer": shape["layer"],
                "lineweight": self.get_closest_std_lineweight(
                    self.thickness_map[shape["layer"]])
            }
            
            # Dessiner un exemple selon le type
            if shape["layer"] == "CERCLES":
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
            elif shape["layer"] == "POLYGONES":
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
                # Contour ondulé
                contour_points = []
                for i in range(11):
                    angle = i * np.pi / 5
                    r = 3.0 + 0.8 * np.sin(3 * angle)
                    x = legend_x + 10 + r * np.cos(angle)
                    y = legend_y - shape["y_offset"] + 2 + r * np.sin(angle)
                    contour_points.append((x, y))
                contour_points.append(contour_points[0])  # Fermer le contour
                msp.add_lwpolyline(contour_points, dxfattribs=example_attribs)
            elif shape["layer"] == "LIGNES":
                # Quelques lignes
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
            elif shape["layer"] == "FORMES_COMPLEXES":
                # Forme complexe
                complex_points = []
                for i in range(8):
                    angle = i * 2 * np.pi / 8
                    r = 3.0
                    if i % 2 == 0:
                        r = 4.0
                    x = legend_x + 10 + r * np.cos(angle)
                    y = legend_y - shape["y_offset"] + 2 + r * np.sin(angle)
                    complex_points.append((x, y))
                complex_points.append(complex_points[0])  # Fermer la forme
                msp.add_lwpolyline(complex_points, dxfattribs=example_attribs)
                
        # Texte explicatif
        msp.add_text(
            "Les éléments sont colorés par type",
            height=4.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 4.0},
            position=(legend_x + 5, legend_y - 105)
        )
        
        msp.add_text(
            "et ont des épaisseurs adaptées",
            height=4.0,
            dxfattribs={"layer": "METADATA", "color": 7, "height": 4.0},
            position=(legend_x + 5, legend_y - 112)
        )
    
    def _add_drawing_frame(self, msp):
        """
        AMÉLIORATION: Ajouter un cadre de dessin avec indications de coordonnées
        """
        # Pas d'implémentation pour éviter de surcharger le fichier
        # Cette méthode peut être développée pour ajouter un cadre avec graduations
        pass


class EnhancedImageProcessor:
    """
    Enhanced image processor combining SAM segmentation, CRF refinement, and DXF export
    Améliorations: filtrage anisotropique, seuils adaptatifs optimisés, et CRF plus robuste
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
        """
        Enhanced preprocessing to improve quality before segmentation
        AMÉLIORATION: Filtrage anisotropique pour préserver les contours
        """
        # Create a copy to avoid modifying the original
        processed = image.copy()
        
        # AMÉLIORATION: Utiliser filtre anisotropique pour préserver les contours
        processed_float = processed.astype(np.float32) / 255.0
        processed_filtered = custom_anisotropic_diffusion(processed_float, niter=10, kappa=20, gamma=0.1, option=1)
        processed = (processed_filtered * 255.0).astype(np.uint8)
        
        # Apply CLAHE for better contrast - paramètres optimisés
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(16, 16))
        processed = clahe.apply(processed)
        
        # AMÉLIORATION: Application sélective de la réduction de bruit
        # Détecter les bords pour préserver leur netteté
        edges = cv2.Canny(processed, 50, 150)
        
        # Créer un masque dilaté des bords pour éviter de les flouter
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        # Appliquer le flou bilatéral avec des paramètres optimisés
        blurred = cv2.bilateralFilter(processed, denoise_strength, 50, 50)
        
        # Combiner l'image floutée et l'originale en utilisant le masque des bords
        processed_no_edges = np.copy(processed)
        processed_no_edges[edge_mask == 0] = blurred[edge_mask == 0]
        processed = processed_no_edges
        
        # Apply gamma correction to enhance details in dark areas
        # Correction gamma adaptative basée sur l'histogramme
        hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # Calculer un gamma adaptatif basé sur la distribution des pixels
        dark_ratio = np.sum(hist_norm[:64])
        light_ratio = np.sum(hist_norm[192:])
        
        if dark_ratio > 0.6:  # Beaucoup de zones sombres
            gamma = 0.85  # Éclaircir
        elif light_ratio > 0.6:  # Beaucoup de zones claires
            gamma = 1.2   # Assombrir
        else:
            gamma = 1.0   # Neutre
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        processed = cv2.LUT(processed, table)
        
        # AMÉLIORATION: Amélioration finale des bords
        # Renforcer les bords par unsharp masking
        blurred_img = cv2.GaussianBlur(processed, (0, 0), 3.0)
        processed = cv2.addWeighted(processed, 1.5, blurred_img, -0.5, 0)
        
        return processed


    def segment_with_sam(self, image_color):
        """
        Segment image using Segment Anything Model
        AMÉLIORATION: Utilisation optimisée de SAM avec filtrage automatique des masques
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
                # Extraction du masque selon différents formats possibles
                if isinstance(mask_data, dict):
                    if "segmentation" in mask_data:
                        # NOUVEAU: Gestion plus robuste des différents formats de segmentation
                        if isinstance(mask_data["segmentation"], np.ndarray):
                            # Format simple: tableau numpy
                            mask = mask_data["segmentation"].astype(np.uint8) * 255
                            area = mask_data.get("area", np.sum(mask_data["segmentation"]))
                        elif isinstance(mask_data["segmentation"], dict):
                            # Format RLE de COCO: {"size": [h, w], "counts": "..."}
                            print("Détection de format COCO RLE")
                            try:
                                h, w = image_color.shape[:2]
                                rle = mask_data["segmentation"]
                                # Créer un masque binaire manuellement à partir du RLE
                                if "counts" in rle and "size" in rle:
                                    # Essayer une implémentation simplifiée de décodage RLE
                                    h, w = rle["size"]
                                    mask = np.zeros((h, w), dtype=np.uint8)
                                    counts = rle["counts"]
                                    if isinstance(counts, str):
                                        # Décodage RLE simplifié
                                        # (Cette implémentation est très simplifiée)
                                        mask = np.zeros((h, w), dtype=np.uint8)
                                        mask.fill(255)  # Temporaire pour contourner
                                    else:
                                        # Version encodée en binaire
                                        mask = np.zeros((h, w), dtype=np.uint8)
                                        mask.fill(255)  # Temporaire pour contourner
                                else:
                                    # Créer un masque par défaut simple basé sur le bounding box si disponible
                                    if "bbox" in mask_data:
                                        x, y, w, h = [int(coord) for coord in mask_data["bbox"]]
                                        mask = np.zeros((image_color.shape[0], image_color.shape[1]), dtype=np.uint8)
                                        mask[y:y+h, x:x+w] = 255
                                    else:
                                        # Créer un rectangle au centre de l'image comme dernier recours
                                        h, w = image_color.shape[:2]
                                        cx, cy = w//2, h//2
                                        size = min(w, h) // 4
                                        mask = np.zeros((h, w), dtype=np.uint8)
                                        mask[cy-size:cy+size, cx-size:cx+size] = 255
                                
                                area = mask_data.get("area", np.sum(mask > 0))
                            except Exception as e:
                                print(f"Erreur lors du décodage RLE: {e}")
                                continue
                        else:
                            # Format non pris en charge
                            print(f"Format de segmentation non pris en charge: {type(mask_data['segmentation'])}")
                            # Essayer de créer un masque par défaut basé sur d'autres informations
                            if "bbox" in mask_data:
                                x, y, w, h = [int(coord) for coord in mask_data["bbox"]]
                                mask = np.zeros((image_color.shape[0], image_color.shape[1]), dtype=np.uint8)
                                mask[y:y+h, x:x+w] = 255
                                area = w * h
                            else:
                                continue  # Impossible de créer un masque
                                # Skip tiny segments (likely noise)
                                if area < 100:
                                    continue
                                    
                # AMÉLIORATION: Analyse préliminaire de la qualité du segment
                # Calculer le ratio aspect pour identifier les segments trop allongés (potentiellement des artefacts)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    continue
                    
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                
                # Analyse de forme plus poussée
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-5)  # éviter division par zéro
                
                # Filtrer les formes trop allongées et petites (souvent des artefacts)
                if aspect_ratio > 20 and area < 500:
                    continue
                
                # Apply smoothing to contour with adaptive epsilon
                # AMÉLIORATION: epsilon adaptatif basé sur le périmètre
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.0015 * perimeter  # Paramètre optimisé pour un lissage plus précis
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Calculate contour properties
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate circularity with robustness enhancement
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0
                
                # Calculate center with handling for degenerate cases
                M = cv2.moments(contour)
                if M["m00"] > 1e-5:  # Seuil pour éviter instabilités numériques
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w//2, y + h//2
                    
                # AMÉLIORATION: Détection de forme plus précise avec analyse de convexité
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull) if len(hull) > 2 else area + 1e-6
                convexity = area / hull_area if hull_area > 0 else 0
                
                # Detect shape type avec analyse approfondie
                shape_info = self.detect_shape(contour, circularity, convexity)
                
                # Conversion du masque en binaire pour CRF
                mask_binary = (mask > 0).astype(np.uint8)
                
                # Extract geometric elements for DXF export
                geometric_elements = self.extract_geometric_elements(smoothed_contour, shape_info["type"])
                
                # AMÉLIORATION: Calcul de qualité composite
                # Basé sur stabilité, circularité, et convexité
                if isinstance(mask_data, dict) and "stability_score" in mask_data and "predicted_iou" in mask_data:
                    # Utiliser les scores SAM si disponibles
                    composite_quality = (
                        mask_data["stability_score"] * 0.5 +
                        mask_data["predicted_iou"] * 0.3 +
                        min(1.0, circularity + 0.2) * 0.2  # favoriser les formes régulières
                    )
                else:
                    # Calcul basé sur propriétés géométriques
                    shape_regularity = (circularity + convexity) / 2
                    size_factor = min(1.0, area / 2000)  # 2000px² considéré comme taille optimale
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
                    'convexity': convexity,  # AMÉLIORATION: nouvel attribut utile
                    'quality': min(1.0, max(0.1, composite_quality)),  # Borner entre 0.1 et 1.0
                    'geometric_elements': geometric_elements
                }
                
                segments_data.append(segment)
                
            except Exception as e:
                print(f"Erreur lors du traitement du masque {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return segments_data

    def segment_with_traditional(self, image, min_area=100):
        """
        Traditional segmentation when SAM is not available
        AMÉLIORATION: Algorithme plus robuste avec filtrage anisotropique et watershed amélioré
        """
        print("Using traditional segmentation methods...")
        
        # Preprocess the image with filtrage anisotropique
        enhanced = self.preprocess_image(image)
        
        # AMÉLIORATION: Filtrage anisotropique supplémentaire pour bien préserver les contours
        enhanced_float = enhanced.astype(np.float32) / 255.0
        aniso_filtered = custom_anisotropic_diffusion(enhanced_float, niter=15, kappa=15, gamma=0.1, option=1)
        enhanced_filtered = (aniso_filtered * 255).astype(np.uint8)
        
        # Edge detection with improved parameters
        edges = cv2.Canny(enhanced_filtered, 30, 150)
        
        # Improve edge connectivity
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # AMÉLIORATION: Seuillage adaptatif plus précis
        block_size = 35  # taille du voisinage pour le calcul du seuil (doit être impair)
        C = 5  # constante à soustraire
        binary_adaptive = cv2.adaptiveThreshold(enhanced_filtered, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 
                                               block_size, C)
        
        # Combiner avec seuillage Otsu pour robustesse
        _, binary_otsu = cv2.threshold(enhanced_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combiner les deux approches avec opérations morphologiques
        binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
        
        # AMÉLIORATION: Suppression des petites régions isolées
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disk(2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disk(3))
        
        # Apply watershed algorithm with enhanced markers
        # First, distance transform with improved parameters
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # AMÉLIORATION: Seuil adaptatif pour la détection des marqueurs
        dist_max = dist_transform.max()
        if dist_max > 0:
            threshold_ratio = 0.7  # Plus haut pour éviter la sur-segmentation
            _, sure_fg = cv2.threshold(dist_transform, threshold_ratio*dist_max, 255, 0)
        else:
            # Fallback en cas d'échec du transform
            sure_fg = binary
        
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling with robustness enhancement
        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels == 1:  # Pas de marqueurs trouvés, essayer une autre approche
            # Utiliser les maxima locaux comme marqueurs
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
            
            # AMÉLIORATION: Nettoyage supplémentaire du masque
            # Supprimer les très petits trous et régions isolées
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
            
            # AMÉLIORATION: Calcul de convexité
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
            
            # AMÉLIORATION: Calcul de score qualité intégrant plusieurs facteurs
            shape_regularity = (circularity + convexity) / 2
            size_factor = min(1.0, area / 2000)  # normaliser par rapport à une taille de référence
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
                'convexity': convexity,  # Nouvel attribut
                'quality': min(1.0, max(0.5, quality_score)),  # Qualité minimum de 0.5 pour segmentation traditionnelle
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
        Apply CRF refinement to all segments
        AMÉLIORATION: Application plus robuste avec filtrage préalable
        """
        print("Applying CRF refinement to all segments...")
        
        # AMÉLIORATION: Préfiltrage gaussien léger de l'image pour réduire le bruit
        filtered_image = cv2.GaussianBlur(image_color, (3, 3), 0.5)
        
        for segment in segments_data:
            mask = segment['mask']
            
            # AMÉLIORATION: Pour les très petits objets, éviter le CRF qui peut les détruire
            if segment['area'] < 200:
                continue
                
            # AMÉLIORATION: Adapter les paramètres CRF selon la forme
            if segment['circularity'] > 0.85:  # Formes très circulaires
                # Préserver les bords réguliers
                crf_params = {
                    'gaussian_sxy': 2,
                    'gaussian_compat': 5,
                    'bilateral_sxy': 50,
                    'bilateral_srgb': 8,
                    'bilateral_compat': 15,
                    'iterations': 8
                }
            else:  # Formes irrégulières
                # Paramètres plus conservateurs
                crf_params = {
                    'gaussian_sxy': 3,
                    'gaussian_compat': 4,
                    'bilateral_sxy': 70,
                    'bilateral_srgb': 13,
                    'bilateral_compat': 10,
                    'iterations': 10
                }
            
            # Appliquer le CRF avec l'image filtrée et les paramètres adaptés
            refined_mask = self.crf_processor.apply_crf(filtered_image, mask, crf_params)
            
            # AMÉLIORATION: Vérifier que le raffinement n'a pas détruit l'objet
            if np.sum(refined_mask) > np.sum(mask) * 0.25:  # Si on conserve au moins 25% de l'objet
                segment['mask'] = refined_mask  # Update with refined mask
            # Sinon garder le masque original
        

    def detect_shape(self, contour, circularity, convexity=None):
        """
        Enhanced shape detection with more precise classifications
        AMÉLIORATION: Ajout du facteur de convexité et détection plus précise
        """
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Get bounding rect for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1
        
        # AMÉLIORATION: Utiliser convexité si disponible
        if convexity is None:
            # Calculer la convexité si non fournie
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if len(hull) > 2 else cv2.contourArea(contour) + 1e-6
            convexity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
        
        # Basic shape properties
        shape_info = {
            'type': 'forme_complexe',
            'regularity': 0.5,
            'vertices': num_vertices,
            'convexity': convexity,  # AMÉLIORATION: Ajout de la convexité
            'approx_contour': approx
        }
        
        # AMÉLIORATION: Détection de forme plus précise
        if circularity > 0.9 and convexity > 0.9:
            shape_info['type'] = 'cercle'
            shape_info['regularity'] = circularity * convexity
        
        elif circularity > 0.8 and num_vertices > 8:
            # Vérifier si c'est une ellipse en analysant les axes
            # Calculer les axes de l'ellipse ajustée
            if len(contour) >= 5:  # Minimum 5 points pour fitEllipse
                try:
                    (center, axes, angle) = cv2.fitEllipse(contour)
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    axis_ratio = minor_axis / (major_axis + 1e-6)
                    
                    if axis_ratio < 0.95:  # Pas trop proche d'un cercle
                        shape_info['type'] = 'ellipse'
                        shape_info['regularity'] = 0.9 * convexity * axis_ratio
                        # Stocker les informations de l'ellipse
                        shape_info['ellipse'] = {
                            'center': center,
                            'axes': axes,
                            'angle': angle
                        }
                    else:
                        shape_info['type'] = 'cercle'
                        shape_info['regularity'] = 0.95
                except:
                    # En cas d'erreur, conserver la classification basée sur la circularité
                    shape_info['type'] = 'forme_complexe' if convexity < 0.8 else 'ellipse'
                    shape_info['regularity'] = 0.8 * convexity
            else:
                shape_info['type'] = 'forme_complexe'
                
        elif num_vertices == 3 or (3 <= num_vertices <= 4 and circularity < 0.6):
            # AMÉLIORATION: Vérification plus stricte pour les triangles
            if convexity > 0.9 and circularity < 0.7:
                shape_info['type'] = 'triangle'
                shape_info['regularity'] = self.calculate_regularity(approx, 3)
            else:
                shape_info['type'] = 'polygone'
                shape_info['regularity'] = convexity * 0.8
            
        elif num_vertices == 4 or (4 <= num_vertices <= 6 and circularity > 0.7):
            # Check if it's a square or rectangle
            if aspect_ratio > 0.9 and convexity > 0.9:
                shape_info['type'] = 'carre'
                shape_info['regularity'] = 0.95 * aspect_ratio
            else:
                shape_info['type'] = 'rectangle'
                shape_info['regularity'] = 0.9 * convexity * aspect_ratio
                
        elif 5 <= num_vertices <= 10:
            # Check if it's a regular polygon
            shape_info['type'] = f'polygone{num_vertices}'
            shape_info['regularity'] = self.calculate_regularity(approx, num_vertices) * convexity
            
        else:
            # Complex shape - improve the contour for DXF export
            shape_info['type'] = 'forme_complexe'
            # Calculate how smooth/regular the contour is
            shape_info['regularity'] = min(0.7, 0.5 + (circularity * convexity) / 2)
            
        return shape_info
    
    def calculate_regularity(self, points, n_sides):
        """
        Calculate how regular/symmetric a polygon is
        AMÉLIORATION: Algorithme plus précis pour évaluer la régularité
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
        
        # AMÉLIORATION: Vérifier aussi les angles
        angles = []
        if len(pts) >= 3:
            for i in range(len(pts)):
                prev_i = (i - 1) % len(pts)
                next_i = (i + 1) % len(pts)
                
                # Vecteurs des côtés
                v1 = pts[prev_i] - pts[i]
                v2 = pts[next_i] - pts[i]
                
                # Normaliser les vecteurs
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
                
                # Produit scalaire pour l'angle
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        # Évaluer la régularité des angles
        angle_regularity = 0.5
        if angles:
            expected_angle = (n_sides - 2) * np.pi / n_sides  # Angle théorique
            angle_errors = [abs(a - expected_angle) for a in angles]
            mean_angle_error = np.mean(angle_errors) / np.pi  # Normaliser par pi
            angle_regularity = max(0, min(1, 1.0 - 2 * mean_angle_error))
        
        # Combiner les deux mesures de régularité
        return 0.7 * regularity + 0.3 * angle_regularity
    
    def extract_geometric_elements(self, contour, shape_type):
        """
        Extract geometric elements for DXF export
        AMÉLIORATION: Extraction plus précise avec meilleure détection des primitives
        """
        elements = []
        
        # Process based on shape type
        if shape_type == 'cercle':
            # AMÉLIORATION: Ajustement plus précis du cercle
            # Utiliser des points équidistants du contour pour un ajustement plus stable
            if len(contour) > 5:
                # Rééchantillonner le contour pour un meilleur ajustement
                perimeter = cv2.arcLength(contour, True)
                num_points = min(20, len(contour))  # Limiter à 20 points max
                epsilon = perimeter / num_points
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # Ajuster le cercle avec les points simplifiés
                (x, y), radius = cv2.minEnclosingCircle(simplified)
            else:
                # Utiliser directement le contour si peu de points
                (x, y), radius = cv2.minEnclosingCircle(contour)
            
            elements.append({
                'type': 'cercle',
                'center': (float(x), float(y)),
                'radius': float(radius)
            })
            
        elif shape_type == 'ellipse':
            if len(contour) >= 5:
                # AMÉLIORATION: Ajustement plus stable de l'ellipse
                # Filtrer les points aberrants pour un meilleur ajustement
                try:
                    # Utiliser fitEllipse de OpenCV
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    elements.append({
                        'type': 'ellipse',
                        'center': center,
                        'axes': axes,
                        'angle': angle
                    })
                except Exception:
                    # Si l'ajustement échoue, revenir à un polygone
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
                
        elif shape_type in ['carre', 'rectangle']:
            # AMÉLIORATION: Utiliser rotatedRect pour une meilleure détection des rectangles orientés
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            elements.append({
                'type': 'rectangle',
                'points': box.tolist(),
                'angle': rect[2]
            })
            
        elif shape_type.startswith('polygone') or shape_type == 'triangle':
            # AMÉLIORATION: Pour les polygones réguliers, optimiser les sommets
            # Extract polygon points
            points = [pt[0].tolist() for pt in contour]
            
            # Pour les polygones réguliers, optimiser la représentation
            if shape_type.startswith('polygone') and len(contour) > 6:
                # Simplifier le polygone si nécessaire
                perimeter = cv2.arcLength(contour, True)
                # Epsilon plus strict pour les polygones réguliers
                epsilon = 0.01 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = [pt[0].tolist() for pt in approx]
            
            elements.append({
                'type': 'polygone',
                'points': points
            })
            
        elif shape_type == 'forme_complexe':
            # Pour les formes complexes, nous fournissons deux représentations: 
            # - Un contour principal pour la forme globale
            # - Une décomposition en segments de ligne pour plus de détails
            
            # AMÉLIORATION: Décomposition optimisée des formes complexes
            # Simplifier le contour de façon adaptative selon sa complexité
            perimeter = cv2.arcLength(contour, True)
            num_points = len(contour)
            
            # Facteur d'epsilon adaptatif: plus strict pour les petits contours
            adaptive_factor = 0.001 if num_points > 100 else (0.005 if num_points > 50 else 0.01)
            epsilon = adaptive_factor * perimeter
            
            # Simplifier le contour
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            points = [pt[0].tolist() for pt in simplified]
            
            # Créer un contour pour la forme principale
            elements.append({
                'type': 'contour',
                'points': points
            })
            
            # Ajouter des segments de ligne pour une représentation plus détaillée
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
        """
        Visualize the segmentation results
        AMÉLIORATION: Visualisation enrichie avec plus d'informations 
        """
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
        
        # AMÉLIORATION: Masques colorés par qualité (code de couleur)
        plt.subplot(232)
        quality_mask = np.zeros(original_image.shape[:2], dtype=np.float32)
        
        # Créer un masque de qualité
        for segment in segments_data:
            mask = segment['mask']
            quality = segment['quality']
            quality_mask[mask > 0] = quality
            
        # Afficher le masque de qualité avec une colormap spécifique
        plt.imshow(quality_mask, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Qualité des segments détectés")
        
        # Shape classification avec légende améliorée
        plt.subplot(233)
        shape_img = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # Préparer un dictionnaire pour la légende
        shape_colors = {}
        color_idx = 50
        
        # Group segments by shape type
        shape_types = {}
        for segment in segments_data:
            shape_type = segment['shape']['type']
            if shape_type not in shape_types:
                shape_types[shape_type] = []
                # Assigner une couleur distincte
                shape_colors[shape_type] = color_idx
                color_idx += 25
                
            shape_types[shape_type].append(segment)
            
        # Colorier l'image selon les types de formes
        for shape_type, segments in shape_types.items():
            intensity = shape_colors[shape_type]
            for segment in segments:
                mask = segment['mask']
                shape_img[mask > 0] = intensity
                
        plt.imshow(shape_img, cmap='jet')
        
        # Ajouter une légende des types de formes
        legend_entries = []
        for shape_type in shape_colors:
            legend_entries.append(f"{shape_type}")
        plt.legend(legend_entries, loc='upper right', fontsize=8)
        plt.title("Classification des formes")
        
        # AMÉLIORATION: Détails géométriques avec annotations
        plt.subplot(234)
        plt.imshow(original_image)
        plt.title("Détails géométriques")
        
        # Ajouter des annotations sur les propriétés géométriques
        for segment in segments_data:
            if segment['quality'] < 0.5:  # Ignorer les segments basse qualité
                continue
                
            center = segment['center']
            shape_type = segment['shape']['type']
            area = segment['area']
            
            # Dessiner un point au centre
            plt.plot(center[0], center[1], 'ro', markersize=4)
            
            # Ajouter des informations géométriques
            info_text = f"{shape_type}\nA={int(area)}px²"
            plt.text(center[0] + 10, center[1], info_text, 
                    fontsize=7, color='white',
                    bbox=dict(facecolor='blue', alpha=0.7))
        
        # AMÉLIORATION: Affichage des contours simplifiés pour DXF
        plt.subplot(235)
        plt.imshow(np.zeros_like(original_image))
        plt.title("Contours simplifiés pour DXF")
        
        # Afficher les éléments géométriques
        for segment in segments_data:
            for element in segment['geometric_elements']:
                if element['type'] == 'cercle':
                    circle = plt.Circle(
                        element['center'], 
                        element['radius'],
                        fill=False, color='cyan'
                    )
                    plt.gca().add_patch(circle)
                    
                elif element['type'] == 'ellipse':
                    try:
                        center = element['center']
                        axes = element['axes']
                        angle = element['angle']
                        
                        ellipse = plt.matplotlib.patches.Ellipse(
                            center, axes[0], axes[1], 
                            angle=angle,
                            fill=False, color='yellow'
                        )
                        plt.gca().add_patch(ellipse)
                    except:
                        pass
                        
                elif element['type'] in ['rectangle', 'polygone', 'contour']:
                    points = np.array(element['points'])
                    if len(points) >= 2:
                        color = 'red' if element['type'] == 'rectangle' else (
                            'green' if element['type'] == 'polygone' else 'magenta')
                        plt.plot(points[:, 0], points[:, 1], color=color, linewidth=1)
                        
                elif element['type'] == 'lignes':
                    for start, end in element['segments']:
                        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=0.5)
        
        plt.axis('equal')
        
        # Afficher des informations de segmentation
        plt.subplot(236)
        plt.axis('off')
        
        # AMÉLIORATION: Statistiques détaillées
        high_q = len([s for s in segments_data if s['quality'] > 0.75])
        med_q = len([s for s in segments_data if 0.5 < s['quality'] <= 0.75])
        low_q = len([s for s in segments_data if s['quality'] <= 0.5])
        
        # Compter par type de forme
        shape_counts = {}
        for segment in segments_data:
            shape_type = segment['shape']['type']
            if shape_type not in shape_counts:
                shape_counts[shape_type] = 0
            shape_counts[shape_type] += 1
        
        # Trier par nombre décroissant
        sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Préparer le texte d'information
        info_text = [
            f"Méthode: {'SAM (Segment Anything)' if method=='sam' else 'Segmentation traditionnelle'}",
            f"Objets détectés: {len(segments_data)}",
            "",
            "Qualité des segments:",
            f"  - Haute qualité: {high_q}",
            f"  - Qualité moyenne: {med_q}",
            f"  - Basse qualité: {low_q}",
            "",
            "Types de formes:",
        ]
        
        # Ajouter les formes avec leur compte
        for shape, count in sorted_shapes:
            info_text.append(f"  - {shape}: {count}")
            
        # Ajouter des informations de traitement
        info_text.extend([
            "",
            f"Date: {CURRENT_DATE}",
            f"Utilisateur: {CURRENT_USER}",
            f"Version: {VERSION}"
        ])
        
        plt.text(0.1, 0.5, "\n".join(info_text), fontsize=10, va='center')
        plt.title("Statistiques de segmentation")
        
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