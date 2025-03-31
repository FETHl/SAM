#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BoxSubdivider: Advanced object detection and segmentation using YOLO-World and SAM

A professional implementation that:
1. Uses YOLO-World to detect objects and generate bounding boxes
2. Applies Segment Anything Model to predict precise object contours
3. Extracts inner contours using SAM's Mask Automatic Generator
4. Refines contours by removing duplicates and handling overlaps
5. Exports results in PNG and SVG formats with thick lines

Author: FETHl
Date: 2025-03-31
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import colorsys
import traceback
import glob
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# Try importing PyDenseCRF for optional CRF refinement
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
    from pydensecrf.utils import create_pairwise_gaussian
    HAS_CRF = True
except ImportError:
    print("Warning: pydensecrf not available. Install with: pip install pydensecrf")
    HAS_CRF = False

# Set default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    """Utility class for clean, colored logging output"""
    INFO = '\033[94m'     # Blue
    SUCCESS = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    ERROR = '\033[91m'    # Red
    ENDC = '\033[0m'      # Reset
    
    @staticmethod
    def info(msg: str) -> None:
        """Log information message in blue"""
        print(f"{Logger.INFO}[INFO] {msg}{Logger.ENDC}")
    
    @staticmethod
    def success(msg: str) -> None:
        """Log success message in green"""
        print(f"{Logger.SUCCESS}[SUCCESS] {msg}{Logger.ENDC}")
    
    @staticmethod
    def warning(msg: str) -> None:
        """Log warning message in yellow"""
        print(f"{Logger.WARNING}[WARNING] {msg}{Logger.ENDC}")
    
    @staticmethod
    def error(msg: str) -> None:
        """Log error message in red"""
        print(f"{Logger.ERROR}[ERROR] {msg}{Logger.ENDC}")


class CRFProcessor:
    """Advanced CRF-based mask refinement for precise boundaries"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize CRF processor
        
        Args:
            debug: Whether to print debug information
        """
        self.debug = debug
    
    def refine_mask(self, image: np.ndarray, mask: np.ndarray, 
                   params: Optional[Dict] = None) -> np.ndarray:
        """
        Apply CRF to refine the mask with custom unary potentials
        
        Args:
            image: RGB image
            mask: Binary mask (0 or 1)
            params: CRF parameters dict
            
        Returns:
            Refined binary mask
        """
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
        
        if self.debug:
            Logger.info(f"Binary mask shape: {binary_mask.shape}, dtype: {binary_mask.dtype}")
        
        try:
            # Create the CRF model
            d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
            
            # Create unary potentials directly to avoid index issues
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


class ContourProcessor:
    """Advanced contour processing with inner contour detection"""
    
    def __init__(self, min_contour_area: int = 100, 
                 min_internal_area: int = 20,
                 max_internal_area_ratio: float = 0.9,
                 debug: bool = False):
        """
        Initialize contour processor
        
        Args:
            min_contour_area: Minimum area for valid external contours
            min_internal_area: Minimum area for valid internal contours
            max_internal_area_ratio: Maximum ratio of internal to external contour area
            debug: Whether to print debug information
        """
        self.min_contour_area = min_contour_area
        self.min_internal_area = min_internal_area
        self.max_internal_area_ratio = max_internal_area_ratio
        self.debug = debug
    
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean a binary mask by removing noise and filling holes
        
        Args:
            mask: Binary mask to clean
            
        Returns:
            Cleaned binary mask
        """
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
    
    def get_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the largest contour from a mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Largest contour or None if no valid contours found
        """
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
    
    def smooth_contour(self, contour: np.ndarray, 
                      smoothing_factor: float = 0.002) -> np.ndarray:
        """
        Smooth a contour for better visualization
        
        Args:
            contour: Contour to smooth
            smoothing_factor: Douglas-Peucker algorithm epsilon factor
            
        Returns:
            Smoothed contour
        """
        if contour is None or len(contour) < 3:
            return contour
            
        # Calculate adaptive epsilon based on contour length
        perimeter = cv2.arcLength(contour, True)
        epsilon = smoothing_factor * perimeter
        
        # Apply Douglas-Peucker algorithm to simplify contour
        smoothed = cv2.approxPolyDP(contour, epsilon, True)
        
        return smoothed
    
    def detect_internal_contours(self, mask: np.ndarray, 
                                image: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Detect internal contours within a mask
        
        Args:
            mask: Binary mask
            image: Original image (optional, enhances detection quality)
            
        Returns:
            List of internal contours
        """
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Find external contour first to get reference area
        external_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not external_contours:
            return []
            
        # Get largest external contour
        external_contour = max(external_contours, key=cv2.contourArea)
        external_area = cv2.contourArea(external_contour)
        
        # If we have an image, use color-based edge detection for better results
        if image is not None:
            # Extract the masked region from the original image
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            
            # Convert to grayscale
            gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            
            # Use Canny edge detection with low thresholds to catch subtle edges
            edges = cv2.Canny(gray, 30, 70)
            
            # Ensure edges are only within our mask
            edges = cv2.bitwise_and(edges, edges, mask=binary_mask)
            
            # Find contours in the edges
            internal_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter internal contours
            filtered_contours = []
            
            for contour in internal_contours:
                area = cv2.contourArea(contour)
                
                # Skip contours that are too small or too large
                if (area < self.min_internal_area or 
                    area > external_area * self.max_internal_area_ratio):
                    continue
                
                # Skip contours that are too similar to the external contour
                # by checking overlap with a small buffer zone from the edge
                temp_mask = np.zeros_like(binary_mask)
                cv2.drawContours(temp_mask, [external_contour], 0, 255, 3)
                
                # Check if this contour overlaps with the edge buffer
                overlap_mask = np.zeros_like(binary_mask)
                cv2.drawContours(overlap_mask, [contour], 0, 255, 1)
                
                overlap = cv2.bitwise_and(temp_mask, overlap_mask)
                if np.count_nonzero(overlap) > 0.5 * np.count_nonzero(overlap_mask):
                    continue
                
                filtered_contours.append(contour)
            
            # Debug output
            if self.debug:
                Logger.info(f"Found {len(filtered_contours)} internal contours using image-based detection")
            
            return filtered_contours
            
        else:
            # Use contour hierarchy for finding internal contours
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
            
            # Debug output
            if self.debug:
                Logger.info(f"Found {len(internal_contours)} internal contours using hierarchy-based detection")
            
            return internal_contours
    
    def smooth_internal_contours(self, contours: List[np.ndarray], 
                                smoothing_factor: float = 0.001) -> List[np.ndarray]:
        """
        Apply smoothing to internal contours for better appearance
        
        Args:
            contours: List of contours to smooth
            smoothing_factor: Douglas-Peucker algorithm epsilon factor
            
        Returns:
            List of smoothed contours
        """
        smoothed = []
        for contour in contours:
            if len(contour) < 4:  # Skip if not enough points
                continue
                
            # Apply smoothing
            perimeter = cv2.arcLength(contour, True)
            epsilon = smoothing_factor * perimeter
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Only include if we still have enough points
            if len(smoothed_contour) >= 3:
                smoothed.append(smoothed_contour)
                
        return smoothed
    
    def process_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a mask to get a single clean contour
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (cleaned mask, smoothed outer contour)
        """
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

class SVGExporter:
    """Export contours to SVG format with enhanced styling and thick lines"""

   
    def export_to_svg(self, contours_data: List[Dict], 
                 image_shape: Tuple[int, ...],
                 output_path: str) -> bool:
        """
        Exporte les contours vers un fichier SVG avec une représentation améliorée pour les contours superposés
        
        Args:
            contours_data: Liste de dictionnaires avec la structure:
                {
                    'outer_contour': contour_externe,
                    'internal_contours': [contours_internes],
                    'label': étiquette,
                    'color': couleur,
                    'opacity': opacité (optionnel),
                    'dash_pattern': motif de tirets (optionnel),
                    'line_thickness': épaisseur de ligne (optionnel),
                    'label_offset': décalage d'étiquette (optionnel)
                }
            image_shape: Dimensions de l'image (h, w)
            output_path: Chemin pour sauvegarder le fichier SVG
            
        Returns:
            Indicateur de succès
        """
        # Extraire les dimensions de l'image
        h, w = image_shape[:2]
        
        # Entête SVG avec métadonnées et styles
        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" version="1.1">
    <metadata>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:dc="http://purl.org/dc/elements/1.1/">
        <rdf:Description>
            <dc:creator>FETHl</dc:creator>
            <dc:date>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</dc:date>
            <dc:description>Généré par BoxSubdivider avec segmentation YOLO-World + SAM</dc:description>
        </rdf:Description>
        </rdf:RDF>
    </metadata>

    <!-- Styles CSS -->
    <style>
        .outer-contour {{ stroke-width: 3px; }}
        .inner-contour {{ stroke-width: 2px; }}
        .highlight {{ filter: drop-shadow(0px 0px 2px rgba(255, 255, 255, 0.7)); }}
        text {{ filter: drop-shadow(0px 0px 1px rgba(0, 0, 0, 0.9)); }}
    </style>

    <!-- Arrière-plan (transparent) -->
    <rect width="{w}" height="{h}" fill="none" />

    '''
        
        # Traiter chaque contour
        for i, data in enumerate(contours_data):
            outer_contour = data['outer_contour']
            internal_contours = data.get('internal_contours', [])
            label = data['label']
            color = data['color']
            
            # Options pour les contours superposés
            opacity = data.get('opacity', 0.3)
            stroke_opacity = min(0.8, opacity + 0.2)  # Légèrement plus visible que le remplissage
            dash_pattern = data.get('dash_pattern', "")
            line_thickness = data.get('line_thickness', 3.0)
            label_offset = data.get('label_offset', (0, 0))
            
            if outer_contour is None or len(outer_contour) < 3:
                continue
                
            # Convertir la couleur en hexadécimal
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )
            
            # Créer une couleur légèrement différente pour les contours internes
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
            
            # Créer un chemin SVG à partir du contour externe
            outer_path_data = "M "
            for j, point in enumerate(outer_contour):
                x, y = point[0]
                outer_path_data += f"{x},{y} "
                if j < len(outer_contour) - 1:
                    outer_path_data += "L "
            outer_path_data += "Z"  # Fermer le chemin
            
            # Commencer le groupe pour cet objet
            svg += f'  <g id="object-{i+1}" class="highlight">\n'
            
            # Ajouter le contour externe au SVG avec un style de ligne épaisse
            dash_attr = f' stroke-dasharray="{dash_pattern}"' if dash_pattern else ''
            svg += f'    <path d="{outer_path_data}" fill="{color_hex}" fill-opacity="{opacity}" ' \
                f'stroke="{color_hex}" stroke-width="{line_thickness}" stroke-opacity="{stroke_opacity}"{dash_attr} />\n'
            
            # Ajouter les contours internes
            for j, internal_contour in enumerate(internal_contours):
                if len(internal_contour) < 3:  # Ignorer les contours trop petits
                    continue
                
                # Créer un chemin pour le contour interne
                internal_path_data = "M "
                for k, point in enumerate(internal_contour):
                    x, y = point[0]
                    internal_path_data += f"{x},{y} "
                    if k < len(internal_contour) - 1:
                        internal_path_data += "L "
                internal_path_data += "Z"  # Fermer le chemin
                
                # Ajouter le contour interne avec un style différent et des lignes épaisses
                internal_thickness = max(1.0, line_thickness - 0.5)
                svg += f'    <path d="{internal_path_data}" fill="none" stroke="{internal_color_hex}" ' \
                    f'stroke-width="{internal_thickness}" stroke-opacity="{stroke_opacity}"{dash_attr} />\n'
            
            # Ajouter l'étiquette
            # Trouver une bonne position pour l'étiquette (près du haut du contour)
            x_values = [p[0][0] for p in outer_contour]
            y_values = [p[0][1] for p in outer_contour]
            
            if x_values and y_values:
                # Utiliser le centroïde pour la position de l'étiquette
                label_x = sum(x_values) / len(x_values) + label_offset[0]
                label_y = min(y_values) - 10 + label_offset[1]
                
                # Rester dans les limites
                label_y = max(15, label_y)
                
                # Ajouter le texte de l'étiquette avec une meilleure visibilité (plus grand, gras)
                svg += f'    <text x="{label_x}" y="{label_y}" font-family="Arial" font-size="16" ' \
                    f'fill="{color_hex}" text-anchor="middle" font-weight="bold">{label}</text>\n'
            
            # Fermer le groupe
            svg += '  </g>\n'
        
        # Fermer le SVG
        svg += '</svg>'
        
        # S'assurer que le répertoire existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sauvegarder dans un fichier
        with open(output_path, 'w') as f:
            f.write(svg)
        
        return True

class ContourMerger:
    """
    Fusionne les contours superposés pour éliminer les duplications
    et améliorer la visualisation des objets.
    """
    
    def __init__(self, iou_threshold=0.5, internal_overlap_threshold=0.8, debug=False):
        """
        Initialisation du fusionneur de contours
        
        Args:
            iou_threshold: Seuil d'IoU pour considérer que des contours se chevauchent
            internal_overlap_threshold: Seuil pour détecter les contours internes en double
            debug: Activer le mode debug
        """
        self.iou_threshold = iou_threshold
        self.internal_overlap_threshold = internal_overlap_threshold
        self.debug = debug
    
    def calculate_iou(self, mask1, mask2):
        """Calcule l'Intersection over Union entre deux masques"""
        if mask1 is None or mask2 is None:
            return 0.0
            
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def calculate_contour_overlap(self, contour1, contour2, shape):
        """
        Calcule le chevauchement entre deux contours en créant des masques
        
        Args:
            contour1: Premier contour
            contour2: Deuxième contour
            shape: Forme du masque (hauteur, largeur)
            
        Returns:
            Valeur de chevauchement entre 0 et 1
        """
        mask1 = np.zeros(shape, dtype=np.uint8)
        mask2 = np.zeros(shape, dtype=np.uint8)
        
        cv2.drawContours(mask1, [contour1], 0, 1, -1)
        cv2.drawContours(mask2, [contour2], 0, 1, -1)
        
        return self.calculate_iou(mask1, mask2)
    
    def deduplicate_internal_contours(self, internal_contours, shape):
        """
        Élimine les contours internes en double
        
        Args:
            internal_contours: Liste des contours internes
            shape: Forme de l'image (h, w)
            
        Returns:
            Liste des contours internes sans doublons
        """
        if not internal_contours or len(internal_contours) <= 1:
            return internal_contours
            
        # Trier les contours par aire (du plus grand au plus petit)
        sorted_contours = sorted(internal_contours, 
                                key=lambda c: cv2.contourArea(c), 
                                reverse=True)
        
        # Liste pour stocker les contours uniques
        unique_contours = []
        
        for contour in sorted_contours:
            # Ignorer les contours trop petits
            if cv2.contourArea(contour) < 20:
                continue
                
            # Vérifier si ce contour est similaire à un contour déjà conservé
            is_duplicate = False
            for existing_contour in unique_contours:
                overlap = self.calculate_contour_overlap(contour, existing_contour, shape)
                if overlap > self.internal_overlap_threshold:
                    is_duplicate = True
                    break
            
            # Si ce n'est pas un doublon, l'ajouter à la liste
            if not is_duplicate:
                unique_contours.append(contour)
                
                # Limiter à 10 contours internes pour éviter la surcharge
                if len(unique_contours) >= 10:
                    break
        
        if self.debug and len(internal_contours) != len(unique_contours):
            print(f"Dédoublonnage: {len(internal_contours)} → {len(unique_contours)} contours internes")
            
        return unique_contours
    
    def merge_overlapping_contours(self, contours_data):
        """
        Fusionne les contours qui se chevauchent et élimine les doublons
        
        Args:
            contours_data: Liste de dictionnaires avec contours, masques, etc.
            
        Returns:
            Liste des contours fusionnés sans doublons
        """
        if not contours_data or len(contours_data) <= 1:
            return contours_data
            
        # Créer une matrice d'IoU pour tous les masques
        n = len(contours_data)
        iou_matrix = np.zeros((n, n))
        
        # Obtenir les dimensions de l'image à partir du premier masque
        image_shape = None
        for data in contours_data:
            if 'mask' in data and data['mask'] is not None:
                image_shape = data['mask'].shape
                break
                
        if image_shape is None:
            # Fallback - utiliser la forme du premier contour
            for data in contours_data:
                if 'outer_contour' in data and data['outer_contour'] is not None:
                    y_max = max([p[0][1] for p in data['outer_contour']])
                    x_max = max([p[0][0] for p in data['outer_contour']])
                    image_shape = (y_max + 10, x_max + 10)
                    break
        
        # Si nous n'avons toujours pas de forme d'image, utiliser une valeur par défaut
        if image_shape is None:
            image_shape = (512, 512)
            
        # Calculer l'IoU entre tous les masques
        for i in range(n):
            for j in range(i+1, n):
                # Si les deux objets ont des masques, utiliser l'IoU des masques
                if ('mask' in contours_data[i] and contours_data[i]['mask'] is not None and
                    'mask' in contours_data[j] and contours_data[j]['mask'] is not None):
                    iou = self.calculate_iou(contours_data[i]['mask'], contours_data[j]['mask'])
                    
                # Sinon, utiliser le chevauchement des contours
                elif ('outer_contour' in contours_data[i] and contours_data[i]['outer_contour'] is not None and
                      'outer_contour' in contours_data[j] and contours_data[j]['outer_contour'] is not None):
                    iou = self.calculate_contour_overlap(
                        contours_data[i]['outer_contour'], 
                        contours_data[j]['outer_contour'],
                        image_shape
                    )
                else:
                    iou = 0.0
                    
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Trouver les groupes de contours à fusionner avec un algorithme d'union-find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Regrouper les contours qui se chevauchent
        for i in range(n):
            for j in range(i+1, n):
                if iou_matrix[i, j] > self.iou_threshold:
                    union(i, j)
        
        # Collecter les groupes de contours
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Fusion des contours dans chaque groupe
        merged_contours_data = []
        
        for group_indices in groups.values():
            if len(group_indices) == 1:
                # Un seul contour dans le groupe - dédupliquer les contours internes
                data = contours_data[group_indices[0]]
                if 'internal_contours' in data and data['internal_contours']:
                    data['internal_contours'] = self.deduplicate_internal_contours(
                        data['internal_contours'], image_shape)
                merged_contours_data.append(data)
            else:
                # Fusionner les masques du groupe
                merged_data = self._merge_contour_group(
                    [contours_data[i] for i in group_indices], image_shape)
                merged_contours_data.append(merged_data)
        
        if self.debug:
            print(f"Fusion: {n} contours → {len(merged_contours_data)} contours fusionnés")
            
        return merged_contours_data
    
    def _merge_contour_group(self, group_data, image_shape):
        """
        Fusionne un groupe de contours superposés
        
        Args:
            group_data: Liste des données de contours à fusionner
            image_shape: Forme de l'image
            
        Returns:
            Données fusionnées
        """
        # Prioriser l'objet avec le score le plus élevé pour les attributs comme la couleur
        group_data = sorted(group_data, key=lambda x: x.get('confidence', 0.0), reverse=True)
        best_data = group_data[0]
        
        # Fusionner les masques
        merged_mask = None
        for data in group_data:
            if 'mask' in data and data['mask'] is not None:
                if merged_mask is None:
                    merged_mask = data['mask'].copy()
                else:
                    merged_mask = np.logical_or(merged_mask, data['mask']).astype(np.uint8)
        
        # Extraire le contour externe du masque fusionné
        outer_contour = None
        if merged_mask is not None:
            contours, _ = cv2.findContours(
                merged_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE
            )
            if contours:
                outer_contour = max(contours, key=cv2.contourArea)
            else:
                outer_contour = best_data.get('outer_contour')
        else:
            outer_contour = best_data.get('outer_contour')
        
        # Collecter tous les contours internes
        all_internal_contours = []
        for data in group_data:
            if 'internal_contours' in data and data['internal_contours']:
                all_internal_contours.extend(data['internal_contours'])
        
        # Dédupliquer les contours internes
        unique_internal_contours = self.deduplicate_internal_contours(
            all_internal_contours, image_shape)
        
        # Fusion des attributs d'étiquette
        # Prendre les noms de classe uniques (sans scores)
        class_names = set()
        for data in group_data:
            if 'label' in data:
                # Extraire le nom de classe (avant l'espace qui sépare du score)
                class_name = data['label'].split(' ')[0]
                class_names.add(class_name)
        
        merged_label = " + ".join(sorted(class_names))
        if 'confidence' in best_data:
            merged_label += f" {best_data['confidence']:.2f}"
        
        # Construire les données fusionnées
        merged_data = {
            'outer_contour': outer_contour,
            'internal_contours': unique_internal_contours,
            'mask': merged_mask,
            'label': merged_label,
            'color': best_data['color'],
            'confidence': max(d.get('confidence', 0.0) for d in group_data)
        }
        
        return merged_data

class OverlappingContoursHandler:
    """
    Gère la détection et la visualisation des contours superposés ou presque superposés
    pour améliorer l'affichage et le traitement
    """
    
    def __init__(self, overlap_threshold=0.5, min_distance=10, debug=False):
        """
        Initialise le gestionnaire de contours superposés
        
        Args:
            overlap_threshold: Seuil d'IoU pour considérer que des contours se chevauchent
            min_distance: Distance minimale entre contours pour les considérer comme presque superposés
            debug: Active le mode debug
        """
        self.overlap_threshold = overlap_threshold
        self.min_distance = min_distance
        self.debug = debug
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calcule l'Intersection over Union (IoU) entre deux masques
        
        Args:
            mask1: Premier masque binaire
            mask2: Deuxième masque binaire
            
        Returns:
            Valeur IoU entre 0 et 1
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0
            
        return intersection / union
    
    def create_contour_mask(self, contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Crée un masque à partir d'un contour
        
        Args:
            contour: Contour à convertir en masque
            shape: Forme du masque (hauteur, largeur)
            
        Returns:
            Masque binaire
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 1, -1)  # Remplir le contour
        return mask
    
    def detect_overlapping_contours(self, contours_data: List[Dict]) -> List[List[int]]:
        """
        Détecte les groupes de contours superposés
        
        Args:
            contours_data: Liste de dictionnaires contenant les contours et leurs masques
            
        Returns:
            Liste des groupes de contours superposés (indices dans contours_data)
        """
        n = len(contours_data)
        if n <= 1:
            return []
            
        # Construire un graphe d'adjacence des contours qui se chevauchent
        overlap_graph = [[] for _ in range(n)]
        
        for i in range(n):
            if 'mask' not in contours_data[i] or contours_data[i]['mask'] is None:
                continue
                
            for j in range(i+1, n):
                if 'mask' not in contours_data[j] or contours_data[j]['mask'] is None:
                    continue
                    
                # Calculer l'IoU entre les masques
                iou = self.calculate_iou(contours_data[i]['mask'], contours_data[j]['mask'])
                
                # Si l'IoU est supérieur au seuil, les contours se chevauchent
                if iou > self.overlap_threshold:
                    overlap_graph[i].append(j)
                    overlap_graph[j].append(i)
        
        # Trouver les composantes connexes (groupes de contours superposés)
        visited = [False] * n
        overlapping_groups = []
        
        for i in range(n):
            if not visited[i]:
                group = []
                self._dfs(i, overlap_graph, visited, group)
                
                if len(group) > 1:  # Ne considérer que les groupes avec au moins 2 contours
                    overlapping_groups.append(group)
        
        return overlapping_groups
    
    def _dfs(self, node: int, graph: List[List[int]], visited: List[bool], group: List[int]):
        """
        Parcours en profondeur (DFS) pour trouver les composantes connexes
        
        Args:
            node: Nœud actuel
            graph: Graphe d'adjacence
            visited: Liste des nœuds visités
            group: Groupe de nœuds connectés actuel
        """
        visited[node] = True
        group.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                self._dfs(neighbor, graph, visited, group)
    
    def process_overlapping_contours(self, contours_data: List[Dict]) -> List[Dict]:
        """
        Traite les contours superposés pour améliorer la visualisation
        
        Args:
            contours_data: Liste des données de contours
            
        Returns:
            Liste mise à jour des données de contours
        """
        # Détecter les groupes de contours superposés
        overlapping_groups = self.detect_overlapping_contours(contours_data)
        
        if not overlapping_groups:
            return contours_data
            
        if self.debug:
            print(f"Détecté {len(overlapping_groups)} groupes de contours superposés")
        
        # Pour chaque groupe de contours superposés
        for group in overlapping_groups:
            # Trier le groupe par score de confiance (si disponible) ou par aire
            sorted_group = sorted(group, key=lambda i: contours_data[i].get('confidence', 0.0) 
                                 if 'confidence' in contours_data[i] 
                                 else cv2.contourArea(contours_data[i]['outer_contour']),
                                 reverse=True)
            
            # Appliquer des styles visuels différents pour chaque contour superposé
            for idx, contour_idx in enumerate(sorted_group):
                # Modifier l'opacité en fonction de la position dans le groupe
                opacity = max(0.2, 1.0 - idx * 0.15)  # Plus faible pour les contours inférieurs
                contours_data[contour_idx]['opacity'] = opacity
                
                # Ajouter un décalage pour les étiquettes
                if 'label_offset' not in contours_data[contour_idx]:
                    contours_data[contour_idx]['label_offset'] = (10 * idx, 5 * idx)
                    
                # Ajouter un style de contour différent (tirets, pointillés, etc.)
                # Pour l'exportation SVG
                dash_patterns = ["", "5,5", "10,5", "5,2,2,2", "10,5,5,5"]
                contours_data[contour_idx]['dash_pattern'] = dash_patterns[min(idx, len(dash_patterns)-1)]
                
                # Ajuster l'épaisseur de ligne
                contours_data[contour_idx]['line_thickness'] = max(1, 3 - idx * 0.5)
        
        return contours_data

class BoxSubdivider:
    """
    BoxSubdivider - Advanced object detection and segmentation using YOLO-World and SAM
    
    This class provides a complete pipeline for:
    1. Detecting objects with YOLO-World
    2. Generating precise contours with SAM
    3. Extracting inner contours
    4. Refining results and removing duplicates
    5. Exporting to PNG and SVG with thick lines
    """
    
    def __init__(self, 
                yolo_config: Optional[str] = None,
                yolo_checkpoint: Optional[str] = None,
                sam_checkpoint: Optional[str] = None,
                device: Optional[torch.device] = None,
                debug: bool = False):
        """
        Initialize BoxSubdivider with model configurations
        
        Args:
            yolo_config: Path to YOLO-World config file
            yolo_checkpoint: Path to YOLO-World checkpoint file
            sam_checkpoint: Path to SAM checkpoint file
            device: Torch device (cuda or cpu)
            debug: Whether to print debug information
        """
        self.debug = debug
        self.device = device if device is not None else DEVICE
        
        # Initialize components
        self.crf_processor = CRFProcessor(debug=debug)
        self.contour_processor = ContourProcessor(
            min_contour_area=100,
            min_internal_area=20,
            max_internal_area_ratio=0.9,
            debug=debug
        )
        self.svg_exporter = SVGExporter()
        
        # Initialize models if paths are provided
        self.yolo_model = None
        self.yolo_pipeline = None
        self.sam_models = None
        
        if yolo_config and yolo_checkpoint:
            self.initialize_yolo_world(yolo_config, yolo_checkpoint)
            
        if sam_checkpoint:
            self.initialize_sam(sam_checkpoint)
    
    def initialize_yolo_world(self, config_file: str, checkpoint_file: str) -> bool:
        """
        Initialize YOLO-World model and pipeline
        
        Args:
            config_file: Path to YOLO-World config file
            checkpoint_file: Path to YOLO-World checkpoint file
            
        Returns:
            Success flag
        """
        try:
            # Import necessary modules
            from mmdet.apis import init_detector
            from mmyolo.utils import register_all_modules
            from mmcv.transforms import Compose
            
            # Register modules
            register_all_modules()
            
            # Initialize detector
            Logger.info(f"Loading YOLO-World model from {checkpoint_file}")
            self.yolo_model = init_detector(config_file, checkpoint_file, device=self.device)
            
            # Create test pipeline as a Compose object
            test_pipeline_cfg = self.yolo_model.cfg.test_dataloader.dataset.pipeline
            self.yolo_pipeline = Compose(test_pipeline_cfg)
            
            Logger.success("YOLO-World model initialized successfully")
            return True
            
        except Exception as e:
            Logger.error(f"Failed to initialize YOLO-World: {e}")
            traceback.print_exc()
            return False
    
    def initialize_sam(self, checkpoint_file: str) -> bool:
        """
        Initialisation améliorée de SAM avec des paramètres optimisés pour les images complexes
        
        Args:
            checkpoint_file: Chemin vers le fichier de checkpoint SAM
            
        Returns:
            Succès de l'initialisation
        """
        try:
            from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
            
            # Déterminer le type de modèle à partir du nom de fichier
            if "vit_h" in checkpoint_file.lower():
                model_type = "vit_h"
            elif "vit_l" in checkpoint_file.lower():
                model_type = "vit_l"
            elif "vit_b" in checkpoint_file.lower():
                model_type = "vit_b"
            else:
                model_type = "vit_h"  # Par défaut le plus grand modèle
            
            # Charger le modèle SAM
            Logger.info(f"Chargement de SAM {model_type} depuis {checkpoint_file}")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_file)
            sam.to(device=self.device)
            
            # Créer le prédicteur pour les contours externes avec des paramètres améliorés
            predictor = SamPredictor(sam)
            
            # Créer le générateur automatique de masques pour les contours internes
            # Paramètres optimisés pour détecter davantage de détails dans les images complexes
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,            # Plus de points pour une meilleure précision (défaut: 32)
                pred_iou_thresh=0.86,          # Seuil IoU légèrement abaissé (défaut: 0.88)
                stability_score_thresh=0.90,   # Seuil de stabilité légèrement abaissé (défaut: 0.95)
                crop_n_layers=1,               # Utiliser 1 couche de recadrage pour les grands objets
                crop_n_points_downscale_factor=2,  # Facteur de mise à l'échelle pour le recadrage
                min_mask_region_area=20,       # Aire minimale plus petite pour capturer les petits détails
                output_mode='binary_mask'      # Mode de sortie en masques binaires
            )
            
            # Stocker les deux modèles
            self.sam_models = {
                "predictor": predictor,
                "mask_generator": mask_generator
            }
            
            Logger.success(f"SAM initialisé avec succès avec le modèle {model_type}")
            return True
            
        except ImportError:
            Logger.error("Échec d'importation de segment_anything. Installez-le avec: pip install segment-anything")
            return False
        except FileNotFoundError:
            Logger.error(f"Fichier de checkpoint SAM non trouvé: {checkpoint_file}")
            return False
        except Exception as e:
            Logger.error(f"Échec d'initialisation de SAM: {str(e)}")
            return False

    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load image with basic preprocessing
        
        Args:
            filepath: Path to image file
            
        Returns:
            RGB image array
        """
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
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement amélioré des images pour une meilleure détection
        
        Args:
            image: Image d'entrée
            
        Returns:
            Image prétraitée
        """
        # Convertir en RGB si en niveaux de gris
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normaliser si nécessaire (s'assurer que les valeurs sont entre 0-255)
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Copier l'image pour ne pas modifier l'original
        processed = image.copy()
        
        # Amélioration du contraste pour une meilleure détection des objets
        # Appliquer CLAHE sur le canal V de HSV pour préserver les couleurs
        hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Améliorer le contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # Recombiner les canaux
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        processed = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        # Réduction légère du bruit avec préservation des bords
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 15)
        
        # Augmenter légèrement la saturation pour mieux distinguer les objets
        hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return processed
    
    def expand_box(self, box: np.ndarray, expand_ratio: float, 
                 image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Expand bounding box with boundary check
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            expand_ratio: Ratio to expand box by
            image_shape: Image shape (h, w, c)
            
        Returns:
            Expanded box [x1, y1, x2, y2]
        """
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
        
        return np.array([x1, y1, x2, y2])


    def detect_objects(self, image: np.ndarray, prompts: List[List[str]], 
                     threshold: float = 0.05) -> Dict:
        """
        Detect objects in an image using YOLO-World with text prompts
        
        Args:
            image: Input image
            prompts: List of text prompts (each as a list of strings)
            threshold: Detection confidence threshold
            
        Returns:
            Dictionary of detection results
        """
        if self.yolo_model is None or self.yolo_pipeline is None:
            raise ValueError("YOLO-World model not initialized")
            
        Logger.info(f"Running YOLO-World detection with {len(prompts)} prompt classes...")
        
        try:
            # Process the image with YOLO-World
            # Flatten prompts - YOLO-World expects a flat list
            flat_prompts = [p[0] for p in prompts]  # Just take first text variant
            
            # Create a temporary file path for the image if needed
            temp_img_path = os.path.join(os.getcwd(), "temp_detection_image.jpg")
            cv2.imwrite(temp_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Prepare input data with proper image path
            data_info = dict(img_id=0, img_path=temp_img_path, texts=flat_prompts)
            data_info = self.yolo_pipeline(data_info)
            data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                            data_samples=[data_info['data_samples']])
            
            # Run model
            with torch.no_grad():
                output = self.yolo_model.test_step(data_batch)[0]
                pred_instances = output.pred_instances
                pred_instances = pred_instances[pred_instances.scores.float() > threshold]
                
            # Clean up the temporary file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
            # Extract detection results
            boxes = pred_instances.bboxes.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            
            # Create text labels and extract class names
            text_labels = []
            class_names = []
            for label_id, score in zip(labels, scores):
                class_name = flat_prompts[label_id]
                class_names.append(class_name)
                text_labels.append(f"{class_name} {score:.2f}")
            
            Logger.success(f"Found {len(boxes)} objects")
            
            # Generate colors for each unique class
            unique_classes = sorted(set(class_names))
            class_counts = {cls: class_names.count(cls) for cls in unique_classes}
            
            # Generate a color for each unique class
            class_colors = {}
            for i, class_name in enumerate(unique_classes):
                hue = (i * 0.618033988749895) % 1.0  # Golden ratio for good color distribution
                class_colors[class_name] = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
            
            return {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'text_labels': text_labels,
                'class_names': class_names,
                'class_colors': class_colors,
                'class_counts': class_counts
            }
            
        except Exception as e:
            Logger.error(f"Error in YOLO-World inference: {e}")
            traceback.print_exc()
            return {
                'boxes': [],
                'labels': [],
                'scores': [],
                'text_labels': [],
                'class_names': [],
                'class_colors': {},
                'class_counts': {}
            }

    def _non_maximum_suppression(self, boxes: np.ndarray, threshold: float = 0.5) -> List[int]:
        """
        Apply non-maximum suppression to remove overlapping boxes
        
        Args:
            boxes: Array of boxes with format [x1, y1, x2, y2, score]
            threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        # If no boxes, return empty list
        if len(boxes) == 0:
            return []
        
        # Extract coordinates and scores
        x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by score (descending)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IoU with other boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Get indices of boxes to keep
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
            
        return keep

    def generate_mask_from_box(self, image: np.ndarray, box: np.ndarray, 
                            expand_ratio: float = 0.15) -> Tuple[np.ndarray, float]:
        """
        Génération améliorée de masques à partir des boîtes avec stratégies multiples
        
        Args:
            image: Image d'entrée
            box: Boîte englobante [x1, y1, x2, y2]
            expand_ratio: Ratio d'expansion de la boîte
            
        Returns:
            Tuple (masque, score)
        """
        if self.sam_models is None or 'predictor' not in self.sam_models:
            return None, 0.0
            
        try:
            # Stratégie 1: Essayer avec la boîte étendue
            expanded_box = self.expand_box(box, expand_ratio, image.shape)
            
            # Prétraiter légèrement l'image pour SAM
            sam_image = image.copy()
            
            # Définir l'image pour le prédicteur SAM
            self.sam_models['predictor'].set_image(sam_image)
            
            # Obtenir le masque du prédicteur SAM avec multimask_output=True pour avoir plusieurs options
            masks, scores, _ = self.sam_models['predictor'].predict(
                point_coords=None,
                point_labels=None,
                box=expanded_box,
                multimask_output=True
            )
            
            # Si nous avons des masques, utiliser le meilleur
            if masks is not None and len(masks) > 0:
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(bool).astype(np.uint8)
                score = scores[best_idx]
                
                # Vérifier si le masque est trop petit (possible échec de SAM)
                mask_area = np.sum(mask)
                box_area = (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1])
                
                # Si le masque est trop petit par rapport à la boîte, essayer une autre stratégie
                if mask_area < box_area * 0.2:
                    # Stratégie 2: Essayer sans expansion de boîte
                    original_box = box.copy()
                    masks, scores, _ = self.sam_models['predictor'].predict(
                        point_coords=None,
                        point_labels=None,
                        box=original_box,
                        multimask_output=True
                    )
                    
                    if masks is not None and len(masks) > 0:
                        best_idx = np.argmax(scores)
                        mask = masks[best_idx].astype(bool).astype(np.uint8)
                        score = scores[best_idx]
                
                # Vérification finale - si le masque est toujours trop petit, utiliser la boîte directement
                mask_area = np.sum(mask)
                if mask_area < box_area * 0.1:
                    # Stratégie 3: Créer un masque à partir de la boîte elle-même
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = 1
                    score = 0.5  # Score moyen car nous utilisons une boîte, pas une prédiction SAM
                    
                return mask, score
            else:
                # Si SAM échoue, créer un masque simple à partir de la boîte
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = 1
                return mask, 0.5
                    
        except Exception as e:
            Logger.error(f"Erreur dans la génération de masque SAM: {e}")
            traceback.print_exc()
            
            # Fallback - créer un masque simple à partir de la boîte
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1
            return mask, 0.1

    def detect_internal_contours(self, mask: np.ndarray, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect internal contours within a mask using SAM's mask generator
        
        Args:
            mask: Binary mask
            image: Original image
            
        Returns:
            List of internal contours
        """
        if self.sam_models is None or 'mask_generator' not in self.sam_models:
            # Fall back to traditional method
            return self.contour_processor.detect_internal_contours(mask, image)
            
        try:
            # Create binary mask
            binary_mask = (mask > 0).astype(np.uint8) * 255
            
            # Extract the masked region from the original image
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            
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
            x_end = min(image.shape[1], x + w + padding_x)
            y_end = min(image.shape[0], y + h + padding_y)
            
            # Crop the masked region
            cropped_mask = binary_mask[y_start:y_end, x_start:x_end]
            cropped_image = image[y_start:y_end, x_start:x_end].copy()
            
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
            sam_masks = self.sam_models['mask_generator'].generate(cropped_image)
            
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
                debug_img = image.copy()
                cv2.drawContours(debug_img, internal_contours, -1, (0,255,0), 1)
                cv2.imwrite(f"{debug_dir}/sam_internal_contours.jpg", 
                        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            
            # If no contours found with SAM, try traditional method
            if len(internal_contours) == 0:
                internal_contours = self.contour_processor.detect_internal_contours(mask, image)
            
            # Smooth the internal contours
            internal_contours = self.contour_processor.smooth_internal_contours(
                internal_contours, smoothing_factor=0.001)
            
            return internal_contours
            
        except Exception as e:
            Logger.error(f"Error in SAM internal contour detection: {e}")
            traceback.print_exc()
            
            # Fall back to traditional method
            return self.contour_processor.detect_internal_contours(mask, image)

    def _draw_merged_contours(self, image: np.ndarray, merged_contours_data: List[Dict]) -> np.ndarray:
        """
        Dessine les contours fusionnés sur l'image
        
        Args:
            image: Image sur laquelle dessiner
            merged_contours_data: Données des contours fusionnés
            
        Returns:
            Image avec contours fusionnés
        """
        # Copie de l'image pour ne pas modifier l'original
        result = image.copy()
        
        for data in merged_contours_data:
            # Extraire les données
            outer_contour = data['outer_contour']
            internal_contours = data.get('internal_contours', [])
            color = data['color']
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            
            # Dessiner le contour externe avec une ligne épaisse
            if outer_contour is not None and len(outer_contour) >= 3:
                # Dessiner le masque avec transparence si disponible
                if 'mask' in data and data['mask'] is not None:
                    mask_overlay = np.zeros_like(result, dtype=np.uint8)
                    mask_overlay[data['mask'] > 0] = [int(c*255) for c in color]
                    alpha = 0.3  # Transparence
                    result = cv2.addWeighted(result, 1, mask_overlay, alpha, 0)
                
                # Dessiner le contour externe avec une ligne épaisse
                cv2.drawContours(result, [outer_contour], -1, color_bgr, 3)
                
                # Dessiner les contours internes si disponibles
                if internal_contours:
                    # Couleur légèrement différente pour les contours internes
                    internal_color_bgr = (
                        min(int(color[2]*255) + 40, 255),
                        min(int(color[1]*255) + 40, 255),
                        min(int(color[0]*255) + 40, 255)
                    )
                    
                    for internal_contour in internal_contours:
                        cv2.drawContours(result, [internal_contour], -1, internal_color_bgr, 2)
                
                # Ajouter l'étiquette
                # Trouver une bonne position pour l'étiquette (près du haut du contour)
                x_values = [p[0][0] for p in outer_contour]
                y_values = [p[0][1] for p in outer_contour]
                
                if x_values and y_values:
                    # Utiliser le centroïde pour la position de l'étiquette
                    label_x = int(sum(x_values) / len(x_values))
                    label_y = int(min(y_values) - 10)
                    
                    # Rester dans les limites
                    label_y = max(15, label_y)
                    
                    # Ajouter un fond semi-transparent pour l'étiquette
                    label_text = data['label']
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    cv2.rectangle(
                        result,
                        (label_x - text_width//2 - 5, label_y - text_height - 5),
                        (label_x + text_width//2 + 5, label_y + 5),
                        (255, 255, 255),
                        -1
                    )
                    
                    # Ajouter le texte de l'étiquette
                    cv2.putText(
                        result,
                        label_text,
                        (label_x - text_width//2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_bgr,
                        2
                    )
        
        return result

    def process_image(self, image_path: str, prompts: List[List[str]], 
                    threshold: float = 0.05, expand_ratio: float = 0.15,
                    use_crf: bool = False, detect_internal: bool = True,
                    output_dir: str = 'outputs') -> Dict:
        """
        Process image with complete segmentation pipeline
        
        Args:
            image_path: Path to input image
            prompts: List of text prompts for YOLO-World
            threshold: Detection confidence threshold
            expand_ratio: Box expansion ratio for SAM
            use_crf: Whether to use CRF refinement
            detect_internal: Whether to detect internal contours
            output_dir: Output directory for results
            
        Returns:
            Dictionary of results
        """
        Logger.info(f"Processing image: {image_path}")
        
        # Get basename for output paths
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Create output directory and SVG directory in advance
        os.makedirs(output_dir, exist_ok=True)
        svg_dir = os.path.join(output_dir, "svg")
        os.makedirs(svg_dir, exist_ok=True)
        
        # Set up SVG path here to avoid the reference-before-assignment error
        svg_path = os.path.join(svg_dir, f"{name_without_ext}.svg")
        
        # Load and preprocess image
        original_image = self.load_image(image_path)
        processed_image = self.preprocess_image(original_image)
        
        # Run object detection
        detections = self.detect_objects(processed_image, prompts, threshold)
        
        # Extract detection information
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']
        text_labels = detections['text_labels']
        class_names = detections['class_names']
        class_colors = detections['class_colors']
        class_counts = detections['class_counts']
        
        # Check if we found any objects
        if len(boxes) == 0:
            Logger.warning("No objects detected")
            return {
                'visualization_path': None,
                'svg_path': None,
                'metadata': {
                    'created_by': 'FETHl',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source_image': image_path
                }
            }
        
        # Map detection to its color based on class name
        colors = [class_colors[class_names[i]] for i in range(len(boxes))]
        
        # Visualization image
        vis_image = original_image.copy()
        
        # Process each detection with SAM for precise contours
        all_masks = []
        all_contours = []
        all_internal_contours = []
        all_labels = []
        all_class_names = []
        
        if self.sam_models is not None:
            Logger.info("Running SAM segmentation for each detection...")
            
            # Process each box
            for i, box in enumerate(boxes):
                Logger.info(f"Processing {text_labels[i]}...")
                
                # Get mask from SAM
                mask, score_sam = self.generate_mask_from_box(processed_image, box, expand_ratio)
                
                if mask is not None:
                    # CRITICAL: Explicitly create a binary mask with only 0 and 1 values
                    binary_mask = np.zeros_like(mask, dtype=np.uint8)
                    binary_mask[mask > 0] = 1
                    
                    # Apply CRF refinement if requested
                    if use_crf and HAS_CRF:
                        Logger.info(f"Applying CRF refinement...")
                        try:
                            binary_mask = self.crf_processor.refine_mask(processed_image, binary_mask)
                        except Exception as e:
                            Logger.error(f"CRF refinement failed: {e}")
                            # Continue without CRF if it fails
                    
                    # Clean the mask and extract a single clean contour
                    cleaned_mask, clean_contour = self.contour_processor.process_mask(binary_mask)
                    
                    # Detect internal contours if requested
                    internal_contours = []
                    if detect_internal:
                        try:
                            Logger.info(f"Detecting internal contours for {text_labels[i]}...")
                            internal_contours = self.detect_internal_contours(cleaned_mask, processed_image)
                            Logger.info(f"Found {len(internal_contours)} internal contours")
                        except Exception as e:
                            Logger.error(f"Error detecting internal contours: {e}")
                            traceback.print_exc()
                    
                    if clean_contour is not None:
                        # Add to results
                        all_masks.append(cleaned_mask)
                        all_contours.append(clean_contour)
                        all_internal_contours.append(internal_contours)
                        all_labels.append(text_labels[i])
                        all_class_names.append(class_names[i])
                    else:
                        Logger.warning(f"Could not extract clean contour for {text_labels[i]}")
                        
                        # Just draw the bounding box if no contour
                        x1, y1, x2, y2 = map(int, box)
                        color = colors[i]
                        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
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
                
                # Draw box with thick line
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 3)  # Thicker line
        
        # Prepare for contour merging if we have contours
        if all_contours:
            # Initialize contour merger if needed
            if not hasattr(self, 'contour_merger'):
                self.contour_merger = ContourMerger(
                    iou_threshold=0.3, 
                    internal_overlap_threshold=0.8,
                    debug=self.debug
                )
                
            # Prepare data for merging
            contours_data_for_merging = []
            for i, (contour, internal_contours, label, class_name) in enumerate(
                zip(all_contours, all_internal_contours, all_labels, all_class_names)):
                mask = all_masks[i] if i < len(all_masks) else None
                contours_data_for_merging.append({
                    'outer_contour': contour,
                    'internal_contours': internal_contours if detect_internal else [],
                    'label': label,
                    'color': class_colors[class_name],
                    'mask': mask,
                    'confidence': scores[i] if i < len(scores) else 0.5
                })
            
            # Merge overlapping contours
            merged_contours_data = self.contour_merger.merge_overlapping_contours(
                contours_data_for_merging
            )
            
            # Draw merged contours on visualization
            vis_image = self._draw_merged_contours(vis_image, merged_contours_data)
            
            # Add legend to the visualization
            vis_image = self._add_legend_to_image(vis_image, class_colors, class_counts)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{name_without_ext}_visualization.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            Logger.success(f"Visualization saved to: {vis_path}")
            
            # Export SVG with merged contours
            try:
                self.svg_exporter.export_to_svg(
                    merged_contours_data,
                    original_image.shape,
                    svg_path
                )
                Logger.success(f"SVG exported to: {svg_path}")
            except Exception as e:
                Logger.error(f"Error exporting SVG: {e}")
                traceback.print_exc()
                svg_path = None
        else:
            # No contours to merge, just save the visualization with boxes
            vis_path = os.path.join(output_dir, f"{name_without_ext}_visualization.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            Logger.success(f"Visualization saved to: {vis_path}")
            svg_path = None  # No SVG to export
        
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
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source_image': image_path
            }
        }

    def _add_legend_to_image(self, image: np.ndarray, 
                           class_colors: Dict[str, Tuple[float, float, float]],
                           class_counts: Dict[str, int]) -> np.ndarray:
        """
        Add a clean legend to the visualization image
        
        Args:
            image: Input image
            class_colors: Dictionary of class names to RGB colors
            class_counts: Dictionary of class names to counts
            
        Returns:
            Image with legend added
        """
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
        legend_height = padding * 2 + num_classes * line_height + 30  # Extra space for title
        
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
            (legend_x + padding, legend_y + padding + 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 0), 
            2
        )
        
        # Add class entries
        y = legend_y + padding + line_height + 20
        
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


def load_prompts(prompt_file: str) -> List[List[str]]:
    """
    Load prompts from a text file
    
    Args:
        prompt_file: Path to text file with one prompt per line
        
    Returns:
        List of prompts in format required by YOLO-World
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r') as f:
        # Each line is a separate prompt
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Format prompts for YOLO-World
    # Each prompt should be a list of strings
    formatted_prompts = [[p] for p in prompts]
    
    return formatted_prompts

def main():
    """Fonction principale pour exécuter BoxSubdivider"""
    # Analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="BoxSubdivider - Segmentation avancée avec YOLO-World et SAM")
    parser.add_argument("config", help="Fichier de configuration YOLO-World")
    parser.add_argument("checkpoint", help="Fichier de checkpoint YOLO-World")
    parser.add_argument("image_path", help="Chemin vers l'image d'entrée ou le répertoire d'images")
    parser.add_argument("prompt_file", help="Chemin vers le fichier texte de prompts")
    parser.add_argument("--sam-checkpoint", help="Chemin vers le fichier de checkpoint SAM", default=None)
    parser.add_argument("--output-dir", help="Répertoire de sortie", default="outputs")
    parser.add_argument("--threshold", help="Seuil de détection", type=float, default=0.035)  # Abaissé de 0.05 à 0.035
    parser.add_argument("--expand-ratio", help="Ratio d'expansion pour SAM", type=float, default=0.2)  # Augmenté de 0.15 à 0.2
    parser.add_argument("--use-crf", help="Appliquer un raffinement CRF", action="store_true")
    parser.add_argument("--detect-internal", help="Détecter les contours internes", action="store_true", default=True)
    parser.add_argument("--debug", help="Activer le mode debug", action="store_true")
    
    args = parser.parse_args()
    
    # En-tête
    print("\n" + "="*80)
    print(f"BoxSubdivider - Détection et Segmentation Avancée")
    print(f"Exécuté sur: {DEVICE}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Initialiser BoxSubdivider
    processor = BoxSubdivider(
        yolo_config=args.config,
        yolo_checkpoint=args.checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        debug=args.debug
    )
    
    # Charger les prompts
    prompts = load_prompts(args.prompt_file)
    Logger.info(f"Chargé {len(prompts)} prompts depuis {args.prompt_file}")
    
    # Traiter les images
    if os.path.isdir(args.image_path):
        # Traiter toutes les images du répertoire
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(args.image_path, f"*{ext.upper()}")))
        
        if not image_files:
            Logger.error(f"Aucun fichier image trouvé dans {args.image_path}")
            return
        
        Logger.info(f"Trouvé {len(image_files)} images à traiter")
        
        # Traiter chaque image
        for image_file in tqdm(image_files, desc="Traitement des images"):
            try:
                processor.process_image(
                    image_file, prompts,
                    threshold=args.threshold,
                    expand_ratio=args.expand_ratio,
                    use_crf=args.use_crf,
                    detect_internal=args.detect_internal,
                    output_dir=args.output_dir
                )
            except Exception as e:
                Logger.error(f"Erreur de traitement de {image_file}: {e}")
                traceback.print_exc()
    else:
        # Traiter une seule image
        if not os.path.exists(args.image_path):
            Logger.error(f"Fichier image non trouvé: {args.image_path}")
            return
        
        try:
            processor.process_image(
                args.image_path, prompts,
                threshold=args.threshold,
                expand_ratio=args.expand_ratio,
                use_crf=args.use_crf,
                detect_internal=args.detect_internal,
                output_dir=args.output_dir
            )
        except Exception as e:
            Logger.error(f"Erreur de traitement de {args.image_path}: {e}")
            traceback.print_exc()
    
    Logger.success("Traitement terminé!")
  

if __name__ == "__main__":
    main()