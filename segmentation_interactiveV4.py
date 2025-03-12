import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import sys
from tqdm import tqdm
from datetime import datetime
import re

# Constants
VERSION = "2.3"
CURRENT_USER = "FETHl"
CURRENT_DATE = "2025-03-12 09:09:45"

# SAM Model Configuration
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        """Affiner les contours d'un masque en utilisant DenseCRF"""
        if not self.crf_available:
            return self.apply_crf_alternative(image, mask)
            
        if crf_params is None:
            crf_params = {
                'gaussian_sxy': 3,         # Largeur du noyau spatial
                'gaussian_compat': 3,      # Compatibilité pour le potentiel gaussien
                'bilateral_sxy': 80,       # Largeur du noyau spatial pour bilateral
                'bilateral_srgb': 13,      # Largeur du noyau RGB
                'bilateral_compat': 10,    # Compatibilité pour le potentiel bilateral
                'iterations': 10           # Nombre d'itérations
            }
        
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
            
            h, w = image.shape[:2]
            image = np.ascontiguousarray(image)

            # Vérifier et convertir le masque si nécessaire
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
                
            # Protection supplémentaire contre les valeurs invalides
            mask[mask > 1] = 1
                
            # Initialiser CRF avec plus de robustesse
            if self.debug:
                print("Application de DenseCRF pour affiner les contours...")
                
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes (objet/fond)

            # Définir l'énergie unaire (probabilité d'appartenance à l'objet) avec gestion des exceptions
            try:
                unary = unary_from_labels(mask, 2, gt_prob=0.7)
                d.setUnaryEnergy(unary)
            except Exception as e:
                print(f"Erreur lors de la définition de l'énergie unaire: {e}")
                return mask  # Retourner le masque original en cas d'erreur

            # Ajouter une contrainte de proximité (gaussienne)
            try:
                d.addPairwiseGaussian(
                    sxy=(crf_params['gaussian_sxy'], crf_params['gaussian_sxy']),
                    compat=crf_params['gaussian_compat'],
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC
                )
            except Exception as e:
                print(f"Erreur lors de l'ajout du potentiel gaussien: {e}")
                # Continuer malgré l'erreur

            # Ajouter une contrainte de couleur (bilatérale)
            try:
                # Vérifier les dimensions de l'image
                if image.shape[2] == 3:  # Seulement pour les images RGB
                    d.addPairwiseBilateral(
                        sxy=(crf_params['bilateral_sxy'], crf_params['bilateral_sxy']),
                        srgb=(crf_params['bilateral_srgb'],) * 3,
                        rgbim=image.astype(np.uint8),  # Assurer que c'est uint8
                        compat=crf_params['bilateral_compat'],
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC
                    )
            except Exception as e:
                print(f"Erreur lors de l'ajout du potentiel bilatéral: {e}")
                # Continuer malgré l'erreur

            # Exécuter l'inférence CRF avec gestion d'erreurs
            try:
                Q = d.inference(crf_params['iterations'])
                
                # Convertir Q en tableau numpy de façon sécurisée
                Q_array = np.array(Q)
                
                # Vérifier la forme et ajuster si nécessaire
                if Q_array.size == h * w * 2:  # Si la taille correspond à ce qu'on attend
                    Q_array = Q_array.reshape((2, h, w))
                    refined_mask = np.argmax(Q_array, axis=0)
                else:
                    print(f"Forme de Q incorrecte: {Q_array.shape}, attendu: (2, {h}, {w})")
                    return mask
            except Exception as e:
                print(f"Erreur lors de l'inférence CRF: {e}")
                return mask

            if self.debug:
                print("Affinage des contours par CRF terminé")
            return refined_mask

        except Exception as e:
            print(f"Erreur générale lors de l'application de CRF: {e}")
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


class InteractiveSegmentationTool:
    """
    Outil interactif pour la sélection de points et la manipulation de contours
    """
    def __init__(self, sam_helper=None, crf_processor=None, debug=False):
        self.sam_helper = sam_helper
        self.crf_processor = crf_processor
        self.debug = debug  # Ajout du paramètre debug
        
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
        
        # Couleurs pour les contours (code couleur RGB)
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
        
        # Contours et leurs couleurs associées
        self.contour_colors = {}  # {contour_idx: color}
        self.polygon_editing_mode = False
        self.edited_polygon = None
        self.edited_polygon_idx = -1
        self.dragged_vertex_idx = -1
        self.vertex_drag_threshold = 10  # pixels
        
    def set_image(self, image):
        """
        Définir l'image à segmenter
        """
        self.image = image.copy()
        self.reset_session()
        
    def reset_session(self):
        """
        Réinitialiser la session interactive
        """
        self.points = []
        self.labels = []
        self.working_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.current_contours = []
        self.selected_contour_idx = -1
        
    def add_point(self, x, y, is_object=True):
        """
        Ajouter un point d'intérêt (foreground ou background)
        """
        self.points.append([x, y])
        self.labels.append(1 if is_object else 0)
        return len(self.points)
        
    def remove_last_point(self):
        """
        Supprimer le dernier point ajouté
        """
        if self.points:
            self.points.pop()
            self.labels.pop()
        return len(self.points)
        
    def generate_mask_from_points(self):
        """
        Générer un masque à partir des points sélectionnés
        """
        if not self.sam_helper or not self.sam_helper.is_initialized():
            print("SAM n'est pas initialisé. Impossible de générer un masque.")
            return None
            
        if not self.points:
            print("Aucun point sélectionné.")
            return None
            
        # Générer les masques avec SAM
        masks, scores, _ = self.sam_helper.generate_masks_from_points(
            self.image, self.points, self.labels
        )
        
        if masks is None or len(masks) == 0:
            print("Échec de la génération du masque.")
            return None
            
        # Prendre le masque avec le meilleur score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(np.uint8)
        
        # Appliquer CRF pour affiner les contours
        refined_mask = self.crf_processor.apply_crf(self.image, mask)
        self.working_mask = refined_mask.astype(np.uint8)
        
        # Mettre à jour les contours
        self._update_contours()
        
        return self.working_mask
    
    def _update_contours(self):
        """
        Mettre à jour la liste des contours à partir du masque actuel
        """
        # Trouver les contours
        contours, _ = cv2.findContours(
            self.working_mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        # Filtrer les contours minuscules
        self.current_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        self.selected_contour_idx = -1
        
        return len(self.current_contours)
    
    def select_contour_at_point(self, x, y):
        """
        Sélectionner un contour en cliquant près ou sur celui-ci
        """
        min_dist = float('inf')
        selected_idx = -1
        
        for i, contour in enumerate(self.current_contours):
            # Calculer la distance minimale du point au contour
            dist = cv2.pointPolygonTest(contour, (x, y), True)
            dist = abs(dist)  # Distance signée (négatif = extérieur)
            
            # Si le point est sur le contour ou à moins de 5 pixels
            if dist < min_dist and dist < 10:
                min_dist = dist
                selected_idx = i
                
        self.selected_contour_idx = selected_idx
        return selected_idx
    
    def remove_selected_contour(self):
        """
        Supprimer le contour sélectionné du masque
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Créer un masque pour ce contour spécifique
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Supprimer ce contour du masque global
        self.working_mask[contour_mask > 0] = 0
        
        # Mettre à jour les contours
        self._update_contours()
        
        return True
    
    def remove_contours_outside_roi(self, roi_x, roi_y, roi_width, roi_height):
        """
        Supprimer tous les contours en dehors d'une région d'intérêt
        """
        if not self.current_contours:
            return False
            
        # Créer un masque ROI
        roi_mask = np.zeros_like(self.working_mask)
        roi_mask[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = 1
        
        # Filtrer les contours
        kept_mask = np.zeros_like(self.working_mask)
        for contour in self.current_contours:
            # Calculer le centre du contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                
                # Si le centre est dans le ROI, conserver ce contour
                if roi_mask[center_y, center_x] > 0:
                    cv2.drawContours(kept_mask, [contour], 0, 1, -1)
        
        # Mettre à jour le masque de travail
        self.working_mask = kept_mask
        
        # Mettre à jour les contours
        self._update_contours()
        
        return True
    
    def split_contour(self, x1, y1, x2, y2, line_thickness=3):
        """
        Diviser un contour en traçant une ligne entre deux points
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Créer un masque pour le contour sélectionné
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Tracer une ligne pour diviser le contour
        split_mask = contour_mask.copy()
        cv2.line(split_mask, (x1, y1), (x2, y2), 0, thickness=line_thickness)
        
        # Mettre à jour le masque de travail
        self.working_mask = np.logical_and(self.working_mask, split_mask).astype(np.uint8)
        
        # Mettre à jour les contours
        self._update_contours()
        
        return True
    
    def smooth_selected_contour(self, epsilon_factor=0.002):
        """
        Lisser le contour sélectionné
        """
        if self.selected_contour_idx < 0 or not self.current_contours:
            return False
            
        # Récupérer le contour
        contour = self.current_contours[self.selected_contour_idx]
        
        # Lisser le contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Créer un masque pour ce contour
        contour_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(contour_mask, [self.current_contours[self.selected_contour_idx]], 0, 1, -1)
        
        # Créer un masque pour le contour lissé
        smooth_mask = np.zeros_like(self.working_mask)
        cv2.drawContours(smooth_mask, [smoothed_contour], 0, 1, -1)
        
        # Remplacer le contour original par le contour lissé dans le masque
        self.working_mask[contour_mask > 0] = 0  # Supprimer l'ancien contour
        self.working_mask[smooth_mask > 0] = 1   # Ajouter le nouveau contour
        
        # Mettre à jour les contours
        self._update_contours()
        
        return True
    
    def set_contour_color(self, contour_idx, color):
        """Change la couleur d'un contour spécifique"""
        if contour_idx >= 0 and contour_idx < len(self.current_contours):
            self.contour_colors[contour_idx] = color
            print(f"Couleur du contour {contour_idx} changée en {color}")
            return True
        return False
    
    def visualize_interactive_state(self, title="Sélection interactive", figsize=(12, 10)):
        """
        Visualiser l'état actuel de la sélection interactive
        """
        if self.image is None:
            return None
        
        plt.figure(figsize=figsize)
        
        # Image originale avec points et contours
        plt.subplot(221)
        plt.imshow(self.image)
        plt.title("Image avec points de sélection")
        
        # Afficher les points
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = 'g' if label == 1 else 'r'
            plt.plot(point[0], point[1], color+'o', markersize=8)
            plt.text(point[0]+5, point[1]+5, str(i), color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.7))
        
        # Masque actuel
        plt.subplot(222)
        plt.imshow(self.working_mask * 255, cmap='gray')
        plt.title("Masque de segmentation")
        
        # Contours sur image originale
        plt.subplot(223)
        overlay = self.image.copy()
        
        # Dessiner tous les contours
        for i, contour in enumerate(self.current_contours):
            color = (0, 255, 0)  # Vert pour les contours normaux
            thickness = 2
            
            # Contour sélectionné en rouge et plus épais
            if i == self.selected_contour_idx:
                color = (255, 0, 0)  # Rouge
                thickness = 3
                
            cv2.drawContours(overlay, [contour], 0, color, thickness)
            
            # Ajouter le numéro du contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(i), (cX, cY),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Contours détectés")
        
        # Vue combinée avec transparence
        plt.subplot(224)
        combined = self.image.copy()
        
        # Créer un masque coloré
        mask_colored = np.zeros_like(combined)
        mask_colored[self.working_mask > 0] = [0, 255, 0]  # Vert semi-transparent
        
        # Fusionner l'image et le masque
        alpha = 0.5
        combined = cv2.addWeighted(combined, 1.0, mask_colored, alpha, 0)
        
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title("Résultat de la segmentation")
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return plt.gcf()
    
    ###########################################################################################
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
            "'g': Generate mask from points\n"
            "'s': Smooth selected contour\n"
            "'r': Remove selected contour\n"
            "'c': Clear all points\n"
            "'q': Quit and save results"
        )
        # Add instructions text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes, 
                    fontsize=9, verticalalignment='top', bbox=props)
        
        # Setup event handlers
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.tight_layout()
        plt.show()
        
        return True
    

   ###############################################################################################     
    def _on_click(self, event):
        """Gestionnaire des clics de souris"""
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Mode d'édition de polygone
        if self.polygon_editing_mode:
            # Bouton gauche: déplacer un sommet
            if event.button == 1:
                # Vérifier double-clic pour ajouter un sommet
                if event.dblclick:
                    # Trouver l'arête la plus proche
                    edge1, edge2 = self.find_nearest_edge(x, y)
                    if edge1 >= 0:
                        # Ajouter un sommet sur cette arête
                        self.add_vertex_at_edge(edge1, edge2)
                        print(f"Sommet ajouté entre les sommets {edge1} et {edge2}")
                else:
                    # Trouver le sommet le plus proche
                    vertex_idx = self.find_nearest_vertex(x, y)
                    if vertex_idx >= 0:
                        # Commencer à faire glisser ce sommet
                        self.dragged_vertex_idx = vertex_idx
            
            # Bouton droit: supprimer un sommet
            elif event.button == 3:
                vertex_idx = self.find_nearest_vertex(x, y)
                if vertex_idx >= 0:
                    if self.delete_vertex(vertex_idx):
                        print(f"Sommet {vertex_idx} supprimé")
        else:
            # Comportement normal
            if event.button == 1:
                self.add_point(x, y, is_object=True)
                
            elif event.button == 3:
                self.add_point(x, y, is_object=False)
                
            elif event.button == 2:
                if self.working_mask is not None:
                    self.select_contour_at_point(x, y)
                    
        self._update_interactive_view()

    def _on_key(self, event):
        """Gestionnaire des événements clavier"""
        # 'p': Commencer/arrêter l'édition de polygone
        if event.key == 'p':
            if self.polygon_editing_mode:
                self.stop_polygon_editing(apply_changes=True)
            elif self.selected_contour_idx >= 0:
                self.start_polygon_editing(self.selected_contour_idx)
        
        # 'escape': Annuler l'édition de polygone
        elif event.key == 'escape':
            if self.polygon_editing_mode:
                self.stop_polygon_editing(apply_changes=False)
        
        # ... autres touches existantes ...
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
                
        # Touches numériques 1-9,0 pour changer la couleur
        elif event.key in '1234567890' and self.selected_contour_idx >= 0:
            color_idx = int(event.key) if event.key != '0' else 9
            if color_idx < len(self.default_contour_colors):
                self.set_contour_color(self.selected_contour_idx, self.default_contour_colors[color_idx])
                
        # 'q': Quit and save results
        elif event.key == 'q':
            plt.close(self.fig)
            return
            
        self._update_interactive_view()

    # Ajouter ce gestionnaire pour le mouvement de la souris
    def _on_mouse_move(self, event):
        """Gestionnaire du mouvement de la souris"""
        if not self.polygon_editing_mode or self.dragged_vertex_idx < 0:
            return
            
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Déplacer le sommet sélectionné
        if self.move_vertex(self.dragged_vertex_idx, x, y):
            self._update_interactive_view()

    # Ajouter ce gestionnaire pour le relâchement de la souris
    def _on_mouse_release(self, event):
        """Gestionnaire du relâchement de la souris"""
        if self.dragged_vertex_idx >= 0:
            self.dragged_vertex_idx = -1

    # Mettre à jour la méthode start_interactive_mode pour connecter ces gestionnaires
    def start_interactive_mode(self):
        """Start the fully interactive GUI mode"""
        if self.image is None:
            print("No image loaded")
            return False
            
        self.interactive_mode = True
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.image)
        self.ax.set_title("Mode de Segmentation Interactive")
        
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
        
        # Afficher les infos d'utilisateur et date en bas de l'interface
        info_text = f"Utilisateur: {CURRENT_USER} | Date: {CURRENT_DATE} | Version: {VERSION}"
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
    ################################################################################################



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
        
        # Mode d'édition de polygone
        if self.polygon_editing_mode:
            # Afficher uniquement le polygone en cours d'édition
            points = self.edited_polygon.reshape(-1, 2)
            
            # Afficher les arêtes
            self.ax.plot(points[:, 0], points[:, 1], 'o-', 
                        color='cyan', linewidth=2, markersize=6)
            
            # Fermer le polygone
            self.ax.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 
                    '-', color='cyan', linewidth=2)
                    
            # Mettre en évidence les sommets
            self.ax.scatter(points[:, 0], points[:, 1], color='yellow', 
                        s=100, marker='o', alpha=0.7)
                        
            # Numéroter les sommets
            for i, (x, y) in enumerate(points):
                self.ax.text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')
                
            # Afficher un message indiquant le mode d'édition
            self.ax.set_title(f"Mode d'édition du polygone {self.edited_polygon_idx}")
            
            # Instructions spécifiques au mode d'édition
            edit_instructions = "Mode édition: Drag=déplacer, Double-clic=ajouter, Clic droit=supprimer, 'p'/'Esc'=quitter"
            self.ax.text(0.5, 0.02, edit_instructions, transform=self.ax.transAxes, 
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(facecolor='yellow', alpha=0.5, pad=2))
        else:
            # Affichage normal des contours
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
        
        # Afficher les infos d'utilisateur et date en bas de l'interface
        info_text = f"Utilisateur: {CURRENT_USER} | Date: {CURRENT_DATE} | Version: {VERSION}"
        self.ax.text(0.02, 0.01, info_text, transform=self.ax.transAxes, 
                    fontsize=8, color='white', alpha=0.7,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
                    
        self.ax.set_title("Mode de Segmentation Interactive" if not self.polygon_editing_mode else f"Édition du polygone {self.edited_polygon_idx}")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()


    #######################################################################################


    def start_polygon_editing(self, contour_idx):
        """Démarre l'édition d'un polygone (contour)"""
        if contour_idx < 0 or contour_idx >= len(self.current_contours):
            print(f"Contour {contour_idx} invalide pour l'édition")
            return False
            
        self.polygon_editing_mode = True
        self.edited_polygon_idx = contour_idx
        self.edited_polygon = self.current_contours[contour_idx].copy()
        self.dragged_vertex_idx = -1
        
        print(f"Mode d'édition du polygone {contour_idx} activé")
        print("Cliquez sur un sommet pour le déplacer, double-cliquez pour l'ajouter/supprimer")
        return True
        
    def stop_polygon_editing(self, apply_changes=True):
        """Arrête l'édition du polygone"""
        if not self.polygon_editing_mode:
            return
            
        if apply_changes and self.edited_polygon_idx >= 0:
            # Appliquer les modifications au contour original
            self.current_contours[self.edited_polygon_idx] = self.edited_polygon.copy()
            # Mettre à jour le masque depuis les contours modifiés
            self._update_mask_from_contours()
            print(f"Modifications du polygone {self.edited_polygon_idx} appliquées")
        else:
            print("Édition annulée, aucune modification appliquée")
            
        # Réinitialiser l'état d'édition
        self.polygon_editing_mode = False
        self.edited_polygon = None
        self.edited_polygon_idx = -1
        self.dragged_vertex_idx = -1
        
    def find_nearest_vertex(self, x, y, max_dist=10):
        """Trouve le sommet le plus proche du point (x, y)"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1
            
        # Récupérer les points du polygone
        points = self.edited_polygon.reshape(-1, 2)
        
        # Calculer les distances au point (x, y)
        distances = np.sqrt(np.sum((points - np.array([x, y])) ** 2, axis=1))
        
        # Trouver l'index du point le plus proche
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # Vérifier si la distance est inférieure au seuil
        if min_dist <= max_dist:
            return min_idx
        else:
            return -1
            
    def find_nearest_edge(self, x, y, max_dist=10):
        """Trouve l'arête la plus proche du point (x, y)"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1, -1
            
        # Récupérer les points du polygone
        points = self.edited_polygon.reshape(-1, 2)
        n_points = len(points)
        
        min_dist = float('inf')
        nearest_edge = (-1, -1)  # (idx1, idx2)
        
        # Vérifier chaque segment
        for i in range(n_points):
            j = (i + 1) % n_points
            
            # Calculer la distance du point au segment
            p1 = points[i]
            p2 = points[j]
            
            # Distance point-ligne
            dist = point_to_line_distance(x, y, p1[0], p1[1], p2[0], p2[1])
            
            if dist < min_dist and dist <= max_dist:
                min_dist = dist
                nearest_edge = (i, j)
                
        return nearest_edge
        
    def add_vertex_at_edge(self, edge_idx1, edge_idx2):
        """Ajoute un sommet au milieu d'une arête"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return -1
            
        # Récupérer les points du polygone
        points = self.edited_polygon.reshape(-1, 2)
        
        # Calculer le point intermédiaire
        p1 = points[edge_idx1]
        p2 = points[edge_idx2]
        new_point = (p1 + p2) / 2
        
        # Créer un nouveau tableau de points avec le nouveau point inséré
        new_points = np.insert(points, edge_idx2, new_point, axis=0)
        
        # Mettre à jour le polygone édité
        self.edited_polygon = new_points.reshape(-1, 1, 2)
        
        return edge_idx2

    def delete_vertex(self, vertex_idx):
        """Supprime un sommet du polygone"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return False
            
        # Récupérer les points du polygone
        points = self.edited_polygon.reshape(-1, 2)
        
        # Vérifier qu'il reste suffisamment de points
        if len(points) <= 3:  # Un triangle est la forme minimale
            print("Impossible de supprimer ce sommet: le polygone doit avoir au moins 3 sommets")
            return False
            
        # Supprimer le sommet
        new_points = np.delete(points, vertex_idx, axis=0)
        
        # Mettre à jour le polygone édité
        self.edited_polygon = new_points.reshape(-1, 1, 2)
        
        return True

    def move_vertex(self, vertex_idx, new_x, new_y):
        """Déplace un sommet vers une nouvelle position"""
        if not self.polygon_editing_mode or self.edited_polygon is None:
            return False
            
        # Récupérer les points du polygone
        points = self.edited_polygon.reshape(-1, 2)
        
        # Vérifier l'index
        if vertex_idx < 0 or vertex_idx >= len(points):
            return False
            
        # Mettre à jour la position
        points[vertex_idx] = [new_x, new_y]
        
        # Mettre à jour le polygone édité
        self.edited_polygon = points.reshape(-1, 1, 2)
        
        return True

    # Fonction utilitaire pour calculer la distance d'un point à une ligne
    def point_to_line_distance(x, y, x1, y1, x2, y2):
        """Calcule la distance d'un point (x, y) à une ligne définie par (x1, y1) et (x2, y2)"""
        # Longueur de la ligne
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if line_length == 0:
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            
        # Projection du point sur la ligne
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length ** 2)
        
        # Limiter t entre 0 et 1 pour rester sur le segment
        t = max(0, min(1, t))
        
        # Point projeté
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # Distance au point projeté
        return np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

class EnhancedImageProcessor:
    """
    Enhanced image processor with SAM and CRF
    """
    def __init__(self, sam_checkpoint=None, debug=False):
        """Initialize the advanced image processor"""
        self.debug = debug
        
        # Initialize helpers
        self.crf_processor = CRFPostProcessor(debug=debug)
        self.sam_helper = SegmentAnythingHelper(checkpoint_path=sam_checkpoint, device=SAM_DEVICE)
        
        # Interactive tools
        self.interactive_tool = InteractiveSegmentationTool(
            sam_helper=self.sam_helper,
            crf_processor=self.crf_processor,
            )
        
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
            
            # Affiner les contours avec CRF
            refined_mask = self.crf_processor.apply_crf(image_color, mask // 255)
            refined_mask = (refined_mask * 255).astype(np.uint8)
            
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
                'mask': refined_mask,  # Using refined mask
                'shape': shape_info,
                'circularity': circularity,
                'quality': mask_data["stability_score"] if "stability_score" in mask_data else 0.8,
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
            
            # Apply CRF refinement to mask
            image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
            refined_mask = self.crf_processor.apply_crf(image_rgb, mask // 255)
            refined_mask = (refined_mask * 255).astype(np.uint8)
            
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
                'mask': refined_mask,  # Using refined mask
                'shape': shape_info,
                'circularity': circularity,
                'quality': 0.9 if circularity > 0.7 else 0.7,  # Estimate quality
            }
            
            segments_data.append(segment)
            
        return segments_data
    
    def segment_image(self, image_color, image_gray, min_area=100):
        """Segment the image using SAM if available, fallback to traditional methods"""
        if self.sam_available:
            # Try SAM segmentation first
            segments_data = self.segment_with_sam(image_color)
            
            if segments_data and len(segments_data) > 0:
                print(f"SAM segmentation successful, found {len(segments_data)} objects.")
                return segments_data, "sam"
                
        # Fallback to traditional methods
        print("Using traditional segmentation methods.")
        segments_data = self.segment_with_traditional(image_gray, min_area)
        print(f"Traditional segmentation found {len(segments_data)} objects.")
        return segments_data, "traditional"
        
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


class InteractiveSegmentationPipeline:
    """
    Pipeline de segmentation interactive avec SAM et CRF
    """
    def __init__(self, sam_checkpoint=None, debug=False):
        self.debug = debug
        self.sam_checkpoint = sam_checkpoint
        self.image_processor = EnhancedImageProcessor(
            sam_checkpoint=sam_checkpoint, 
            debug=debug
        )
        
    def process_image(self, input_image, output_dir):
        """Pipeline: Chargement -> Segmentation interactive -> Affichage"""
        print("\n=== DÉMARRAGE DU PIPELINE ===\n")
        
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Image d'entrée non trouvée: {input_image}")
            
        print(f"Traitement de l'image: {input_image}")
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Définir les chemins de sortie
        basename = os.path.splitext(os.path.basename(input_image))[0]
        output_viz = os.path.join(output_dir, f"{basename}_visualisation.png")
        
        # 1. Chargement de l'image
        print("\n1. Chargement de l'image...")
        try:
            image_color, image_gray = self.image_processor.load_image(input_image)
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
            
        # 2. Initialisation de l'outil interactif
        print("\n2. Initialisation de l'outil interactif...")
        try:
            self.image_processor.interactive_tool.set_image(image_color)
        except Exception as e:
            print(f"Erreur lors de l'initialisation de l'outil interactif: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
            
        # 3. Démarrage de la segmentation interactive
        print("\n3. Prêt pour la segmentation interactive")
        print("   Utilisez la fonction 'add_point(x, y, is_object)' pour ajouter des points d'intérêt")
        print("   Utilisez 'generate_mask_from_points()' pour générer un masque")
        print("   Utilisez 'select_contour_at_point(x, y)' pour sélectionner un contour")
        print("   Utilisez 'remove_selected_contour()' pour supprimer un contour non souhaité")
        
        # Exécution réussie
        print("\n=== INITIALISATION TERMINÉE ===\n")
        return True
        

def demonstration_interactive():
    """Démonstration du mode interactif (pour développement)"""
    # Créer le pipeline
    pipeline = InteractiveSegmentationPipeline(
        sam_checkpoint=SAM_CHECKPOINT,
        debug=True
    )
    
    # Chemins de test
    image_path = "test_image.jpg"  # A remplacer par un chemin réel
    output_dir = "./output"
    
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"Image de test non trouvée: {image_path}")
        print("Veuillez fournir un chemin d'image valide")
        return
        
    # Initialiser le pipeline
    pipeline.process_image(image_path, output_dir)
    
    # Référence à l'outil interactif pour plus de commodité
    tool = pipeline.image_processor.interactive_tool
    
    # Démonstration d'utilisation
    print("\n=== DÉMONSTRATION D'UTILISATION ===\n")
    
    # 1. Ajouter des points d'intérêt
    print("1. Ajout de points d'intérêt...")
    h, w = tool.image.shape[:2]
    
    # Points au centre et autres positions stratégiques
    tool.add_point(w//2, h//2, is_object=True)  # Centre - objet
    tool.add_point(w//4, h//4, is_object=True)  # Coin supérieur gauche - objet
    tool.add_point(3*w//4, h//4, is_object=False)  # Coin supérieur droit - fond
    tool.add_point(w//4, 3*h//4, is_object=False)  # Coin inférieur gauche - fond
    
    # 2. Générer un masque
    print("2. Génération du masque...")
    mask = tool.generate_mask_from_points()
    
    if mask is not None:
        print(f"  - Masque généré avec succès. {len(tool.current_contours)} contours détectés.")
    else:
        print("  - Échec de la génération du masque.")
        
    # 3. Visualiser l'état
    print("3. Visualisation de l'état...")
    fig = tool.visualize_interactive_state("Démonstration - Résultats initiaux")
    plt.savefig(os.path.join(output_dir, "demo_initial.png"))
    
    # 4. Sélectionner et manipuler des contours
    if tool.current_contours:
        print("4. Manipulation des contours...")
        
        # Sélectionner le premier contour
        first_contour = tool.current_contours[0]
        M = cv2.moments(first_contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            print(f"  - Sélection du contour au point ({cX}, {cY})...")
            selected_idx = tool.select_contour_at_point(cX, cY)
            
            if selected_idx >= 0:
                print(f"  - Contour {selected_idx} sélectionné.")
                
                # Lisser le contour sélectionné
                print("  - Lissage du contour...")
                tool.smooth_selected_contour(epsilon_factor=0.005)
                
                # Visualiser après lissage
                fig = tool.visualize_interactive_state("Démonstration - Après lissage")
                plt.savefig(os.path.join(output_dir, "demo_smoothed.png"))
                
                # Diviser le contour (si c'est un grand contour)
                area = cv2.contourArea(tool.current_contours[tool.selected_contour_idx])
                if area > 10000:
                    print("  - Division du contour...")
                    # Trouver des points pour diviser
                    x, y, w, h = cv2.boundingRect(tool.current_contours[tool.selected_contour_idx])
                    tool.split_contour(x, y+h//2, x+w, y+h//2)
                    
                    # Visualiser après division
                    fig = tool.visualize_interactive_state("Démonstration - Après division")
                    plt.savefig(os.path.join(output_dir, "demo_split.png"))
                
                # Supprimer le contour sélectionné
                print("  - Suppression du contour...")
                tool.remove_selected_contour()
                
                # Visualiser après suppression
                fig = tool.visualize_interactive_state("Démonstration - Après suppression")
                plt.savefig(os.path.join(output_dir, "demo_removed.png"))
    
    print("\n=== DÉMONSTRATION TERMINÉE ===\n")
    

def main():
    """Main function for command line execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Segmentation interactive avancée avec CRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Chemin vers l'image à traiter", nargs="?")
    parser.add_argument("-o", "--output", help="Dossier de sortie pour les fichiers générés", default="./output")
    parser.add_argument("--viz", help="Générer une visualisation", action="store_true", default=True)
    
    # Advanced options
    parser.add_argument("--min-area", type=int, default=200, 
                        help="Surface minimale (en pixels²) pour les objets à détecter")
    parser.add_argument("--sam-checkpoint", type=str, default=SAM_CHECKPOINT, 
                        help="Chemin vers les poids du modèle SAM")
    
    # Display options
    parser.add_argument("--show", action="store_true", 
                        help="Afficher les visualisations")
    parser.add_argument("--debug", action="store_true", 
                        help="Activer le mode debug")
    parser.add_argument("--demo", action="store_true",
                        help="Exécuter la démonstration interactive")
    parser.add_argument("--interactive", action="store_true",
                        help="Démarrer le mode interactif GUI avec souris et clavier")
    
    args = parser.parse_args()
    
    # Afficher la bannière de démarrage
    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║  Segmentation Interactive Avancée v{VERSION:<17} ║")
    print(f"║  Développée par: {CURRENT_USER:<41} ║")
    print(f"║  Date: {CURRENT_DATE:<46} ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    
    # Mode démonstration
    if args.demo:
        demonstration_interactive()
        return 0
        
    # Vérifier si un chemin d'image est fourni
    if not args.image_path:
        print("Erreur: Aucun chemin d'image fourni. Utilisez --demo pour la démonstration.")
        return 1
        
    # Initialiser le pipeline
    pipeline = InteractiveSegmentationPipeline(
        sam_checkpoint=args.sam_checkpoint,
        debug=args.debug
    )
    
    # Exécuter le pipeline
    try:
        success = pipeline.process_image(args.image_path, args.output)
        
        if success:
            print("\nInitialisation réussie. Vous pouvez maintenant utiliser les fonctions interactives.")
            
            # Référence à l'outil interactif pour plus de facilité
            tool = pipeline.image_processor.interactive_tool
            
            # Mode GUI interactif complet (souris et clavier)
            if args.interactive:
                print("\nDémarrage du mode interactif GUI...")
                tool.start_interactive_mode()
                return 0
            
            # Afficher un message d'aide
            print("\nUtilisation interactive:")
            print("  tool.add_point(x, y, is_object=True)  # Ajouter un point d'intérêt")
            print("  tool.remove_last_point()              # Supprimer le dernier point")
            print("  tool.generate_mask_from_points()      # Générer un masque à partir des points")
            print("  tool.select_contour_at_point(x, y)    # Sélectionner un contour")
            print("  tool.remove_selected_contour()        # Supprimer le contour sélectionné")
            print("  tool.split_contour(x1, y1, x2, y2)    # Diviser un contour")
            print("  tool.smooth_selected_contour()        # Lisser le contour sélectionné")
            print("  fig = tool.visualize_interactive_state() # Visualiser l'état actuel")
            print("  plt.show()                            # Afficher la visualisation")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nErreur fatale lors du traitement: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())