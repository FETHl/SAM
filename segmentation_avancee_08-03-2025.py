#!/usr/bin/env python3
"""
Pipeline complet de segmentation d'image haute précision:
- Super-résolution avec Real-ESRGAN
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

# Constants
CURRENT_USER = "FETHl"
CURRENT_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
VERSION = "4.0.0"

# SAM Model Configuration
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SuperResolutionHelper:
    """
    Classe pour améliorer la résolution des images avec Real-ESRGAN
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.realesrgan_available = self._check_realesrgan()
        
    def _check_realesrgan(self):
        """Vérifier si Real-ESRGAN est installé"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            return True
        except ImportError:
            print("AVERTISSEMENT: Real-ESRGAN n'est pas installé. Utilisation de l'alternative simple.")
            print("Pour installer: git clone https://github.com/xinntao/Real-ESRGAN && cd Real-ESRGAN && pip install -e . && cd ..")
            return False
    
    def enhance_resolution(self, image_path, output_path, scale=4):
        """Améliore la résolution de l'image avec Real-ESRGAN ou une méthode alternative"""
        if not self.realesrgan_available:
            return self.enhance_resolution_simple(image_path, output_path, scale)
        
        try:
            print(f"Chargement de l'image: {image_path}")
            
            # Configurer le device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Utilisation de: {device}")
            
            # Chercher les poids du modèle
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model_path = None
            possible_paths = [
                'weights/RealESRGAN_x4plus.pth',
                'Real-ESRGAN/weights/RealESRGAN_x4plus.pth',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/RealESRGAN_x4plus.pth')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Utilisation du modèle: {model_path}")
                    break
                    
            if model_path is None:
                print("ERREUR: Poids du modèle Real-ESRGAN non trouvés.")
                print("Téléchargez-les depuis: https://github.com/xinntao/Real-ESRGAN/releases")
                return self.enhance_resolution_simple(image_path, output_path, scale)
            
            # Créer le modèle RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            
            # Initialiser l'upsampler
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=512,  # Utiliser tiling pour les grandes images
                tile_pad=10,
                pre_pad=0,
                half=True if torch.cuda.is_available() else False,  # Utiliser FP16 si CUDA
                device=device
            )
            
            print("Amélioration de la résolution en cours...")
            start_time = time.time()
            
            # Lire et améliorer l'image
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Impossible de lire l'image: {image_path}")
                
            h, w = img.shape[:2]
            print(f"Dimensions de l'image: {w}x{h}")
            
            # Limiter la taille pour éviter les problèmes de mémoire
            max_size = 1500  # taille max d'entrée
            if max(h, w) > max_size:
                scale_factor = max_size / max(h, w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Image redimensionnée à {new_w}x{new_h} pour éviter les problèmes de mémoire")
                
            # Exécuter Real-ESRGAN
            output, _ = upsampler.enhance(img, outscale=scale)
            
            # Sauvegarder l'image améliorée
            cv2.imwrite(output_path, output)
            
            print(f"Amélioration terminée en {time.time() - start_time:.1f} secondes")
            print(f"Image sauvegardée: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Erreur lors de l'amélioration de la résolution: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return self.enhance_resolution_simple(image_path, output_path, scale)

    def enhance_resolution_simple(self, image_path, output_path, scale=2):
        """Méthode simple d'amélioration de la résolution avec OpenCV"""
        print("Utilisation de la méthode simple d'amélioration (OpenCV)...")
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de lire l'image: {image_path}")
                
            h, w = img.shape[:2]
            print(f"Dimensions originales: {w}x{h}")
            
            # Redimensionnement avec Lanczos pour une meilleure qualité
            img_upscaled = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)
            
            # Amélioration des détails avec unsharp mask
            kernel_size = (0, 0)  # Taille automatique
            sigma = 1.0 + (scale - 1) * 0.5  # Ajuster sigma selon le scale
            blurred = cv2.GaussianBlur(img_upscaled, kernel_size, sigma)
            img_upscaled = cv2.addWeighted(img_upscaled, 1.5, blurred, -0.5, 0)
            
            cv2.imwrite(output_path, img_upscaled)
            print(f"Image améliorée simple sauvegardée: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Erreur lors de l'amélioration simple: {e}")
            return image_path


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

            # Convertir le masque en étiquettes
            mask_labels = mask.astype(np.uint32)

            # Initialiser CRF
            if self.debug:
                print("Application de DenseCRF pour affiner les contours...")
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes (objet/fond)

            # Définir l'énergie unaire (probabilité d'appartenance à l'objet)
            unary = unary_from_labels(mask_labels, 2, gt_prob=0.7)
            d.setUnaryEnergy(unary)

            # Ajouter une contrainte de proximité (gaussienne)
            d.addPairwiseGaussian(
                sxy=(crf_params['gaussian_sxy'], crf_params['gaussian_sxy']),
                compat=crf_params['gaussian_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )

            # Ajouter une contrainte de couleur (bilatérale)
            d.addPairwiseBilateral(
                sxy=(crf_params['bilateral_sxy'], crf_params['bilateral_sxy']),
                srgb=(crf_params['bilateral_srgb'],) * 3,
                rgbim=image,
                compat=crf_params['bilateral_compat'],
                kernel=dcrf.DIAG_KERNEL,
                normalization=dcrf.NORMALIZE_SYMMETRIC
            )

            # Exécuter l'inférence CRF
            Q = d.inference(crf_params['iterations'])
            refined_mask = np.argmax(Q, axis=0).reshape((h, w))

            if self.debug:
                print("Affinage des contours par CRF terminé")
            return refined_mask

        except Exception as e:
            print(f"Erreur lors de l'application de CRF: {e}")
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


class EnhancedImageProcessor:
    """
    Enhanced image processor combining Super-Resolution, SAM and CRF
    """
    def __init__(self, sam_checkpoint=None, debug=False, line_thickness=0.35):
        """Initialize the advanced image processor"""
        self.debug = debug
        
        # Initialize helpers
        self.super_resolution = SuperResolutionHelper(debug=debug)
        self.crf_processor = CRFPostProcessor(debug=debug)
        self.sam_helper = SegmentAnythingHelper(checkpoint_path=sam_checkpoint, device=SAM_DEVICE)
        self.dxf_exporter = EnhancedDXFExporter(line_thickness=line_thickness)
        
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

    def enhance_image_resolution(self, image_path, output_path, scale=2):
        """Améliorer la résolution de l'image"""
        print("\nAmélioration de la résolution avec Real-ESRGAN...")
        return self.super_resolution.enhance_resolution(image_path, output_path, scale)

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
            
            # Raffinement CRF du masque
            refined_mask = self.crf_processor.apply_crf(image_color, mask // 255)
            refined_mask = (refined_mask * 255).astype(np.uint8)
            
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
                'mask': refined_mask,  # Using refined mask
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
            
            # Apply CRF refinement to mask
            image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
            refined_mask = self.crf_processor.apply_crf(image_rgb, mask // 255)
            refined_mask = (refined_mask * 255).astype(np.uint8)
            
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
                'mask': refined_mask,  # Using refined mask
                'shape': shape_info,
                'circularity': circularity,
                'quality': 0.9 if circularity > 0.7 else 0.7,  # Estimate quality
                'geometric_elements': geometric_elements
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
            # CORRECTION: Replace np.int0 with np.int32
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
        
        # Use our enhanced DXF exporter with line thickness
        self.dxf_exporter.line_thickness = line_thickness
        doc, msp, layer_defs = self.dxf_exporter.create_dxf_document()
        
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
        
        if self.debug:
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
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Add circle entity
            msp.add_circle(
                (center[0], center[1], 0),
                radius,
                dxfattribs=dxf_attribs
            )
            
            # Add center point for reference
            msp.add_point(
                (center[0], center[1], 0),
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
                major_axis * np.cos(angle_rad),
                major_axis * np.sin(angle_rad),
                0
            )
            
            # Create ellipse with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Add ellipse entity
            msp.add_ellipse(
                (center[0], center[1], 0),
                major_axis_vector,
                ratio,
                0, 2*np.pi,
                dxfattribs=dxf_attribs
            )
            
        elif element_type == 'rectangle':
            layer_name = "RECTANGLES"
            points = element['points']
            
            # Extract 2D points
            points_2d = [(p[0], p[1]) for p in points]
            
            # Create rectangle with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Add polyline entity
            msp.add_lwpolyline(
                points_2d,
                close=True,
                dxfattribs=dxf_attribs
            )
            
        elif element_type == 'polygone':
            # Determine if it's a triangle or other polygon
            points = element['points']
            if len(points) == 3:
                layer_name = "TRIANGLES"
            else:
                layer_name = "POLYGONES"
                
            # Extract 2D points
            points_2d = [(p[0], p[1]) for p in points]
            
            # Create polygon with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Add polyline entity
            msp.add_lwpolyline(
                points_2d,
                close=True,
                dxfattribs=dxf_attribs
            )
            
        elif element_type == 'contour':
            layer_name = "CONTOURS"
            points = element['points']
            
            # Create contour with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Extract 2D points
            points_2d = [(p[0], p[1]) for p in points]
            
            # Add polyline entity
            msp.add_lwpolyline(
                points_2d,
                close=True,
                dxfattribs=dxf_attribs
            )
            
        elif element_type == 'lignes':
            layer_name = "LIGNES"
            segments = element['segments']
            
            # Create lines with enhanced thickness
            dxf_attribs = {
                "layer": layer_name,
                "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                    self.dxf_exporter.thickness_map[layer_name])
            }
            
            # Add individual line segments
            for segment in segments:
                start, end = segment
                msp.add_line(
                    (start[0], start[1], 0),
                    (end[0], end[1], 0),
                    dxfattribs=dxf_attribs
                )
                
        else:  # Unknown type - use complex shape layer
            layer_name = "FORMES_COMPLEXES"
            
            # Try to extract points or segments
            if 'points' in element:
                points = element['points']
                points_2d = [(p[0], p[1]) for p in points]
                
                # Create complex shape with enhanced thickness
                dxf_attribs = {
                    "layer": layer_name,
                    "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                        self.dxf_exporter.thickness_map[layer_name])
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
                    "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                        self.dxf_exporter.thickness_map[layer_name])
                }
                
                # Add line segments
                for segment in segments:
                    start, end = segment
                    msp.add_line(
                        (start[0], start[1], 0),
                        (end[0], end[1], 0),
                        dxfattribs=dxf_attribs
                    )
        
        # Add identifier text if segment_id is provided
        if segment_id is not None:
            # Find position for label - use center of element
            if element_type == 'cercle' or element_type == 'ellipse':
                center_pos = (element['center'][0], element['center'][1], 0)
            elif 'points' in element:
                points = element['points']
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                center_pos = (center_x, center_y, 0)
            elif 'segments' in element and element['segments']:
                start = element['segments'][0][0]
                center_pos = (start[0], start[1], 0)
            else:
                center_pos = (0, 0, 0)
            
            # Create label with segment ID
            msp.add_text(
                str(segment_id),
                height=5.0,
                dxfattribs={
                    "layer": "ANNOTATIONS",
                    "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                        self.dxf_exporter.thickness_map["ANNOTATIONS"])
                },
                insert=center_pos
            )
            
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
            "lineweight": self.dxf_exporter.get_closest_std_lineweight(
                self.dxf_exporter.thickness_map["METADATA"])
        }
        
        # Main title
        msp.add_text(
            "Plan de segmentation avancée avec SAM",
            height=12.0,
            dxfattribs=dxf_attribs,
            insert=(0, -20, 0)
        )
        
        # Creator and date
        msp.add_text(
            f"Créé par: {CURRENT_USER} • Date: {CURRENT_DATE}",
            height=8.0,
            dxfattribs=dxf_attribs,
            insert=(0, -35, 0)
        )
        
        # Version information
        msp.add_text(
            f"Enhanced SAM Segmentation v{VERSION}",
            height=6.0,
            dxfattribs=dxf_attribs,
            insert=(0, -50, 0)
        )
        

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


class SuperResolutionAndSegmentationPipeline:
    """
    Pipeline complet combinant super-résolution et segmentation avancée
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
        
    def process_image(self, input_image, output_dir, scale=2):
        """Pipeline complet: Super-Résolution + SAM + CRF + Export DXF"""
        print("\n=== DÉMARRAGE DU PIPELINE ===\n")
        
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Image d'entrée non trouvée: {input_image}")
            
        print(f"Traitement de l'image: {input_image}")
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Définir les chemins de sortie
        basename = os.path.splitext(os.path.basename(input_image))[0]
        high_res_image = os.path.join(output_dir, f"{basename}_haute_resolution.png")
        output_dxf = os.path.join(output_dir, f"{basename}_segmentation.dxf")
        output_viz = os.path.join(output_dir, f"{basename}_visualisation.png")
        
        # 1. Amélioration de la résolution
        print("\n1. Amélioration de la résolution avec Real-ESRGAN...")
        try:
            high_res_path = self.image_processor.enhance_image_resolution(input_image, high_res_image, scale=scale)
        except Exception as e:
            print(f"Erreur lors de l'amélioration: {e}")
            high_res_path = input_image
            
        # 2. Chargement de l'image haute résolution
        print("\n2. Chargement de l'image améliorée...")
        image_color, image_gray = self.image_processor.load_image(high_res_path)
        
        # 3. Segmentation avancée
        print("\n3. Segmentation avancée avec SAM et CRF...")
        try:
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
        print("\n4. Génération des visualisations...")
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
        print("\n5. Export DXF avec lignes épaisses...")
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
        print(f"  - Image originale: {input_image}")
        print(f"  - Image haute résolution: {high_res_image}")
        print(f"  - Fichier DXF: {output_dxf}")
        print(f"  - Visualisation: {output_viz}")
        print(f"  - Objets détectés: {len(segments_data)}")
        
        return True


def main():
    """Main function for command line execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Super-résolution et segmentation avancée avec export DXF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Chemin vers l'image à traiter")
    parser.add_argument("-o", "--output", help="Dossier de sortie pour les fichiers générés", default="./exports")
    parser.add_argument("--dxf", help="Générer un fichier DXF", action="store_true", default=True)
    parser.add_argument("--viz", help="Générer une visualisation", action="store_true", default=True)
    parser.add_argument("--no-upscale", help="Ne pas effectuer la super-résolution", action="store_true")
    
    # Advanced options
    parser.add_argument("--scale", type=int, default=2, 
                        help="Facteur d'échelle pour la super-résolution (2 ou 4)")
    parser.add_argument("--min-area", type=int, default=200, 
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
    
    # Afficher la bannière de démarrage
    print(f"╔════════════════════════════════════════════════════════════════╗")
    print(f"║  Super-résolution et Segmentation Avancée v{VERSION:<17} ║")
    print(f"║  Démarré par: {CURRENT_USER:<41} ║")
    print(f"║  Date: {CURRENT_DATE:<46} ║")
    print(f"╚════════════════════════════════════════════════════════════════╝")
    
    # Initialiser le pipeline
    pipeline = SuperResolutionAndSegmentationPipeline(
        sam_checkpoint=args.sam_checkpoint,
        debug=args.debug,
        line_thickness=args.line_thickness
    )
    
    # Exécuter le pipeline
    try:
        scale = 1 if args.no_upscale else args.scale
        success = pipeline.process_image(args.image_path, args.output, scale=scale)
        
        if success and args.show:
            # Montrer la visualisation
            viz_path = os.path.join(args.output, 
                                  f"{os.path.splitext(os.path.basename(args.image_path))[0]}_visualisation.png")
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


