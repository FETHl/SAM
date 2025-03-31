# YOLO-World + SAM Clean Segmentation

## Description Détaillée

Un système avancé de segmentation d'image qui combine la puissance de YOLO-World pour la détection d'objets avec SAM (Segment Anything Model) pour une segmentation précise des contours externes et internes. Ce projet permet d'obtenir des masques de segmentation de haute qualité avec détection automatique des caractéristiques internes des objets.

**Auteur:** FETHl  
**Date de la dernière mise à jour:** 2025-03-31

![Exemple de Segmentation](assets/example_segmentation.jpg)

## Caractéristiques Principales

- **Détection d'objets par YOLO-World** : Détection précise d'objets basée sur des prompts textuels
- **Segmentation en deux étapes** :
  - **SamPredictor** pour les contours externes précis des objets détectés
  - **SamAutomaticMaskGenerator** pour les détails internes des objets
- **Post-traitement CRF** : Raffinement des masques avec Conditional Random Fields
- **Détection de contours internes** : Algorithmes multiples pour la détection des caractéristiques internes
- **Exportation SVG** : Génération de représentations vectorielles propres avec contours internes et externes
- **Visualisation avec légende** : Affichage des résultats avec code couleur par classe
- **Pipeline modulaire** : Architecture flexible permettant la personnalisation de chaque étape

## Prérequis

### Matériel

- GPU CUDA recommandé pour les performances optimales
- Minimum 8 Go de RAM (16 Go recommandés)
- Espace disque : 10 Go pour les modèles et dépendances

### Logiciel

- Python 3.8+
- PyTorch 1.10+
- CUDA Toolkit 11.3+ (pour l'accélération GPU)
- OpenCV 4.5+

## Installation

### Configuration de l'environnement

```bash
# Créer un environnement virtuel
conda create -n yolo_sam python=3.10
conda activate yolo_sam

# Installer les dépendances principales
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy tqdm matplotlib

# Installer MMCV et MMDetection
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet

# Installer MMYOLO
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -v -e .
cd ..

# Installer Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git

# Installer pydensecrf (optionnel, pour le raffinement CRF)
pip install pydensecrf


Téléchargement des modèles préentraînés
bash
# Créer le répertoire pour les poids
mkdir -p weights checkpoints

# Télécharger le modèle YOLO-World
wget -P weights https://download.openmmlab.com/mmyolo/v0/yolo_world/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth

# Télécharger le modèle SAM
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
Structure du Projet
Code
yolo_world_clean_segmentation/
├── configs/                    # Fichiers de configuration pour YOLO-World
├── weights/                    # Poids préentraînés de YOLO-World
├── checkpoints/                # Modèles SAM préentraînés
├── demo/                       # Images et exemples de démonstration
│   └── sample_images/          # Images d'exemple
├── outputs/                    # Résultats de segmentation générés
│   └── svg/                    # Fichiers SVG exportés
├── debug_outputs/              # Sorties de débogage (si activé)
├── yolo_world_clean_segmentation.py  # Script principal
├── prompts1.txt                # Exemples de prompts pour la détection
└── README.md                   # Documentation du projet
Guide d'Utilisation
Préparation des prompts
Créez un fichier texte avec une classe par ligne. Par exemple, prompts1.txt :

Code
person
helmet
glove
glasses
chair
table
car
truck
bike
dog
Commande de base
bash
python yolo_world_clean_segmentation.py [config_file] [checkpoint_file] [image_path] [prompt_file] --sam-checkpoint [sam_checkpoint] [options]
Exemple d'utilisation
bash
python yolo_world_clean_segmentation.py configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth demo/sample_images/Seq1.png prompts1.txt --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth --debug
Options disponibles
Option	Description	Valeur par défaut
--output-dir	Répertoire de sortie	outputs
--threshold	Seuil de détection pour YOLO-World	0.05
--expand-ratio	Ratio d'expansion pour les boîtes englobantes	0.15
--use-crf	Appliquer le raffinement CRF	False
--detect-internal	Activer la détection des contours internes	True
--min-internal-area	Surface minimale des contours internes	50
--debug	Activer le mode débogage avec sorties visuelles	False
--debug-internal	Débogage spécifique pour les contours internes	False
--smooth-factor	Facteur de lissage pour les contours internes	0.001
--no-artificial-contours	Désactiver les contours artificiels	False
Architecture Technique
Pipeline de Traitement
Chargement et prétraitement d'image

Conversion des espaces de couleur
Amélioration du contraste via CLAHE
Débruitage préservant les contours
Détection avec YOLO-World

Détection basée sur des prompts textuels
Filtrage des détections par seuil de confiance
Segmentation extérieure avec SamPredictor

Utilisation des boîtes englobantes de YOLO-World
Expansion des boîtes pour meilleur contexte
Génération de masques précis pour chaque objet
Détection des contours internes

Approche double :
SamAutomaticMaskGenerator pour les détails internes
Méthodes traditionnelles de CV en fallback (multi-méthode)
Filtrage des contours par taille, forme et naturalité
Lissage des contours pour une apparence naturelle
Post-traitement

Raffinement optionnel CRF pour des contours précis
Nettoyage et simplification des contours
Visualisation et Exportation

Génération d'overlay de masques avec transparence
Dessin des contours externes et internes
Exportation au format SVG avec métadonnées
Légende avec code couleur par classe d'objet
Classes Principales
CleanSegmentationProcessor
Classe principale qui coordonne le pipeline de traitement complet.

InternalContourDetector
Spécialisée dans la détection des caractéristiques internes des objets.

CRFPostProcessor
Applique le raffinement CRF aux masques pour des contours plus précis.

ContourCleaner
Nettoie et simplifie les contours pour une meilleure visualisation.

CleanSVGExporter
Exporte les contours en SVG propre avec représentation vectorielle.

Fonctionnement du Mode Dual SAM
Le système utilise une approche hybride qui combine deux modes de SAM :

SamPredictor : Utilisé pour la segmentation principale des objets, en prenant les boîtes englobantes de YOLO-World. Cette approche est précise et contrôlable pour l'extraction des contours externes.

SamAutomaticMaskGenerator : Appliqué uniquement à l'intérieur des objets détectés pour trouver les caractéristiques internes comme les plis de vêtements, les boutons, ou les parties distinctes d'objets. Cette approche est optimisée pour la détection automatique des détails.

Cette combinaison offre un excellent équilibre entre précision et temps de traitement, en concentrant l'analyse automatique des masques uniquement sur les régions pertinentes.

Résolution des Problèmes Courants
Erreur "CUDA out of memory"
Réduisez la taille des images d'entrée
Utilisez un modèle SAM plus léger (VIT-B au lieu de VIT-H)
Traitez moins d'images simultanément
Contours internes non détectés
Augmentez la sensibilité avec --min-internal-area plus bas
Activez le mode debug avec --debug --debug-internal
Vérifiez que l'image a suffisamment de contraste
Segmentation imprécise
Essayez d'activer le CRF avec --use-crf
Augmentez --expand-ratio pour capturer plus de contexte
Utilisez des prompts plus spécifiques
Exemples de Résultats
Type d'image	Résultats attendus
Portraits	Détection précise du visage, des vêtements avec plis et textures
Scènes urbaines	Délimitation claire des véhicules, bâtiments avec fenêtres et portes
Vêtements	Segmentation fine avec détection des coutures, boutons et motifs
Animaux	Contours précis avec détection des caractéristiques faciales et textures
Personnalisation Avancée
Ajustement des Paramètres SAM
Python
# Dans la fonction initialize_sam
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,           # Plus élevé = plus de détails, mais plus lent
    pred_iou_thresh=0.86,         # Seuil de qualité des masques
    stability_score_thresh=0.92,  # Seuil de stabilité des masques
    crop_n_layers=0,              # Couches de recadrage
    min_mask_region_area=20,      # Surface minimale des régions
)
Amélioration de la Détection des Contours Internes
Python
# Dans la méthode _filter_contours de InternalContourDetector
# Modifier les poids pour privilégier différentes caractéristiques
naturalness = (0.4 * convexity) + (0.3 * (1.0 - min(1.0, complexity/100))) + (0.3 * min(1.0, points_density))
Publications et Références
YOLO-World: https://github.com/AILab-CVC/YOLO-World
Segment Anything Model (SAM): https://segment-anything.com/
DenseCRF: https://github.com/lucasb-eyer/pydensecrf
Licence et Attribution
Ce projet est distribué sous licence MIT. Veuillez citer les projets originaux (YOLO-World et SAM) selon leurs licences respectives lors de l'utilisation de ce code.

Contributeurs
Développement Principal: FETHl
Date de création: 2025-03-26