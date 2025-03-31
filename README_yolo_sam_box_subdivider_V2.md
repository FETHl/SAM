# BoxSubdivider

## Segmentation avancée d'objets avec YOLO-World et SAM

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Auteur:** FETHl  
**Date de création:** 2025-03-31  
**Version:** 1.0.0

---

## 📌 Sommaire

- [Introduction](#introduction)
- [Caractéristiques principales](#caractéristiques-principales)
- [Architecture du système](#architecture-du-système)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemples avancés](#exemples-avancés)
- [Paramètres](#paramètres)
- [Structure des fichiers](#structure-des-fichiers)
- [Résolution des problèmes](#résolution-des-problèmes)
- [Performances](#performances)
- [Licence](#licence)

---

## 📋 Introduction

BoxSubdivider est un outil de segmentation d'objets de pointe qui combine la puissance de YOLO-World pour la détection d'objets et le modèle Segment Anything (SAM) de Meta AI pour la segmentation précise. Cette combinaison permet une détection et une segmentation d'objets hautement précises basées sur des descriptions textuelles.

Le système est conçu pour générer des contours d'objets détaillés, en identifiant à la fois les contours externes et internes, puis en produisant des visualisations de haute qualité en formats PNG et SVG avec des lignes épaisses pour une meilleure visibilité.

## ✨ Caractéristiques principales

- **Détection d'objets guidée par le texte** avec YOLO-World
- **Segmentation précise des contours** avec SAM
- **Détection des contours internes** pour les détails complexes
- **Fusion intelligente des contours superposés** pour éviter les duplications
- **Raffinement des masques avec CRF** (Conditional Random Fields)
- **Export en PNG et SVG** avec des contours épais pour une meilleure visualisation
- **Traitement par lots** pour traiter plusieurs images
- **Visualisation avancée** avec légendes et code couleur

## 🔍 Architecture du système

BoxSubdivider utilise une approche en pipeline à plusieurs étapes:

1. **Détection d'objets avec YOLO-World** - Identifie les objets dans l'image en fonction des prompts textuels
2. **Segmentation avec SAM** - Génère des masques précis pour chaque objet détecté
3. **Détection de contours internes** - Identifie les structures internes des objets
4. **Fusion de contours** - Fusionne intelligemment les contours superposés
5. **Raffinement** - Applique CRF pour améliorer les bords des masques
6. **Exportation** - Génère des visualisations PNG et des fichiers SVG scalables

## 📦 Prérequis

- Python 3.8 ou supérieur
- PyTorch 1.10 ou supérieur
- OpenCV 4.5 ou supérieur
- GPU CUDA recommandé pour les performances (4GB+ VRAM)

### Modèles nécessaires

- **YOLO-World** - [Télécharger le modèle préentraîné](https://github.com/mmdetection/mmdetection/tree/master/configs/yolo)
- **SAM** - [Télécharger le modèle vit_h](https://github.com/facebookresearch/segment-anything#model-checkpoints)

## 💻 Installation

### 1. Installation des dépendances

```bash
# Créer un environnement virtuel
conda create -n boxsubdivider python=3.8
conda activate boxsubdivider

# Installer PyTorch avec CUDA
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Installer les dépendances principales
pip install opencv-python numpy tqdm pillow matplotlib colorsys

# Installer MMCV et MMDetection pour YOLO-World
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
pip install mmyolo

# Installer SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Installer pydensecrf (optionnel pour le raffinement CRF)
pip install git+https://github.com/lucasb-eyer/pydensecrf.git