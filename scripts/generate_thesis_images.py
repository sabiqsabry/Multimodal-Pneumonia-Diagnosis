import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.gridspec import GridSpec

# Setup paths
FIG_DIR = "outputs/figures"
TRAIN_IMAGES_DIR = "data/raw/rsna/Training/Images"
TRAIN_METADATA = "data/raw/rsna/stage2_train_metadata.csv"
EXPLAIN_DIR = "outputs/figures/explainability"

os.makedirs(FIG_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(TRAIN_METADATA)

# 1. Figure 1: Representative pneumonia-positive CXR
print("Creating Figure 1: Representative pneumonia-positive CXR...")
pneumonia_cases = df[df['Target'] == 1]['patientId'].unique()
if len(pneumonia_cases) > 0:
    # Pick first pneumonia case
    pneumonia_id = pneumonia_cases[0]
    pneumonia_path = os.path.join(TRAIN_IMAGES_DIR, f"{pneumonia_id}.png")
    
    if os.path.exists(pneumonia_path):
        img = Image.open(pneumonia_path)
        plt.figure(figsize=(6, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f"Representative Pneumonia-Positive CXR\nPatient ID: {pneumonia_id}", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        fig1_path = os.path.join(FIG_DIR, "figure1_pneumonia_cxr.png")
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig1_path}")
    else:
        print(f"Pneumonia image not found: {pneumonia_path}")

# 2. Literature Review: Pneumonia vs Normal CXR comparison
print("Creating Literature Review: Pneumonia vs Normal CXR comparison...")
normal_cases = df[df['Target'] == 0]['patientId'].unique()

if len(pneumonia_cases) > 0 and len(normal_cases) > 0:
    # Select one of each
    pneumonia_id = pneumonia_cases[0]
    normal_id = normal_cases[0]
    
    pneumonia_path = os.path.join(TRAIN_IMAGES_DIR, f"{pneumonia_id}.png")
    normal_path = os.path.join(TRAIN_IMAGES_DIR, f"{normal_id}.png")
    
    if os.path.exists(pneumonia_path) and os.path.exists(normal_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Pneumonia CXR
        img_pneumonia = Image.open(pneumonia_path)
        ax1.imshow(img_pneumonia, cmap='gray')
        ax1.set_title("Pneumonia-Positive CXR", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Normal CXR
        img_normal = Image.open(normal_path)
        ax2.imshow(img_normal, cmap='gray')
        ax2.set_title("Normal CXR", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(FIG_DIR, "literature_review_cxr_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {comparison_path}")
    else:
        print("One or both CXR images not found")

# 3. Explainability: Grad-CAM heatmap
print("Creating Explainability: Grad-CAM heatmap...")
gradcam_files = [f for f in os.listdir(EXPLAIN_DIR) if f.startswith('gradcam_')]
if gradcam_files:
    # Use the first available Grad-CAM image
    gradcam_path = os.path.join(EXPLAIN_DIR, gradcam_files[0])
    img = Image.open(gradcam_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title("Grad-CAM Heatmap Visualization\n(Highlighting regions important for pneumonia detection)", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    gradcam_out = os.path.join(FIG_DIR, "explainability_gradcam.png")
    plt.savefig(gradcam_out, dpi=300, bbox_inches='tight')
    print(f"Saved: {gradcam_out}")
else:
    print("No Grad-CAM images found in explainability folder")

# 4. Explainability: SHAP summary plot
print("Creating Explainability: SHAP summary plot...")
shap_files = [f for f in os.listdir(EXPLAIN_DIR) if f.startswith('shap_')]
if shap_files:
    # Use the first available SHAP image
    shap_path = os.path.join(EXPLAIN_DIR, shap_files[0])
    img = Image.open(shap_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title("SHAP Summary Plot\n(Feature importance for EMR-based prediction)", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    shap_out = os.path.join(FIG_DIR, "explainability_shap.png")
    plt.savefig(shap_out, dpi=300, bbox_inches='tight')
    print(f"Saved: {shap_out}")
else:
    print("No SHAP images found in explainability folder")

# 5. Combined explainability figure
print("Creating combined explainability figure...")
if gradcam_files and shap_files:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grad-CAM
    gradcam_path = os.path.join(EXPLAIN_DIR, gradcam_files[0])
    img_gradcam = Image.open(gradcam_path)
    ax1.imshow(img_gradcam)
    ax1.set_title("Grad-CAM Heatmap\n(CXR-based attention)", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # SHAP
    shap_path = os.path.join(EXPLAIN_DIR, shap_files[0])
    img_shap = Image.open(shap_path)
    ax2.imshow(img_shap)
    ax2.set_title("SHAP Summary Plot\n(EMR feature importance)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    combined_explain_path = os.path.join(FIG_DIR, "explainability_combined.png")
    plt.savefig(combined_explain_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {combined_explain_path}")

print("\nAll thesis images generated successfully!")




