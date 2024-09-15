import os
import cv2
import pytesseract
import torch
import re
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from transformers import SwinForImageClassification
from torchvision import transforms
from constants import entity_unit_map  # Assuming constants.py is imported here

# Model Loading - Swin Transformer for Feature Extraction
swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')

# Set Tesseract Path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define allowed units for validation (from constants.py)
allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Preprocess Image for Swin Transformer
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)

# Image to Text Extraction Using Tesseract
def extract_text_from_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return ""  # If the image is corrupted or unreadable
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Tesseract OCR
    extracted_text = pytesseract.image_to_string(gray)
    
    return extracted_text

# Extract number and unit based on the folder name (entity)
def extract_dimension_value(text, entity_name):
    pattern = r'(\d+(\.\d+)?)\s*([a-zA-Z]+)'  # Pattern to extract value and unit
    matches = re.findall(pattern, text)
    
    if matches:
        for match in matches:
            value, _, unit = match
            unit = unit.lower()
            
            # Check if the unit is valid for the entity_name
            if unit in entity_unit_map[entity_name]:
                return f"{value} {unit}"
    
    return ""


# Main Prediction Function
def predict_dimension(image_path, entity_name):
    try:
        # Step 1: Preprocess the image for Swin Transformer
        input_tensor = preprocess_image(image_path)
        
        # Step 2: Perform inference using Swin Transformer
        with torch.no_grad():
            outputs = swin_model(input_tensor)
    except (UnidentifiedImageError, OSError):
        # If the image is corrupted or unreadable
        return ""
    
    # Step 3: Apply OCR (Tesseract) on the image
    extracted_text = extract_text_from_image(image_path)
    
    # Step 4: Extract dimension value based on entity_name and extracted text
    dimension_value = extract_dimension_value(extracted_text, entity_name)
    
    return dimension_value or ""  # Return empty if no valid prediction


# Function to Process a Set of Images in Different Entity Folders
def process_images_in_folders(image_folders, output_csv_path):
    results = []
    
    for entity_name in image_folders:
        folder_path = os.path.join('../images', entity_name)
        
        if os.path.exists(folder_path):
            for img_filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, img_filename)
                
                # Predict the dimension value
                prediction = predict_dimension(image_path, entity_name)
                
                results.append({
                    'image': img_filename,
                    'entity_name': entity_name,
                    'prediction': prediction if prediction else "No valid data"
                })
        else:
            print(f"Folder {folder_path} does not exist.")
    
    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")


# Call the function for your folder structure
image_folders = ['width', 'depth', 'height', 'voltage', 'wattage', 'item_weight', 'maximum_weight_recommendation', 'item_volume']
process_images_in_folders(image_folders, 'predictions.csv')
