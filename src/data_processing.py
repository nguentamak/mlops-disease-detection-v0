import os
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

# Chemin du dataset
dataset_path = "data/raw_images/" 
processed_path = "data/processed_images/"
os.makedirs(processed_path, exist_ok=True)

# Fonction de transformation
def preprocess_images(input_dir, output_dir, target_size=(128, 128)):
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('jpg', 'png')):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path).resize(target_size)
                img.save(os.path.join(output_dir, file))
preprocess_images(dataset_path, processed_path)

# Division des données
image_paths = [os.path.join(processed_path, img) for img in os.listdir(processed_path)]
labels = [...]  # Charger vos étiquettes depuis un fichier CSV ou annoter manuellement
data = pd.DataFrame({'image_path': image_paths, 'label': labels})

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
