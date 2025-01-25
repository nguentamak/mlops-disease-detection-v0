import os, csv, shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

# Chemin du dataset
dataset_path = "data/raw_images/" 
processed_path = "data/processed_images/"
annoted_path = "data/annoted_images/annoted_images.csv"
output_dir_train = 'data/train'
output_dir_test = 'data/test'
data = []
os.makedirs(processed_path, exist_ok=True)

# Fonction de transformation
def preprocess_images(input_dir, output_dir, data, target_size=(128, 128)):
    
    for subdir, _, files in os.walk(input_dir): 
       for file in files:
            if file.endswith(('jpg', 'png', 'JPG', 'PNG')):
               img_path = os.path.join(subdir, file)
               img_path = img_path.replace("\\", "/")

               # Extraire le nom du dernier répertoire
               last_dir = os.path.basename(os.path.dirname(img_path))
            
               # Ajouter les données à la liste
               file_path = os.path.join(output_dir, file)
               data.append([file_path, last_dir])

               img = Image.open(img_path).resize(target_size)
               img.save(file_path)

def generate_csv_from_directory(data_list, output_csv):
    
    # Écrire les données dans un fichier CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Écrire l'en-tête
        writer.writerow(["image_path", "label"])
        
        # Écrire les données
        writer.writerows(data_list)
    
    print(f"Fichier CSV généré avec succès : {output_csv}")

def organize_files_by_annotation(df, output_dir):
    """
    Organize files into directories based on annotations in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with two columns - 'file_path' and 'annotation'.
        output_dir (str): Path to the output directory where files will be organized.

    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _, row in df.iterrows():
        file_path = row['image_path']
        annotation = row['label']

        # Create a directory for the annotation if it doesn't exist
        annotation_dir = os.path.join(output_dir, str(annotation))
        if not os.path.exists(annotation_dir):
            os.makedirs(annotation_dir)

        # Copy the file to the annotation directory
        if os.path.exists(file_path):
            shutil.copy(file_path, annotation_dir)
        else:
            print(f"File not found: {file_path}")


preprocess_images(dataset_path, processed_path, data)

generate_csv_from_directory(data, annoted_path)

# Division des données
#image_paths = [os.path.join(processed_path, img) for img in os.listdir(processed_path)]

#data = pd.DataFrame({'image_path': image_paths, 'label': labels})
data_df = pd.DataFrame(data, columns=['image_path', 'label'])

train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

# Organize files
organize_files_by_annotation(train_data, output_dir_train)
organize_files_by_annotation(test_data, output_dir_test)
labels = [...]  # Charger vos étiquettes depuis un fichier CSV ou annoter manuellement