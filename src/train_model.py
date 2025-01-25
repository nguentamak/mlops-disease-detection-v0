import tensorflow as tf
import mlflow as ml
print(ml.__version__)
import mlflow.keras
model_path ="models/my_model.keras"
# Charger les données
train_dataset = tf.keras.preprocessing.image_dataset_from_directory("data/train")
test_dataset = tf.keras.preprocessing.image_dataset_from_directory("data/test")

# Assurez-vous de définir correctement les labels
labels = ['Bacterial_spot', 
          'Early_blight', 
          'Late_blight', 
          'Leaf_Mold', 
          'Mosaic_virus',
          'Septoria_leaf_spot',
          'Spider_mites Two-spotted_spider_mite',
          'Target_Spot',
          'Tomato_healthy',
          'Yellow_Leaf_Curl_Virus']  # Liste des labels, par exemple : ['cat', 'dog', 'bird']

train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, (128, 128)), y))
test_dataset = test_dataset.map(lambda x, y: (tf.image.resize(x, (128, 128)), y))

# Définir le modèle CNN
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),  # Images redimensionnées
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),  # Réduction des dimensions
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Sortie pour 10 classes
])

model.summary()

# Compiler et entraîner le modèle
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

mlflow.start_run()
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)
mlflow.keras.log_model(model, "disease_detection_model")
mlflow.log_metric("accuracy", history.history['accuracy'][-1])
model.save(model_path)
mlflow.end_run()
