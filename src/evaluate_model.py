import tensorflow as tf
model_path ="models/my_model.keras"
output_dir_test = 'data/test'
def evaluate_model(model_path, test_dataset):
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(test_dataset)
    print(f"Test Accuracy: {results[1] * 100:.2f}%")

test_dataset = tf.keras.preprocessing.image_dataset_from_directory("data/test")
evaluate_model(model_path, test_dataset)