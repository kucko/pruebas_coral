import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Cargar el modelo de TensorFlow Lite
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocesar la imagen antes de pasarla al modelo
def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalizar la imagen
    return img

# Ejecutar la inferencia con TensorFlow Lite
def run_inference(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter
