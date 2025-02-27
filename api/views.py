import os
import numpy as np
import pickle
import cv2
from django.http import JsonResponse
from rest_framework.decorators import api_view
from tensorflow.keras.preprocessing.image import img_to_array

# Paths to your model and label transformer
MODEL_PATH = r"C:\Users\waqas\Desktop\plant_disease_classification_model.pkl"
LABEL_TRANSFORMER_PATH = r"C:\Users\waqas\Desktop\plant_disease_label_transform.pkl"

# Load the model and label transformer
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(LABEL_TRANSFORMER_PATH, "rb") as label_file:
    label_transformer = pickle.load(label_file)

DEFAULT_IMAGE_SIZE = (256, 256)


def preprocess_image(image_path):
    """Load and preprocess the image for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
    image = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension


@api_view(['POST'])
def predict_disease(request):
    """API endpoint to predict plant disease from an uploaded image."""
    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image uploaded"}, status=400)

    image_file = request.FILES['image']
    image_path = os.path.join("media/temp", image_file.name)

    os.makedirs("media/temp", exist_ok=True)

    with open(image_path, "wb") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    processed_image = preprocess_image(image_path)

    if processed_image is None:
        return JsonResponse({"error": "Invalid image format"}, status=400)

    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    disease_name = label_transformer.classes_[predicted_class]
    os.remove(image_path)

    return JsonResponse({"disease": disease_name, "confidence": float(np.max(prediction))})
