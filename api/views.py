import os
import gdown
import numpy as np
import pickle
import cv2
from django.http import JsonResponse
from rest_framework.decorators import api_view
from tensorflow.keras.preprocessing.image import img_to_array

# Google Drive File IDs for model files
MODEL_FILE_ID = "1QpqjXkntRg01t-1HWmzFvrMtOJKObHZY"
LABEL_FILE_ID = "1sGvbcwq92Ry83HFDqzFZZOVkMhSxkNzH"

# Base directory (EC2 compatible)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model and label transformer paths (Linux compatible)
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_classification_model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "plant_disease_label_transform.pkl")


def download_file_from_gdrive(file_id, output_path):
    """Download a file from Google Drive if it does not exist."""
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)


# Ensure model and label files are downloaded
download_file_from_gdrive(MODEL_FILE_ID, MODEL_PATH)
download_file_from_gdrive(LABEL_FILE_ID, LABEL_PATH)

# Load the model and label transformer
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(LABEL_PATH, "rb") as label_file:
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

    # Ensure media/temp directory exists
    temp_dir = os.path.join(BASE_DIR, "media/temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Save uploaded image
    image_file = request.FILES['image']
    image_path = os.path.join(temp_dir, image_file.name)

    with open(image_path, "wb") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Preprocess image
    processed_image = preprocess_image(image_path)

    if processed_image is None:
        return JsonResponse({"error": "Invalid image format"}, status=400)

    # Predict disease
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    disease_name = label_transformer.classes_[predicted_class]

    # Clean up temp file
    os.remove(image_path)

    return JsonResponse({"disease": disease_name, "confidence": float(np.max(prediction))})
