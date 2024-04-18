import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.shortcuts import render
from django.http import HttpResponse
from myapp.classes import classes
import base64

# Load the Keras model
model = load_model('myapp/models/model.keras')

def preprocess_image(image_path, IMG_HEIGHT, IMG_WIDTH):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalization
    return image

def home(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            IMG_HEIGHT = 30
            IMG_WIDTH = 30
            uploaded_file = request.FILES['file']
            image_bytes = uploaded_file.read()

            # Resize and preprocess the uploaded image
            image = tf.image.decode_image(image_bytes, channels=3)
            image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
            image = tf.expand_dims(image, axis=0)
            image = image / 255.0  # Normalization

            # Make prediction
            prediction = model.predict(image)
            predicted_class_idx = np.argmax(prediction)  # Get the index of the class with the highest probability
            predicted_class_name = classes[predicted_class_idx]  # Get the class name corresponding to the predicted index

            # Get the selected class from the dropdown
            selected_class_id = int(request.POST.get('class'))

            # Check if the selected class is valid
            if selected_class_id not in classes:
                return HttpResponse("Error: Selected class not found in classes.")

            # Check if the predicted class matches the selected class
            is_correct = (predicted_class_idx == selected_class_id)

            class_label = f"Predicted Class: {predicted_class_name}"
            accuracy = 1 if is_correct else 0

            # Convert the uploaded image to base64 for display in the result
            image_64 = base64.b64encode(image_bytes).decode('utf-8')
            image_src = f"data:image/jpeg;base64,{image_64}"

            return render(request, "result.html", {'predicted_class_index': predicted_class_idx, 'class_label': class_label, 'image_src': image_src, 'accuracy': accuracy})
        except Exception as e:
            # Handle errors that occur during prediction
            return HttpResponse(f"Error during prediction: {e}")
    return render(request, "home.html", {'classes': classes})
