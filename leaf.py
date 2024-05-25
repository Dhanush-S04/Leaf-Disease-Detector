import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('D:\Leaf disease detection\leaf-cnn.h5')  # Replace with your model path

# Define the disease labels
disease_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                  'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
                  'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
                  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
                  'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
                  'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 
                  'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 
                  'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                  'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the required input size of the model
    image = cv2.resize(image, (224, 224))
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_disease(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = disease_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_label, confidence

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image using PIL
        image = Image.open(file_path)
        # Resize the image if needed
        # Resize the image if needed
        image = image.resize((300, 300), resample=Image.LANCZOS)

     
        # Convert PIL image to Tkinter PhotoImage
        photo = ImageTk.PhotoImage(image)
        # Display the image on the label
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        
        # Perform prediction on the uploaded image
        img_cv2 = cv2.imread(file_path)
        prediction, confidence = predict_disease(img_cv2)
        # Update the prediction label
        prediction_label.config(text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}")

# Create the main window
root = tk.Tk()
root.title("Leaf Disease Detector")
root.configure(bg='white')

# Set window size to screen resolution
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (screen_width, screen_height))

# Set the title with specific font, size, and color
title_label = tk.Label(root, text="Leaf Disease Detector".upper(), font=("Sylfaen", 37, "bold"), fg="black", bg='white')
title_label.pack(pady=20)

# Create labels for image and prediction
image_label = tk.Label(root)
image_label.pack(pady=10)
prediction_label = tk.Label(root, text="", font=("Arial", 12), bg='white')
prediction_label.pack(pady=10)

# Create a button for image upload
upload_button = tk.Button(root, text="Upload Image".upper(), command=upload_image, font=("Times New Roman", 15), fg="Blue", width=15, height=2)
upload_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
