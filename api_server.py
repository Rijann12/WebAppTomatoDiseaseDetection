    # api_server.py
from flask import Flask, request, jsonify
import google.generativeai as genai
import tensorflow as tf
import numpy as np
import db_utils # Make sure db_utils.py is in the same directory
import os
from PIL import Image # Import Pillow for image processing

app = Flask(__name__)

    # --- Configuration ---
    # Initialize DB
db_utils.init_db()

    # Configure Gemini (IMPORTANT: In a real app, use environment variables for API keys!)
genai.configure(api_key="AIzaSyCNOj54zXIYg2N4tQni5rNfiHL5K-i5I6o") # Use your actual Gemini API key
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    # Load TensorFlow Model (load once when app starts to avoid reloading for every request)
try:
        # Ensure the model path is correct relative to api_server.py
        model_path = "trained_plant_disease_model.h5"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            plant_disease_model = None
        else:
            plant_disease_model = tf.keras.models.load_model(model_path)
            print("TensorFlow model loaded successfully.")
except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        plant_disease_model = None # Handle this gracefully

    # Define class names (must match your model's output classes)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

    # --- API Endpoints ---

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
        # 1. Check if model is loaded
        if plant_disease_model is None:
            return jsonify({"error": "Server error: AI model not loaded."}), 500

        # 2. Check for image file in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in the request."}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file in the request."}), 400

        try:
            # 3. Read image data directly from file stream
            # Using PIL (Pillow) to open image from bytes
            img = Image.open(file.stream).convert('RGB') # Ensure 3 channels
            img = img.resize((128, 128)) # Resize to model's input size

            # Convert PIL image to numpy array
            input_arr = np.array(img)
            input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

            # 4. Model Prediction
            predictions = plant_disease_model.predict(input_arr)
            predicted_index = np.argmax(predictions)
            predicted_disease = class_names[predicted_index]

            # 5. Get Precaution from Gemini
            prompt = f"Precaution steps and remedies for the plant disease: '{predicted_disease}'?"
            gemini_response = gemini_model.generate_content(prompt)
            precaution_text = gemini_response.text

            # 6. Save to DB
            db_utils.save_prediction(predicted_disease)
            db_utils.save_precaution(predicted_disease, precaution_text)

            # 7. Return JSON response
            return jsonify({
                "disease": predicted_disease,
                "precaution": precaution_text
            })

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def get_history():
        try:
            prediction_history = db_utils.get_prediction_history()
            precaution_history = db_utils.get_precaution_history()

            # Format history for JSON response
            formatted_predictions = [{"disease": d, "timestamp": t} for d, t in prediction_history]
            formatted_precautions = [{"disease": d, "precaution": p, "timestamp": t} for d, p, t in precaution_history]

            return jsonify({
                "predictions": formatted_predictions,
                "precautions": formatted_precautions
            })
        except Exception as e:
            print(f"Error fetching history: {e}")
            return jsonify({"error": f"Could not retrieve history: {str(e)}"}), 500

    # --- Run the Flask App ---
if __name__ == '__main__':
        # IMPORTANT: For local testing, use '0.0.0.0' to make it accessible from your mobile device/emulator
        # In production, you'd use a WSGI server like Gunicorn.
        app.run(host='0.0.0.0', port=5000, debug=True) # debug=True for development, set to False for production
    