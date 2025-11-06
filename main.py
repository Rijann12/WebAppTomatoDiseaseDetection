import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image 


class_names = [  
                'Tomato___Target_Spot',
                'Tomato___Spider_mites-spotted_spider_mite', 
                'Tomato___Leaf_mold', 
                'Tomato___Late_Blight', 
                'Tomato___Septoria_leaf_spot',
                'Tomato___healthy', 
                'Tomato_yellow_curl',
                'Tomato___Bacterial_spot', 
                'Not Leaf or Not Known',
                'Tomato___Early_Blight',
                'Not Known Disease',
                'Tomato__Healthy']


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, 256, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load your trained PyTorch model
cnn_model = torch.load("best_model_2.pt", map_location=torch.device("cpu"))


infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Resize to IMAGE_SIZE
    transforms.CenterCrop((224, 224)),    # Center crop
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # DEFAULT_MEAN
        std=[0.229, 0.224, 0.225]        # DEFAULT_STD
    )
])
# # Image transform 
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# Model Prediction Function 
# def model_prediction(test_image, threshold=0.7):
#     image = Image.open(test_image).convert("RGB")
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     with torch.no_grad():
#         outputs = cnn_model(image)
#         probabilities = torch.softmax(outputs, dim=1)[0]
#         predicted_index = torch.argmax(probabilities).item()
#         confidence = probabilities[predicted_index].item()

#     if confidence < threshold:
#         return "Not Known", confidence
#     else:
#         predicted_class = class_names[predicted_index]
#         return predicted_class, confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
cnn_model.eval()  # ensure evaluation mode

def model_prediction(test_image, threshold=0.7):
    image = Image.open(test_image).convert("RGB")
       # Apply the same transform as validation
    x = infer_transform(image).unsqueeze(0).to(device)  # add batch dimension
    # image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = cnn_model(x)  # use cnn_model here
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    if confidence < threshold:
        return "Not Known", confidence
    else:
        predicted_class = class_names[predicted_index]
        return predicted_class, confidence
#Database connection
import db_utils

db_utils.init_db() 

# #Model Prediction
# def model_prediction(test_image, threshold=0.7):
#     model = tf.keras.models.load_model("best1_model.h5")
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.expand_dims(input_arr, axis=0)  # convert to batch
#     predictions = model.predict(input_arr)

#     predicted_index = np.argmax(predictions)
#     confidence = float(predictions[0][predicted_index]) # probability of predicted class

#     if confidence< threshold:
#         return "Not Known", confidence
#     else:
#         predicted_class=class_names[predicted_index]
#         return predicted_class, confidence
    
    
#Precaution / Prevention tips
genai.configure(api_key="AIzaSyCNOj54zXIYg2N4tQni5rNfiHL5K-i5I6o")
# model = genai.GenerativeModel("gemini-1.5-flash")
import requests
def get_precaution_from_ai(disease_name):
    try:
        res = requests.post(
            "http://localhost:5001/precaution",
            json={"disease": disease_name},
            timeout=10
        )
        if res.status_code == 200:
            return res.json().get("Precaution", " Could not parse suggestions.")
        else:
            return " Could not fetch suggestions right now. Please try again later."
    except Exception as e:  
        return f" Gemini service unreachable: {str(e)}"
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","View History"])

#Main Page

if(app_mode=="Home"):
    st.header("DeepLeaf")
    st.subheader("Smart Plant Disease Detection System")
    image_path = "home_page1.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
     ## Welcome to **DeepLeaf** – Your AI-Powered Plant Health Companion!

    Say goodbye to guesswork and hello to smart farming.  
    **DeepLeaf** uses deep learning to detect plant diseases from images — fast, simple, and accessible to everyone.

    ###  What is DeepLeaf?
    DeepLeaf is an intelligent plant disease detection platform that helps users identify crop issues by simply uploading a leaf image. Whether you're growing vegetables at home or managing a farm,
     DeepLeaf gives you quick insights and actionable suggestions to protect your plants.
    ###  What Does This System Do?
                
    This platform allows you to:
    -  Upload images of affected plant leaves
    -  Detect potential plant diseases using a trained CNN model
    -  Get instant prevention and treatment suggestions powered by AI

    ###  How It Works:
    1. **Capture or Select a Leaf Image** — Preferably with visible symptoms.
    2. **Upload the Image** in the **Disease Recognition** section.
    3. **Receive Diagnosis** — The model predicts the disease, and you'll get tailored advice.

    ###  Key Features:
    -  **Accurate Predictions** using a Convolutional Neural Network (CNN)
    -  **Real-time Analysis** for faster decision-making
    -  **Integrated AI Suggestions** for remedies and preventive actions
    -  **Simple, Clean Interface** with no login or complex setup required

    ###  Need Help?
    Visit the **About** page to learn more about the project, its development team, and the technology stack used.

    ---
     Ready to identify plant diseases?  
    Head over to the **Disease Recognition** tab in the sidebar to get started!
    """)
elif app_mode == "View History":
    st.header(" Prediction & Precaution History")

    # --- Prediction History ---
    st.subheader(" Model Prediction History")
    prediction_data = db_utils.get_prediction_history()
    if prediction_data:
        st.table(
            [{"Disease": row[0], "Timestamp": row[1]} for row in prediction_data]
        )
    else:
        st.info("No prediction history found.")

    # --- Precaution History ---
    st.subheader(" Gemini Precaution History")
    precaution_data = db_utils.get_precaution_history()
    if precaution_data:
        st.table(
            [{"Disease": row[0], "Precaution": row[1], "Timestamp": row[2]} for row in precaution_data]
        )
    else:
        st.info("No precaution history found.")
#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image:")

    if test_image:
        file_type = test_image.type
        valid_types = ['image/jpeg', 'image/png']

        if file_type not in valid_types:
            st.error(" Invalid file type. Please upload a JPEG or PNG image.")
        else:
            if st.button("Show Image"):
                st.image(test_image, use_column_width=True)

            if st.button("Predict"):
                st.subheader(" Our Prediction")

                predicted_disease, confidence  = model_prediction(test_image)

                if predicted_disease == "Not Leaf":
                    st.warning(" Not leaf detected. Unable to provide recommendations.")
                    precaution = "N/A"
                if predicted_disease == "Not Known":
                    st.warning(f"Prediction below confidence threshold.Output: **Not Known**")
                    precaution = "N/A"
                else:
                    st.success(f" Model Prediction: **{predicted_disease}** ")

                if predicted_disease == "Unknown Diseases":
                    st.warning(" Unknown disease detected. Unable to provide recommendations.")
                    precaution = "N/A"
                else:

                    with st.spinner(" Generating precaution tips using Gemini AI..."):
                        precaution = get_precaution_from_ai(predicted_disease)

                    st.markdown(f" **Precaution & Treatment Tips for {predicted_disease}:**\n\n{precaution}")                
# Save prediction and precaution to the database
                db_utils.save_prediction(predicted_disease)
                db_utils.save_precaution(predicted_disease, precaution)
        
elif app_mode == "View History":
    st.markdown("##  Prediction History")
    prediction_history = db_utils.get_prediction_history()
    if prediction_history:
        for disease, timestamp in prediction_history:
            st.markdown(f"-  **{disease}** – `{timestamp}`")
    else:
        st.info("No prediction history available.")

    st.markdown("---")
    st.markdown("##  Precaution History")
    precaution_history = db_utils.get_precaution_history()
    if precaution_history:
        for disease, precaution, timestamp in precaution_history:
            st.markdown(f"-  **{disease}** – `{timestamp}`  ↳ _{precaution}_")
    else:
        st.info("No precaution history available.")

