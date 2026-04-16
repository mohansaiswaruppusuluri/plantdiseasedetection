# '''import os
# import json
# from PIL import Image
#
#
# import numpy as np
# import tensorflow as tf
# import streamlit as st
#
# working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# #loading pre-trained model
# model = tf.keras.models.load_model(model_path)
# class_indices = json.load(open(f"{working_dir}/class_indices.json"))
#
# #fucntion to load and preprocess the image using pillow
# def load_and_preprocess_image(image_path, target_size=(224, 224)):
#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     #adding batch dimensions
#     img_array = np.expand_dims(img_array,axis=0)
#     #scaling the image values to (0,1)
#     img_array = img_array.astype('float32') / 255.0
#     return img_array
# #function for predicting the class of an image
#
# def predict_image_class(model, image_path, class_indices):
#     preprocessed_image = load_and_preprocess_image(image_path)
#     prediction = model.predict(preprocessed_image)
#     predicted_class_index = np.argmax(prediction)
#     predicted_class_name = class_indices[str(predicted_class_index)] # Class indices might be string keys when loaded from JSON
#     return predicted_class_name
#
# #streamlit app
# st.title("Plant Disease Classifier")
# uploaded_image = st.file_uploader("Upload an image...",type=("jpg","jpeg",'png'))
#
# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     col1,col2 = st.columns(2)
#
#     with col1:
#         resized_img = image.resize((150,150))
#         st.image(resized_img)
#
#     with col2:
#         if st.button('Classfiy'):
#             #preprocess the uploaded image and predict the class
#             prediction = predict_image_class(model,uploaded_image,class_indices)
#             st.success(f"Prediction:{str(prediction)}")
#
# '''
#
#
# #gemini code
# '''
# def load_and_preprocess_image_pycharm(image_path, target_size=(224, 224)):
#     img = Image.open(image_path)
#     img = img.resize(target_size)
#     img_array = np.array(img)
#     # Ensure image has 3 channels if it's grayscale
#     if img_array.ndim == 2:
#         img_array = np.stack((img_array,) * 3, axis=-1)
#     elif img_array.shape[-1] == 4: # Handle RGBA images
#         img_array = img_array[..., :3]
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array
#
# # --- Step 2: Define the prediction function ---
# def predict_image_class_pycharm(model, image_path, class_indices):
#     preprocessed_image = load_and_preprocess_image_pycharm(image_path)
#     prediction = model.predict(preprocessed_image)
#     predicted_class_index = np.argmax(prediction)
#     predicted_class_name = class_indices[str(predicted_class_index)] # Class indices might be string keys when loaded from JSON
#     return predicted_class_name
#
# # --- Step 3: Example usage in PyCharm ---
#
# # Define the path where you downloaded your model and class_indices.json
# # You might need to adjust this path based on your project structure in PyCharm
# model_path_pycharm = 'plant_disease_prediction_model.keras' # Or 'plant_disease_prediction_model.h5'
# class_indices_path_pycharm = 'class_indices.json'
#
# # Load the trained model
# try:
#     loaded_model = tf.keras.models.load_model(model_path_pycharm)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     print("Please ensure 'plant_disease_prediction_model.keras' (or .h5) is in the correct directory.")
#     exit()
#
# # Load the class indices
# try:
#     with open(class_indices_path_pycharm, 'r') as f:
#         loaded_class_indices = json.load(f)
#     print("Class indices loaded successfully!")
# except Exception as e:
#     print(f"Error loading class indices: {e}")
#     print("Please ensure 'class_indices.json' is in the correct directory.")
# # '''
# import os
# # import json
# # from PIL import Image
# # import numpy as np
# # import tensorflow as tf
# # import streamlit as st
#
# # ---------------- CONFIG ----------------
# st.set_page_config(
#     page_title="🌿 Plant Disease Detector",
#     page_icon="🌱",
#     layout="wide"
# )
#
# # ---------------- CUSTOM CSS ----------------
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0e1117;
#     }
#     .title {
#         text-align: center;
#         font-size: 40px;
#         font-weight: bold;
#         color: #00ffcc;
#     }
#     .subtitle {
#         text-align: center;
#         color: #aaaaaa;
#         margin-bottom: 20px;
#     }
#     .stButton>button {
#         background-color: #00ffcc;
#         color: black;
#         border-radius: 10px;
#         height: 3em;
#         width: 100%;
#         font-size: 18px;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # ---------------- LOAD MODEL ----------------
# # working_dir = os.path.dirname(os.path.abspath(__file__))
# # model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# #
# # model = tf.keras.models.load_model(model_path)
# # class_indices = json.load(open(f"{working_dir}/class_indices.json"))
# from huggingface_hub import hf_hub_download
#
# # 🔽 Download model from Hugging Face
# # model_path = hf_hub_download(
# #     repo_id="Mohanchowdary/plant-disease-model",
# #     filename="plant_disease_prediction_model.h5"
# # )
#
# # 🔽 Download class indices
# # class_indices_path = hf_hub_download(
# #     repo_id="Mohanchowdary/plant-disease-model",
# #     filename="class_indices.json"
# # )
#
# # 🔽 Load model
# # model = tf.keras.models.load_model(model_path)
# #
# # # 🔽 Load class indices
# # with open(class_indices_path, "r") as f:
# #     class_indices = json.load(f)
#
#
#
#
#
#
# # ---------------- IMAGE PREPROCESS ----------------
# def load_and_preprocess_image(image, target_size=(224, 224)):
#     img = Image.open(image).convert("RGB")
#     img = img.resize(target_size)
#     img_array = np.array(img)
#
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array.astype('float32') / 255.0
#
#     return img_array
#
# # ---------------- PREDICTION ----------------
# def predict_image_class(model, image, class_indices):
#     preprocessed_image = load_and_preprocess_image(image)
#     prediction = model.predict(preprocessed_image)
#     predicted_class_index = np.argmax(prediction)
#     return class_indices[str(predicted_class_index)]
#
# # ---------------- UI ----------------
# st.markdown('<div class="title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
# st.markdown('<div class="subtitle">Upload a plant leaf image and detect disease instantly</div>', unsafe_allow_html=True)
#
# uploaded_image = st.file_uploader("📤 Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])
#
# if uploaded_image is not None:
#     col1, col2 = st.columns([1, 1])
#
#     with col1:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="🖼 Uploaded Image", use_container_width=True)
#
#     with col2:
#         st.markdown("### 🔍 Prediction Result")
#
#         if st.button("🚀 Classify"):
#             prediction = predict_image_class(model, uploaded_image, class_indices)
#
#             st.success(f"✅ **Prediction:** {prediction}")
#
#             # 🌱 Suggestion Section
#             st.markdown("### 🌱 Suggestion")
#
#             if "healthy" in prediction.lower():
#                 st.info("🌿 Your plant looks healthy! Keep maintaining proper watering and sunlight.")
#             else:
#                 st.warning(
#                     "⚠️ Disease detected. Consider using proper pesticides, "
#                     "remove affected leaves, and consult agricultural experts."
#                 )
#
# # ---------------- FOOTER ----------------
# st.markdown("---")
# st.markdown("💡 Built with ❤️ for Smart Agriculture | By Sai")




# gradio version of code

import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# ---------------- DOWNLOAD MODEL ----------------
MODEL_PATH = "plant_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1KVw_K1TAHMU_lU95C460GzLtghvQ8swz"
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

# ---------------- LOAD CLASS INDICES ----------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0

    return img_array

# ---------------- PREDICTION ----------------
def predict(image):
    img = preprocess_image(image)

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_class = class_indices[str(predicted_index)]

    if "healthy" in predicted_class.lower():
        suggestion = "🌿 Your plant looks healthy! Maintain proper watering and sunlight."
    else:
        suggestion = (
            "⚠️ Disease detected!\n"
            "- Remove infected leaves\n"
            "- Use suitable pesticide\n"
            "- Avoid overwatering\n"
            "- Consult agricultural expert if needed"
        )

    return predicted_class, confidence, suggestion

# ---------------- UI ----------------

st.title("🌿 Plant Disease Detector")
st.markdown("Upload a plant leaf image and detect disease instantly")

uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_image)
        st.image(image, caption="🖼 Uploaded Image", width=400)

    with col2:
        if st.button("🚀 Classify"):
            with st.spinner("Analyzing image..."):
                label, confidence, suggestion = predict(image)

            st.success(f"🌱 Prediction: {label}")
            st.info(f"📊 Confidence: {confidence:.2f}%")

            st.markdown("### 💡 Suggestion")
            st.warning(suggestion)

st.markdown("---")
st.markdown("💚 Built with AI for Smart Agriculture | By Sai")
