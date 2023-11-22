import streamlit as st
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load the model
model = load_model("./model/v3_pred_cott_dis.h5")
st.write('@@ Model loaded')

def pred_cot_dieas(cott_plant):
    st.write("@@ Got Image for prediction")
    test_image = load_img(cott_plant, target_size=(150, 150))
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image).round(3)
    st.write('@@ Raw result = ', result)
    pred = np.argmax(result)

    if pred == 0:
        return "Aphids_disease", 'Aphids_disease.html'
    elif pred == 1:
        return "Army_worm", 'Army_worm.html'
    elif pred == 2:
        return "Bacterial_Blight", 'Bacterial_Blight.html'
    elif pred == 3:
        return "Healthy Cotton Plant", 'healthy_plant.html'
    elif pred == 4:
        return "Powdery Mildew", 'Powdery_Mildew.html'
    else:
        return "Target spot", 'Target_spot.html'

# Render the main page
def home():
    st.title("Plant Disease Prediction")
    st.write("Upload an image of a cotton plant to predict the disease.")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Perform prediction and display the result
        pred, output_page = pred_cot_dieas(cott_plant=uploaded_file)
        st.success(f"The plant is classified as: {pred}")
        st.balloons()

# Run the app
if __name__ == '__main__':
    home()

