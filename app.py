from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline

#pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

st.title("Adult Content Detector")
load_dotenv()
model = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

image_path = st.file_uploader("Choose a image",type=['jpg','jpeg','png'])

if st.button("Checking"):
    with  st.spinner('Checking Content...'):
        img = Image.open(image_path)
        result = model(images=img)
        nsfw_score = next((item['score'] for item in result if item['label']=='nsfw'),None)
        st.write(nsfw_score)

        st.subheader(f"Adult Content Probability : {str(round(nsfw_score*100,2))} %")
        st.slider("",0,100,int(nsfw_score*100),1)

        if nsfw_score>0.1:
            st.subheader("Your Content is not safe❌")
            st.text("Cannot Display the Image")
        else:
            st.subheader("Your Content is Safe")
            st.image(img)    

