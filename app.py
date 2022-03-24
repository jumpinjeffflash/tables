import tensorflow as tf
model = tf.keras.models.load_model('tables.hdf5')

from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from cv2 import cv2

import streamlit as st

st.title('Welcome to the table classifier!')

st.markdown("This dashboard takes an image of a table and classifies whether it's tricky to tranport, which will help with setting tranportation fees")

@st.cache

def import_and_predict(image_data, model):
    
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)

        return prediction

file = st.file_uploader("Please upload your image...", type=["png","jpg","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=150)
    prediction = import_and_predict(image, model)
    
    if prediction[0]>0.55:
        st.write("""### This table looks tricky to move - and we should charge extra for transporting it""")
    else:
        st.write("""### This table doesn't look tricky to move, so we don't need to charge extra for transporting it""")
    
    percentage = prediction*100
    out_arr = np.array_str(percentage,precision=2,suppress_small=True)
    
    probability = out_arr.strip("[").strip("]")
    probability_finessed = probability+"%"

    st.markdown("For context, here's our model's prediction of how tricky this table will be to move:")
    st.write(probability_finessed)
    st.markdown("(Scale: 100% = really tricky | 0% = easy peasy)")
 
