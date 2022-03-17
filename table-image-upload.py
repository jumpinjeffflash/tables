import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from PIL import Image
from keras.layers import BatchNormalization

from tqdm import tqdm

st.title('Table classifier')

st.markdown("This model takes a picture of a table and returns a probability whether it's a pain in the ass to move...")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Please upload your table image")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Don't tease me! Upload a picture, please!!")
        else:
            with st.spinner('Hmmmmmmm...I believe that this table is...'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)
                
def predict(image):
    classifier_model = "table_model_HDF5_format.h5"
    IMAGE_SHAPE = (350, 350, 3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((350,350))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'lightweight',
          'middleweight',
          'heavyweight']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'lightweight': 0,
          'middleweight': 0,
          'heavyweight': 0, 
}
    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result
    
if __name__ == "__main__":
    main()
       
