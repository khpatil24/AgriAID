import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('D:/AgriAID/plantvillage dataset/cotton_trained.keras')
import cv2
image_path = "C:/Users/khpat/Downloads/cotton_target.jpg"
img = cv2.imread(image_path) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
input_arr.shape
prediction = model.predict(input_arr)
result_index = np.argmax(prediction)
class_name = [
    "Aphids",
    "Army Worms",
    "Bacterial Blight",
    "Healthy",
    "Powdery Mildew",
    "Target Spot"
]
model_pred = class_name[result_index]
print(model_pred)
