import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, send_from_directory
import cv2

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from io import BytesIO
import urllib


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('link.html')



@app.route('/home', methods=['POST'])
def home():
    if request.method == 'POST':
        global COUNT
        img = request.files['image']
        model = tf.keras.models.load_model('model/CM_Classifier_1')
        img.save('static/{}.jpg'.format(COUNT))    
        img_arr = image.load_img('static/{}.jpg'.format(COUNT), target_size=(150, 150))
        x=image.img_to_array(img_arr)
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
        prediction = model.predict(images)
        preds = np.array(prediction)
        COUNT += 1
        return render_template('prediction.html', data=preds)



@app.route('/main', methods=['POST'])
def main():
    if request.method == 'POST':
        model = tf.keras.models.load_model('model/CM_Classifier_1')
        preds = 0
        global COUNT
        URL = request.form['URL']
        with urllib.request.urlopen(URL) as url:
            img = image.load_img(BytesIO(url.read()), target_size=(150, 150))
        img.save('static/{}.jpg'.format(COUNT))
        x=image.img_to_array(img)
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
        prediction = model.predict(images)
        preds = np.array(prediction)
        COUNT += 1
        return render_template('prediction.html', data=preds)


@app.route('/load_imge')
def load_imge():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug = True)