from flask import Flask, redirect, url_for, request,Response,send_file
import tensorflow as tf
# from model import *
from PIL import Image
import cv2
import os
import numpy as np
import io

def divide_img(img):
	imgs = []
	for i in range(2):
		for j in range(2):
			imgs.append(img[256*i:256*(i+1),256*j:256*(j+1),:])
	return np.array(imgs)

def sum_img(img):
	result_imgs = np.zeros((512,512,3),dtype=np.float32)
	indexing = 0 
	for i in range(2):
		for j in range(2):
			result_imgs[256*i:256*(i+1),256*j:256*(j+1),:] = img[indexing]
			indexing +=1 
	return result_imgs

def run_model(sess, imgs):
	outputs = []
	for i in range(4):	
		input_img = imgs[i].reshape(1,256,256,3)
		output = sess.run("Model/g_A/t1:0", feed_dict={"input_A:0": input_img})
		output = np.array(output)
		outputs.append(output)
	return np.array(outputs)

def save_image(imgs):
	result = imgs * 255
	result = result.astype(np.uint8)
	result = Image.fromarray(result)
	result.save("uploads/result.jpg")
	return;

app = Flask(__name__)


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

loader = tf.train.import_meta_graph('/home/suka/git/Transformer/server/model/cyclegan-67.meta')
loader.restore(sess, tf.train.latest_checkpoint('/home/suka/git/Transformer/server/model'))

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

   
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file = request.files['file']
    img = Image.open(file)
    arr = file.stream.read()
    img = img.resize((512,512))
    img.save("uploads/before.jpg")

    img = divide_img(np.array(img))
    outputs = run_model(sess,img)
    outputs = outputs.reshape(4,256,256,3)
    result = sum_img(outputs)
    save_image(result)
    with open("uploads/result.jpg", 'rb') as bites:
        return send_file(
                     io.BytesIO(bites.read()),
                     attachment_filename='result.jpg',
                     mimetype='image/jpg'
               )
    

    
if __name__ == "__main__":
	app.run(host='192.168.10.103', port=8000, debug=True)

