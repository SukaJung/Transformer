from flask import Flask, redirect, url_for, request,Response,send_file
import tensorflow as tf
# from model import *
from PIL import Image
import cv2
import os
import numpy as np
import io


app = Flask(__name__)


# loader = tf.train.import_meta_graph('/home/suka/resnet50/model.ckpt.meta')
# loader.restore(sess, tf.train.latest_checkpoint('/home/suka/resnet50'))

first_image = Image.open("./uploads/1.jpg")
second_image = Image.open("./uploads/2.jpg")

first_image = np.array(Image.open("./uploads/1.jpg"))
second_image = np.array(Image.open("./uploads/2.jpg"))

gpu_options = tf.GPUOptions(allow_growth=True)

detection_graph = tf.Graph(config=tf.ConfigProto(gpu_options=gpu_options))
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('/home/suka/resnet50/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    

run_inference_for_single_image(first_image,detection_graph)
# MODEL = DeepLabModel("./deeplabv3_mnv2_pascal_train_aug.tar.gz")

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def eval():
	return "A"

   
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file = request.files['file']
    img = Image.open(file)
    arr = file.stream.read()
    img = img.resize((64,64))
    img.save("uploads/before.jpg")
    img = np.asarray(img)
    
    img = img.reshape(1,64,64,3)
    print(img.shape)

    output = sess.run("Image-Output:0", feed_dict={"Image-Input0:0": img})
    output = np.array(output)
    print(output.shape)
    output = output[0].reshape(64, 64, 3)
    print(output.shape)
    output = output * 255
    output = output.astype(np.uint8)
    result = Image.fromarray(output)
    result.save("uploads/result.jpg")
#     content = open("uploads/result.jpg").read()
    with open("uploads/result.jpg", 'rb') as bites:
        return send_file(
                     io.BytesIO(bites.read()),
                     attachment_filename='result.jpg',
                     mimetype='image/jpg'
               )
    

    
if __name__ == "__main__":
	app.run(host='192.168.10.103', port=8000, debug=True)

