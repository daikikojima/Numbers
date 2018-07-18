import os

from flask import Flask, render_template, request, url_for, redirect, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
import string
import random
from datetime import datetime

from DNN.mnist_recog import predict_num

app = Flask(__name__)
app.secret_key = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(20)])

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method=="POST":
        img = request.files['img']
        if img and allowed_file(img.filename):
            # Save img file
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "." + secure_filename(img.filename).split(".")[-1]
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_url = '/uploads/' + filename
            # Predict from img
            image = Image.open(img.stream)
            label = predict_num(image)
            return render_template('result.html',img_url = img_url, label = label)
    flash("画像の拡張子はpng, jpg, gifだけだよ!")
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
