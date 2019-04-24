import os
import time
import hashlib
import numpy as np
import pandas as pd
import torch.nn as nn

from fastai.vision import *
from fastai.metrics import error_rate

from flask import Flask, render_template, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

TEMPLATE_DIR = os.path.abspath('../templates/default')

# Directory where model files are (with export.pkl)
DIR_DAMAGE = Path('./damage-model')
DIR_DAMAGE_SIDE = Path('./damage-side-model')
DIR_DAMAGE_LEVEL = Path('./damage-level-model')

defaults.device = torch.device('cpu')

app = Flask(__name__, static_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

learner_damage = load_learner(DIR_DAMAGE)
learner_side = load_learner(DIR_DAMAGE_SIDE)
learner_level = load_learner(DIR_DAMAGE_LEVEL)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image Only!'), FileRequired(u'Choose a file!')])
    submit = SubmitField(u'Upload')


def get_resource_as_string(name, charset='utf-8'):
    with app.open_resource(name) as f:
        return f.read().decode(charset)


app.jinja_env.globals['get_resource_as_string'] = get_resource_as_string


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        for filename in request.files.getlist('photo'):
            open_name = 'admin' + str(time.time())
            name = hashlib.md5(open_name.encode('utf-8')).hexdigest()[:15]
            photos.save(filename, name=name + '.')
        success = True
    else:
        success = False
    return render_template('index.html', form=form, success=success)


@app.route('/manage')
def manage_file():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    return render_template('manage.html', files_list=files_list)


@app.route('/open/<filename>')
def open_file(filename):
    file_url = photos.url(filename)
    path = 'uploads/' + filename
    img = open_image(path)
    pred_class, pred_idx, outputs = learner_damage.predict(img)
    print_stats('CAR DAMAGE', learner_damage.data.classes, pred_class, pred_idx, outputs)

    if pred_idx.item() == 0:
        # Car is damaged, make predictions about the details
        side_class, side_idx, side_outputs = learner_side.predict(img)
        print_stats('CAR DAMAGE SIDE', learner_side.data.classes, side_class, side_idx, side_outputs)
        level_class, level_idx, level_outputs = learner_level.predict(img)
        print_stats('CAR DAMAGE LEVEL', learner_level.data.classes, level_class, level_idx, level_outputs)
        return render_template('browser.html', file_url=file_url,
                           pred_class=pred_class, reliability=outputs[pred_idx],
                           side_class=side_class, side_reliability=side_outputs[side_idx],
                           level_class=level_class, level_reliability=level_outputs[level_idx])
    else:
        return render_template('browser.html', file_url=file_url, damaged=pred_idx.item(),
                               pred_class=pred_class, reliability=outputs[pred_idx])



@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file'))


def print_stats(header, classes, pred_class, pred_idx, outputs):
    print(f'======== {header} ========')
    print('Data classes: ' + str(classes))
    print('Predicted class: ' + str(pred_class))
    print('Predicted index: ' + str(pred_idx.item()))
    print('Outputs: ' + str(outputs))


if __name__ == '__main__':
    app.run(debug=True)
