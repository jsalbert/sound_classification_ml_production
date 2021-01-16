import os
import sys
import numpy as np


from keras.models import load_model
from werkzeug.utils import secure_filename
from utils.utils import process_predictions
from feature_extraction.feature_extractor import FeatureExtractor
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}


# Create Flask App
app = Flask(__name__)

# Limit content size
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(filepath):
    feature_extractor = FeatureExtractor('config_files/feature_extraction.json')
    return feature_extractor.extract_features(filepath)


# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results',
                filename=filename))
    return render_template("index.html")


# Classify and show results
@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    # Compute audio signal features
    features = extract_features(filename)
    features = np.expand_dims(features, 0)
    # Load model and perform inference
    model = load_model('models/best_model.hdf5')
    predictions = model.predict(features)[0]
    # Process predictions and render results
    predictions_probability, prediction_classes = process_predictions(predictions,
                                                                    'config_files/classes.json')

    predictions_to_render = {prediction_classes[i]:"{}%".format(
                                round(predictions_probability[i]*100, 3)) for i in range(3)}
    # Delete uploaded file
    os.remove(filename)
    # Render results
    return render_template("results.html",
        filename=filename,
        predictions_to_render=predictions_to_render)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
