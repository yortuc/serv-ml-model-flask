from model import MLModel
from flask import Flask
app = Flask(__name__)


MODEL = MLModel()


@app.route('/')
def api_index():
    return 'Welcome to awesome machine learning rest api! try /predict/12'


@app.route('/predict/<input_x>')
def api_predict(input_x):
	return MODEL.predict(input_x)
