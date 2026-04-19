from flask import Flask
from flask import request
import requests
import json
from FlaskModelStorage import import_model_fre
from FlaskModelStorage import import_model_spa
import tensorflow as tf

modelfre=import_model_fre()
modelspa=import_model_spa()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/fre")
def translate_french():
    text=request.args.get("text",0)
    result = modelfre.translate(tf.constant([text]))
    return "<p>"+result+"</p>"

@app.route("/spa")
def translate_spanish():
    text=request.args.get("text",0)
    result = modelspa.translate(tf.constant([text]))
    return "<p>"+result+"</p>"