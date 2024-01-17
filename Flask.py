# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:23:39 2023

@author: mecheste
"""

from flask import Flask 

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello World!<p>"
app.run()