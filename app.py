# from itertools import Predicate
from flask import Flask, render_template, request, jsonify

import os
from flask.wrappers import Response
import yaml
import joblib
import numpy as np
from prediction_service import prediction

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # If it's a request from webapp form.
            if request.form:
                data_req = dict(request.form)
                response = prediction.form_response(data_req)
                return render_template("index.html", response=response)

            # If it's a request from an api.
            elif request.json:
                data_req = dict(request.json)
                response = prediction.api_response(data_req)
                return jsonify(response)
        except Exception as e:
            print(e)
            # error = {"error": "Something went wrong! Plz try again."}
            error = {"error": e}
            return render_template("404.html", error=error)
    elif request.method == "GET":
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
