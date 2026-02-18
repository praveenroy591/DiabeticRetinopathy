import os
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

from cloudant.client import Cloudant

app = Flask(__name__)
app.secret_key = "dr_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- Load Model --------
model = load_model("Updated-Xception-diabetic-retinopathy.h5")

# -------- Cloudant DB --------
ACCOUNT_NAME = "b9a4698c-e779-4915-9de2-6ffde75c34cf-bluemix"
USERNAME = "b9a4698c-e779-4915-9de2-6ffde75c34cf-bluemix"
API_KEY = "Tp1KVRqBgKrAReRboeI-mnk4vU9O3sKnTWc8WPQ-hVdN"

client = Cloudant.iam(
    account_name=ACCOUNT_NAME,
    username=USERNAME,
    api_key=API_KEY,
    connect=True
)

db = client.create_database("my_database", throw_on_exists=False)

# -------- Routes --------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/afterreg", methods=["POST"])
def afterreg():
    name = request.form["name"]
    userid = request.form["userid"]
    password = request.form["password"]

    query = {"_id": {"$eq": userid}}
    docs = db.get_query_result(query)

    if len(docs.all()) == 0:
        data = {
            "_id": userid,
            "name": name,
            "password": password
        }
        db.create_document(data)
        return render_template("register.html", pred="Registration Successful! Please Login.")
    else:
        return render_template("register.html", pred="User already exists!")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/afterlogin", methods=["POST"])
def afterlogin():
    userid = request.form["userid"]
    password = request.form["password"]

    query = {"_id": {"$eq": userid}}
    docs = db.get_query_result(query)

    if len(docs.all()) == 0:
        return render_template("login.html", pred="Invalid Username")
    else:
        if password == docs[0][0]["password"]:
            session["user"] = userid
            return redirect(url_for("prediction"))
        else:
            return render_template("login.html", pred="Invalid Password")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    result = ""

    if request.method == "POST":
        f = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        class_index = np.argmax(preds)

        labels = [
            "No Diabetic Retinopathy",
            "Mild Diabetic Retinopathy",
            "Moderate Diabetic Retinopathy",
            "Severe Diabetic Retinopathy",
            "Proliferative Diabetic Retinopathy"
        ]

        result = labels[class_index]

    return render_template("prediction.html", prediction=result)


@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout.html")


if __name__ == "__main__":
    app.run(debug=True)
