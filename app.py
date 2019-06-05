import base64
import sqlite3
from io import BytesIO
import flask
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = flask.Flask(__name__)
model = None
DATA_BASE_PATH = 'D:\\Flask_api\\food_recogn.db'
MODEL_PATH = 'D:\\Flask_api\\_resources\\FoodRecognModel.h5'
CLASSES_PATH = 'D:\\Flask_api\\_resources\\classes.txt'
USERS_IMAGES_PATH = 'D:\\Flask_api\\users_images\\'
IMAGE_INDEX = 0

'''
Keras model part of API
'''


def loadmodel():
    """
    loads keras model from h5 file
    :return: global model loaded
    """
    global model
    model = load_model(filepath=MODEL_PATH)


def get_classes():
    """
    gets classes from txt file
    :return:
    """
    with open(CLASSES_PATH, 'r') as x_txt:
        x_classes = [l.strip() for l in x_txt.readlines()]
        x_ix_to_class = dict(zip(range(len(x_classes)), x_classes))
        x_class_to_ix = {v: k for k, v in x_ix_to_class.items()}
    return x_class_to_ix


def preprocess_input(x):
    """
    preproceses input for model.predict()
    :param x: input as np array
    :return: x - preprocessed
    """
    x /= 255.
    x -= 0.5
    x *= 2
    return x


def prepare_image(image, target):
    """
    transforms an image intro input data for model.predict()
    :param image: image as bytes
    :param target: dimensions for model input in this case 299x299
    :return: preprocessed numpy array
    """
    # if img is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def best_prediction(preds):
    """
    :param preds: numpy array of predictions
    :return: top prediction
    """
    best_pred = 0.0
    ind = 0
    for i in range(0, 100):
        if best_pred < preds[0][i] <= 1:
            best_pred = preds[0][i]
            ind = i
    return best_pred, ind


"""
Sql functions
"""


def add_photo_to_db(filename, label, cal, owner, percentage, date):
    """
    Insert photo informations to PHOTOINFO tabel in DB
    :param filename:
    :param label:
    :param cal:
    :param owner:
    :param percentage:
    :param date:
    :return:
    """
    connection = sqlite3.connect(DATA_BASE_PATH)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO PHOTOINFO (filename, label, kcal, owner, date) VALUES (?, ?, ?, ?, ?)",
                   (filename, label, cal, owner, date))
    connection.commit()
    connection.close()


def add_points_to_user(username, points):
    connection = sqlite3.connect(DATA_BASE_PATH)
    cursor = connection.cursor()
    cursor.execute("UPDATE USER SET points = points + ? WHERE username = ?", (points, username))
    connection.commit()
    connection.close()


"""
API routes
"""


@app.route("/predict", methods=["POST"])
def predict():
    global IMAGE_INDEX
    response_data = {"success": False, "label": "none", "cal": "none", "quantity": "none", "percentage": "none"}
    request_data = flask.request.get_json()

    filename = request_data["filename"]
    owner = request_data["owner"]
    date = request_data["date"]
    image = request_data["image"]
    multi = float(request_data["multi"])
    # decode image
    image = Image.open(BytesIO(base64.b64decode(image)))

    # save_image to disk
    image.save(USERS_IMAGES_PATH + owner + "_" + filename + "_" + str(IMAGE_INDEX) + ".png")
    IMAGE_INDEX = IMAGE_INDEX + 1
    image = prepare_image(image, target=(299, 299))

    predictions = model.predict(image, 32, 1)
    predictions = predictions.tolist()
    print(predictions)
    best_pred, ind = best_prediction(predictions)
    classes = get_classes()
    print(best_pred, " ", ind)

    print(classes)
    result = ""
    for i in classes:
        if classes[i] == ind:
            result = i

    response_data["label"] = result.split(" ")[0]
    response_data["cal"] = str(float(result.split(" ")[1].split("/")[0].split("_")[0]) * multi)
    response_data["quantity"] = result.split(" ")[1].split("/")[1]
    response_data["percentage"] = str(best_pred * 100)
    response_data["success"] = True

    if response_data["success"]:
        add_photo_to_db(filename, response_data["label"], response_data["cal"] + "_cal", owner,
                        response_data["percentage"],
                        date)
        add_points_to_user(owner, 1)

    return flask.jsonify(response_data)


@app.route("/get_user_points", methods=["POST"])
def get_user_points():
    response = {"points": 0}
    request_data = flask.request.get_json()
    username = request_data["username"]

    connection = sqlite3.connect(DATA_BASE_PATH)
    row = connection.execute('SELECT points FROM USER WHERE username = \'' + username + '\'')
    p = str(int(row.fetchone()[0]))
    response["points"] = int(p)
    connection.close()

    return flask.jsonify(response)


@app.route("/get_graph_data", methods=["POST"])
def get_graph():
    graph_data = []
    request_data = flask.request.get_json()
    username = request_data[0]["username"]
    week = request_data[0]["week"]
    connection = sqlite3.connect(DATA_BASE_PATH)
    rows = connection.execute("SELECT * FROM PHOTOINFO WHERE owner = '" + username + "' ")
    rows = rows.fetchall()
    for row in rows:
        if week in row[5]:
            graph_data.append({"day": row[5].split("_")[1], "cal": row[2].split("_")[0]})
    connection.close()
    return flask.jsonify(graph_data)


@app.route("/user_labeled", methods=["POST"])
def user_labeled():
    global IMAGE_INDEX
    response_data = {"success": False, "label": "none", "cal": "none", "quantity": "none", "percentage": "none"}
    request_data = flask.request.get_json()

    filename = request_data["filename"]
    owner = request_data["owner"]
    date = request_data["date"]
    image = request_data["image"]
    label = response_data["label"]
    cal = response_data["cal"]
    quantity = response_data["quantity"]
    percentage = "100%"

    # decode image
    image = Image.open(BytesIO(base64.b64decode(image)))

    # save_image to disk
    image.save(USERS_IMAGES_PATH + owner + "_" + filename + "_" + str(IMAGE_INDEX) + ".png")
    IMAGE_INDEX = IMAGE_INDEX + 1

    response_data["success"] = True
    response_data["label"] = label
    response_data["cal"] = cal
    response_data["quantity"] = quantity
    response_data["percentage"] = percentage

    if response_data["success"]:
        add_photo_to_db(filename, response_data["label"], response_data["cal"], owner, response_data["percentage"],
                        date)

    return flask.jsonify(response_data)


@app.route("/login", methods=["POST"])
def login_function():
    data = {"login": False}
    content = flask.request.get_json(silent=True)
    username = content['username']
    password = content['password']
    connection = sqlite3.connect(DATA_BASE_PATH)
    rows = connection.execute(
        'SELECT * FROM USER WHERE username = \'' + username + '\' AND password = \'' + password + '\'')
    rows = rows.fetchall()
    if len(rows) == 1 and rows[0][0] == username and rows[0][1] == password:
        data['login'] = True
    connection.close()
    return flask.jsonify(data)


@app.route("/register", methods=["POST"])
def register_function():
    data = {"registered": False, "error": '0'}
    content = flask.request.get_json(silent=True)
    username = content['username']
    password = content['password']
    email = content['email']
    connection = sqlite3.connect(DATA_BASE_PATH)
    rows = connection.execute(
        'SELECT * FROM USER WHERE username = \'' + username + '\'')
    rows = rows.fetchall()
    if len(rows) > 0:
        data['error'] = 'invalid username'
    rows = connection.execute(
        'SELECT * FROM USER WHERE email = \'' + email + '\'')
    rows = rows.fetchall()
    if len(rows) > 0:
        data['error'] = 'invalid email'
    if data['error'] == '0':
        try:
            cur = connection.cursor()
            cur.execute('INSERT INTO USER (username, password, email) VALUES (?,?,?)', (username, password, email))
            connection.commit()
            data['registered'] = True
        except:
            connection.rollback()
            data['error'] = 'database error'
            data['registered'] = False

    connection.close()
    return flask.jsonify(data)


@app.route("/")
def index():
    return flask.render_template("Index.html")


if __name__ == '__main__':
    print("Server is starting ...")
    loadmodel()
    print("Server started: ...")
    app.run(host="192.168.100.4", port=8081)
