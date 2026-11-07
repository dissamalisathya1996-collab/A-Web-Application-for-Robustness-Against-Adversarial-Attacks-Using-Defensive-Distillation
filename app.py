from flask import Flask, json, render_template, request, redirect, session, flash, url_for
from flask_mongoengine import MongoEngine
from werkzeug.utils import secure_filename
from flask_session import Session
from ML.attacks import fgsm_attack, pgd_attack
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from mongoengine import Document, StringField, FloatField, DateTimeField
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from flask import send_file
from ML.visualize_fgsm import visualize_fgsm_accuracy
from ML.visualize_pgd import visualize_pgd_accuracy
from utils import preprocess_image, apply_attack, get_prediction_confidence
from ML.visualize_confidence import visualize_confidence


app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = os.path.join('static', 'Uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Session Configuration
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# MongoDB Configuration
app.config['MONGODB_SETTINGS'] = {
    'db': 'DeepDefendAI',
    'host': 'host.docker.internal',
    'port': 27017
}
db = MongoEngine(app)

# User Model
class users(db.Document):
    name = db.StringField(required=True)
    email = db.StringField(required=True)
    password = db.StringField(required=True)

# Collection for uploaded images
class UploadedImage(Document):
    filename = StringField(required=True)
    upload_time = DateTimeField(default=datetime.utcnow)

# Collection for accuracy scores
class ModelAccuracyScore(Document):
    teacher_accuracy = FloatField(required=True)
    student_accuracy = FloatField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

# Collection for attack accuracy scores
class AttackAccuracyScore(Document):
    attack_type = StringField(required=True)
    teacher_accuracy = FloatField(required=True)
    student_accuracy = FloatField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)    

@app.route('/')
def home():
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/view_results', methods=['GET', 'POST'])
def view_results():
    if request.method == 'POST':
        file = request.files['imageUpload']
        attack_type = request.form.get('attackType')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Save to MongoDB
            UploadedImage(filename=filename).save()

            session['uploaded_image'] = file_path
            session['attack_type'] = attack_type

            # Load and preprocess the image to numpy array
            image = Image.open(file_path).convert('RGB')
            image = image.resize((32, 32))  # Match your model's expected input
            image_array = np.array(image) / 255.0  # Normalize

            np.save(os.path.join(UPLOAD_FOLDER, 'latest_image.npy'), image_array)

            flash('Results generated successfully!', 'success')

    return render_template('view_results.html')

@app.route('/model_comparison')
def model_comparison():
    # Load saved test data
    X_test = np.load('static/X_test.npy')
    y_test = np.load('static/y_test.npy')

    # Load models (with compile fix)
    teacher_model = load_model('models/teacher_model.h5', compile=False)
    student_model = load_model('models/student_model.h5', compile=False)

    teacher_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    student_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate models
    teacher_acc = teacher_model.evaluate(X_test, y_test, verbose=0)[1] * 100
    student_acc = student_model.evaluate(X_test, y_test, verbose=0)[1] * 100

    ModelAccuracyScore(teacher_accuracy=teacher_acc, student_accuracy=student_acc).save()

    return render_template(
        'model_comparison.html',
        teacher_acc=f"{teacher_acc:.2f}",
        student_acc=f"{student_acc:.2f}",
        chart_path = url_for('static', filename='accuracy_comparison.png')
    )

@app.route('/attack_performance')
def attack_performance():
    attack_type = session.get('attack_type')
    
    if not attack_type:
        return "Missing attack type", 400

    # Load full test set
    X_test = np.load('static/X_test.npy')
    y_test = np.load('static/y_test.npy')

    indices = random.sample(range(len(X_test)), 500)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # Load models (with compile fix)
    teacher_model = load_model('models/teacher_model.h5', compile=False)
    student_model = load_model('models/student_model.h5', compile=False)

    teacher_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    student_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    # Apply attack on the full test set
    if attack_type == 'fgsm':
        adv_teacher = fgsm_attack(teacher_model, X_sample, y_sample, epsilon=0.1)
        adv_student = fgsm_attack(student_model, X_sample, y_sample, epsilon=0.1)
    elif attack_type == 'pgd':
        adv_teacher = pgd_attack(teacher_model, X_sample, y_sample, epsilon=0.3, alpha=0.01, steps=40)
        adv_student = pgd_attack(student_model, X_sample, y_sample, epsilon=0.3, alpha=0.01, steps=40)
    else:
        return "Invalid attack type", 400

    # Evaluate
    teacher_acc = teacher_model.evaluate(adv_teacher, y_sample, verbose=0)[1] * 100
    student_acc = student_model.evaluate(adv_student, y_sample, verbose=0)[1] * 100

    AttackAccuracyScore(
        attack_type=attack_type,
        teacher_accuracy=teacher_acc,
        student_accuracy=student_acc
    ).save()

    if attack_type == 'fgsm':
       visualization_path = visualize_fgsm_accuracy(teacher_acc, student_acc)
    elif attack_type == 'pgd':
       visualization_path = visualize_pgd_accuracy(teacher_acc, student_acc)

    return render_template('attack_performance.html',
                           teacher_accuracy=teacher_acc,
                           student_accuracy=student_acc,
                           attack_type=attack_type.upper(),
                           visualization_image=visualization_path.split("static/")[1])

import matplotlib.pyplot as plt

@app.route('/confidence_comparison')
def confidence_comparison():
    attack_type = session.get('attack_type')

    # Load model
    teacher_model = load_model('models/teacher_model.h5')

    # Load image
    image_path = os.path.join('static', 'Uploads', 'latest_image.npy')
    original_image = np.load(image_path)

    # Get label
    label, _ = get_prediction_confidence(teacher_model, original_image)

    # Apply attack
    attacked_image = apply_attack(original_image.copy(), teacher_model, attack_type, label=label)

    # Get confidence scores
    _, original_conf = get_prediction_confidence(teacher_model, original_image)
    _, attacked_conf = get_prediction_confidence(teacher_model, attacked_image)

    # 🎯 Create visualization
    graph_path = visualize_confidence(original_conf * 100, attacked_conf * 100, 
                                      output_path=f'static/confidence_comparison_{attack_type}.png')

    return render_template('confidence_comparison.html',
                           attack_type=attack_type.upper(),
                           original_conf=round(original_conf * 100, 2),
                           attacked_conf=round(attacked_conf * 100, 2),
                           graph_filename=os.path.basename(graph_path))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Query user from DB
        user = users.objects(email=email, password=password).first()

        if user:
            session['user_id'] = str(user.id)
            session['username'] = user.name  # Store username to greet
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect('login')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")

        # Check if all fields are filled
        if not name or not email or not password or not confirm_password:
            return render_template("signup.html", error_message="All fields are required!")

        # Check if passwords match
        if password != confirm_password:
            return render_template("signup.html", error_message="Passwords do not match!")

        # Check if email already exists
        existing_user = users.objects(email=email).first()
        if existing_user:
            return render_template("signup.html", error_message="Email already registered!")

        # Save new user
        user = users(name=name, email=email, password=password)
        user.save()

        return redirect(url_for('login'))  # Redirect to login page after signup

    return render_template('signup.html', error_message='')


if __name__ == '__main__':
    app.run(debug=True)
