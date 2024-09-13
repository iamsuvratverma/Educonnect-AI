from flask import Flask, flash, redirect, render_template, request, url_for, session, jsonify, make_response
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_mail import Mail, Message
import io
import os
import base64
import cv2
import numpy as np
from PIL import Image
from services import equ_solver, Computertext, image_extract_api, summarization
from services.webcam_detect import sign_detection
from static.database.databases import get_db
from functools import wraps
from flask_session import Session
from datetime import timedelta

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'suvratverma20@gmail.com'
app.config['MAIL_PASSWORD'] = 'rwjj orxh zvxs ioab'
app.config['MAIL_DEFAULT_SENDER'] = 'suvratverma20@gmail.com'
app.secret_key = '1023456987'

# Flask-Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
Session(app)

mail = Mail(app)
socketio = SocketIO(app, cors_allowed_origins="*")


# Decorator to prevent caching
def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return no_cache


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        subject = request.form['subject']
        
        msg = Message(subject='New Contact Form Submission',
                      sender=email,
                      recipients=['sobhitv5@gmail.com'])
        msg.body = f"""
        Name: {name}
        Email: {email}
        Subject: {subject}
        Message:
        {message}
        """
        
        try:
            mail.send(msg)
            flash('Thank you for your message. We will get back to you shortly!', 'success')
        except Exception as e:
            flash('There was an issue sending your message. Please try again later.', 'error')
            print(f"Error sending email: {e}")

        return redirect('/contact')

    return render_template('contact.html')


@socketio.on('image')
def image(data_image):
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    
    frame, letter, prediction_score = sign_detection(frame)
    frame = cv2.putText(frame, 'CV', (480, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    info = {'frame': jpg_as_text, 'letter': letter, 'prediction_score': prediction_score}
    socketio.emit('processed_frame', info)


@app.route('/outputsign', methods=['POST', 'GET'])
@login_required
@nocache
def outputsign():
    return render_template('outputsign.html')


@app.route('/index', methods=['POST', 'GET'])
@login_required
@nocache
def index():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('login.html')


@app.route("/login")
@nocache
def login():
    return render_template("login.html")


@app.route("/signup")
@nocache
def signup():
    return render_template("signup.html")


@app.route('/about')
@nocache
def about():
    return render_template('about.html')


@app.route("/createaccount", methods=["POST"])
def create_account():
    username = request.form["username"]
    password = request.form["password"]
    db = get_db()
    others = db.fetch(username)
    if others:
        flash("Username already exists", "error")
        return redirect(url_for("signup"))
    if len(password) < 8:
        flash("Password must be at least 8 characters long", "error")
        return redirect(url_for("signup"))
    
    try:
        db.insert(username, password)
        flash("Account created successfully, please login", "success")
        return redirect(url_for("login"))
    except Exception as e:
        flash("An error occurred", "error")
        print(f"Error creating account: {e}")
        return redirect(url_for("signup"))


@app.route("/login_account", methods=["POST"])
def login_account():
    username = request.form["username"]
    password = request.form["password"]
    db = get_db()
    user = db.login(username, password)
    
    if user:
        session['username'] = username
        return redirect(url_for("index"))
    else:
        flash("Incorrect username or password", "error")
        return redirect(url_for("login"))


@app.route("/logout")
@nocache
def logout():
    print("Session before logout:", session)
    session.clear()  # Clear all session data
    resp = redirect(url_for("login"))
    resp.set_cookie('session', '', expires=0)  # Clear the session cookie
    print("Cookies after clearing:", request.cookies)
    print("Session after logout:", session)
    flash("You have been logged out.", "success")
    return resp


@app.route('/signlang')
@login_required
@nocache
def signlang():
    return render_template('signlang.html')


@app.route('/textext')
@login_required
@nocache
def textext():
    return render_template('textext.html')


@app.route('/equation', methods=['GET', 'POST'])
@login_required
@nocache
def equation():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        response = equ_solver.solve_equation(file)
        return jsonify(response)

    return render_template('equation.html')


@app.route('/computertext', methods=['GET', 'POST'])
@login_required
@nocache
def service_two():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        extracted_text = Computertext.process_image(file_path)
        response = {"extracted_text": extracted_text}
        return jsonify(response)
    
    return render_template('textexctcomp.html')


@app.route('/handout', methods=['GET', 'POST'])
@login_required
@nocache
def handout():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        try:
            response = image_extract_api.get_text(file)
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('handout.html')


@app.route('/textsum', methods=['GET', 'POST'])
@login_required
@nocache
def textsum():
    if request.method == 'POST':
        text = request.form.get('text')
        summary = summarization.summarize_text(text)
        return jsonify(summary=summary)

    return render_template('textsum.html')


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
