from flask import Flask, render_template, request, jsonify
from services import equ_solver, Computertext,image_extract_api,summarization
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import io
import base64
from PIL import Image
import cv2
import numpy as np
from services.webcam_detect import sign_detection
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set upload folder

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


socketio = SocketIO(app, cors_allowed_origins="*")




@socketio.on('image')
def image(data_image):
    
    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    
    #Detection
    frame, letter, prediction_score = sign_detection(frame) 
                                
    
    frame = cv2.putText(frame, 'CV', (480,390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2, cv2.LINE_AA)
    
     # Encode the frame as base64 string
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    #Dictionary to be emitted
    info = {'frame': jpg_as_text, 'letter' : letter, 'prediction_score' : prediction_score}
    
    # Emit the frame data back to JavaScript client
    socketio.emit('processed_frame', info)

@app.route('/outputsign', methods=['POST', 'GET'])
def outputsign():
    return render_template('outputsign.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/signlang')
def signlang():
    return render_template('signlang.html')

@app.route('/textext')
def textext():
    return render_template('textext.html')

# @app.route('/textsum')
# def textsum():
#     return render_template('textsum.html')

@app.route('/equation', methods=['GET', 'POST'])
def equation():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        # Call the function from equ_solver
        response = equ_solver.solve_equation(file)
        return jsonify(response)

    return render_template('equation.html')

@app.route('/computertext', methods=['GET', 'POST'])
def service_two():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Call the OCR processing function
        extracted_text = Computertext.process_image(file_path)
        
        response = {"extracted_text": extracted_text}
        return jsonify(response)
    
    return render_template('textexctcomp.html')

@app.route('/handout', methods=['GET', 'POST'])
def handout():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        try:
            # Call the function from image_extract_api
            response = image_extract_api.get_text(file)
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('handout.html')

@app.route('/textsum', methods=['GET', 'POST'])
def textsum():
    if request.method == 'POST':
        text = request.form.get('text')
        summary = summarization.summarize_text(text)  # Replace with your summarization function
        return jsonify(summary=summary)

    return render_template('textsum.html')




if __name__ == '__main__':
   socketio.run(app,debug=True, port=5000)
