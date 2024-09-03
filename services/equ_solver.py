import mimetypes
import os
import time  # Import time for sleep function
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyAryw1N_4tfE3-EDreiC8dYHNSTZAc24Tg")

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

def upload_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")
    return genai.upload_file(file_path, mime_type=mime_type)

def solve_equation(file):
    # Save the file to the upload folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        uploaded_file = upload_image(file_path)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                },
            ]
        )

        response = chat_session.send_message("solve this step by step")

        # Clean up the uploaded file
        os.remove(file_path)

        return {"response": response.text}

    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": "An error occurred while processing the file."}
