import os
import time
import mimetypes
from PIL import Image
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyAryw1N_4tfE3-EDreiC8dYHNSTZAc24Tg")

UPLOAD_FOLDER = './images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

# Function to upload a file and determine its MIME type
def upload_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")
    return genai.upload_file(file_path, mime_type=mime_type)

# Function to print response word by word
def print_word_by_word(text, delay=0.3):
    words = text.split()
    for word in words:
        print(word, end=' ', flush=True)
        time.sleep(delay)
    print()  # For a new line after the full response

def get_text(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        uploaded_file = upload_image(file_path)

        # Define the prompt and generation configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Create the model with the generation configuration
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        # Start a chat session and include both the image and the prompt
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [uploaded_file],
                },
            ]
        )

        # Send the prompt message and get the response
        response = chat_session.send_message("Extract the whole text")

        # Print the response from the model word by word
        # print_word_by_word(response.text)

        return {"response": response.text}

    except Exception as e:
        print(f"Error processing file: {e}")
        return {"error": "An error occurred while processing the file."}

    finally:
        # Ensure file is removed after processing
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
