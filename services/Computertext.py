import cv2
import easyocr
import os
from matplotlib import pyplot as plt


def create_path(img):
    ''' Creates an absolute path for the image file. '''
    return os.path.join(os.getcwd(), img)


def recognize_text(img_path):
    ''' Loads an image and recognizes text. '''
    reader = easyocr.Reader(['hi', 'en']) 
    return reader.readtext(img_path)


def overlay_ocr_text(img_path, save_name):
    ''' Loads an image, recognizes text, and overlays the text on the image. '''
    
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Recognize text
    result = recognize_text(img_path)
    
    # If OCR prob is over 0.2, overlay bounding box and text
    for (bbox, text, prob) in result:
        if prob >= 0.2:
            print(f'Detected text: {text} (Probability: {prob:.2f})')
            
            # Get top-left and bottom-right bbox vertices
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            # Create a rectangle for bbox display
            cv2.rectangle(img, top_left, bottom_right, color=(255, 0, 0), thickness=2)
            
            # Put recognized text
            cv2.putText(img, text, (top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)

    # # Save the image with overlaid text
    # plt.imshow(img)
    # plt.axis('off')
    # plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    # plt.close()


# def ocr_text(img_path):
#     ''' Extracts text from an image and prints it as a paragraph. '''
#     result = recognize_text(img_path)
    
#     # Concatenate all the recognized text into a single string
#     extracted_text = ""
#     for (bbox, text, prob) in result:
#         if prob:
#             extracted_text += text + " "
    
#     print(extracted_text.strip())  # Print the extracted text


def process_image(image_path):
    ''' Process image and return OCR result as a space-separated string of words. '''
    result = recognize_text(image_path)

    # Extract words from OCR result
    words = []
    for (bbox, text, prob) in result:
        if prob:
            words.extend(text.split())  # Split text into words

    # Join words with spaces to maintain proper spacing
    return ' '.join(words)  # Convert list of words into a single string with spaces
