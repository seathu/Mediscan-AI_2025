
import numpy as np
import cv2
# import cv2
import pytesseract


def preprocess_image(img):    
    img = cv2.imread(img) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    processed_image = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        61,
        11
    )   
    return processed_image

def get_text_from_url(URL):        
    # URL = URL[1:]    
    # below URL path might change according to your installed path of tesseract
    pytesseract.pytesseract.tesseract_cmd = r'E:\text_extract\tesseract.exe'
    image = cv2.imread(URL)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    text = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
    return text

def testing(URL):
    img = preprocess_image(URL)
    text = pytesseract.image_to_string(img, lang='eng',config='--psm 6')

    return text


# image_url = "E:\\karthik_drugs_recomendations\\mongo\\images\\prescriptions\\1.jpg"
image_url = "E:\\karthik_drugs_recomendations\\mongo\\mongo\\new0.jpg"
print(get_text_from_url(image_url))
print("----------------------")
print(testing(image_url))
print("----------------------")
import pytesseract
from PIL import Image
print("----------------------")

pytesseract.pytesseract.tesseract_cmd = r'E:\text_extract\tesseract.exe'
image = "E:\\karthik_drugs_recomendations\\mongo\\mongo\\new0.jpg"
text = pytesseract.image_to_string(image,lang='eng',config='--psm 6')
print(text)
print("----------------------")
# image = image.convert('L')
# image = image.resize((800, 600))
# image = image.point(lambda p: p > 180 and 255)
# text = pytesseract.image_to_string(image)
# print(text)