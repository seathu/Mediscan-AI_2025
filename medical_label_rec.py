# pip install torch torchvision torchaudio
# pip install easyocr
# pip install python-Levenshtein

from flask import Flask ,Response,render_template,url_for,request
import pymongo
import json
import numpy as np
from bson.objectid import ObjectId
import datetime
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import Flask ,Response,render_template,url_for,request,send_from_directory,url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from wtforms import Form, BooleanField, StringField, PasswordField, validators
##########
import cv2 as cv
import pytesseract
from PIL import Image
import sys  # to access the system
import pandas as pd
import cv2
import pytz.tzinfo
#############
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
#######################
from PIL import Image 
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel 
# import os
import requests
import time
from fpdf import FPDF
import ctypes  # An included library with Python install.   
from difflib import SequenceMatcher
from waitress import serve

import shutil

import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

from fuzzywuzzy import fuzz

import webbrowser
from more_itertools import locate

app = Flask(__name__)
app.config['SECRET_KEY'] = "asldfkjlj"
app.config['UPLOADED_PHOTOS_DEST'] = 'mongo'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

# resizing of image before cropping starts
def get_new_valuenwdt(value):
    if value <= 650:
        height = value
    elif value > 650 and value <= 800:
        height = value * 0.8
    elif value >= 800 and value <= 1000:
        height = value * 0.5
    elif value >= 1000 and value <= 1200:
        height = value * 0.65
    elif value >= 1200 and value <= 1800:
        height = 1000
    else:
        height = value * 0.3
    return int(height)
# resiging of image before cropping ends.


#### crop code starts
class MouseCrop:
    def __init__(self, image):
        print('in main image showing')
        # new_size = (get_new_valuenwdt(self.image.shape[0]), get_new_valuenwdt(self.image.shape[1]))     
        self.image = image
        print(self.image.shape)
        new_size = (get_new_valuenwdt(self.image.shape[0]), get_new_valuenwdt(self.image.shape[1]))        
        print(new_size)        
        print(str(self.image.shape[0])+" : "+str(self.image.shape[1]))
        self.image = cv2.resize(self.image, new_size)
        self.coordinates = []
        self.width = 0
        self.height = 0
        self.cropping = False
    def handle_clicks(self, event, x, y, flags, params):
        # below print statement occurs multiple times          
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coordinates = [(x, y)]
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.width = abs(x - self.coordinates[0][0])
            self.height = abs(y - self.coordinates[0][1])
        elif event == cv2.EVENT_LBUTTONUP and self.cropping:
            self.coordinates.append((x, y))
            self.cropping = False
            self.width = abs(self.coordinates[1][0] - self.coordinates[0][0])
            self.height = abs(self.coordinates[1][1] - self.coordinates[0][1])
            cv2.rectangle(self.image, self.coordinates[0], self.coordinates[1], (0, 255, 0), 2)
            cv2.putText(self.image, f"Width: {self.width}px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(self.image, f"Height: {self.height}px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Image', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.crop_image()
    def crop_image(self):
        print('in crop image showing')
        # print(self.image.shape)  
        if self.width > 0 and self.height > 0:
            cropped_image = self.image[self.coordinates[0][1]: self.coordinates[0][1] + self.height,
                                       self.coordinates[0][0]: self.coordinates[0][0] + self.width]
            print(cropped_image.shape)
            # cropped_image = cv2.resize(cropped_image, (211,88))
            cv2.imshow("Cropped Image", cropped_image)
            cv2.imwrite('mongo/current.jpg', cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # print('in enhanced image showing')
            # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # cropped_image_sharpening = cv2.filter2D(cropped_image, -1, kernel)
            # cv2.imshow("Cropped Image after enhance", cropped_image_sharpening)
            # cv2.imwrite('mongo/current_enhance.jpg', cropped_image_sharpening)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(cropped_image_sharpening.shape)

        else:
            print("Please select the image region to crop")
    def show_image(self):  
        print('in sub image showing')
        print(self.image.shape)   
        cv2.imshow("Image", self.image)
        cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback("Image", self.handle_clicks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#### crop code ends

def get_text_from_url(URL):        
    URL = URL[1:]    
    # below URL path might change according to your installed path of tesseract
    # C:\Program Files\Tesseract-OCR\
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = cv2.imread(URL)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    text = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    return text

# splits the cropped image line by line as sub images
def split_image_line_by_line(IMG_URL, rows_count):   
    IMG_URL = IMG_URL[1:]        
    image = cv2.imread(IMG_URL)
    if image is None:
        print('image not found')
    else:
        # print(image.shape)
        print('')
    rows = rows_count
    height, width, _ = image.shape
    split_height = height // rows
    split_width = width // 1
    for row in range(rows):
        for col in range(1):
            start_y = row * split_height
            end_y = (row + 1) * split_height
            start_x = col * split_width
            end_x = (col + 1) * split_width        
            cropped_image = image[start_y:end_y, start_x:end_x]
            path = "mongo/new" + str(row) + ".jpg"
            cv2.imwrite(path, cropped_image)
            # image sharpening code starts...
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            cropped_image = cv2.filter2D(cropped_image, -1, kernel)
            # image sharpening code ends..
            path = "mongo/cropeed" + str(row) + ".jpg"                  
            cv2.imwrite(path, cropped_image)
    return ""

# get all newly generated split images paths
def get_split_image_paths():
    files_with_prefix = []
    for root, dirs, files in os.walk("mongo"):
        for file in files:
            if file.startswith("new"):
                files_with_prefix.append(os.path.join(root, file))
    return files_with_prefix

# get text from split image using trOCR with good accuracy
def run_trOCR(model_name="microsoft/trocr-base-printed", images="",path=""):    
    processor = TrOCRProcessor.from_pretrained(model_name,use_fast=True)    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    print(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    model.to(device)  # Move model to GPU
    pixel_values = processor(path, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=1000)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]       
    return generated_text

# creats the final pdf to end user
def createPdf(ftxt):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 255)
    pdf.multi_cell(w=0, h=10, txt=ftxt, border=1, align='C')  # w=0 means full width
    pdf.output("mongo/output.pdf")
    return ""   

# for dataset reading and retrieving needed columns starts...
def get_csv_df(file_path):
    df = pd.read_csv(file_path)
    return df

def get_csv_column_names_asList(file_path):    
    df = pd.read_csv(file_path)    
    return df.columns.tolist()

def get_column_values_asList(file_path, col_name):
    col_values = []
    df = get_csv_df(file_path)
    col_values = df[col_name]
    col_values = list(col_values)
    return col_values
# for dataset reading and retrieving needed columns ends...

# retrives percentage of matching between 2 strings starts..
def cleaned_string(mystring):
    level1 = re.sub(r'[^a-zA-Z0-9]', '', mystring)
    final_cleaned_string = ''.join([i for i in level1 if not i.isdigit()])
    final_cleaned_string = final_cleaned_string.lower()
    return final_cleaned_string

def get_sim_percentage_bw_2strings(str1, str2):
    value = fuzz.ratio(cleaned_string(str1), str2(string2))
    return value
# retrives percentage of matching between 2 strings ends..


# functions for uname ,pwd checks and adding new users starts...
def check_exising_user(uname):
    status = ''
    all_unames_pwds = []    
    file_path = "users.txt"    
    with open(file_path, 'r') as file:
        for line in file:
            if line.endswith("\n"):                
                line = line.replace("\n", "")
                all_unames_pwds.append(line)
            else:                
                all_unames_pwds.append(line)
    for user in all_unames_pwds:
        if (user.startswith(uname)):
            status = 'y'
            break
        else:
            status = 'n'
    return status

def add_user(uname, pwd):
    status = ""   
    file_path = "users.txt"   
    if (check_exising_user(uname) == 'n'):
        to_append = "\n"+ uname + " " + pwd
        with open(file_path, "a") as file:
            file.write(to_append)
            status = "yes"
    else:
        status = "no"
    return status

def ret_pwd(uname):
    pwd = ""
    all_unames_pwds = []    
    file_path = "users.txt"
    with open(file_path, 'r') as file:
        for line in file:
            if line.endswith("\n"):                
                line = line.replace("\n", "")
                all_unames_pwds.append(line)
            else:                
                all_unames_pwds.append(line)        
    for user in all_unames_pwds:
        if (user.startswith(uname)):
            index = user.index(" ")
            pwd = user[index + 1:len(user)]            
    return pwd

def get_uid_pwd_status(uname,password):
    status = ""
    pwd = ""
    further = check_exising_user(uname)
    if (further == 'n'):
        status = 'n'
    else:
        pwd = ret_pwd(uname)
        if (pwd == password):
            status = 'y'  
    return status
# functions for uname ,pwd checks and adding new users ends...

class UploadForm(FlaskForm):    
    photo = FileField(
        validators=[
            FileAllowed(photos, 'only images allowed'),
            FileRequired('FileField should not be empty')
        ]
    )
    submit = SubmitField('Display Selected Image for cropping,extraction,analysis and recommendations.....')

class UploadForm1(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email Address', [validators.Length(min=6, max=35)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    submit = SubmitField('analysis')


@app.route('/mongo/<filename>')
def get_file(filename):    
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],filename)

@app.route("/",methods = ['GET', 'POST'])
# def home():    
#     fpath = "/mongo/logo.jpg"    
#     return render_template("welcome_medical_label_rec.html",fpath=fpath)
def home():    
    fpath = "/mongo/logo.jpg"    
    return render_template("home_medical_label_rec.html",fpath=fpath)

@app.route("/login_au", methods=['GET', 'POST'])
def login_au():
    status = 'y'
    output = request.form.to_dict()    
    uname = output["uname"] 
    pword = output["pword"]    
    fpath = "/mongo/logo.jpg"   
    if (uname == "admin" and pword == "admin"):
        return render_template("success_medical_label_rec.html",uname=uname,pword=pword,fpath=fpath)
    else:
        normal_user_status = get_uid_pwd_status(uname, pword)
        if(normal_user_status == 'n'):
            return render_template("unsuccess_medical_label_rec.html",uname=uname,pword=pword,fpath=fpath)
        else:
            return render_template("success_medical_label_rec.html",uname=uname,pword=pword,fpath=fpath)


@app.route("/ucreation", methods=['GET', 'POST'])
def ucreation():
    return render_template("create_medical_label_rec.html")

@app.route("/registration", methods=['GET', 'POST'])
def registration():    
    output = request.form.to_dict()
    fname = output["fname"]    
    lname = output["lname"]
    gender = output["gender"]    
    ctime = datetime.datetime.now()
    user = {"fname": fname, "lName": lname ,"gender": gender, "ctime": ctime}   
    fpath = "/mongo/logo.jpg"   
    status = add_user(fname, lname)
    if(status == 'yes'):
        return render_template('create_medical_label_rec.html', fname=fname, lname=lname, gender=gender, ctime=ctime)
    else:
        return render_template("unsuccess_reg_medical_label_rec.html", fname=fname, lname=lname, gender=gender, ctime=ctime,fpath=fpath)


@app.route("/admin",methods = ['GET', 'POST'])
def alog():
    return render_template("admin_medical_label_rec.html")

@app.route("/alogin",methods = ['GET', 'POST'])
def alogin():
    output = request.form.to_dict()    
    uname = output["uname"] 
    pword = output["pword"]    
    if (uname == "admin" and pword == "admin"):
        return render_template("success.html",uname=uname,pword=pword)
    else: 
        return render_template("unsuccess.html")


@app.route("/create",methods = ['GET', 'POST'])
def create_user():
    return render_template("create_medical_label_rec.html")

@app.route('/create_login',methods=['POST', 'GET'])
def create_login():   
    output = request.form.to_dict()    
    fname = output["fname"] 
    lname = output["lname"]
    gender = output["gender"]    
    ctime = datetime.datetime.now()
    user = {"fname": fname, "lName": lname ,"gender": gender, "ctime": ctime}   

    status = add_user(fname, lname)
    if(status == 'yes'):
        return render_template('create_medical_label_rec.html', fname=fname, lname=lname, gender=gender, ctime=ctime)
    else:
        return render_template("unsuccess.html")
    


@app.route("/simage",methods = ['GET', 'POST'])
def simage():
    folder_path = r"mongo"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)        
        if os.path.isfile(file_path):
            if "logo" in file_path or "background" in file_path:
                print('')
            else:
                os.remove(file_path)      
    form = UploadForm()
    form1 = UploadForm1(request.form)
    if form1.validate_on_submit():
        # print('second form validate')
        print('')
    else:
        # print('not validate')
        print('')
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)              
        to_crop_url = file_url[1:]        
        image = cv2.imread(to_crop_url)
        if image is not None:
            mouse_click_handler = MouseCrop(image)
            mouse_click_handler.show_image()
        else:
            print("Could not load image")        
        sel_file_path = file_url        
        text = get_text_from_url(file_url)        
        data1=[
            {
                'file_name':file_url,
                'text_text': text                
            }
        ]        
        file1 = open("myfile.txt", "w")
        file1.write(text)
        file2 = open("selfile.txt", "w")
        file2.write(sel_file_path)
        file3 = open("selfile1.txt", "w")
        file3.write("/mongo/current.jpg")
        cropped_text = get_text_from_url("/mongo/current.jpg")        
        file4 = open("croptextfile.txt", "w")
        file4.write(cropped_text)
        
    else:
        file_url = None
    return render_template('upload_medical_label_rec.html', form=form, file_name=file_url)    

@app.route("/analyze",methods = ['GET', 'POST'])
def analyze():
    ext_text = ""
    all_extract_times = []  
    text = ""
    text1 = ""
    main_text = []
    crop_text1 = []
    delimiter = "\r\n"
    with open('myfile.txt', 'r') as file:
        for line in file:
            text += line + "\r\n"
            line = line.translate({ord('\n'): None, ord('\r'): None})
            main_text.append(line)     
    main_text = list(filter(None, main_text))    
    f = open("selfile.txt", "r+")
    sel_file_path = f.readline()
    f1 = open("selfile1.txt", "r+")
    sel_file_path1 = f1.readline()
    crop_text = get_text_from_url(sel_file_path1)

    with open('croptextfile.txt', 'r') as file1:
        for line1 in file1:
            line1 = line1.translate({ord('\n'): None, ord('\r'): None})            
            crop_text1.append(line1)
    split_image_line_by_line(sel_file_path1,len(crop_text1))
    # below line to remove empty strings from list
    crop_text1 = [item for item in crop_text1 if item]
    split_files = []
    split_files = get_split_image_paths()
    ctypes.windll.user32.MessageBoxExW(None, str(split_files), "Total split files", 1)
    split_files_count = len(split_files)
    count = 1
    print("total split files/images count is: "+str(split_files_count))
    drugs_extracted = []
    splitter_img_paths = []
    start_time = time.time()
    output_text = "Extracted Medicines from prescriptions: \n"
    
    for split_path in split_files:
        s_time = time.time()     
        temp_path = "/" + split_path
        temp_path = temp_path.replace('\\',' / ')
        temp_path = temp_path.replace(' ','')
        splitter_img_paths.append(temp_path)
        model_id = "microsoft/trocr-large-handwritten"  # indus tre, This is a sample of text
        IMG=Image.open(split_path).convert("RGB")
        print("Split file/image:"+str(count)+" to text extractions starts....")       
        txt_extracted=run_trOCR(model_id, IMG, IMG)
        print("Split file/image:"+str(count)+" to text extractions ends....")
        count = count + 1  
        output_text += txt_extracted+"\n"
        drugs_extracted.append(txt_extracted)
        ext_text += txt_extracted+"\n"
        e_time=time.time()
        all_extract_times.append(e_time - s_time)
    # 27th code evening starts..
    data = {
        'Category': split_files,
        'Values': all_extract_times
    }
    df=pd.DataFrame(data)
    bar_chart = go.Figure(data=[
        go.Bar(x=df['Category'], y=df['Values'], marker_color='blue')
    ])
    bar_chart.update_layout(title='MLRR', xaxis_title='Category', yaxis_title='Values')
    chart_html = pio.to_html(bar_chart, full_html=False)
    # 27th code evening ends..
    file_text = open("exttext.txt", "w")
    file_text.write(ext_text)

    elapsed_time="total medicine extracted time wrt images is(in seconds): "+str(time.time() - start_time)    
    output_text += "\n\n Related Medicines & Suggesions:\n"
    # createPdf(output_text)

    
    # below 2 lines to display pdf in webbroser
    # pdf_path = "mongo/output.pdf" 
    # webbrowser.open_new(f"file://{os.path.abspath(pdf_path)}")
    # os.system(pdf_path)
    return render_template('extract_medical_label_rec.html',text=text,crop_text1=crop_text1,sel_file_path1=sel_file_path1,drugs_extracted=drugs_extracted,splitter_img_paths=splitter_img_paths,elapsed_time=elapsed_time,chart=chart_html)

@app.route("/predictions", methods=['GET', 'POST'])
def predictions():
    drug_names = []
    reasons = []
    descriptions = []
    drug_names = openCSV('Drug_Name')
    reasons = openCSV('Reason')
    descriptions = openCSV('Description')    
    similar_metrics_drugs = []
    similar_metrics_drugs_indexes = []
    for drug in drug_names:
        res = SequenceMatcher(None, drug, 'Alphamix').ratio()
        if (res > 0.5):
            similar_metrics_drugs.append(res)
            similar_metrics_drugs_indexes.append(drug_names.index(drug))
    # print(similar_metrics_drugs)
    # print(similar_metrics_drugs_indexes)
    return render_template("recs_medical_label_rec.html")

def get_similarity_drug_from_main_list(dName, mList):
    sim_percentage = []
    for item in mList:
        ratio_perfect = fuzz.ratio(cleaned_string(dName), cleaned_string(item))
        sim_percentage.append(ratio_perfect)
    return sim_percentage

def get_similarity_drug_from_main_list_max_indexes(max, valueList):
    index = 0
    all_indexes = []
    for value in valueList:
        if (value == max):
            all_indexes.append(index)
            index = index + 1
        else:
            index = index + 1
    return all_indexes

def get_significant_drug_max_match_index(dnames_cleaned ,maxIndexes, dName):
    max_index = 0
    # print(len(dnames_cleaned))
    # print(len(maxIndexes))
    first_three = dName[:2]
    for index in maxIndexes:
        drugName_at_index = dnames_cleaned[index]
        # print(drugName_at_index+"   "+first_three+" "+str(index)+" "+dName)   
        # if (drugName_at_index.contains(first_three)):
        if first_three in drugName_at_index:
            # print(drugName_at_index+"   "+first_three+" "+str(index)+" "+dName)
            max_index = index
    return max_index


@app.route("/frecommendations", methods=['GET', 'POST'])
def frecommendations():
    html_text = []
    crop_text = []
    cleaned_text = []
    fpath = "/mongo/logo.jpg"
    with open('exttext.txt', 'r') as file1:
        for line1 in file1:
            line1 = line1.translate({ord('\n'): None, ord('\r'): None})            
            crop_text.append(line1)
    # print(crop_text)
    for item in crop_text:
        cleaned_one = cleaned_string(item)
        cleaned_text.append(cleaned_one)
    # print("------")
    # print(cleaned_text)
    # final recommendations starts...
    data_file_name = 'Drug_Data.csv'
    drug_names = []
    prescribed_for = []
    drug_names_cleaned = []
    prescribed_for_cleaned = []
    matching_percentages = []
    drug_names = get_column_values_asList(data_file_name, 'drugName')
    prescribed_for = get_column_values_asList(data_file_name, 'Prescribed_for')
    for drug in drug_names:         
        cl_one = cleaned_string(drug)        
        drug_names_cleaned.append(cl_one)
    for presc in prescribed_for:
        cl_two = cleaned_string(str(presc))        
        prescribed_for_cleaned.append(cl_two)
    # matching_percentages1 = get_similarity_drug_from_main_list(cleaned_text[0], drug_names_cleaned)
    # matching_percentages2 = get_similarity_drug_from_main_list(cleaned_text[1], drug_names_cleaned)
    # matching_percentages3 = get_similarity_drug_from_main_list(cleaned_text[2], drug_names_cleaned)
    # print(matching_percentages)
    # print(max(matching_percentages1))
    # print(max(matching_percentages2))
    # print(max(matching_percentages3))
    # print(get_similarity_drug_from_main_list_max_indexes(max(matching_percentages1),matching_percentages1))
    # print(get_similarity_drug_from_main_list_max_indexes(max(matching_percentages2),matching_percentages2))
    # print(get_similarity_drug_from_main_list_max_indexes(max(matching_percentages3),matching_percentages3))

    output_text = "Extracted Medicines from prescriptions: \n"
    html_text.append("Extracted Medicines from prescriptions: ")

    results_prescribed = []
    
    ##############################
    for dName in cleaned_text:        
        max_match_percentage = get_similarity_drug_from_main_list(dName, drug_names_cleaned)
        max_match_percentage_of_drug = max(max_match_percentage)
        # print(dName)  
        # print(dName + " and max mathing percentage is " + str(max_match_percentage_of_drug))
        matching_percentage_indexes = get_similarity_drug_from_main_list_max_indexes(max_match_percentage_of_drug, max_match_percentage)
        # print(matching_percentage_indexes)
        max_significant_index = get_significant_drug_max_match_index(drug_names_cleaned, matching_percentage_indexes, dName)
        # print("is " + str(max_significant_index))   
        # print(prescribed_for[max_significant_index])

        if (max_significant_index > 0):
            output_text += drug_names[max_significant_index] + ":    "
            html_text.append(drug_names[max_significant_index] + ":    " +prescribed_for[max_significant_index])
            output_text += prescribed_for[max_significant_index] + "\n"
            # html_text.append(prescribed_for[max_significant_index])
            results_prescribed.append(prescribed_for[max_significant_index])

    output_text += "\n\nRelated Medicines & Suggesions:\n"
    html_text.append("Related Medicines & Suggesions:")
    # print(output_text)
    # createPdf(output_text)

    # print(results_prescribed)

    pres_sugg_indexes = []
    if (len(results_prescribed) > 0):
        for pres in results_prescribed:
            pres_sugg_indexes = list(locate(get_column_values_asList(data_file_name, 'Prescribed_for'), lambda x: x == pres))
            if (len(pres_sugg_indexes) > 2):
                # print('yes')
                output_text += drug_names[pres_sugg_indexes[0]] + ":    "
                html_text.append(drug_names[pres_sugg_indexes[0]] + ":    "+prescribed_for[pres_sugg_indexes[0]])
                output_text += prescribed_for[pres_sugg_indexes[0]] + "\n"
                # html_text.append(prescribed_for[pres_sugg_indexes[0]])
                output_text += drug_names[pres_sugg_indexes[1]] + ":    "
                html_text.append(drug_names[pres_sugg_indexes[1]] + ":    "+prescribed_for[pres_sugg_indexes[1]])
                output_text += prescribed_for[pres_sugg_indexes[1]] + "\n"
                # html_text.append(prescribed_for[pres_sugg_indexes[1]])
                output_text += "\n\n"


    createPdf(output_text)
    pdf_path = "mongo/output.pdf"
    webbrowser.open_new(f"file://{os.path.abspath(pdf_path)}")
    output_text = output_text.replace("\n", "<br>")


    return render_template('recommendations_medical_label_rec.html',fpath=fpath,crop_text=crop_text,cleaned_text=cleaned_text,html_text=html_text)

 ############
 # create user code goes in here
 ############   

if __name__ == "__main__":   
    app.run(debug=True) # Run the development server

# if __name__ == "__main__":
#     serve(app, host="0.0.0.0", port=8080)
