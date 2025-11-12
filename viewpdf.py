import os
import webbrowser
pdf_path = "E:\\karthik_drugs_recomendations\\mongo\\mongo\\output.pdf" 
webbrowser.open_new(f"file://{os.path.abspath(pdf_path)}")

os.system(pdf_path)
