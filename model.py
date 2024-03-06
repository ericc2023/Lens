# Math/deep learning libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path

import os

# Data processing
import pandas as pd
from torch.utils.data import random_split

# Data transformations
import torchvision
from torchvision import transforms
# from torchvision.transforms import v2

# Plotting
import matplotlib.pyplot as plt

# Other
import time
from datetime import date

import pypdfium2 as pdfium
import io
from PIL import Image
from io import BytesIO

# def convert_pdf_to_images(file_path, scale=300/72):
    
#     pdf_file = pdfium.PdfDocument(file_path)  
#     page_indices = [i for i in range(len(pdf_file))]
    
#     renderer = pdf_file.render(
#         pdfium.PdfBitmap.to_pil,
#         page_indices = page_indices, 
#         scale = scale,
#     )
    
#     list_final_images = [] 
    
#     for i, image in zip(page_indices, renderer):
        
#         image_byte_array = BytesIO()
#         image.save(image_byte_array, format='jpeg', optimize=True)
#         image_byte_array = image_byte_array.getvalue()
#         list_final_images.append(dict({i:image_byte_array}))
    
#     return list_final_images

def convert_pdf_to_images(pdf_path):
        # Convert your PDF to images
    images = convert_from_path(pdf_path)
    
    # Convert images to numpy arrays if required
    image_arrays = [np.array(image) for image in images]

    return image_arrays

def convert_pdf_to_numpy(pdf_path):
    image_array = convert_pdf_to_images(pdf_path)
    np_array = np.array(image_array)
    return np_array

pdf = np.array
pdfs_path = "data/pdf"

for p in os.listdir(pdfs_path):
    p = os.path.join(pdfs_path, p)
    new_pdf = convert_pdf_to_numpy(p)
    np.append(pdf, new_pdf)

class Model(nn.Module):
    def __init__(self):
        '''
        What do we want this model to look like?


        '''
        super(Model,self).__init__()

