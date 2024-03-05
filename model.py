# Math/deep learning libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
from PIL import Image
from io import BytesIO



def convert_pdf_to_images(file_path, scale=300/72):
    
    pdf_file = pdfium.PdfDocument(file_path)  
    page_indices = [i for i in range(len(pdf_file))]
    
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices, 
        scale = scale,
    )
    
    list_final_images = [] 
    
    for i, image in zip(page_indices, renderer):
        
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(dict({i:image_byte_array}))
    
    return list_final_images

class Model(nn.Module):
    def __init__(self):
        '''
        What do we want this model to look like?


        '''
        super(Model,self).__init__()

