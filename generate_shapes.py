import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import zipfile
import io
import datetime
import math

OUTPUT_ZIP_FILE = "images_28x28.zip"
LABEL_ZIP_FILE = "labels_28x28.zip"
CONTRAST_THRESHOLD = 120 #limiar mínimo de contraste, o máximo é 441 aprox

def color_distance(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)

def random_color(brilho_min = 150):
    return tuple(np.random.randint(0, 256, size=3))

def generate_random_shape_params(img_size):
    """
    Generates random parameters for a single shape.
    
    Returns:
        shape_type (str): "rectangle" or "ellipse"
        x0, y0, x1, y1 (int): coordinates of the bounding box (integers)
        color (tuple): RGB color
    """
    shape_type = random.choice(["rectangle", "ellipse"])
    x0, y0 = np.random.randint(0, img_size - 1 - 2, size=2)
    x1, y1 = np.random.randint(2, img_size - 1, size=2)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    if x1 == x0:
        x0-=1
        x1+=1

    if y1 == y0:
        y0 -= 1
        y1 += 1
    
    if x1 - x0 == 1:
        if random.choice([0,1]) == 0:
            x0 -= 1
        else:
            x1 += 1
            
    if y1 - y0 == 1:
        if random.choice([0,1]) == 0:
            y0 -= 1
        else:
            y1 += 1

    color = random_color()
    return shape_type, x0, y0, x1, y1, color

def draw_shape(draw, shape_type, x0, y0, x1, y1, color):
    """Draws a shape (rectangle or ellipse) on the given draw object."""
    if shape_type == "rectangle":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:
        draw.ellipse([x0, y0, x1, y1], fill=color)

def generate_and_add_to_zip(i, images_zip, labels_zip, img_size):
    """
    Generates one image with a single random shape and its corresponding label file.
    
    The label file format is:
        type x0 y0 x1 y1 R G B
        
    Where:
        - type: 0 = rectangle, 1 = ellipse
        - x0, y0, x1, y1: bounding box coordinates (integers)
        - R G B: color values (0–255)
    """
    background_color = random_color()
    shape_color = random_color()

    while color_distance(background_color, shape_color) < CONTRAST_THRESHOLD:
        shape_color = random_color()

    shape_type, x0, y0, x1, y1, color = generate_random_shape_params(img_size)
    img = Image.new("RGB", (img_size, img_size), background_color)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape_type, x0, y0, x1, y1, shape_color)
    
    img_buff = io.BytesIO()
    img.save(img_buff, format='PNG')
    img_filename = f"{i:06}.png"
    images_zip.writestr(img_filename, img_buff.getvalue())

    shape_id = 0 if shape_type == "rectangle" else 1
    label_content = f"{shape_id} {x0} {y0} {x1} {y1} {shape_color[0]} {shape_color[1]} {shape_color[2]}\n"
    label_filename = f"{i:06}.txt"
    labels_zip.writestr(label_filename, label_content)

def generate_data(num_images, img_size):
    curr_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")

    new_dir = f"dataset_{curr_time}_{num_images}-imgs"

    os.makedirs(new_dir, exist_ok=True)

    output_images_path = os.path.join(new_dir, OUTPUT_ZIP_FILE)
    output_labels_path = os.path.join(new_dir, LABEL_ZIP_FILE)

    with zipfile.ZipFile(output_images_path, 'w', zipfile.ZIP_DEFLATED) as images_zip, \
        zipfile.ZipFile(output_labels_path, 'w', zipfile.ZIP_DEFLATED) as labels_zip:
        for i in tqdm(range(num_images), desc="Gerando imagens"):
            generate_and_add_to_zip(i, images_zip, labels_zip, img_size)
        
    return new_dir

#if __name__ == "__main__":
#    generate_data(num_images=100, img_size=28)