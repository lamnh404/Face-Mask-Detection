import shutil
import os
from glob import iglob
import random
import xml.etree.ElementTree as ET


SOURCE_DIR= "face_data"
PATH= 'datasets'


def process_data():
    if os.path.isdir(PATH):
        shutil.rmtree(PATH)
    os.makedirs(PATH)
    images_path=os.path.join(PATH,'images')
    labels_path=os.path.join(PATH,'labels')
    os.makedirs(images_path)
    os.makedirs(labels_path)
    os.makedirs(os.path.join(images_path,'train'))
    os.makedirs(os.path.join(images_path,'val'))
    os.makedirs(os.path.join(labels_path,'train'))
    os.makedirs(os.path.join(labels_path,'val'))
    file_path=list(iglob(os.path.join(SOURCE_DIR,'images','*.png')))
    file_name= [file_name.replace('.png','').split('/')[-1] for file_name in file_path]

    random.seed(42)
    random.shuffle(file_name)
    train_file_name=file_name[:int(len(file_name)*0.8)]
    val_file_name=file_name[int(len(file_name)*0.8):]
    category_list=['with_mask','without_mask','mask_weared_incorrect']
    for file_name in train_file_name:
        shutil.copy(os.path.join(SOURCE_DIR,'images',file_name+'.png'),os.path.join(images_path , 'train',file_name+'.png'))
        try:
            xml_path = os.path.join(SOURCE_DIR, 'annotations', file_name + '.xml')
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size= root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in root.iter('object'):
                    category = obj.find('name').text
                    bbox=obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    _width = (xmax - xmin) / width
                    _height = (ymax - ymin) / height
                    with open(os.path.join(labels_path,'train',file_name+'.txt'),'a') as f:
                        f.write(f"{category_list.index(category)} {x_center:.6f} {y_center:.6f} {_width:.6f} {_height:.6f}\n")
            else:
                print(f"XML file not found: {xml_path}")
        except Exception as e:
            print(f"Error processing XML for {file_name}: {e}")
    for file_name in val_file_name:
        shutil.copy(os.path.join(SOURCE_DIR,'images',file_name+'.png'),os.path.join(images_path , 'val',file_name+'.png'))
        try:
            xml_path = os.path.join(SOURCE_DIR, 'annotations', file_name + '.xml')
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size= root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                for obj in root.iter('object'):
                    category = obj.find('name').text
                    bbox=obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    _width = (xmax - xmin) / width
                    _height = (ymax - ymin) / height
                    with open(os.path.join(labels_path,'val',file_name+'.txt'),'a') as f:
                        f.write(f"{category_list.index(category)} {x_center:.6f} {y_center:.6f} {_width:.6f} {_height:.6f}\n")
            else:
                print(f"XML file not found: {xml_path}")
        except Exception as e:
            print(f"Error processing XML for {file_name}: {e}")