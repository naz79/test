#
#
#

#-------------------------------------------


import streamlit as st
import torch

from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time


from pathlib import Path

import sys
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances

# import torch
# torch.__version__
import torchvision
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2.data.datasets import register_coco_instances

try:
    register_coco_instances("my_dataset_train", {}, "./data/_annotations.coco.json", "./data")
    
except:
    print("already registered")
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = 'cpu'
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500
predictor = DefaultPredictor(cfg)





## CFG
cfg_model_path = "models/yourModel.pt" 

cfg_enable_url_download = True
if cfg_enable_url_download:
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt" #Configure this if you set cfg_enable_url_download to True
    cfg_model_path = f"models/{url.split('/')[-1:]}" #config model path from url name








def imageInput(src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
#         cv2.imwrite("upload.png",image_file)
        print()
        col1, col2 = st.columns(2)
        if image_file is not None:
            with col1:
                img = Image.open(image_file)
                img.save("upload.png","PNG")
                st.image(img, caption='Selected Image', use_column_width='always')
            with col2:
            
                if image_file is not None :


                    os.system("python ./yolov7/detect.py --weights yolov7_best.pt --img 416 --conf 0.4 --source {}".format("upload.png"))
                    #--Display predicton
                    img_ = Image.open('./result_v7.png')
                    st.image(img_, caption='Plane Detection Yolov7')
                
            col3, col4 = st.columns(2)
            with col1:
                img = Image.open(image_file)
                st.image(img, caption='Selected Image', use_column_width='always')
            with col2:            
                if image_file is not None :

   
                    os.system("python ./yolov5/detect.py --weights Best.pt --img 416 --conf 0.4 --source {}".format("upload.png"))
                    img_ = Image.open("result.png")
                    st.image(img_, caption='Plane Detection Yolov5')
                                        
            # col5, col6 = st.columns(2)
            # with col5:
            #     img = Image.open(image_file)
            #     st.image(img, caption='Selected Image', use_column_width='always')
            # with col6:            
            #     if image_file is not None and submit:
            #         #--Display predicton
            #         os.system("python ./yolor/detect.py --weights yolor_p6.pt --img 416 --conf 0.4 --device cpu --cfg ./yolor/cfg/yolor_p6.cfg --names ./yolor/data/coco.names --source {}".format(image_file))
            #         img_ = Image.open("result_r.png")
            #         st.image(img_, caption='Plane Detection YoloR')    

            col7, col8 = st.columns(2)
            with col7:
                img = Image.open(image_file)
                st.image(img, caption='Selected Image', use_column_width='always')
            with col8:            
                if image_file is not None :

    
                    im=cv2.imread("upload.png")
                    outputs = predictor(im)
                    v = Visualizer(im[:, :, ::-1],
                            metadata=my_dataset_train_metadata, 
                            scale=0.8
                             )
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    st.image(out.get_image(), caption='Plane Detection Faster rcnn')
            
            

                
    elif src == 'From test set.': 
        
        # Image selector slider
        imgpath = glob.glob('.\pic\*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Detect")


        col3, col4 = st.columns(2)
        with col3:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col4:
            
            if image_file is not None and submit:
                
                
                os.system("python ./yolov7/detect.py --weights yolov7_best.pt --img 416 --conf 0.4 --source {}".format(image_file))
                #--Display predicton
                img_ = Image.open('./result_v7.png')
                st.image(img_, caption='Plane Detection Yolov7')
                    
                    
          
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:


                #--Display predicton
                os.system("python ./yolov5/detect.py --weights Best.pt --img 416 --conf 0.4 --source {}".format(image_file))
                img_ = Image.open("result.png")
                st.image(img_, caption='Plane Detection Yolov5')
               
        
        
        col7, col8 = st.columns(2)
        with col7:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col8:            
            if image_file is not None and submit:
                

                #--Display predicton
                im=cv2.imread(image_file)
                # im = cv2.imread(imageName)
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1],
                        metadata=my_dataset_train_metadata, 
                        scale=0.8
                         )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                st.image(out.get_image(), caption='Plane Detection Faster rcnn')

        col5, col6 = st.columns(2)
        with col5:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col6:            
            if image_file is not None and submit:
                
                
                #--Display predicton
                os.system("python ./yolor/detect.py --weights yolor_p6.pt --img 416 --conf 0.4 --device cpu --cfg ./yolor/cfg/yolor_p6.cfg --names ./yolor/data/coco.names --source {}".format(image_file))
                img_ = Image.open("result_r.png")
                st.image(img_, caption='Plane Detection YoloR')



# def main():
    # -- Sidebar
st.sidebar.title('⚙️Options')
datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])





st.header('Planes Detection')
# image = Image.open('Picture1.jpg')
# st.image(image, caption='plane',width=300,)
st.subheader('Select Data options.')


imageInput(datasrc)
    

    

# if __name__ == '__main__':
  
#     main()

