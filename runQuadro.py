import sys
from pathlib import Path
import random
import os

import numpy
from Quadro import Quadro
import pandas as pd
import torch

random.seed(41)

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]


class QuadroCopter(Quadro):
    def __init__(self,conf_path: str):
       super().__init__(conf_path) 
       self.save_txt=True
       self.src=TEST_IMAGES_PATH,
       self.results=[]

    def save_labels(self,txtpath,cls,xywh,conf,txt_path,frame):
       xc = xywh[0]
       yc = xywh[1]
       w = xywh[2]
       h = xywh[3]
       score=conf.cpu().detach().numpy().tolist()
       label=cls.cpu().detach().numpy().tolist()
       label=int(label)
       image_id = os.path.basename(txt_path)
       xc = round(xc*10000)/10000
       yc = round(yc*10000)/10000
       w = round(w*100000)/10000
       h = round(h*10000)/10000
       score = round(score*10000)/10000
       print(xc,yc,w,h,image_id,label,score)
       result = {
          'image_id': image_id,
          'xc': xc,
          'yc': yc,
          'w': w,
          'h': h,
          'label': label,
          'score': score
       }
       self.results.append(result)


def main():
    q=QuadroCopter('/yolov5/lost.yaml')
    q.get_source()
    q.set_savedir()
    q.load_model()
    q.get_dataloader()
    q.run_inference()
    test_df = pd.DataFrame(q.results, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
    test_df.to_csv(SAVE_PATH, index=False)

if __name__ == '__main__':
    main()
