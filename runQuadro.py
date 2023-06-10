import sys
from pathlib import Path
import random

import numpy
from Quadro import Quadro
import pandas as pd

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
       image_id = txtpath

       result = {
          'image_id': image_id,
          'xc': round(xc, 4),
          'yc': round(yc, 4),
          'w': round(w, 4),
          'h': round(h, 4),
          'label': 0,
          'score': round(conf, 4)
       }
       self.results.append(result)


def main():
    q=QuadroCopter('quadro.yaml')
    q.get_source()
    q.set_savedir()
    q.load_model()
    q.get_dataloader()
    q.run_inference()
    test_df = pd.DataFrame(q.results, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
    test_df.to_csv(SAVE_PATH, index=False)


if __name__ == '__main__':
    main()
