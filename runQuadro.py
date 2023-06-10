import numpy
from Quadro import Quadro

class QuadroCopter(Quadro):
    def __init__(self,conf_path: str):
       super().__init__(conf_path) 
       self.save_txt=True

    def save_labels(self,txtpath,cls,xywh,conf,txt_path,frame):
       #line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
       line = (frame,cls,conf,*xywh)  # label format
       with open(f'{txt_path}.txt', 'a') as f:
           f.write(('%g ' * len(line)).rstrip() % line + '\n')
   

def main():
    q=QuadroCopter('quadro.yaml')
    q.get_source()
    q.set_savedir()
    q.load_model()
    q.get_dataloader()
    q.run_inference()

if __name__ == "__main__":
    main()
    print('finish')
