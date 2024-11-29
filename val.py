import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/root/autodl-fs/runs/comparisontrain/rtdetr-r50-m-IRSTD-1k-epoch100-b8-w4/weights/best.pt')
    model.val(data='/root/autodl-fs/IRSTD-1k/data.yaml',
              split='val',
              imgsz=640,
              batch=8,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )