import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    model.val(data='dataset/weed.yaml',
    # model.val(data='dataset/4weed.yaml',
    # model.val(data='dataset/cotton12.yaml',
              split='test',
              imgsz=640,
              batch=8,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )