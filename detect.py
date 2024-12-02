import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp382/weights/best.pt') # select your model.pt path
    model.predict(source='/home/ge107552201346/datasets/test/detect',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  conf=0.11,
                  # visualize=True # visualize model features maps
                )