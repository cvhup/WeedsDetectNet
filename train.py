import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/WeedsDetectNet/WeedsDetectNetn-weed.yaml')
    # model = YOLO('ultralytics/cfg/models/WeedsDetectNet/WeedsDetectNetn-cotton12.yaml')
    # model = YOLO('ultralytics/cfg/models/WeedsDetectNet/WeedsDetectNetn-4weed.yaml')

    model.train(data='dataset/weed.yaml',
    # model.train(data='dataset/cotton12.yaml',
    # model.train(data='dataset/4weed.yaml',
                # cache=False,
                cache=True,
                imgsz=640,
                epochs=500,
                batch=8,
                close_mosaic=10,
                workers=2,
                device='5',
                optimizer='SGD', # using SGD
                # resume='/runs/train/exp/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )