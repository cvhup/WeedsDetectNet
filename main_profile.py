import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('ultralytics/cfg/models/v5/yolov5n-GA-bifpn-ours64-dyhead128n2-CARAFE-35128.yaml')
    # model = YOLO('ultralytics/cfg/models/v5/yolov5n-W.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()