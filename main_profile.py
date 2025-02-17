import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/home/claude/Pictures/CUPID-ICRA2025/ultralytics/ultralytics/cfg/models/v8/CUPID.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[1024, 1024])
    model.fuse()