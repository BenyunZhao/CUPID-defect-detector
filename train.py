import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO('/home/claude/Documents/GitHub/CUPID-defect-detection/ultralytics/cfg/models/v8/yolov8x-CUPID.yaml')  # Train without pretrained weights
    # model.load('/home/claude/Documents/GitHub/yolov8/pretrained/yolov8x.pt')  # Load pretrained weights

    # Training parameters ----------------------------------------------------------------------------------------------
    model.train(
        data='/home/claude/Documents/GitHub/CUPID-defect-detection/ultralytics/cfg/datasets/CUBIT2024.yaml',
        epochs=50,  # (int) Number of training epochs
        time=None,  # (float, optional) Total training time in hours, overrides epochs if set
        patience=100,  # (int) Number of epochs to wait for no apparent improvement for early stopping
        batch=16,  # (int) Number of images per batch (-1 for autobatch)
        imgsz=640,  # (int | list) Input image size as int or [h, w] list
        save=True,  # (bool) Save train checkpoints and predict results
        save_period=-1,  # (int) Save checkpoint every x epochs (disabled if < 1)
        cache=True,  # (bool) True/ram, disk or False. Cache data for faster training
        device='0',  # (int | str | list, optional) Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
        workers=12,  # (int) Number of worker threads for data loading (per DDP process)
        project='runs-for-test',  # (str, optional) Project name
        name='CUPIDn',  # (str, optional) Experiment name, results saved to 'project/name'
        exist_ok=False,  # (bool) Whether to overwrite existing experiment
        pretrained=None,  # (bool | str) Whether to use pretrained model (bool) or model path to load weights from (str)
        optimizer='SGD',  # (str) Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose=True,  # (bool) Whether to print verbose output
        seed=0,  # (int) Random seed for reproducibility
        deterministic=True,  # (bool) Whether to enable deterministic mode
        single_cls=False,  # (bool) Train multi-class data as single-class
        rect=False,  # (bool) Rectangular training if mode='train', rectangular validation if mode='val'
        cos_lr=False,  # (bool) Use cosine learning rate scheduler
        close_mosaic=0,  # (int) Disable mosaic augmentation for final epochs
        resume=True,  # (bool) Resume training from last checkpoint
        amp=True,  # (bool) Automatic Mixed Precision training, choices=[True, False]
        fraction=1.0,  # (float) Dataset fraction to train on (default is 1.0, all images)
        profile=False,  # (bool) Profile ONNX and TensorRT speeds during training for logger
        freeze=None,  # (int | list, optional) Freeze first n layers or freeze layers with specified indices
        multi_scale=False,  # (bool) Vary image size during training

        # Validation/Test settings
        val=True,  # (bool) Validate/test during training
        split='val',  # (str) Dataset split to use for validation, i.e. 'val', 'test' or 'train'
        save_json=False,  # (bool) Save results to JSON file
        save_hybrid=False,  # (bool) Save hybrid version of labels (labels + additional predictions)
        conf=0.001,  # (float, optional) Object confidence threshold (default 0.25 predict, 0.001 val)
        iou=0.7,  # (float) Intersection over Union (IoU) threshold for NMS
        max_det=300,  # (int) Maximum number of detections per image
        half=False,  # (bool) Use half precision (FP16)
        dnn=False,  # (bool) Use OpenCV DNN for ONNX inference
        plots=True,  # (bool) Save plots and images during train/val

        # Hyperparameters
        lr0=0.01,  # (float) Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01,  # (float) Final learning rate (lr0 * lrf)
        momentum=0.937,  # (float) SGD momentum/Adam beta1
        weight_decay=0.0005,  # (float) Optimizer weight decay 5e-4
        warmup_epochs=3.0,  # (float) Warmup epochs (fractions ok)
        warmup_momentum=0.8,  # (float) Warmup initial momentum
        warmup_bias_lr=0.1,  # (float) Warmup initial bias learning rate
        box=7.5,  # (float) Box loss gain
        cls=0.5,  # (float) Class loss gain (scale with pixels)
        dfl=1.5,  # (float) dfl loss gain
        pose=12.0,  # (float) Pose loss gain
        kobj=1.0,  # (float) Keypoint object loss gain
        label_smoothing=0.0,  # (float) Label smoothing (fraction)
        nbs=64,  # (int) Nominal batch size
        hsv_h=0.015,  # (float) Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # (float) Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # (float) Image HSV-Value augmentation (fraction)
        degrees=0.0,  # (float) Image rotation (+/- deg)
        translate=0.1,  # (float) Image translation (+/- fraction)
        scale=0.5,  # (float) Image scale (+/- gain)
        shear=0.0,  # (float) Image shear (+/- deg)
        perspective=0.0,  # (float) Image perspective (+/- fraction), range 0-0.001
        flipud=0.5,  # (float) Image flip up-down (probability)
        fliplr=0.5,  # (float) Image flip left-right (probability)
        mosaic=1.0,  # (float) Image mosaic (probability)
        mixup=0.5,  # (float) Image mixup (probability)
        copy_paste=0.0,  # (float) Segment copy-paste (probability)
        auto_augment='randaugment',  # (str) Auto augmentation policy for classification
)