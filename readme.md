
# CSFPR-RTDETR

The corresponding paper title for this project is “CSFPR-RTDETR : Real-Time Small Object Detection Network for UAV Images Based on Cross Spatial Frequency Domain and Position Relation”.

## Train

Single GPU training

```python
# tools/train.py
from ultralytics import RTDETR
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml')
    model.train(data='dataset/visdrone.yaml',
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='0',
                project='visdrone',
                name="CSFPR-RTDETR",
                )

```

## Val

```python
# tools/val.py
from ultralytics import YOLOv10
if __name__ == '__main__':
	model = RTDETR('your_weight')
	model.val(data='../dataset/visdrone.yaml',
              split='test',
              batch=8,
              )
```
