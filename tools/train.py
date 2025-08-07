import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':

    model = RTDETR('../ultralytics/cfg/modelY/CSFPR-RTDETR.yaml')
    model.train(data='../dataset/visdrone.yaml',
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='0',
                project='visdrone',
                name="CSFPR-RTDETR",
                )
    model = RTDETR('../ultralytics/cfg/modelY/CSFPR-RTDETR.yaml')
    model.train(data='../dataset/HITUAV.yaml',
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                device='0',
                project='HITUAV',
                name="CSFPR-RTDETR",

                )
    model = RTDETR('../ultralytics/cfg/modelY/CSFPR-RTDETR.yaml')
    model.train(data='../dataset/AITOD.yaml',
                imgsz=640,
                epochs=300,
                batch=2,
                workers=2,
                device='0',
                project='AITOD',
                name="CSFPR-RTDETR",
                )
