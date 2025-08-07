import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':

    model = RTDETR('../weights/visdrone.pt')
    model.val(data='../dataset/visdrone.yaml',
              split='test', 
              batch=8,
              )
