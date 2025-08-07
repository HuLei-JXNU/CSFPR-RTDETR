import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('') # select your model.pt path
    model.predict(source=' ',
                  conf=0.25,
                  project='vis/dete',
                  name='dete',
                  save=True,
                  line_width=2, 
                  show_conf=True, 
                  show_labels=True, 
                  save_conf=True,
                  save_txt=True, 

                  )
#