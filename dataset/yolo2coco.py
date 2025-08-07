import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# classes = ['Person','Car', 'Bicycle', 'OtherVehicle', 'DontCare']

classes =[
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
              'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
              'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='F:/datasets_yaogan/DIOR-yolo/test/images',type=str, help="path of images")
parser.add_argument('--label_path', default='F:/datasets_yaogan/DIOR-yolo/test/labels',type=str, help="path of labels .txt")
parser.add_argument('--save_path', type=str,default='F:/datasets_yaogan/DIOR-yolo/test/test.json', help="if not split the dataset, give a path to a json file")
arg = parser.parse_args()

def yolo2coco(arg):
    print("Loading data from ", arg.image_path, arg.label_path)

    assert os.path.exists(arg.image_path)
    assert os.path.exists(arg.label_path)
    
    originImagesDir = arg.image_path                                   
    originLabelsDir = arg.label_path
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    

    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):

        txtFile = f'{index[:index.rfind(".")]}.txt'
        stem = index[:index.rfind(".")]

        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f'{os.path.join(originImagesDir, index)} read error.\nerror:{e}')

        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):

            continue
        dataset['images'].append({'file_name': index,
                            'id': stem,
                            'width': width,
                            'height': height})
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])


                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                cls_id = int(label[0])   
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': stem,
                    'iscrowd': 0,

                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1


    with open(arg.save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(arg.save_path))

if __name__ == "__main__":
    yolo2coco(arg)