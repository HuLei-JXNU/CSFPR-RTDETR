
import os
import cv2
 

colors = [(56, 56, 255), (151, 157, 255),(31, 112, 255),(29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61),(52, 147, 26),  (187, 212, 0), (168, 153, 44),(255, 194, 0),  (147, 69, 52), (255, 115, 100),
          (236, 24, 0),  (255, 56, 132), (133, 0, 82),  (255, 56, 203),(200, 149, 255),(199, 55, 255)]
 
class Annotator:
    def __init__(self, im, line_width=None):
        """Initialize the Annotator class with image and line width."""
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
 
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
 
 
def draw_curr_img(detection_boxes, curr_img_path, save_img_path, line_width=10, show_labels=True):
    print(f"开始处理{curr_img_path}文件")
    image = cv2.imread(curr_img_path)  # 使用OpenCV读取图像
    h, w = image.shape[:2]
    annotator = Annotator(image, line_width=line_width)
    for detection_box in detection_boxes:
        class_id, x_center, y_center, bbox_width, bbox_height = detection_box
        class_id=int(class_id)
        x_center *= w
        y_center *= h
        bbox_width *= w
        bbox_height *= h
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_width / 2)
        # 开始画当前检测框
        label = str(detect_object[class_id]) if show_labels else ""
        annotator.box_label([x1, y1, x2, y2], label=label, color=colors[class_id])
 
    cv2.imwrite(save_img_path, image)
    print(f"{curr_img_path} 检测框已经画完啦!!!")
 
def init_detect(show_labels=True):
    image_names = os.listdir(input_img_dir)
    # 检查目录是否存在
    if not os.path.exists(save_img_dir):
        # 如果目录不存在，则创建它
        os.makedirs(save_img_dir)
        print(f"{save_img_dir}目录不存在，已经成功新建该文件夹")
    for image_name in image_names:
        # file_type=image_name.split(".")[1]
        file_type=image_name.split(".")[-1].lower()
        if file_type in ["jpg", "png", "gif", "bmp", "tif","tiff", "webp", "heif", "svg", "raw", "ico"]:
            # input_txt_name = image_name.split(".")[0] + ".txt"
            input_txt_name = os.path.splitext(image_name)[0] + ".txt"
            input_img_path = os.path.join(input_img_dir, image_name)
            input_txt_path = os.path.join(input_txt_dir, input_txt_name)
            save_img_path = os.path.join(save_img_dir, image_name)
            if os.path.exists(input_txt_path):
                # 读取文件并解析为列表
                with open(input_txt_path, 'r') as file:
                    lines = file.readlines()
                    detection_boxes = [list(map(float, line.strip().split())) for line in lines]
            else:
                detection_boxes = []
            draw_curr_img(detection_boxes, input_img_path, save_img_path, line_width, show_labels)
        else:
            print("当前图片格式不正确，已经成功跳过！！！")
 
if __name__ == "__main__":

    input_img_dir=r"F:/VisDrone2019-DET-test-dev/images"
    input_txt_dir=r"F:/VisDrone2019-DET-test-dev/labels"
    save_img_dir =r"F:/VisDrone2019-DET-test-dev/save_img"

    detect_object= ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
   

    line_width=2
    show_labels=True
    init_detect()