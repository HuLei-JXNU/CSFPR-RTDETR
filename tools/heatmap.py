
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
import torch.fft
from scipy import ndimage
import matplotlib.pyplot as plt


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return logits_[indices], boxes_[indices], xywh2xyxy(boxes_[indices]).cpu().detach().numpy()
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0][0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class rtdetr_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class rtdetr_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize, enable_frequency_analysis=True, plot_title_prefix='Aggregated'):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        target = rtdetr_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.__dict__.update(locals())
    
    def frequency_analysis(self, feature_map, num_channels=3):
        """分析特征图的频率域响应，简化为低中高三个频率通道"""
        # 确保特征图在正确的设备上
        if hasattr(self, 'device'):
            feature_map = feature_map.to(self.device)
            
        # 对特征图进行FFT变换
        fft_feature = torch.fft.fft2(feature_map)
        fft_magnitude = torch.abs(fft_feature)
        
        # 将频率域分为三个频率带
        h, w = fft_magnitude.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        frequency_bands = []
        band_names = ['Low', 'Mid', 'High']
        
        for i in range(3):  # 固定为3个通道
            # 创建频率带掩码 - 确保在正确设备上
            mask = torch.zeros_like(fft_magnitude[0, 0], device=self.device)
            
            if i == 0:  # 低频 - 包括DC和附近的低频分量
                radius = min(h, w) // 6  # 使用1/6的半径表示低频
                y, x = torch.meshgrid(torch.arange(h, device=self.device), 
                                     torch.arange(w, device=self.device), 
                                     indexing='ij')
                dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
                mask = (dist < radius).float()
                
            elif i == 1:  # 中频
                radius_inner = min(h, w) // 6
                radius_outer = min(h, w) // 3  # 1/3作为中频带的外半径
                
                y, x = torch.meshgrid(torch.arange(h, device=self.device), 
                                     torch.arange(w, device=self.device), 
                                     indexing='ij')
                dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
                mask = ((dist >= radius_inner) & (dist < radius_outer)).float()
                
            else:  # 高频 - 剩余的频率范围
                radius_inner = min(h, w) // 3
                
                y, x = torch.meshgrid(torch.arange(h, device=self.device), 
                                     torch.arange(w, device=self.device), 
                                     indexing='ij')
                dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
                mask = (dist >= radius_inner).float()
            
            # 应用掩码并逆FFT
            masked_fft = fft_feature * mask.unsqueeze(0).unsqueeze(0)
            filtered_feature = torch.fft.ifft2(masked_fft).real
            
            frequency_bands.append({
                'name': band_names[i],
                'feature': filtered_feature,
                'magnitude': torch.mean(torch.abs(filtered_feature), dim=[0, 1])
            })
        
        return frequency_bands
    
    def visualize_frequency_responses(self, img, feature_maps, save_path):
        """可视化三个频率通道的响应"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 分析第一个特征层的频率响应作为示例
        if len(feature_maps) > 0:
            feature_map = feature_maps[0].to(self.device)
                
            # 对特征图求平均(如果有多个通道)
            if len(feature_map.shape) > 2:
                if len(feature_map.shape) == 4:  # [batch, channel, height, width]
                    feature_avg = torch.mean(feature_map[0], dim=0)
                else:
                    feature_avg = torch.mean(feature_map, dim=0)
            else:
                feature_avg = feature_map
            
            # 频率域分析
            freq_bands = self.frequency_analysis(feature_avg.unsqueeze(0).unsqueeze(0))
            
            # 显示每个频率带的响应
            for i, (ax, band) in enumerate(zip(axes, freq_bands)):
                band_response = torch.abs(band['feature'][0, 0])
                band_response = (band_response - band_response.min()) / (band_response.max() - band_response.min() + 1e-8)
                
                im = ax.imshow(band_response.cpu().numpy(), cmap='jet')
                magnitude_value = float(band["magnitude"].mean().cpu().item())
                ax.set_title(f'{band["name"]} Frequency\nMag: {magnitude_value:.4f}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/frequency_analysis.png', dpi=600)
        plt.close()
    
    def process_with_frequency_analysis(self, img_path, save_path):


        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        img = letterbox(img, auto=False, scaleFill=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        # 获取特征图
        feature_maps = []
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        # 注册hook
        hooks = []
        for layer_idx in self.layer:
            hook = self.model.model[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            pred = self.model(tensor)
        
        # 移除hook
        for hook in hooks:
            hook.remove()
        
        # 生成频率域分析图
        self.visualize_frequency_responses(img, feature_maps, save_path)
        
        # 原有的热力图生成
        try:
            grayscale_cam = self.method(tensor, [self.target])
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            pred_processed = self.post_process(pred[0][0], img.shape[:2])
            if self.renormalize:
                cam_image = self.renormalize_cam_in_bounding_boxes(
                    pred_processed[:, :4].cpu().detach().numpy().astype(np.int32), 
                    img, grayscale_cam
                )
            
            if self.show_box:
                for data in pred_processed:
                    data = data.cpu().detach().numpy()
                    cam_image = self.draw_detections(
                        data[:4], 
                        self.colors[int(data[4:].argmax())], 
                        f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}', 
                        cam_image
                    )
            
            cam_image = cv2.resize(cam_image, (ori_w, ori_h))
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/heatmap.png')
            
        except AttributeError as e:
            print(f"Error generating heatmap: {e}")


    def post_process(self, result, shape):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        
        # filter
        score, cls = logits_.max(1, keepdim=True)
        idx = (score > self.conf_threshold).squeeze()
        logits_, boxes_ = logits_[idx], boxes_[idx]
        
        # xywh -> xyxy
        h, w = shape
        boxes_ = xywh2xyxy(boxes_)
        boxes_[:, 0] *= w
        boxes_[:, 2] *= w
        boxes_[:, 1] *= w
        boxes_[:, 3] *= w
        
        return torch.cat([boxes_, logits_], dim=1)
    
    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        h, w, _ = image_float_np.shape
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1 , 0) , max(y1, 0) 
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2) , min(grayscale_cam.shape[0] - 1, y2) 
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]
        img = letterbox(img, auto=False, scaleFill=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        pred = self.model(tensor)[0][0]
        pred = self.post_process(pred, img.shape[:2])
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())], f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}', cam_image)
        cam_image = cv2.resize(cam_image, (ori_w, ori_h))
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
    
    def plot_frequency_bands_for_map(self, processed_feature_map, title_prefix, save_path_full):
        """
        Generates and saves a plot of frequency bands for a given processed feature map.
        processed_feature_map: A 2D tensor [H, W] representing the feature map to analyze.
        title_prefix: A string to prepend to the plot title.
        save_path_full: Full path to save the image.
        """
        # Ensure feature map is on the correct device
        processed_feature_map = processed_feature_map.to(self.device)
        
        # Frequency domain analysis - expects [batch, channel, height, width]
        # Add batch and channel dimensions
        freq_bands = self.frequency_analysis(processed_feature_map.unsqueeze(0).unsqueeze(0))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Adjust rect to make space for suptitle if it's long or figure is small
        fig.suptitle(f'{title_prefix}  Frequency Channel Responses', fontsize=20, y=0.94) # 默认值通常是 0.98 或 1.0，尝试更小的值
        
        for i, (ax, band) in enumerate(zip(axes, freq_bands)):
            response = torch.abs(band['feature'][0, 0]) # [0,0] because input to frequency_analysis was [1,1,H,W]
            response_np = response.cpu().numpy()
            
            # Normalize for visualization
            response_np_min = response_np.min()
            response_np_max = response_np.max()
            if response_np_max - response_np_min < 1e-8: # Avoid division by zero if flat
                 response_np = np.zeros_like(response_np)
            else:
                response_np = (response_np - response_np_min) / (response_np_max - response_np_min)
            
            im = ax.imshow(response_np, cmap='jet', vmin=0, vmax=1) # Ensure consistent color scale
            
            magnitude_value = float(band["magnitude"].mean().cpu().item())
            
            # ax.set_title(f'{band["name"]} Channel\nMag: {magnitude_value:.4f}')
            ax.set_title(f'{band["name"]} Channel',fontsize=15)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # 如果调整了 suptitle 的 y 值，可能需要相应调整 tight_layout 的 rect
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # 如果 suptitle 的 y 值减小了，这里的 top 可能也需要相应调整
        plt.savefig(save_path_full, dpi=1200) 
        plt.close(fig)

    def generate_aggregated_frequency_response(self, img_path, save_path_dir):
        """
        Generates an aggregated frequency response plot from multiple layers.
        Feature maps are upsampled to the resolution of the first specified layer, averaged, 
        and then frequency analysis is performed.
        """
        img = cv2.imread(img_path)
        img = letterbox(img, auto=False, scaleFill=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        feature_maps_from_hooks = {}
        def make_hook(name):
            def hook_fn(module, input, output):
                feature_maps_from_hooks[name] = output.detach().clone()
            return hook_fn
        
        hooks = []
        layer_names_ordered = []
        for layer_idx in self.layer:
            layer_name = f'layer_{layer_idx}'
            layer_names_ordered.append(layer_name)
            hook = self.model.model[layer_idx].register_forward_hook(make_hook(layer_name))
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(tensor)
        
        for hook in hooks:
            hook.remove()

        if not feature_maps_from_hooks or not layer_names_ordered:
            print("No feature maps captured for aggregated frequency analysis.")
            return

        processed_maps_to_aggregate = []
        target_h, target_w = -1, -1

        # Process the first layer's feature map to determine target size
        first_layer_name = layer_names_ordered[0]
        if first_layer_name not in feature_maps_from_hooks:
            print(f"Feature map for the first target layer ({first_layer_name}) not found.")
            return
            
        first_fm_tensor = feature_maps_from_hooks[first_layer_name].to(self.device)
        
        if len(first_fm_tensor.shape) == 4: # B, C, H, W
            first_fm_processed = torch.mean(first_fm_tensor[0], dim=0) # Avg channels, take 1st batch
        elif len(first_fm_tensor.shape) == 3: # C, H, W
            first_fm_processed = torch.mean(first_fm_tensor, dim=0)
        else: # H, W
            first_fm_processed = first_fm_tensor
        
        target_h, target_w = first_fm_processed.shape[-2], first_fm_processed.shape[-1]
        processed_maps_to_aggregate.append(first_fm_processed)

        # Process and resize remaining feature maps
        for layer_name in layer_names_ordered[1:]:
            if layer_name not in feature_maps_from_hooks:
                print(f"Warning: Feature map for layer {layer_name} not found. Skipping.")
                continue
            fm_tensor = feature_maps_from_hooks[layer_name].to(self.device)
            
            if len(fm_tensor.shape) == 4:
                current_fm_processed = torch.mean(fm_tensor[0], dim=0)
            elif len(fm_tensor.shape) == 3:
                current_fm_processed = torch.mean(fm_tensor, dim=0)
            else:
                current_fm_processed = fm_tensor
            
            if current_fm_processed.shape[-2] != target_h or current_fm_processed.shape[-1] != target_w:
                current_fm_processed = torch.nn.functional.interpolate(
                    current_fm_processed.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            processed_maps_to_aggregate.append(current_fm_processed)
        
        if not processed_maps_to_aggregate:
            print("No processed feature maps to aggregate after filtering.")
            return

        aggregated_map = torch.mean(torch.stack(processed_maps_to_aggregate), dim=0)

        save_file_path = os.path.join(save_path_dir, 'aggregated_layers_frequency_channels.png')
        self.plot_frequency_bands_for_map(aggregated_map, self.plot_title_prefix, save_file_path)

    def __call__(self, img_path, save_path):
        # Create base save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_name in os.listdir(img_path):
                img_full_path = os.path.join(img_path, img_name)
                img_base_name_iter = os.path.splitext(img_name)[0]
                current_img_save_dir_iter = os.path.join(save_path, img_base_name_iter)
                os.makedirs(current_img_save_dir_iter, exist_ok=True)

                self.process(img_full_path, os.path.join(current_img_save_dir_iter, 'heatmap.png'))
                
                if self.enable_frequency_analysis:
                    # For detailed analysis per layer (if needed later)
                    # self.detailed_frequency_analysis(img_full_path, current_img_save_dir_iter) 
                    # For aggregated frequency response
                    self.generate_aggregated_frequency_response(img_full_path, current_img_save_dir_iter)
        else: # Single image processing
            img_base_name = os.path.splitext(os.path.basename(img_path))[0]
            current_img_save_dir = os.path.join(save_path, img_base_name)
            os.makedirs(current_img_save_dir, exist_ok=True)

            self.process(img_path, os.path.join(current_img_save_dir, 'heatmap.png'))
            
            if self.enable_frequency_analysis:
                # For detailed analysis per layer (if needed later)
                # self.detailed_frequency_analysis(img_path, current_img_save_dir)
                # For aggregated frequency response
                self.generate_aggregated_frequency_response(img_path, current_img_save_dir)

def get_params():
    params = {

        'weight': '',
        'device': 'cuda:0',
        'method': 'EigenCAM',

        'layer': [5, 6, 7, 8],
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_box': False,
        'renormalize': False,
         'enable_frequency_analysis': True,
         'plot_title_prefix': 'Aggregated'
    }
    return params




if __name__ == '__main__':

    param_sets = [
        {
            'weight': 'runs/visdrone/r18/weights/best.pt',
            'layer': [15,19,22,25],
            'output_dir_suffix': 'param_set_1_r18',
            'plot_title_prefix': 'Baseline'
        },
        {
            'weight': 'others/temp/rtdetr-position+(C2f-SFHF)+MANet_HFERB+p22/weights/best.pt',
            'layer': [24, 27, 30, 33],
            'output_dir_suffix': 'param_set_2_our',
            'plot_title_prefix': 'CSFPR-RTDETR'
        }
    ]

    base_output_dir = 'heat/fr'
    input_image_path = r'F:/visdrone2019/VisDrone2019-DET-test-dev/images/9999952_00000_d_0000204.jpg'
    
    generated_heatmap_paths = []
    generated_agg_freq_plot_paths = []
    input_image_basename = os.path.splitext(os.path.basename(input_image_path))[0]

    for i, param_config in enumerate(param_sets):
        print(f"\nProcessing with parameter set {i+1}:")
        print(f"  Weight: {param_config['weight']}")
        print(f"  Layers: {param_config['layer']}")
        print(f"  Plot Title Prefix: {param_config['plot_title_prefix']}")

        current_params = get_params()
        current_params['weight'] = param_config['weight']
        current_params['layer'] = param_config['layer']
        current_params['plot_title_prefix'] = param_config['plot_title_prefix']
        current_params['enable_frequency_analysis'] = True # 确保为每个参数集都启用频率分析


        output_image_specific_dir = os.path.join(base_output_dir, param_config['output_dir_suffix'], input_image_basename)
        
        try:
            model_instance = rtdetr_heatmap(**current_params)

            model_instance(input_image_path, os.path.join(base_output_dir, param_config['output_dir_suffix']))
            
            heatmap_path = os.path.join(output_image_specific_dir, 'heatmap.png')
            if os.path.exists(heatmap_path):
                generated_heatmap_paths.append(heatmap_path)
            else:
                print(f"Warning: Heatmap not found at {heatmap_path} for parameter set {i+1}")

            # 收集聚合频率图的路径
            agg_freq_plot_path = os.path.join(output_image_specific_dir, 'aggregated_layers_frequency_channels.png')
            if os.path.exists(agg_freq_plot_path):
                generated_agg_freq_plot_paths.append(agg_freq_plot_path)
                print(f"Aggregated frequency plot for parameter set {i+1} found at: {agg_freq_plot_path}")
            else:
                print(f"Warning: Aggregated frequency plot not found at {agg_freq_plot_path} for parameter set {i+1}")

        except Exception as e:
            print(f"Error processing parameter set {i+1} ({param_config['output_dir_suffix']}): {e}")


    if len(generated_heatmap_paths) == 2:
        try:
            img1 = Image.open(generated_heatmap_paths[0])
            img2 = Image.open(generated_heatmap_paths[1])
            if img1.width != img2.width:
                print(f"Warning: Heatmap widths are different ({img1.width} vs {img2.width}). Resizing second image.")
                img2 = img2.resize((img1.width, int(img2.height * img1.width / img2.width)))
            
            total_height = img1.height + img2.height
            combined_img = Image.new('RGB', (img1.width, total_height))
            combined_img.paste(img1, (0, 0))
            combined_img.paste(img2, (0, img1.height))
            
            combined_save_path = os.path.join(base_output_dir, f'{input_image_basename}_combined_heatmaps.png')
            combined_img.save(combined_save_path)
            print(f"\nCombined heatmap saved to: {combined_save_path}")
        except FileNotFoundError:
            print("Error: One or both heatmap images not found for combining.")
        except Exception as e:
            print(f"Error combining heatmap images: {e}")
    elif len(generated_heatmap_paths) < 2:
        print(f"\nNot enough heatmaps generated to combine. Need 2, got {len(generated_heatmap_paths)}.")

    # 新增：拼接聚合频率图
    if len(generated_agg_freq_plot_paths) == 2:
        try:
            img1_agg = Image.open(generated_agg_freq_plot_paths[0])
            img2_agg = Image.open(generated_agg_freq_plot_paths[1])


            if img1_agg.width != img2_agg.width:
                print(f"Warning: Aggregated frequency plot widths are different ({img1_agg.width} vs {img2_agg.width}). Resizing second image.")

                img2_agg = img2_agg.resize((img1_agg.width, int(img2_agg.height * img1_agg.width / img2_agg.width)))
            
            total_height_agg = img1_agg.height + img2_agg.height
            combined_img_agg = Image.new('RGB', (img1_agg.width, total_height_agg))
            
            combined_img_agg.paste(img1_agg, (0, 0))
            combined_img_agg.paste(img2_agg, (0, img1_agg.height))
            
            combined_agg_save_path = os.path.join(base_output_dir, f'{input_image_basename}_combined_aggregated_frequency.png')
            combined_img_agg.save(combined_agg_save_path)
            print(f"\nCombined aggregated frequency plot saved to: {combined_agg_save_path}")

        except FileNotFoundError:
            print("Error: One or both aggregated frequency plot images not found for combining.")
        except Exception as e:
            print(f"Error combining aggregated frequency plot images: {e}")
    elif len(generated_agg_freq_plot_paths) < 2:
        print(f"\nNot enough aggregated frequency plots generated to combine. Need 2, got {len(generated_agg_freq_plot_paths)}.")