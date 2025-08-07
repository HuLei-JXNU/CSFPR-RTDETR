import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from PIL import Image # 用于加载图像
from torchvision import transforms # 用于图像预处理

# 将项目根目录添加到 sys.path 以允许从 ultralytics 导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入必要的模块
try:
    # 假设您的完整 RT-DETR 模型类在 ultralytics.models.rtdetr.RTDETR 中定义
    # 或者您有自己的模型定义文件
    from ultralytics.models.rtdetr import RTDETR  # 示例：假设这是您的模型类
    # from ultralytics.models.yolo.detect.predict import DetectionPredictor # 如果需要预测器
    from ultralytics.nn.modules.transformer import (
        PositionRelationEmbedding,
        # 以下模块可能不是直接需要，但为了完整性保留
        Deformable_position_TransformerDecoder,
        DeformableTransformerDecoderLayer,
        MLP,
    )
    from ultralytics.nn.modules.position_encoding import Conv2dNormActivation
    from ultralytics.data.augment import LetterBox # 用于图像预处理
    from ultralytics.utils import ops
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保此脚本位于 RTDETR-main 目录中，或者 ultralytics 包已正确安装且模型定义可访问。")
    sys.exit(1)

# 用于存储捕获的权重的全局列表
captured_pos_relation_outputs = []
# 控制何时捕获的标志
capture_enabled = False
# 控制捕获哪个调用（例如，只捕获第一次调用）
capture_call_limit = 1 # 设置为1表示只捕获第一次调用，None表示捕获所有
current_capture_count = 0

def position_relation_hook_fn(module, input_tensors, output_tensor):
    global captured_pos_relation_outputs, capture_enabled, capture_call_limit, current_capture_count
    
    if capture_enabled and not module.training:
        if capture_call_limit is None or current_capture_count < capture_call_limit:
            # output_tensor 是 PositionRelationEmbedding.forward() 的直接输出
            # 形状: [batch_size, num_heads, num_src_boxes, num_tgt_boxes]
            captured_pos_relation_outputs.append(output_tensor.detach().cpu().clone())
            current_capture_count += 1
            print(f"Hook: 已捕获 PositionRelationEmbedding 输出 (调用次数: {current_capture_count})。形状: {output_tensor.shape}")
        if capture_call_limit is not None and current_capture_count >= capture_call_limit:
            print(f"Hook: 已达到捕获上限 ({capture_call_limit})。")


def visualize_weights_heatmap(weights_tensor, call_idx, head_idx, save_dir="pos_relation_heatmaps_real_image", save_txt=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取权重张量的实际大小
    h, w = weights_tensor.shape
    
    # 设置规整的坐标轴刻度，使用25的间隔
    # 确保1和300都包含在内
    h_ticks = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    w_ticks = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    
    # 1. 标准版带有突出显示的正值
    plt.figure(figsize=(12, 10))
    # 使用 RdBu_r 颜色映射，蓝色表示正值，红色表示负值，并将中心点设置为0
    ax = sns.heatmap(weights_tensor, annot=False, cmap="RdBu_r", center=0, cbar=True)
    plt.title(f"Position Relation Weights", fontsize=16) # 增大标题字体
    plt.xlabel("Output Bounding Box Index (Layer 1)", fontsize=14) # 增大X轴标签字体
    plt.ylabel("Output Bounding Box Index (Layer 2)", fontsize=14) # 增大Y轴标签字体
    
    # 设置x轴和y轴刻度位置和标签
    # 将刻度位置映射到实际的矩阵索引
    x_positions = [(x-1) * (w-1) / 299 for x in w_ticks]
    y_positions = [(y-1) * (h-1) / 299 for y in h_ticks]
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(w_ticks, fontsize=15) # 增大X轴刻度字体
    ax.set_yticks(y_positions)
    ax.set_yticklabels(h_ticks, fontsize=15) # 增大Y轴刻度字体
    
    base_name = f"pos_relation_inference_call{call_idx}_head{head_idx}"
    save_path = os.path.join(save_dir, f"{base_name}.png")
    plt.savefig(save_path,dpi=1200)
    plt.close()
    print(f"已将热图保存到: {save_path}")
    
    # 2. 只显示正值的版本
    plt.figure(figsize=(12, 10))
    # 创建一个掩码，将负值和零设为True（将被掩盖）
    mask = weights_tensor <= 0
    # 使用热色谱显示正值，越大颜色越暖
    ax = sns.heatmap(weights_tensor, mask=mask, annot=False, cmap="YlOrRd", cbar=True)
    plt.title(f"Position Relation Weights - 仅正值 (Call {call_idx}, Head {head_idx})", fontsize=16) # 增大标题字体
    plt.xlabel("Key/Target Boxes Index", fontsize=17) # 增大X轴标签字体
    plt.ylabel("Query/Source Boxes Index", fontsize=17) # 增大Y轴标签字体
    
    # 设置相同的刻度
    ax.set_xticks(x_positions)
    ax.set_xticklabels(w_ticks, fontsize=15) # 增大X轴刻度字体
    ax.set_yticks(y_positions)
    ax.set_yticklabels(h_ticks, fontsize=15) # 增大Y轴刻度字体
    
    positive_base_name = f"pos_relation_inference_call{call_idx}_head{head_idx}_positive_only"
    positive_save_path = os.path.join(save_dir, f"{positive_base_name}.png")
    plt.savefig(positive_save_path)
    plt.close()
    print(f"已将只显示正值的热图保存到: {positive_save_path}")
    
    # txt文件保存部分保持不变
    if save_txt:
        txt_save_path = os.path.join(save_dir, f"{base_name}.txt")
        with open(txt_save_path, 'w') as f:
            f.write(f"# Position Relation Weights - Call {call_idx}, Head {head_idx}\n")
            f.write(f"# Shape: {weights_tensor.shape}\n")
            f.write("# Format: 每行一个位置关系权重，每行格式为: query_idx, key_idx, weight_value\n\n")
            
            for q_idx in range(weights_tensor.shape[0]):
                for k_idx in range(weights_tensor.shape[1]):
                    weight_val = weights_tensor[q_idx, k_idx]
                    f.write(f"{q_idx}\t{k_idx}\t{weight_val:.6f}\n")
        
        print(f"已将权重数据保存到: {txt_save_path}")
        
        positive_txt_save_path = os.path.join(save_dir, f"{positive_base_name}.txt")
        with open(positive_txt_save_path, 'w') as f:
            f.write(f"# Position Relation Weights (仅正值) - Call {call_idx}, Head {head_idx}\n")
            f.write(f"# Shape: {weights_tensor.shape}\n")
            f.write("# Format: 每行一个位置关系权重，每行格式为: query_idx, key_idx, weight_value\n\n")
            
            for q_idx in range(weights_tensor.shape[0]):
                for k_idx in range(weights_tensor.shape[1]):
                    weight_val = weights_tensor[q_idx, k_idx]
                    if weight_val > 0:
                        f.write(f"{q_idx}\t{k_idx}\t{weight_val:.6f}\n")
        
        print(f"已将正值权重数据保存到: {positive_txt_save_path}")


def preprocess_image(image_path, img_size=640, stride=32, device='cpu'):
    """
    加载并预处理单个图像，使其与 RT-DETR 的输入兼容。
    参考 ultralytics.data.dataset.DetectionDataset.load_image 和 LetterBox
    """
    img0 = Image.open(image_path).convert("RGB")
    img = LetterBox(img_size, auto=True, stride=stride)(image=img0) # 填充和缩放
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB (如果LetterBox输出BGR)
                                          # 如果LetterBox直接输出RGB的CHW，则不需要::-1
    img = img.transpose(1, 2, 0) # CHW to HWC for ToTensor if needed, or ensure it's CHW
    # 确保 img 是 CHW
    if img.shape[2] == 3 and img.shape[0] !=3 : # HWC
        img = img.transpose(2,0,1)


    img = torch.from_numpy(img.copy()).to(device)
    img = img.float() / 255.0  # 归一化到 [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加 batch 维度
    return img, img0 # 返回预处理后的张量和原始图像 (PIL)


def run_inference_and_visualize(model_weights_path: str, test_image_path: str, img_size: int = 640, head_idx_to_visualize: int = 0):
    global captured_pos_relation_outputs, capture_enabled, current_capture_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 实例化并加载完整的 RT-DETR 模型 ---
    try:
        print(f"正在从 '{model_weights_path}' 加载 RT-DETR 模型...")
        # 直接从 .pt 文件加载模型，不需要额外的参数
        model = RTDETR(model_weights_path)
        model.to(device)
        # 不调用 model.eval() - 让 predict 处理这个
        print(f"模型 '{model_weights_path}' 加载完成。")

    except Exception as e:
        print(f"加载模型或权重时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. 定位目标模块并注册 Hook ---
    #    在调用 predict() 前先注册 hook
    hook_handle = None
    try:
        # 尝试首先在 model.model[-1].decoder 中查找
        decoder_module = None
        if hasattr(model, 'model') and isinstance(model.model, nn.ModuleList) and len(model.model) > 0:
            detection_model_candidate = model.model[-1] 
            if hasattr(detection_model_candidate, 'decoder') and \
               isinstance(detection_model_candidate.decoder, Deformable_position_TransformerDecoder):
                decoder_module = detection_model_candidate.decoder
            else:
                print(f"警告: 在 model.model[-1] 中未找到预期的解码器。检测到的类型: {type(getattr(detection_model_candidate, 'decoder', None))}")
        
        # 备用: 检查 model.decoder
        if not decoder_module and hasattr(model, 'decoder') and isinstance(model.decoder, Deformable_position_TransformerDecoder):
             decoder_module = model.decoder
        
        # 如果仍然找不到解码器，尝试打印模型结构
        if not decoder_module:
            print("错误: 未能在模型中定位到 Deformable_position_TransformerDecoder。")
            print("\n尝试打印模型结构以帮助调试:")
            for name, module_item in model.named_modules():
                if isinstance(module_item, Deformable_position_TransformerDecoder):
                    print(f"找到解码器: {name}")
                    decoder_module = module_item
                    break
                elif 'decoder' in name.lower() and hasattr(module_item, 'position_relation_embedding'):
                    print(f"找到可能的解码器: {name}")
                    decoder_module = module_item
                    break
            
            if not decoder_module:
                print("无法找到合适的解码器模块。无法继续。")
                return

        # 找到解码器后，获取并注册 hook
        if hasattr(decoder_module, 'position_relation_embedding'):
            target_hook_module = decoder_module.position_relation_embedding
            if isinstance(target_hook_module, PositionRelationEmbedding):
                hook_handle = target_hook_module.register_forward_hook(position_relation_hook_fn)
                print(f"已在 '{type(target_hook_module).__name__}' 模块上注册 forward hook。")
            else:
                print(f"错误:找到的 'position_relation_embedding' 类型不正确: {type(target_hook_module)}。")
                return
        else:
            print(f"错误: 解码器中未找到 'position_relation_embedding' 属性。")
            return
            
    except Exception as e:
        print(f"注册 hook 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. 直接使用 model.predict() 执行推理 ---
    print("正在对图像执行模型推理...")
    captured_pos_relation_outputs.clear()
    current_capture_count = 0 
    capture_enabled = True    # 启用捕获
    
    try:
        # 直接使用 predict 方法，它会内部自动处理评估模式
        print(f"使用 model.predict(source='{test_image_path}') 进行推理...")
        results = model.predict(source=test_image_path, stream=False)
        
        if results:
            print(f"模型推理完成。检测到 {len(results[0].boxes) if results and hasattr(results[0], 'boxes') else 0} 个目标。")
        else:
            print("模型推理完成，但未返回结果或结果为空。")

    except Exception as e:
        print(f"模型推理期间出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture_enabled = False # 禁用捕获
        # 移除 Hook
        if hook_handle:
            hook_handle.remove()
            print("已移除 forward hook。")

    # --- 4. 处理并可视化捕获的权重 ---
    if not captured_pos_relation_outputs:
        print("未捕获到位置关系权重。可能是 hook 未正确触发或模块未执行。")
    else:
        print(f"总共捕获到 {len(captured_pos_relation_outputs)} 组位置关系权重。")
        for call_idx, weights_for_call in enumerate(captured_pos_relation_outputs):
            print(f"  处理第 {call_idx+1} 次捕获的权重 - 形状: {weights_for_call.shape}")
            
            batch_to_visualize = 0 # 默认第一个 batch
            # 检查指定的 head_idx 是否有效
            num_heads_in_capture = weights_for_call.shape[1]
            
            if head_idx_to_visualize >= num_heads_in_capture:
                print(f"  警告: 指定的 head_idx ({head_idx_to_visualize}) 超出范围 (共有 {num_heads_in_capture} 个头)。")
                print(f"  自动使用第一个头 (head_idx=0) 代替。")
                head_idx_to_visualize = 0
            
            if weights_for_call.shape[0] > batch_to_visualize:
                selected_weights_single_head = weights_for_call[batch_to_visualize, head_idx_to_visualize, :, :]
                print(f"  可视化 batch {batch_to_visualize}, head {head_idx_to_visualize} 的权重")
                visualize_weights_heatmap(
                    selected_weights_single_head.numpy(), 
                    call_idx=call_idx + 1, 
                    head_idx=head_idx_to_visualize
                )
            else:
                print(f"  捕获的权重中没有足够的批次来可视化 B{batch_to_visualize} H{head_idx_to_visualize}。")

if __name__ == "__main__":
    # --- 用户需要配置这些路径 ---
    
    # 请替换为您的实际路径
    path_to_rtdetr_weights = "others/temp/rtdetr-position+(C2f-SFHF)+MANet_HFERB+p22/weights/best.pt" # <--- 修改这里
    path_to_test_image = "F:/visdrone2019/VisDrone2019-DET-test-dev/images/9999952_00000_d_0000204.jpg" # <--- 修改这里
    image_input_size = 640 # RT-DETR 通常使用 640x640
    head_idx = 7  # <--- 指定要可视化的头索引 (0,1,2,...)

    if not os.path.exists(path_to_rtdetr_weights):
        print(f"错误: 权重文件未找到: {path_to_rtdetr_weights}")
        print("请确保路径正确。")
    elif not os.path.exists(path_to_test_image):
        print(f"错误: 测试图片未找到: {path_to_test_image}")
        print("请确保路径正确。")
    else:
        run_inference_and_visualize(
            model_weights_path=path_to_rtdetr_weights,
            test_image_path=path_to_test_image,
            img_size=image_input_size,
            head_idx_to_visualize=head_idx  # 新增参数: 指定要可视化的头索引
        )