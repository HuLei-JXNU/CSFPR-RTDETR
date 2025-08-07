import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('../weights/AITOD.pt')
    model.val(data='../dataset/AITOD.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=8,
              )
    model = RTDETR('../weights/visdrone.pt')
    model.val(data='../dataset/visdrone.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=8,
              )
    model = RTDETR('../weights/HITUAV.pt')
    model.val(data='../dataset/HITUAV.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=8,
              )
#
# import torch
# import torch.nn as nn
#
# # 下载模型参数
# net = torch.load('../weights/AITOD.pt')  # 再加载网络的参数
#
# print(net)



# import torch
#
# # 1. 定义文件路径
# original_file_path = '../weights/AITOD_modified.pt'
# new_file_path = '../weights/AITOD_modified2.pt' # 建议保存为新文件名
#
# # 2. 加载完整的 checkpoint 字典
# print(f"正在加载原始文件: {original_file_path}")
# checkpoint = torch.load(original_file_path)
#
# # (可选) 打印原始值，确认路径正确
# print("原始模型配置路径:")
# print(checkpoint['train_args']['name'])
#
# # 3. 修改字典中的值
# # 你想设置的新路径
# new_model_config_path = 'CSFPR-RTDETR'
# # 访问嵌套的字典并赋新值
# checkpoint['train_args']['name'] = new_model_config_path
#
# print("\n...值已在内存中修改...")
#
# # 4. 保存修改后的 checkpoint 字典到新文件
# print(f"正在保存修改后的文件到: {new_file_path}")
# torch.save(checkpoint, new_file_path)
#
# print("保存成功！")
#
# # 5. (验证) 重新加载新文件，检查是否修改成功
# print("\n--- 验证修改结果 ---")
# modified_checkpoint = torch.load(new_file_path)
# print("新的模型配置路径:")
# print(modified_checkpoint['train_args']['name'])
#
# #
