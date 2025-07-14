from transformers import AutoProcessor

# 以一个常见的 Qwen2-VL 模型为例
model_path = "/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct" 

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 打印 image_processor 的配置信息
# 您会看到关于尺寸的详细设置
print(processor.image_processor)

# 直接访问尺寸属性
# .size 是一个字典，包含 'height' 和 'width'
image_size_config = processor.image_processor.size
print(f"\nRecommended target size: {image_size_config}")