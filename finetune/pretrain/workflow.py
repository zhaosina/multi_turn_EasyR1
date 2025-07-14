#!/usr/bin/env python3
"""
基于Qwen-VL2.5的迭代式grounding系统
实现自我纠正的视觉定位功能
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import re
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("✅ 成功导入 Qwen2.5-VL 相关组件")
except ImportError as e:
    print(f"❌ 无法导入Qwen2.5-VL组件: {e}")
    print("请安装: pip install git+https://github.com/huggingface/transformers accelerate")
    raise

try:
    from qwen_vl_utils import process_vision_info
    print("✅ 成功导入 qwen_vl_utils")
    HAS_QWEN_VL_UTILS = True
except ImportError:
    print("⚠️ qwen_vl_utils 不可用，使用内置处理方法")
    HAS_QWEN_VL_UTILS = False

class QwenVL25IterativeGrounding:
    def __init__(self, model_path='Qwen/Qwen2.5-VL-7B-Instruct'):
        """初始化Qwen-VL2.5模型"""
        print("🚀 加载Qwen-VL2.5模型...")
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        success = self._load_model()
        if not success:
            raise RuntimeError("❌ 无法加载Qwen-VL2.5模型")

    def _load_model(self):
        """加载Qwen-VL2.5模型"""
        try:
            print(f"🔄 从 {self.model_path} 加载模型...")
            
            # 加载模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            print("✅ Qwen-VL2.5模型加载成功!")
            print(f"📊 模型设备: {next(self.model.parameters()).device}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("🔧 请检查:")
            print("   1. 模型路径是否正确")
            print("   2. 是否已安装正确版本的transformers: pip install git+https://github.com/huggingface/transformers accelerate")
            print("   3. 是否已安装qwen-vl-utils: pip install qwen-vl-utils")
            print("   4. 是否有足够的GPU内存")
            return False

    def _process_vision_info(self, messages):
        """处理视觉信息"""
        if HAS_QWEN_VL_UTILS:
            try:
                return process_vision_info(messages)
            except Exception as e:
                print(f"⚠️ qwen_vl_utils处理失败: {e}，使用内置方法")
        
        # 内置处理方法
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message.get('content'), list):
                for content in message['content']:
                    if content.get('type') == 'image':
                        image_path = content.get('image')
                        if image_path and os.path.exists(image_path):
                            image_inputs.append(Image.open(image_path).convert('RGB'))
        
        return image_inputs, video_inputs

    def get_prediction(self, image_path, task_instruction, iteration=0, feedback=""):
        """获取模型预测"""
        
        if iteration == 0:
            # 初次预测使用标准prompt
            sys_prompt = """你是一个专业的视觉定位助手。你的任务是分析屏幕截图并预测最佳的点击位置。

基本动作规范:
- CLICK: 点击指定位置
- 格式: CLICK <point>[[x坐标, y坐标]]</point>
- 示例: CLICK <point>[[101, 872]]</point>

请严格按照以下格式回答:
思考: 详细说明你的推理过程
动作: 具体的点击动作

任务指令: """
            
            full_prompt = sys_prompt + task_instruction
        else:
            # 纠正时的prompt
            try:
                img = Image.open(image_path)
                img_width, img_height = img.size
            except:
                img_width, img_height = 1080, 2400  # 默认尺寸
                
            sys_prompt = f"""这是第{iteration + 1}次位置预测。你的前一次预测被验证为不准确，现在需要重新分析并预测一个完全不同的位置。

**图像信息:**
- 图像尺寸: {img_width} x {img_height} 像素
- 红色圆圈标记了你上一次的错误预测位置

**关键指示:**
1. 你上一次的预测是错误的，必须避免再次预测相同或相近的位置
2. 仔细观察图像中的红圈，那个位置是错误的
3. 重新分析整个界面，寻找正确的目标元素
4. 预测一个与红圈位置明显不同的新坐标

**反馈信息:**
{feedback}

**要求:**
- 必须预测与之前不同的坐标位置
- 仔细分析目标元素的实际位置
- 不要重复之前的错误

动作格式: CLICK <point>[[x坐标, y坐标]]</point>

请严格按照以下格式回答:
思考: 分析为什么上次预测错误，以及如何找到正确位置 DPO 本质上是一个有监督的微调过程（使用特定的损失函数），其稳定性和可复现性远高于在线 RL。
动作: 提供一个全新的点击位置

任务指令: """
            
            full_prompt = sys_prompt + task_instruction + f"\n位置分析和纠正: {feedback}"

        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        try:
            # 处理输入
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # 移到正确设备
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 生成预测 - 为迭代增加随机性
            with torch.no_grad():
                if iteration == 0:
                    # 首次预测使用较低温度保证稳定性
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9
                    )
                else:
                    # 后续迭代使用更高温度增加多样性
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        repetition_penalty=1.1
                    )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if isinstance(output_text, list) else output_text
            
        except Exception as e:
            print(f"❌ 生成预测时出错: {e}")
            return "思考: 无法处理图像，使用默认位置。\n动作: CLICK <point>[[400, 400]]</point>"

    def extract_coordinates(self, text):
        """从文本中提取坐标"""
        # 多种坐标格式的正则表达式
        patterns = [
            r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>',  # 标准格式
            r'<point>\[\[(\d+),\s*(\d+)\]\]</point>',        # 简化格式1
            r'\[\[(\d+),\s*(\d+)\]\]',                        # 简化格式2
            r'\((\d+),\s*(\d+)\)',                            # 括号格式
            r'(\d+),\s*(\d+)'                                 # 纯数字对
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 对于最后一个模式，需要验证数字是否合理
                if pattern == r'(\d+),\s*(\d+)':
                    for x_str, y_str in matches:
                        x, y = int(x_str), int(y_str)
                        if 0 <= x <= 2000 and 0 <= y <= 3000:  # 合理的屏幕坐标范围
                            return x, y
                else:
                    x, y = int(matches[0][0]), int(matches[0][1])
                    return x, y
        
        print(f"⚠️ 无法从文本中提取坐标: {text[:200]}...")
        return None, None

    def mark_image(self, image_path, x, y, output_path):
        """在图像上标记预测位置"""
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 画红色圆圈
        radius = 20
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline='red', width=5)
        
        # 添加标签
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((x+radius+5, y-radius), "预测", fill='red', font=font)
        
        img.save(output_path)
        print(f"✅ 标记位置 ({x}, {y}) 保存到 {output_path}")
        return img.size

    def verify_position(self, marked_image_path, task_instruction, predicted_x, predicted_y):
        """验证预测位置并提供改进建议"""
        
        verify_prompt = f"""你是一个严格的UI位置验证专家。你需要非常仔细地验证一个点击位置的准确性。

任务: "{task_instruction}"

图像中有一个红色圆圈标记了预测位置 ({predicted_x}, {predicted_y})。

**严格验证标准:**
1. 红圈必须精确覆盖目标UI元素的中心区域
2. 如果红圈只是接近目标但没有准确覆盖，则视为不准确
3. 如果目标元素很小，红圈必须在元素内部
4. 如果目标元素较大，红圈必须在元素的可点击区域内

**详细分析要求:**
- 仔细观察红圈的位置
- 识别任务要求的目标元素在哪里
- 判断红圈是否准确命中目标元素
- 如果不准确，明确指出正确的目标位置

**回答格式 (严格遵守):**
如果红圈准确命中目标元素: "验证通过"
如果红圈没有准确命中: "验证失败。目标元素位于: CLICK <point>[[准确的x坐标, 准确的y坐标]]</point>"

请进行严格验证，不要过于宽松。"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": marked_image_path},
                    {"type": "text", "text": verify_prompt},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            verification_text = output_text[0] if isinstance(output_text, list) else output_text
            
        except Exception as e:
            print(f"❌ 验证位置时出错: {e}")
            img = Image.open(marked_image_path)
            img_width, img_height = img.size
            center_x, center_y = img_width // 2, img_height // 2
            verification_text = f"由于技术问题无法分析。尝试中心位置: CLICK <point>[[{center_x}, {center_y}]]</point>"
        
        # 解析验证结果 - 使用更严格的判断逻辑
        verify_lower = verification_text.lower()
        
        print(f"🔍 验证模型回复: {verification_text}")
        
        # 更严格的成功判断条件
        success_indicators = ["验证通过", "位置正确", "准确命中", "correctly positioned"]
        failure_indicators = ["验证失败", "位置不正确", "没有准确命中", "not accurate", "incorrect"]
        
        is_success = False
        is_explicit_failure = False
        
        # 检查明确的成功指示
        for indicator in success_indicators:
            if indicator in verification_text:
                is_success = True
                break
        
        # 检查明确的失败指示    
        for indicator in failure_indicators:
            if indicator in verification_text:
                is_explicit_failure = True
                break
        
        # 默认为失败，除非有明确的成功指示
        if is_success and not is_explicit_failure:
            print(f"✅ 位置验证通过! 位置: ({predicted_x}, {predicted_y})")
            return True, predicted_x, predicted_y
        else:
            print(f"❌ 位置验证失败，需要调整...")
            
            # 尝试提取建议的新位置
            suggested_x, suggested_y = predicted_x, predicted_y
            
            # 查找CLICK格式的坐标
            click_patterns = [
                r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>',
                r'<point>\[\[(\d+),\s*(\d+)\]\]</point>',
                r'\[\[(\d+),\s*(\d+)\]\]'
            ]
            
            for pattern in click_patterns:
                matches = re.findall(pattern, verification_text)
                if matches:
                    suggested_x, suggested_y = int(matches[0][0]), int(matches[0][1])
                    print(f"🎯 模型建议新位置: ({suggested_x}, {suggested_y})")
                    break
            else:
                # 如果没找到建议坐标，标记为明确失败
                print(f"⚠️ 模型未提供明确的修正位置，将重新分析")
                # 不要使用中心点作为默认，而是保持原位置让下次迭代处理
                suggested_x, suggested_y = predicted_x, predicted_y
            
            return False, suggested_x, suggested_y

    def _is_reasonable_position(self, x, y, image_path):
        """检查位置是否在图像范围内且合理"""
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # 检查坐标是否在图像范围内，并留有一定边距
            margin = 10
            if (margin <= x <= img_width - margin and 
                margin <= y <= img_height - margin):
                return True
            else:
                print(f"⚠️ 建议位置 ({x}, {y}) 超出图像范围 ({img_width}x{img_height})")
                return False
        except Exception as e:
            print(f"⚠️ 无法验证位置合理性: {e}")
            return False

    def run_iterative_grounding(self, image_path, task_instruction, max_iterations=3):
        """运行迭代grounding流程"""
        print(f"\n🎯 任务: {task_instruction}")
        print("="*50)
        
        original_image = image_path
        current_image = image_path
        feedback = ""
        
        for iteration in range(max_iterations):
            print(f"\n🔄 第 {iteration + 1} 次迭代")
            print("-" * 30)
            
            # 步骤1: 获取预测
            print("📍 获取位置预测...")
            prediction_text = self.get_prediction(current_image, task_instruction, iteration, feedback)
            print(f"模型回答: {prediction_text}")
            
            # 提取坐标
            x, y = self.extract_coordinates(prediction_text)
            if x is None or y is None:
                print("❌ 无法提取有效坐标，停止迭代")
                return None
            
            print(f"🎯 预测位置: ({x}, {y})")
            
            # 步骤2: 标记图像
            print("🖍️ 标记预测位置...")
            marked_path = f"qwenvl25_marked_iter_{iteration + 1}.jpg"
            img_size = self.mark_image(original_image, x, y, marked_path)
            
            # 步骤3: 验证准确性
            print("✅ 验证位置准确性...")
            is_accurate, suggested_x, suggested_y = self.verify_position(marked_path, task_instruction, x, y)
            
            if is_accurate:
                print("🎉 位置准确！任务完成。")
                print(f"✨ 最终位置: ({suggested_x}, {suggested_y})")
                print(f"📝 使用了 {iteration + 1} 次迭代")
                return {
                    "success": True,
                    "final_position": (suggested_x, suggested_y),
                    "iterations": iteration + 1,
                    "final_image": marked_path
                }
            else:
                print("❌ 位置验证失败，准备下一次迭代...")
                
                # 只有当建议位置与当前位置不同且看起来合理时，才验证建议位置
                if (suggested_x, suggested_y) != (x, y) and self._is_reasonable_position(suggested_x, suggested_y, original_image):
                    print(f"🔍 验证模型建议的位置: ({suggested_x}, {suggested_y})")
                    next_marked_path = f"qwenvl25_marked_iter_{iteration + 1}_suggested.jpg"
                    self.mark_image(original_image, suggested_x, suggested_y, next_marked_path)
                    
                    # 验证建议位置，但使用更严格的标准
                    is_accurate_suggested, _, _ = self.verify_position(next_marked_path, task_instruction, suggested_x, suggested_y)
                    if is_accurate_suggested:
                        print("🎉 建议位置验证通过！任务完成。")
                        print(f"✨ 最终位置: ({suggested_x}, {suggested_y})")
                        return {
                            "success": True,
                            "final_position": (suggested_x, suggested_y),
                            "iterations": iteration + 1,
                            "final_image": next_marked_path
                        }
                    else:
                        print("❌ 建议位置也未通过验证")
                else:
                    print("⚠️ 建议位置无效或与当前位置相同，继续迭代")
                
                # 设置下一次迭代的反馈 - 提供更具体的指导
                failed_positions = [(x, y)]
                if iteration > 0:
                    # 收集之前所有失败的位置
                    if "失败位置历史:" in feedback:
                        prev_positions = re.findall(r'\((\d+), (\d+)\)', feedback.split("失败位置历史:")[1])
                        failed_positions.extend([(int(px), int(py)) for px, py in prev_positions])
                
                failed_pos_str = ", ".join([f"({px}, {py})" for px, py in failed_positions])
                
                if (suggested_x, suggested_y) != (x, y):
                    feedback = f"""第{iteration + 1}次预测 ({x}, {y}) 验证失败。
验证模型建议位置: ({suggested_x}, {suggested_y})
失败位置历史: {failed_pos_str}

错误分析: 你一直预测错误的位置。请完全重新分析图像，寻找与这些失败位置明显不同的目标元素。
指导建议: 尝试图像的其他区域，仔细观察UI元素的布局和层次结构。"""
                else:
                    feedback = f"""第{iteration + 1}次预测 ({x}, {y}) 验证失败。
失败位置历史: {failed_pos_str}

错误分析: 验证模型未提供明确的位置建议，说明当前预测严重偏离目标。
指导建议: 重新审视任务要求，在图像中寻找符合任务描述的UI元素，尝试完全不同的区域。"""
                
                current_image = marked_path
        
        print(f"⏰ 达到最大迭代次数 ({max_iterations})，停止尝试")
        return {
            "success": False,
            "final_position": (x, y),
            "iterations": max_iterations,
            "final_image": marked_path
        }

def main():
    """演示使用"""
    print("🚀 基于Qwen2.5-VL的迭代式视觉Grounding系统")
    print("="*60)
    print("功能：图像分析 → 位置预测 → 自我验证 → 迭代优化")
    print("="*60)
    
    try:
        # 初始化系统
        grounding = QwenVL25IterativeGrounding()
        
        # 测试参数
        image_path = "/home/wgb/mobile_images/0a0f9f9d-cc60-48e6-9450-c1b18f57fabc.png"
        task = "how to create the recipes?"
        
        if os.path.exists(image_path):
            print(f"✅ 找到图像文件: {image_path}")
            
            # 运行迭代grounding
            result = grounding.run_iterative_grounding(image_path, task, max_iterations=3)
            
            print("\n" + "="*50)
            print("🏁 最终结果:")
            if result and result["success"]:
                print(f"✅ 成功找到位置: {result['final_position']}")
                print(f"📊 迭代次数: {result['iterations']}")
                print(f"🖼️ 最终标记图像: {result['final_image']}")
            else:
                print("❌ 未能找到准确位置")
                if result:
                    print(f"📊 尝试了 {result['iterations']} 次迭代")
                    print(f"🖼️ 最后的标记图像: {result['final_image']}")
        else:
            print(f"❌ 图像文件不存在: {image_path}")
            print("请修改image_path为有效的图像路径")
            
    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        print("\n🔧 故障排除建议:")
        print("1. 确保已安装: pip install git+https://github.com/huggingface/transformers accelerate")
        print("2. 确保已安装: pip install qwen-vl-utils")  
        print("3. 确保已安装: pip install torch>=2.0.0")
        print("4. 检查Qwen2.5-VL模型是否已下载")
        print("5. 确保有足够的GPU内存")

if __name__ == "__main__":
    main() 