from transformers import Qwen2VLProcessor
processor = Qwen2VLProcessor.from_pretrained("/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct")
tok = processor.tokenizer
print("pad_token_id:", tok.pad_token_id)
# 调用 pad()，不报错即成功
batch = tok.pad({"input_ids": [tok.encode("示例")],
                 "attention_mask": [ [1,1,1] ]},
                padding="longest", return_tensors="pt")
print(batch)
