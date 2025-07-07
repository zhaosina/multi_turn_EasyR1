import logging
import torch
import yaml
from pathlib import Path

# 导入我们正在调试的核心模块
from verl.trainer.config import PPOConfig 
from verl.trainer.data_loader import create_mgrpo_v_dataloader
from verl.trainer.ray_trainer import RayPPOTrainer 
from verl.protocol import pad_dataproto_to_divisor

# 从 transformers 加载必要的组件
from transformers import AutoProcessor, AutoTokenizer

# 配置日志记录，方便观察输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置区 ---
# 请根据您的环境修改这些路径
CONFIG_PATH ="/home/zhq/workdir/GUI/EasyR1/examples/config.yaml"
# MODEL_PATH 将从配置文件中读取，如果配置文件没有，请在此处指定
# MODEL_PATH = "/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct" 

class MockWorkerGroup:
    """一个模拟的 WorkerGroup，只为了提供 world_size 属性。"""
    def __init__(self, world_size):
        self.world_size = world_size

    def prepare_rollout_engine(self):
        # 模拟空操作
        pass

    def release_rollout_engine(self):
        # 模拟空操作
        pass

def debug_data_pipeline():
    """
    主调试函数，模拟 fit() 函数的数据处理流程。
    """
    logging.info("==================== 开始调试数据处理流程 ====================")

    try:
        # --- 步骤 1: 加载配置和组件 ---
        logging.info(f"正在从 {CONFIG_PATH} 加载配置...")
        with open(CONFIG_PATH, "r") as f:
            cfg_dict = yaml.safe_load(f)
        ppo_config = PPOConfig.from_dict(cfg_dict)
        logging.info("配置加载成功。")

        model_path = ppo_config.worker.actor.model_name_or_path
        logging.info(f"正在从 {model_path} 加载 processor 和 tokenizer...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Processor 和 tokenizer 加载成功。")

        # --- 步骤 2: 创建 Dataloader ---
        logging.info("正在创建训练数据加载器 (train_dataloader)...")
        train_dataloader, _ = create_mgrpo_v_dataloader(ppo_config.data, tokenizer, processor)
        logging.info("Dataloader 创建成功。")

        # --- 步骤 3: 模拟实例化 Trainer ---
        # 我们不需要一个完整的、能运行的 trainer，只需要一个能调用其方法的对象。
        # 对于分布式相关的部分，我们传入 None 或模拟对象。
        logging.info("正在模拟实例化 RayPPOTrainer...")
        # 注意：这里我们假设可以传入None来绕过完整的初始化，如果不行则需要更复杂的Mock
        trainer = RayPPOTrainer(
            config=ppo_config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=None, # 验证流程不是本次调试重点
            role_worker_mapping=None,
            resource_pool_manager=None,
        )
        # 手动设置模拟的 worker_group，以提供 world_size
        # 这里的 world_size 必须与您实际训练时 FSDP 的并行数一致
        # 从之前的报错 "chunk 11" 来看，这个值很可能是 11
        world_size_to_check = 11 
        trainer.actor_rollout_ref_wg = MockWorkerGroup(world_size=world_size_to_check)
        trainer.data_iterator = iter(train_dataloader)
        logging.info(f"Trainer 模拟实例化成功，设置 world_size = {world_size_to_check} 用于测试。")


        # --- 步骤 4: 完整模拟 fit() 函数的核心数据处理流程 ---
        logging.info("\n==================== 模拟 fit() 循环第一步 ====================")
        
        # 4.1 调用 _make_batch_data
        logging.info("--> 正在调用 trainer._make_batch_data()...")
        batch = trainer._make_batch_data(metrics={})
        if not batch or len(batch) == 0:
            logging.error("错误: _make_batch_data 返回了一个空的批次，调试中止。")
            return
        original_len = len(batch)
        logging.info(f"<-- _make_batch_data() 完成。返回的批次长度: {original_len}")

        # 4.2 调用 _balance_batch
        logging.info("--> 正在调用 trainer._balance_batch()...")
        trainer._balance_batch(batch, metrics={})
        len_after_balance = len(batch)
        logging.info(f"<-- _balance_batch() 完成。批次长度: {len_after_balance}")
        assert original_len == len_after_balance, "错误: _balance_batch 改变了批次长度！"
        
        # 4.3 调用 pad_dataproto_to_divisor (这是我们修复的核心)
        logging.info(f"--> 正在调用 pad_dataproto_to_divisor(..., world_size={world_size_to_check})...")
        padded_batch, pad_size = pad_dataproto_to_divisor(batch, world_size_to_check)
        len_after_padding = len(padded_batch)
        logging.info(f"<-- pad_dataproto_to_divisor() 完成。填充了 {pad_size} 个样本，填充后长度: {len_after_padding}")
        
        # --- 步骤 5: 最终验证 ---
        logging.info("\n==================== 最终验证 ====================")
        final_len = len(padded_batch)
        divisor = world_size_to_check
        is_divisible = (final_len % divisor == 0)
        
        logging.info(f"最终批次长度: {final_len}")
        logging.info(f"要求的除数 (world_size): {divisor}")
        logging.info(f"是否可以整除? ({final_len} % {divisor} == 0): {is_divisible}")

        if is_divisible:
            logging.info("\n✅ 成功！最终批次的长度可以被 world_size 整除。AssertionError 已修复。")
        else:
            logging.error(f"\n❌ 失败！最终批次长度 {final_len} 仍然无法被 {divisor} 整除。请检查代码逻辑。")
            assert False, "Divisibility check failed!"

    except Exception as e:
        logging.error(f"调试过程中发生意外错误: {e}", exc_info=True)

if __name__ == "__main__":
    debug_data_pipeline()