import torch
import os
import json

# 1. 定义 controller 需要的默认路径常量 (修复 ImportError)
DATA_PATH_DEFAULT = "./data"


class LMTextBatcher:
    """
    适配 controller.py 的文本数据加载器。

    功能：
    1. 修复 ImportError: 实现了 controller 所需的 LMTextBatcher 类名。
    2. 数据兜底: 如果找不到真实数据文件，会自动生成随机数据，确保代码能跑通。
    3. 接口适配: 实现了 next_batch() 方法供训练循环调用。
    """

    def __init__(self, data_path, split, batch_size, block_size):
        # 对应 controller.py 第 534 行的初始化参数
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        self.block_size = block_size
        self.ptr = 0

        # 加载词表和数据
        self.vocab, self.data = self._load_or_create_data()

    def _load_or_create_data(self):
        """内部方法：加载数据，如果失败则生成随机数据"""
        vocab = {}
        data_tensor = None
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # --- 1. 尝试加载词表 (vocab.json) ---
        vocab_file = os.path.join(current_dir, "vocab.json")
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
            except Exception as e:
                print(f"[Dataset] Warning: vocab.json exists but load failed: {e}")

        # 如果没加载到，给一个默认词表大小，避免 crash
        vocab_size = len(vocab) if vocab else 12000

        # --- 2. 尝试加载文本数据 (input.txt) ---
        input_file = os.path.join(current_dir, "input.txt")
        if os.path.exists(input_file):
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    text = f.read()
                # 简单的字符/词转ID
                tokens = []
                for c in text:
                    # 优先查表，查不到给0
                    tokens.append(vocab.get(c, 0))

                # 只有当数据长度足够一个 batch 时才使用
                if len(tokens) > self.batch_size * self.block_size:
                    print(f"[{self.split}] Loaded real data from {input_file} (len={len(tokens)})")
                    data_tensor = torch.tensor(tokens, dtype=torch.long)
            except Exception as e:
                print(f"[Dataset] Warning: input.txt load failed: {e}")

        # --- 3. 兜底方案：生成随机数据 ---
        # 如果上面没读到数据，或者数据太少，就生成随机数。
        # 这样可以保证你不需要下载数据集，直接运行 controller.py 也能跑通流程。
        if data_tensor is None:
            print(f"[{self.split}] Data not found or insufficient. Generating DUMMY random data...")
            # 生成 10万个 token 供测试
            data_tensor = torch.randint(0, vocab_size, (100000,), dtype=torch.long)

        return vocab, data_tensor

    def next_batch(self):
        """
        核心方法：对应 controller.py 第 475 行的调用: x, y = batcher.next_batch()
        """
        B = self.batch_size
        T = self.block_size

        # 如果数据读完了，重置指针 (Loop)
        if self.ptr + B * T + 1 > len(self.data):
            self.ptr = 0

        # 截取一段数据
        chunk = self.data[self.ptr: self.ptr + B * T + 1]
        self.ptr += B * T

        # x 是输入，y 是预测目标（向后移一位）
        x = chunk[:-1].view(B, T)
        y = chunk[1:].view(B, T)

        return x, y