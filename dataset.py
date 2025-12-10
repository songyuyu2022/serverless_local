# dataset.py
import os
import json
import torch

DATA_PATH_DEFAULT = "input.txt"
VOCAB_PATH_DEFAULT = "vocab.json"


def build_char_vocab(
    txt_path: str = DATA_PATH_DEFAULT, vocab_path: str = VOCAB_PATH_DEFAULT
):
    """
    从 txt 构建字符级词表。如果 vocab.json 已经存在则直接加载。
    返回: (stoi, itos)
    """
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(stoi, f, ensure_ascii=False, indent=2)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class LMTextBatcher:
    """
    语言模型批次生成器：
    - 把整篇文本编码成一个长的 id 序列
    - 支持按比例划分 train / val 分片
    - 每次 get_batch() 返回 (x, y)，形状都是 [B, T]
      其中 y 是 x 的右移一位（next-token 预测）

    兼容两种调用方式：
      1) LMTextBatcher(txt_path="input.txt", block_size=64, ...)
      2) LMTextBatcher(data_path="input.txt", seq_len=64, ...)
    """

    def __init__(
        self,
        # 原始参数
        txt_path: str = DATA_PATH_DEFAULT,
        batch_size: int = 4,
        block_size: int = 64,
        device: str = "cpu",
        vocab_path: str = VOCAB_PATH_DEFAULT,
        split: str = "train",         # "train" or "val"
        split_ratio: float = 0.9,     # 90% 训练，10% 验证
        # 兼容 controller 的命名
        data_path: str = None,
        seq_len: int = None,
    ):
        """
        data_path: 若不为 None，则覆盖 txt_path（兼容 controller 中 data_path 参数）
        seq_len:   若不为 None，则覆盖 block_size（兼容 controller 中 seq_len 参数）
        """
        # 兼容 data_path / seq_len 名称
        if data_path is not None:
            txt_path = data_path
        if seq_len is not None:
            block_size = seq_len

        self.txt_path = txt_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split
        self.split_ratio = split_ratio

        # 构建/加载词表
        self.stoi, self.itos = build_char_vocab(txt_path, vocab_path)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 整个语料编码为 id 序列
        full_ids = torch.tensor(
            [self.stoi[ch] for ch in text],
            dtype=torch.long,
        )
        full_len = full_ids.numel()

        # 按比例切分：前 split_ratio 做训练，后部分做验证
        split_idx = int(full_len * split_ratio)
        if split == "train":
            self.data_ids = full_ids[:split_idx]
        elif split == "val":
            self.data_ids = full_ids[split_idx:]
        else:
            self.data_ids = full_ids  # 未知类型就用全量

        self.data_len = self.data_ids.numel()
        self.pos = 0  # 当前指针

        print(
            f"[dataset] loaded {txt_path}, split={split}, "
            f"length={self.data_len}, vocab_size={len(self.stoi)}, "
            f"block_size={self.block_size}, batch_size={self.batch_size}"
        )

    def get_batch(self):
        """
        返回:
            x: [B, T]  输入 token ids
            y: [B, T]  目标 token ids（x 右移一位）
        """
        B, T = self.batch_size, self.block_size
        # 我们需要 B*(T+1) 个 token，用来构造 x 和 y
        need = B * (T + 1)

        if self.pos + need >= self.data_len:
            # 一个 "epoch" 走完，重新从这段 split 开头来
            self.pos = 0

        chunk = self.data_ids[self.pos : self.pos + need]
        self.pos += need

        chunk = chunk.view(B, T + 1)
        x = chunk[:, :-1].to(self.device)  # [B, T]
        y = chunk[:, 1:].to(self.device)   # [B, T]
        return x, y

    def next_batch(self):
        """兼容 controller.py 调用，等价于 get_batch()."""
        return self.get_batch()
