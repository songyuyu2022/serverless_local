import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_len=10):
        self.data = []
        self.vocab = vocab or {"<UNK>": 0}

        # 简单读取并构建词表 (真实场景请用 Tokenizer)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                # 简单的 Label 生成逻辑：根据长度或内容生成假标签供训练
                label = len(tokens) % 2
                token_ids = []
                for t in tokens:
                    if t not in self.vocab:
                        self.vocab[t] = len(self.vocab)
                    token_ids.append(self.vocab[t])

                # Padding / Truncating
                if len(token_ids) < max_len:
                    token_ids += [0] * (max_len - len(token_ids))
                else:
                    token_ids = token_ids[:max_len]

                self.data.append((torch.tensor(token_ids), torch.tensor(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(file_path, batch_size=4):
    dataset = TextDataset(file_path)
    # 返回 DataLoader 和 词表大小 (用于初始化模型)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), len(dataset.vocab)