import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np

# 1. 自定义数据集类（示例）
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 定义BitFit训练函数
def train_bitfit(model, train_loader, device, epochs=3, lr=1e-4):
    # 仅选择偏置参数进行训练
    params_to_train = [
        p for n, p in model.named_parameters() 
        if 'bias' in n and p.requires_grad  # 仅选择偏置项
    ]
    optimizer = AdamW(params_to_train, lr=lr)  # 优化器仅更新偏置参数

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 3. 模型推理函数
def infer(model, tokenizer, text, device, max_length=128):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    return pred

# 4. 主函数流程
if __name__ == "__main__":
    # 配置参数（适配消费级GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'  # 使用较小的BERT-base模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    batch_size = 8  # 小批次减少显存占用

    # 示例数据（替换为真实数据）
    train_texts = ["I love NLP!", "This is a test sentence."]
    train_labels = [1, 0]
    dataset = CustomDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 加载模型并配置BitFit
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    # 冻结所有非偏置参数
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param.requires_grad = False  # 冻结非偏置参数

    # 训练模型
    train_bitfit(model, train_loader, device, epochs=3, lr=1e-4)

    # 推理示例
    test_text = "BitFit works well!"
    prediction = infer(model, tokenizer, test_text, device)
    print(f"Prediction: {prediction}")

    # 保存模型（仅保存偏置参数）
    torch.save(model.state_dict(), 'bitfit_model.pth')