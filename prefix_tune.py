
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

# 1. 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.inputs = []
        for text in texts:
            encoding = tokenizer(
                text, 
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.inputs.append(encoding.input_ids.squeeze())
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

# 2. 实现Prefix-Tuning模型
class PrefixTuningGPT2(GPT2LMHeadModel):
    def __init__(self, config, prefix_length=10):
        super().__init__(config)
        self.prefix_length = prefix_length
        self.hidden_size = config.hidden_size
        
        # 定义可训练前缀参数
        self.prefix_embedding = torch.nn.Embedding(prefix_length, self.hidden_size)
        
        # 冻结原始模型参数
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_embedding.requires_grad_(True)  # 仅训练前缀

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 生成时自动处理前缀
        attention_mask = kwargs.get('attention_mask', None)
        inputs_embeds = self._add_prefix_to_input(input_ids)
        
        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask
        }
    def _add_prefix_to_input(self, input_ids):
        batch_size = input_ids.shape[0]
        prefix_ids = torch.arange(self.prefix_length).to(input_ids.device)
        prefix_embeds = self.prefix_embedding(prefix_ids)
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.transformer.wte(input_ids)
        return torch.cat([prefix_embeds, inputs_embeds], dim=1)

    def forward(self, inputs_embeds=None, **kwargs):
        # 覆盖forward以支持自定义嵌入
        if inputs_embeds is not None:
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        return super().forward(**kwargs)

# 3. 训练函数
def train():
    # 初始化模型和分词器
    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(model_name)

    # model = PrefixTuningGPT2(base_model, prefix_length=10)
    model = PrefixTuningGPT2(base_model.config, prefix_length=10)
    model.load_state_dict(base_model.state_dict(), strict=False)  # 加载预训练权重
    
    # 示例数据（实际应替换为真实数据）
    train_texts = [
        "Generate description: name: Starbucks | type: coffee shop => Starbucks is a popular coffee shop chain.",
        "Generate description: name: The Eagle | type: restaurant => The Eagle is a well-known restaurant."
    ]
    train_dataset = TextDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 优化器（只优化前缀参数）
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    for epoch in range(3):  # 示例训练3个epoch
        total_loss = 0
        for batch in train_loader:
            inputs = batch.to(device)
            outputs = model(inputs, attention_mask=(inputs != tokenizer.pad_token_id))
            
            # 计算损失（仅计算目标部分的loss）
            prefix_len = model.prefix_length
            shift_logits = outputs.logits[:, prefix_len:-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()  # [batch, seq_len-1]
                       # 验证维度
            assert shift_logits.shape[1] == shift_labels.shape[1], \
                f"Logits length {shift_logits.shape[1]} != Labels length {shift_labels.shape[1]}"
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")
    
    # 保存前缀参数
    torch.save(model.prefix_embedding.state_dict(), 'prefix_weights.pth')
    print("Prefix weights saved!")

# 4. 推理函数
def inference():
    # 加载基础模型
    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # 直接加载自定义模型
    model = PrefixTuningGPT2.from_pretrained(
        model_name,
        prefix_length=10
    )
    model.prefix_embedding.load_state_dict(torch.load('prefix_weights.pth'))
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成时需要创建注意力掩码
    input_text = "Generate description: name: Book Cafe | type: bookstore"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    train()   # 运行训练
    inference()  # 运行推理