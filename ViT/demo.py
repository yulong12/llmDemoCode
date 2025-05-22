from functools import partial
import torch
import torch.nn as nn

# 自定义函数：将输入转换为二元组（兼容PyTorch风格）
def to_2tuple(x):
    """将输入转换为二元组，如果是整数则复制为(x, x)"""
    return (x, x) if isinstance(x, int) else x

# 创建带默认参数的LayerNorm层（eps防止除以零）
LayerNorm = partial(nn.LayerNorm, eps=1e-6)

class PatchEmbedding(nn.Module):
    """将图像分割为块并进行嵌入的模块
    
    Args:
        img_size (int): 输入图像尺寸，默认224
        patch_size (int): 每个块的大小，默认16
        in_channels (int): 输入通道数，默认3
        embed_dim (int): 嵌入维度，默认768
        norm_layer (nn.Module): 标准化层类型，默认LayerNorm
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, norm_layer=LayerNorm):
        super().__init__()
        self.img_size = to_2tuple(img_size)        # 转换为(H, W)格式
        self.patch_size = to_2tuple(patch_size)     # 转换为(P, P)格式
        self.embed_dim = embed_dim
        
        # 使用卷积层实现块分割：kernel和stride都等于patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)
        
        # 标准化层（如果未指定则使用恒等映射）
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """前向传播流程说明：
        1. 卷积投影 -> 2. 展平空间维度 -> 3. 转置维度 -> 4. 标准化
        """
        x = self.proj(x)  # 输出形状 (B, C, H/P, W/P)
        x = x.flatten(2)  # 展平为 (B, C, N) N=块数
        x = x.transpose(1, 2)  # 转置为 (B, N, C)
        x = self.norm(x)  # 对每个块进行标准化
        return x

if __name__ == "__main__":
    # 参数配置
    img_size = 224      # 标准ViT输入尺寸
    patch_size = 16     # 每个块16x16像素
    in_channels = 3     # RGB图像
    embed_dim = 768     # 标准嵌入维度
    batch_size = 2      # 测试批量大小

    # 实例化模块
    patch_embed = PatchEmbedding(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim
    )

    # 生成随机测试数据 (batch_size, channels, height, width)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # 前向传播
    output = patch_embed(x)
    
    # 验证输出形状应为：
    # (batch_size, num_patches, embed_dim)
    # num_patches = (224/16)^2 = 14x14 = 196
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)  # 预期输出 (2, 196, 768)