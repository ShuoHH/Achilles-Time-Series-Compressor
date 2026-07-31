"""
解码器注册表与工厂函数。

使用方式：
    from cross_models.decoders import create_decoder, DECODER_REGISTRY

    decoder = create_decoder('siren', dim=32, trend_dim=1, hidden_dim=512, num_blocks=5)

添加新解码器：
    1. 在 decoders/ 目录下新建文件，继承 BaseDecoder
    2. 在本文件的 DECODER_REGISTRY 中注册
    3. 命令行 --decoder_type <name> 即可切换
"""

from .base_decoder import BaseDecoder
from .tcn_decoder import TcnDecoder
from .fourier_decoder import FourierDecoder
from .gaussian_mlp_decoder import GaussianMLPDecoder
from .attn_decoder import AttnDecoder

# =========================================================================
# 解码器注册表：name -> class
# 添加新解码器只需在此处加一行
# =========================================================================
DECODER_REGISTRY = {
    'tcn': TcnDecoder,
    'fourier': FourierDecoder,
    'gaussian_mlp': GaussianMLPDecoder,   # 非频率 MLP INR（高斯激活）
    'attn': AttnDecoder,                  # 注意力（Transformer）INR
}


def create_decoder(decoder_type: str, **kwargs) -> BaseDecoder:
    """
    工厂函数：根据名称创建解码器实例。
    
    Args:
        decoder_type: 解码器类型名称（必须在 DECODER_REGISTRY 中注册）
        **kwargs: 传递给解码器构造函数的参数
            - dim (int): 状态向量总维度
            - trend_dim (int): 趋势分量维度
            - hidden_dim (int): 隐藏层维度
            - 以及各解码器特有的参数
    
    Returns:
        BaseDecoder 实例
        
    Raises:
        ValueError: 未注册的解码器类型
    """
    if decoder_type not in DECODER_REGISTRY:
        available = ', '.join(sorted(DECODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown decoder type '{decoder_type}'. "
            f"Available: [{available}]"
        )
    
    decoder_cls = DECODER_REGISTRY[decoder_type]
    return decoder_cls(**kwargs)


def list_decoders():
    """列出所有已注册的解码器。"""
    for name, cls in sorted(DECODER_REGISTRY.items()):
        print(f"  {name:20s} -> {cls.__module__}.{cls.__name__}")
