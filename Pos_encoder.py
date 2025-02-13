import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEmbeddingSine(nn.Module):
    """
    Taken from official PyTorch code for Vision Transformer.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) #(3, 8, 8)
        x_embed = not_mask.cumsum(2, dtype=torch.float32) #(3, 8, 8)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) 
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) #(64,) (dim_t // 2 will change dim_t from (0,1,2,3,4,5,6,7,..)----->(0,0,1,1,2,2,3,3,..) cuz to have same frequency for sin and cos)

        pos_x = x_embed[:, :, :, None] / dim_t #(3, 8, 8, 64)
        pos_y = y_embed[:, :, :, None] / dim_t #(3, 8, 8, 64)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # Select even indices -> (3, 8, 8, 32 , 2)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3) # Select odd indices -> (3, 8, 8, 32 , 2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos # (3, 128, 8, 8)

# x = torch.randn(2, 512, 19, 25)
# po = PositionEmbeddingSine(256)
# mask = torch.ones(2, 19, 25).bool()
# pos_embed = po(x, mask)
# print(pos_embed.shape)
