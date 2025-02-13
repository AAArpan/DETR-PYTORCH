import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add the parent directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes scaled dot-product attention.
    Args:
        Q: Query tensor (batch_size, num_heads, seq_len, d_k=d_model/heads)
        K: Key tensor (batch_size, num_heads, seq_len, d_k=d_model/heads)
        V: Value tensor (batch_size, num_heads, seq_len, d_k=d_model/heads)
        mask: Optional mask (useful for preventing attention to certain tokens)
    Returns:
        Output tensor and attention weights.
    """
    d_k = Q.shape[-1]  # Dimension of keys
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # QK^T / sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))  

    attention_weights = F.softmax(scores, dim=-1) 
    output = torch.matmul(attention_weights, V) 
    return output, attention_weights


class MultiheadAttention(nn.Module):
    """
    Computes the multihead attention.
    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, HW, d_model).
        K (torch.Tensor): Key tensor of shape (batch_size, HW, d_model).
        V (torch.Tensor): Value tensor of shape (batch_size, HW, d_model).
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len). Default is None.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, HW, d_model).
    """
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
    
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        if Q.dim() == 2 or K.dim() == 2 or V.dim() == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)
        
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # print(Q.shape)
        # print(K.shape)
        # print(V.shape)

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        #print(output.shape) # (batch_size, num_heads, seq_len, d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        #print(output.shape)
        return self.W_O(output)
    
class AddNormalization(nn.Module):
    """
    Computes the layer normalization.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, HW, d_model).
        sublayer_x (torch.Tensor): Output tensor of the sublayer.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, HW, d_model).
    """
    def __init__(self, d_model):
        super(AddNormalization, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_x):
        return self.norm(x + sublayer_x)
    
class FeedForward(nn.Module):
    """
    Computes the feedforward layer.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, HW, d_model).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, HW, d_model).
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class EncoderLayer(nn.Module):
    """
    Computes the encoder layer.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, HW, d_model).
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len). Default is None.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, HW, d_model).
    """
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiheadAttention(d_model, num_heads)
        self.add_norm1 = AddNormalization(d_model)
        self.feed_forward = FeedForward(d_model)
        self.add_norm2 = AddNormalization(d_model)

    def forward(self, x, pos_embed, mask=None):
        sublayer_x = self.multihead_attention(x, x, x, mask=None)
        x = self.add_norm1(x, sublayer_x)
        sublayer_x = self.feed_forward(x)
        return self.add_norm2(x, sublayer_x)
    
class Encoder(nn.Module):
    """
    Computes the encoder.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, HW, d_model).
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len). Default is None.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, HW, d_model).
    """
    def __init__(self, num_layers, d_model, num_heads):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, pos_embed, mask=None):
        x = x + pos_embed
        x = x.flatten(2).permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x, pos_embed, mask)
        return x
    
class DecoderLayer(nn.Module):
    """
    Computes the decoder layer.
    Args:
        x(torch.Tensor): Input tensor of shape (batch_size, Object_queries, d_model).
        encoder_output (torch.Tensor): Memory tensor of shape (batch_size, HW, d_model).
        src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, seq_len). Default is None.
        tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, seq_len). Default is None.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size,Object_queries, d_model).
    """
    def __init__(self, d_model, num_heads, rate):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiheadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(rate)
        self.add_norm1 = AddNormalization(d_model)
        self.multihead_attention2 = MultiheadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(rate)
        self.add_norm2 = AddNormalization(d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout3 = nn.Dropout(rate)
        self.add_norm3 = AddNormalization(d_model)

    def forward(self, target, query_embed, encoder_output, src_mask=None, tgt_mask=None):
        x = target + query_embed
        sublayer_x = self.multihead_attention1(x, x, x, mask=tgt_mask)
        sublayer_x = self.dropout1(sublayer_x)
        x = self.add_norm1(x, sublayer_x)
        sublayer_x = self.multihead_attention2(x, encoder_output, encoder_output, mask=src_mask)
        sublayer_x = self.dropout2(sublayer_x)
        x = self.add_norm2(x, sublayer_x)
        sublayer_x = self.feed_forward(x)
        sublayer_x = self.dropout3(sublayer_x)
        return self.add_norm3(x, sublayer_x)

# class Decoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, rate):
#         super(Decoder, self).__init__()
#         self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, rate) for _ in range(num_layers)])

#     def forward(self, target, query_embed, encoder_output,src_mask=None, tgt_mask=None):
#         for layer in self.decoder_layers:
#             x = layer(target, query_embed, encoder_output,src_mask, tgt_mask)

#         return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, rate):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, rate) for _ in range(num_layers)])

    def forward(self, target, query_embed, encoder_output, src_mask=None, tgt_mask=None):
        outputs = []  
        x = target 

        for layer in self.decoder_layers:
            x = layer(x, query_embed, encoder_output, src_mask, tgt_mask)
            outputs.append(x)  

        outputs = torch.stack(outputs)
        return outputs     

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, rate):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads)
        self.decoder = Decoder(num_layers, d_model, num_heads, rate)
        
    def forward(self, x, query_embed, pos_embed, mask=None):
        bs, c, h, w = x.shape
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        target = torch.zeros_like(query_embed)
        encoder_output = self.encoder(x, pos_embed, mask)
        decoder_output = self.decoder(target, query_embed , encoder_output, src_mask=None, tgt_mask=None)
        return decoder_output


# x = torch.randn(2, 512, 7, 7)
# po = PositionEmbeddingSine(256)
# mask = torch.ones(2, 7, 7).bool()
# pos_embed = po(x, mask)
# query_embed = nn.Embedding(100, 512)

# output = transformer(x, query_embed.weight, pos_embed, mask)





# # Test multihead attention
# d_model = 512
# num_heads = 8
# seq_len = 10
# batch_size = 32
# ma = MultiheadAttention(d_model, num_heads)
# Q = torch.randn(batch_size, seq_len, d_model)
# K = torch.randn(batch_size, seq_len, d_model)
# V = torch.randn(batch_size, seq_len, d_model)
# output = ma(Q, K, V)
# print(output.shape)
# print(output)