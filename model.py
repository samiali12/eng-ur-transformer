import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        self.d_model =  d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 0::1] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :] 
        return self.dropout(x)
    

class LayerNormalizaion(nn.Module):
    def __init__(self, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (self.eps + std) + self.beta

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model:  int, d_ff: int, dropout: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 



class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: int):
        self.d_model = d_model
        self.num_head = num_head
    
        assert d_model % num_head == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model / num_head
        self.q_w = nn.Linear(d_model, d_model)
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.o_w = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if mask is not None:
            attention_score.masked_fill(mask==0, -1e9)
            attention_score = attention_score.softmax(dim=-1) # corrected line
        if dropout is not None:
            attention_score = dropout(attention_score)
            return (attention_score @ value), attention_score

    def forward(self, query, key, value, mask=None):
        query = self.q_w(query)
        key = self.k_w(key)
        value = self.v_w(value)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)

        x, self.attention_score = MultiheadAttention.attention(query, key, value, mask ,self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        return self.o_w(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):    
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention: MultiheadAttention, feedforward: FeedForwardLayer, dropout: float):
        self.multihead_attention = multihead_attention
        self.feedforward = feedforward
        self.residual_connections = ResidualConnection(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connections(x, lambda x: self.multihead_attention(x, x, x, src_mask))
        x = self.feedforward(x, lambda x: self.feedforward(x))
        return x        

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizaion()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.ModuleList):
    def __init__(self, multihead_attention: MultiheadAttention, 
                 cross_attention: MultiheadAttention, 
                 feedforward: FeedForwardLayer, dropout: float):
        self.multihead_attention = multihead_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for i in range(3))

    def forward(self, x, encoder_output, src_mask, tar_mask):
        x = self.residual_connections[0](x, lambda x: self.multihead_attention(x, x, x, tar_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feedforward(x))
        return x 
    

class Decoder(nn.ModuleList):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizaion()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    return torch.log_softmax(self.proj(x), dim=-1)
  

class Transformer(nn.Module):
    def __init__(self, input_embedding: InputEmbedding, output_embedding: InputEmbedding,
                input_pe: PositionalEncoding, output_pe: PositionalEncoding,
                encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer ):
        super().__init__()
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.input_pe = input_pe
        self.output_pe = output_pe 
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encoder(self, input_embd, mask):
        src = self.input_embedding(input_embd)
        src = self.input_pe(src)
        return self.encoder(src)
    
    def decode(self, output_embd, encoder_ouput, src_mask,  tar_mask):
        tar = self.output_embedding(output_embd)
        tar = self.output_embedding(encoder_ouput)
        return self.decoder(tar, encoder_ouput, src_mask, tar_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    


def build_transformer(input_vocab_size: int, 
                    output_vocab_size: int, 
                    input_seq_length: int, 
                    output_seq_length: int,
                    d_model: int, 
                    ffn_hidden_dim: int, 
                    num_heads: int, 
                    dropout: float):
    input_embedding = InputEmbedding(vocab_size=input_vocab_size, d_model=d_model)
    output_embedding = InputEmbedding(vocab_size=output_embedding, d_model=d_model)

    input_pe = PositionalEncoding(d_model=d_model, seq_len=input_seq_length, dropout=dropout)
    output_pe = PositionalEncoding(d_model=d_model, seq_len=output_seq_length, dropout=dropout)

    encoder_blocks = []
    for i in range(6):
        multihead_attention = MultiheadAttention(d_model=d_model, num_head=num_heads, dropout=dropout)
        feedforward = FeedForwardLayer(d_model=d_model, d_ff=ffn_hidden_dim, dropout=dropout)
        encoder = EncoderBlock(multihead_attention=multihead_attention, feedforward=feedforward, dropout=dropout)
        encoder_blocks.append(encoder)

    
    decoder_blocks = []
    for i in range(6):
        multihead_attention = MultiheadAttention(d_model=d_model, num_head=num_heads, dropout=dropout)
        cross_multihead_attention = MultiheadAttention(d_model=d_model, num_head=num_heads, dropout=dropout)
        feedforward = FeedForwardLayer(d_model=d_model, d_ff=ffn_hidden_dim, dropout=dropout)
        decoder = DecoderBlock(multihead_attention=multihead_attention, cross_attention=cross_multihead_attention, 
                               feedforward=feedforward, dropout=dropout)
        decoder_blocks.append(decoder)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=output_vocab_size)

    transformer = Transformer(input_embedding, output_embedding, input_pe, output_pe, encoder, decoder, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


