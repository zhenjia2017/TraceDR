import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim=32, output_dim=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.aggregation = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x, mask=None, device='cpu'):
        # x: (batch_size, num_symptoms, embed_dim) 或 (num_symptoms, embed_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_symptoms, embed_dim]
        elif x.dim() != 3:
            raise ValueError(f"输入张量必须是二维或三维，实际维度为 {x.dim()}")

        weight = self.aggregation(x)  # [batch_size, num_symptoms, 1]
        weight = torch.tanh(weight)
        if mask is not None:
            weight = weight.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        weight = F.softmax(weight, dim=1)  # [batch_size, num_symptoms, 1]
        agg_embeds = torch.bmm(x.transpose(1, 2), weight).squeeze(-1)  # [batch_size, embed_dim]
        if agg_embeds.dim() == 2 and agg_embeds.shape[0] == 1:
            agg_embeds = agg_embeds.squeeze(0)  # [embed_dim] 如果 batch_size=1
        return agg_embeds