import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoder(nn.Module):
    def __init__(self, output_dim):
        super(PositionEncoder, self).__init__()
        self.fc = nn.Linear(2, output_dim)

    def forward(self, pos):
        # pos: [batch_size, 2]
        pos_emb = self.fc(pos)
        # pos_emb: [batch_size, output_dim]
        return pos_emb


class StructureEncoder(nn.Module):
    def __init__(self, output_dim):
        super(StructureEncoder, self).__init__()
        self.conv = nn.Sequential(
            # TODO: structure channel number
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [batch_size, 1, 1024, 512]
        features = self.conv(x)
        # features: [batch_size, 64, 1, 1]
        features = features.view(features.size(0), -1)
        # features: [batch_size, 64]
        out = self.fc(features)
        # out: [batch_size, output_dim]
        return out


class MixedViewDiff(nn.Module):
    def __init__(self, position_dim=64, structure_dim=64, hidden_dim=128):
        super(MixedViewDiff, self).__init__()
        self.position_encoder = PositionEncoder(position_dim)
        self.structure_encoder = StructureEncoder(structure_dim)
        self.query_mlp = nn.Linear(
            position_dim + structure_dim, hidden_dim)  # TODO: how mlp works
        self.key_mlp = nn.Linear(position_dim + structure_dim, hidden_dim)
        # self.value_conv = nn.Conv2d(3, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, target_pos, target_structure, surround_pos, surround_structure, surround_panoramas):
        batch_size = target_pos.size(0)
        t = surround_pos.size(1)

        # === Target ===
        target_pos_emb = self.position_encoder(target_pos)
        # target_pos_emb: [batch_size, position_dim]

        target_structure_emb = self.structure_encoder(target_structure)
        # target_structure_emb: [batch_size, structure_dim]

        target_features = torch.cat(
            [target_pos_emb, target_structure_emb], dim=1)
        # target_features: [batch_size, position_dim + structure_dim]

        Q = self.query_mlp(target_features)
        # Q: [batch_size, hidden_dim]
        Q = Q.unsqueeze(1)
        # Q: [batch_size, 1, hidden_dim]

        # === Surroundings ===
        surround_pos_flat = surround_pos.view(batch_size * t, 2)
        # surround_pos_flat: [batch_size * t, 2]
        surround_pos_emb = self.position_encoder(surround_pos_flat)
        # surround_pos_emb: [batch_size * t, position_dim]

        surround_structure_flat = surround_structure.view(
            batch_size * t, 1, 1024, 512)
        # surround_structure_flat: [batch_size * t, 1, 1024, 512]
        surround_structure_emb = self.structure_encoder(
            surround_structure_flat)
        # surround_structure_emb: [batch_size * t, structure_dim]

        surround_features = torch.cat(
            [surround_pos_emb, surround_structure_emb], dim=1)
        # surround_features: [batch_size * t, position_dim + structure_dim]
        K = self.key_mlp(surround_features)
        # K: [batch_size * t, hidden_dim]
        K = K.view(batch_size, t, -1)
        # K: [batch_size, t, hidden_dim]

        surround_panoramas_flat = surround_panoramas.view(
            batch_size, t, 3, 1024, 512)
        # surround_panoramas_flat: [batch_size, t, 3, 1024, 512]
        V = surround_panoramas_flat
        # V: [batch_size, t, 3, 1024, 512]

        # === Cross Attention Calculation ===

        attention_scores = torch.matmul(Q, K.transpose(1, 2))
        # attention_scores: [batch_size, 1, t]
        attention_weights = F.softmax(
            attention_scores / (self.hidden_dim ** 0.5), dim=-1)
        # attention_weights: [batch_size, 1, t]

        attention_weights = attention_weights.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1)
        # attention_weights: [batch_size, 1, t, 1, 1]

        V = V.permute(0, 2, 1, 3, 4)
        # V: [batch_size, 3, t, 1024, 512]

        weighted_V = torch.sum(V * attention_weights, dim=2)
        # weighted_V: [batch_size, 3, 1024, 512]

        output = weighted_V
        # output: [batch_size, 3, 1024, 512]

        return output
