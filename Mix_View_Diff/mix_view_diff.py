import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super(CustomMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(self.kdim, embed_dim)
        self.value_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query_proj(query)  # [batch_size, seq_len_q, embed_dim]
        K = self.key_proj(key)      # [batch_size, seq_len_k, embed_dim]
        V = self.value_proj(value)  # [batch_size, seq_len_v, embed_dim]

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len_q, head_dim]

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(attn_output)  # [batch_size, seq_len_q, embed_dim]
        return output, attn_weights


class PositionEncoder(nn.Module):  # Nerf position encoder
    def __init__(self, output_dim, input_dim=2):
        super(PositionEncoder, self).__init__()
        self.input_dim = input_dim
        L_embed = int(output_dim / 2)
        self.L_embed = L_embed
        self.freq_bands = 2.0 ** torch.linspace(0, L_embed - 1, L_embed)

    def forward(self, pos):
        rets = []
        for freq in self.freq_bands:
            rets.append(torch.sin(freq * pos))
            rets.append(torch.cos(freq * pos))
        pos_enc = torch.cat(rets, dim=-1)
        return pos_enc


class NearbyEncoder(nn.Module):
    def __init__(self, pos_hidden_dim=12, num_heads=4, surround_panoramas=3):
        super(NearbyEncoder, self).__init__()

        self.surround_panoramas = surround_panoramas
        self.pos_hidden_dim = pos_hidden_dim

        base_resnet = models.resnet18() # Use resnet18
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet.children())[1:-2],
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.position_encoder = PositionEncoder(output_dim=pos_hidden_dim // 2, input_dim=2)

        self.panorama_encoder = nn.Sequential(
            *list(models.resnet18().children())[:-2] # Use resnet18
        )

        self.multi_attention = CustomMultiHeadAttention(
            embed_dim=512 + pos_hidden_dim, # Adjusted embed_dim
            num_heads=num_heads,
            kdim=512 + pos_hidden_dim, # Adjusted kdim
            vdim=512*16*16           # Adjusted vdim
        )

        self.attn_to_2d = nn.Linear(512 + pos_hidden_dim, 512 * 16 * 16) # Adjusted Linear layer

        self.combined_hint_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted channels
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 32),
             nn.ConvTranspose2d(32, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )

    def forward(self, target_pos, target_structure, surround_pos, surround_structure, surround_panoramas, verbose=False):
        target_pos_cut = target_pos[:, :, 0, 0]   
        target_pos_features = self.position_encoder(target_pos_cut)  

        target_structure_features = self.structure_encoder(target_structure).squeeze(-1).squeeze(-1)

        target_combined_features = torch.cat([target_structure_features, target_pos_features], dim=1)  
        target_combined_features = target_combined_features.unsqueeze(1)
        
        batch_size, T, C, H, W = surround_structure.shape
        surround_structure_reshaped = surround_structure.reshape(batch_size * T, C, H, W)
        surround_structure_features_reshaped = self.structure_encoder(surround_structure_reshaped).squeeze(-1).squeeze(-1)
        surround_structure_features = surround_structure_features_reshaped.view(batch_size, T, 512) # Adjusted dim

        surround_pos_cut = surround_pos[:, :, 0, 0]
        surround_pos_reshaped = surround_pos_cut.reshape(batch_size * T, 2)
        surround_pos_features_reshaped = self.position_encoder(surround_pos_reshaped)
        surround_pos_features = surround_pos_features_reshaped.view(batch_size, T, self.pos_hidden_dim)

        surround_combined_features = torch.cat([surround_structure_features, surround_pos_features], dim=2)

        panoramas_reshaped = surround_panoramas.view(batch_size * T, 3, 512, 512)
        panoramas_reshaped_features = self.panorama_encoder(panoramas_reshaped).view(batch_size, T, 512 * 16 * 16) # Adjusted dim

        attn_output, attn_weights = self.multi_attention(
            query=target_combined_features,
            key=surround_combined_features,
            value=panoramas_reshaped_features
        )

        attn_output_1d = self.attn_to_2d(attn_output.squeeze(1))
        attn_output_2d = attn_output_1d.view(batch_size, 512, 16, 16) # Adjusted dim

        decoded_output = self.combined_hint_decoder(attn_output_2d)
        return decoded_output


class CrossViewEncoder(nn.Module):
    def __init__(self):
        super(CrossViewEncoder, self).__init__()
        base_resnet_sat = models.resnet18() # Use resnet18
        self.sat_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet_sat.children())[1:-2]
        )

        base_resnet_str = models.resnet18() # Use resnet18
        self.str_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(base_resnet_str.children())[1:-2]
        )
        # Output [batch_size,512,16,16]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Adjusted channels
            nn.GroupNorm(16, 128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted channels
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted channels
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )

    def forward(self, target_structure, target_satellite, verbose=False):
        # target_satellite: [batch_size,3,512,512]
        # target_structure: [batch_size,1,512,512]

        sat_feat = self.sat_encoder(target_satellite)      
        # [batch_size,512,16,16]
        str_feat = self.str_encoder(target_structure)      
        # [batch_size,512,16,16]

        if verbose: 
            print("sat_feat:", sat_feat.shape)
            print("str_feat:", str_feat.shape)
    
        combined_feat = torch.cat([sat_feat, str_feat], dim=1)
        # [batch_size,1024,16,16]
        if verbose: print("combined_feat (CrossView):", combined_feat.shape)

        out = self.decoder(combined_feat)
        # [batch_size,128,512,512]
        if verbose: print("crossview_decoder_output:", out.shape)

        return out


class MixedViewDiff(nn.Module):
    def __init__(self):
        super(MixedViewDiff, self).__init__()
        self.nearby_encoder = NearbyEncoder()
        self.crossview_encoder = CrossViewEncoder()
        # self.view_result = nn.Sequential(
        #     nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh()
        # )

    def forward(self, target_pos, target_structure, target_satellite, surround_pos, surround_structure, surround_panoramas, verbose=False, view_result=False):
        crossview_embedding = self.crossview_encoder(target_structure, target_satellite, verbose=verbose)  
        # [batch_size,128,512,512]
        nearby_embedding = self.nearby_encoder(target_pos, target_structure, surround_pos, surround_structure, surround_panoramas, verbose=verbose)
        # [batch_size,128,512,512]

        if verbose: 
            print("crossview_embedding:", crossview_embedding.shape)
            print("nearby_embedding:", nearby_embedding.shape)

        output = torch.cat([crossview_embedding, nearby_embedding], dim=1)

        # if view_result:
        #     output = self.view_result(output)
        # [batch_size,256,512,512]
        if verbose: print("final_output:", output.shape)
        return output


def main(verbose=True):
    batch_size = 2
    T = 3
    target_pos = torch.randn(batch_size, 2, 512, 512)            # [batch_size,2,512,512]
    target_structure = torch.randn(batch_size, 1, 512, 512)  # [batch_size,1,512,512]
    target_satellite = torch.randn(batch_size, 3, 512,512)    # [batch_size,3,512,512]
    surround_pos = torch.randn(batch_size, T * 2,1,1)          # [batch_size,T * 2,1,1]
    surround_structure = torch.randn(batch_size, T, 1,512,512)# [batch_size,T,1,512,512]
    surround_panoramas = torch.randn(batch_size, T, 3,512,512)# [batch_size,T,3,512,512]

    model = MixedViewDiff()
    output = model(target_pos, target_structure, target_satellite, surround_pos, surround_structure, surround_panoramas, verbose=verbose)
    if verbose: print("output:", output.shape)

if __name__ == "__main__":
    main(verbose=True)