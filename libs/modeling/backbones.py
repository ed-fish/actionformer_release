import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock,ModalLocalMaskedMHCA, ModalTransformerBlock, MaskedConv1D,MaskedConv2D,
                     ConvBlock, LayerNorm, TemporalMaxer, TemporalMaxerMHA)


class AxialAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(AxialAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
    
    def forward(self, x, y, mask=None):
        B, C, T = x.size()
        x = x.permute(2, 0, 1)  # [T, B, C]
        y = y.permute(2, 0, 1)
        if mask is not None:
            mask  = mask.squeeze(1)
         
        # Axial attention along the temporal axis
        x, _ = self.attention(x, y, y, key_padding_mask=mask)
        
        x = x.permute(1, 2, 0)  # [B, C, T]
        return x, mask

class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim=256):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.fc1 = nn.Linear(16384, 512)
        
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))

        # Flatten spatial dimensions to get 512 features per time step
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, height]
        x = x.view(x.size(0), x.size(1), -1)  # [batch, time, 512]
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@register_backbone("convTransformerAudio")
class ConvTransformerAudioBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv2D(c0, c1, (n_embd_ks, n_embd_ks)) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        # self.embd = nn.ModuleList()
        # self.embd_norm = nn.ModuleList()
        # for idx in range(arch[0]):
        #     n_in = n_embd if idx > 0 else n_in
        #     self.embd.append(
        #         MaskedConv2D(
        #             n_in, n_embd, (n_embd_ks, n_embd_ks),
        #             stride=1, padding=(n_embd_ks//2,n_embd_ks//2), bias=(not with_ln)
        #         )
        #     )
        #     if with_ln:
        #         self.embd_norm.append(LayerNorm(n_embd))
        #     else:
        #         self.embd_norm.append(nn.Identity())
        
        self.embd = CNNEmbedding(n_embd)

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
            
        self.cross_att = nn.ModuleList()
        
        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
            
        
        self.pooling_layer = nn.AvgPool1d(kernel_size=5, stride=5)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
            
             
        # embedding network
        x = self.pooling_layer(x)
        x = x.unsqueeze(1)
        
        x = self.embd(x)
        x = x.permute(0, 2, 1)
            # mask = torch.zeros_like(x)
        
        # for idx in range(len(self.embd)):
        #     x, mask = self.embd[idx](x, mask)
        #     # x = F.max_pool2d(x, kernel_size=(128, 1), stride=(1, 1))
        #     # x = x.squeeze(2)
        #     # mask = mask.squeeze(2)
        #     # x = self.relu(self.embd_norm[idx](x))
        #     x = self.relu(x)
            
            
        # x = F.max_pool2d(x, kernel_size=(64, 1), stride=(1, 1))
        # x = x.squeeze()
        
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)
            
        # stem transformer
        # mask = mask.squeeze(1)
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
            
        # x = (2, 512, 2304)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )
        
        for param in self.branch[idx].parameters():
            assert not torch.isnan(param).any(), f"NaNs in params of TransformerBlock at idx {idx}"
        
        # main branch with downsampling - NANS appear here :/
        for idx in range(len(self.branch)):
            assert not torch.isnan(x).any(), f"NaNs in x before TransformerBlock at idx {idx}"
            assert not torch.isnan(mask).any(), f"NaNs in mask before TransformerBlock at idx {idx}"
            p_x = x.clone()
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )
            
            if torch.isnan(x).any():
                breakpoint()
                print("problem", p_x, p_x.shape)
            
            assert not torch.isnan(x).any(), f"NaNs in x after TransformerBlock at idx {idx}"
            assert not torch.isnan(mask).any(), f"NaNs in mask after TransformerBlock at idx {idx}"
            
                
        
        return out_feats, out_masks
    
    
@register_backbone("crossmodalTransformer")
class ConvModalTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_audio_embd,
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (1, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        with_ln = True

        # feature projection
        self.n_in = n_in
        n_a_in = n_audio_embd
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            # n_in = n_embd if idx == 2 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())
                
                 
        self.a_embd = nn.ModuleList()
        self.a_embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_a_in = n_embd if idx > 0 else n_a_in
            self.a_embd.append(
                MaskedConv1D(
                    n_a_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.a_embd_norm.append(nn.LayerNorm(n_embd))
            else:
                self.a_embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                ModalTransformerBlock(
                    n_embd, n_audio_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
            
        
        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                ModalTransformerBlock(
                    n_embd, n_audio_embd,n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x,audio,  mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        B_a, C_a, T_a = audio.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
         
        # embedding network
         
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
            audio, audio_mask = self.a_embd[idx](audio, mask)
            audio = self.relu(self.a_embd_norm[idx](audio))
            
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, audio, mask = self.stem[idx](x, audio, mask)
             
        # prep for outputs
        x_ = torch.cat((x, audio), dim=1)
        
        out_feats = (x_, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, audio, mask = self.branch[idx](x, audio, mask)
            x_ = torch.cat((x, audio), dim=1)
            
            out_feats += (x_, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        
        
        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
            

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )
            
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        # original!
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
            
        # x = (2, 512, 2304)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )
        

        # # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(nn.GroupNorm(16, n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
    

@register_backbone("convPooler")
class ConvPoolerBackbone(nn.Module):
    """
        A backbone that combines convolutions with TemporalMaxer
    """

    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch=(2, 5),           # (#convs, #branch TemporalMaxer)
        scale_factor=2,       # dowsampling rate for the branch
        with_ln=False,        # if to attach layernorm after conv
        **kwargs,
    ):
        super().__init__()
        assert len(arch) == 2
        self.n_in = n_in
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.GELU()
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)))

            if with_ln:
                self.embd_norm.append(nn.GroupNorm(16, n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # main branch using TemporalMaxer
        self.branch = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch.append(TemporalMaxer(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
    
@register_backbone("convPoolerMHA")
class ConvPoolerBackbone(nn.Module):
    """
        A backbone that combines convolutions with TemporalMaxer
    """

    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch=(2, 5),           # (#convs, #branch TemporalMaxer)
        scale_factor=2,       # dowsampling rate for the branch
        with_ln=False,        # if to attach layernorm after conv
        **kwargs,
    ):
        super().__init__()
        assert len(arch) == 2
        self.n_in = n_in
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)))

            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())
                 
        self.a_embd = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else 128
            self.a_embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)))

        # main branch using TemporalMaxer
        self.branch = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch.append(TemporalMaxerMHA(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, y, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
            y, _= self.a_embd[idx](y, mask)
            y = self.relu(self.embd_norm[idx](y))
            
        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x,y, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
    
@register_backbone("SGP")
class SGPBackbone(nn.Module):
    """
        A backbone that combines SGP layer with transformers
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            sgp_mlp_dim,  # the numnber of dim in SGP
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            path_pdrop=0.0,  # droput rate for drop path
            downsample_type='max',  # how to downsample feature in FPN
            sgp_win_size=[-1] * 6,  # size of local window for mha
            k=1.5,  # the K in SGP
            init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
            use_abs_pe=False,  # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(sgp_win_size) == (1 + arch[2])
        self.arch = arch
        self.sgp_win_size = sgp_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                SGPBlock(n_embd, 1, 1, n_hidden=sgp_mlp_dim, k=k, init_conv_vars=init_conv_vars))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(SGPBlock(n_embd, self.sgp_win_size[1 + idx], self.scale_factor, path_pdrop=path_pdrop,
                                        n_hidden=sgp_mlp_dim, downsample_type=downsample_type, k=k,
                                        init_conv_vars=init_conv_vars))
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem network
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
    
@register_backbone("unAV")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in_V,                # input visual feature dimension
        n_in_A,                # input audio feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]-1): 
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        for idx in range(arch[2]):
            self.cross_att_Va.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.cross_att_Av.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C_V, T = x_V.size()
        mask_V = mask_A = mask
        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # stem transformer
        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

        x_Va, mask_V = self.ori_cross_att_Va(x_V, x_A, mask_V) 
        x_Av, mask_V = self.ori_cross_att_Av(x_A, x_V, mask_A) 

        # prep for outputs
        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks_V = tuple()
        out_masks_A = tuple()
        # 1x resolution
        out_feats_V += (x_Va, )
        out_masks_V += (mask_V, )
        out_feats_A += (x_Av, )
        out_masks_A += (mask_A, )

        # main branch with downsampling
        for idx in range(len(self.cross_att_Va)):
            x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], out_feats_A[idx], mask_V)
            out_feats_V += (x_V, )
            out_masks_V += (mask_V, )

            x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], out_feats_V[idx], mask_A)
            out_feats_A += (x_A, )
            out_masks_A += (mask_A, )

        return out_feats_V, out_feats_A, out_masks_V
        