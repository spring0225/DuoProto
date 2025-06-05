import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ViT import ViT, ViTMIL

class MultiphaseViTMIL(nn.Module):
    """
    Multi-phase Vision Transformer Multiple Instance Learning model.
    Extends ViTMIL to handle multiple phases (pre, arterial, PV, delay) as input.
    """
    def __init__(
        self,
        in_channels=1,
        img_size=(192, 192, 192),
        patch_size=(32, 32, 32),
        pooling_type="avg",
        mil_type="embedding",
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        pos_embed="conv",
        proj_type="conv",
        pos_embed_type="learnable",
        classification=False,
        num_classes=2,
        dropout_rate=0.0,
        spatial_dims=3,
        post_activation="Tanh",
        qkv_bias=False,
        save_attn=False,
        mode="train",
        k=1.0,
        fusion_method="concat",  # 'concat', 'attention', or 'gated'
    ):
        """
        Initialize the Multi-phase ViTMIL model.
        
        Args:
            in_channels (int): Number of input channels per phase
            fusion_method (str): Method to fuse features from different phases 
                                ('concat', 'attention', or 'gated')
            Other arguments are the same as ViTMIL
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        self.pooling_type = pooling_type
        self.mil_type = mil_type
        self.num_classes = num_classes
        self.mode = mode
        self.k = k
        
        # Create a separate ViT for each phase
        self.pre_vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=False,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            k=k
        )
        
        self.arterial_vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=False,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            k=k
        )
        
        self.pv_vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=False,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            k=k
        )
        
        self.delay_vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=False,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
            k=k
        )
        
        if fusion_method == "concat":
            self.concat_proj = nn.Linear(hidden_size * 4, hidden_size)  
        else:
            self.concat_proj = nn.Identity()

        
        # Phase Fusion Module
        if fusion_method == "concat":
            # Concatenate features from all phases
            feature_size = hidden_size * 4  
            self.concat_proj = nn.Linear(feature_size, hidden_size)  
            feature_size = hidden_size  
        elif fusion_method == "attention":
            # Attention-based fusion
            feature_size = hidden_size
            self.fusion = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.fusion_norm = nn.LayerNorm(hidden_size)
        elif fusion_method == "gated":
            # Gated fusion
            feature_size = hidden_size
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 4, 4),
                nn.Softmax(dim=-1)
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Main ER classification head
        self.classifier = nn.Linear(feature_size, num_classes)
        
        # Attention for MIL
        self.attention = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.Tanh(),
            nn.Linear(feature_size // 2, 1),
            nn.Dropout(0.1)
        )
        
    def save_patch_probs(self, x):
        self.patch_probs = x

    def get_patch_probs(self):
        return self.patch_probs
    
    def save_softmax_bag_probs(self, x):
        self.softmax_probs = x

    def get_softmax_bag_probs(self):
        return self.softmax_probs
    
    def get_activations(self, x):
        return self.pv_vit.get_activations(x)

    def get_activations_gradient(self):
        return self.pv_vit.get_activations_gradient()

    def activations_hook(self, grad):
        self.gradients = grad

    
    
    def AvgPooling(self, representation):
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the features."""
            pooled_feature = torch.mean(representation, dim=1)
            return pooled_feature
        elif self.mil_type == "instance":
            pooled_represent = torch.mean(representation, dim=1)
            pooled_probs = torch.softmax(pooled_represent, dim=1)
            return pooled_represent, pooled_probs
    
    def TopKPooling(self, representation, topk):
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the top k features."""
            pooled_feature, _ = torch.topk(representation, k=topk, dim=1)
            pooled_feature = torch.mean(pooled_feature, dim=1)
            return pooled_feature
        elif self.mil_type == "instance":
            if self.num_classes == 2:
                representation = torch.softmax(representation, dim=2)
                pooled_probs, pooled_idxs = torch.topk(representation[:,:,0], k=topk, dim=1)
                pooled_probs = representation[torch.arange(pooled_probs.shape[0]).unsqueeze(1), pooled_idxs]
                pooled_probs = torch.mean(pooled_probs, dim=1)
            else:
                pooled_scores, pooled_idxs = torch.topk(representation, k=topk, dim=1)
                pooled_scores = torch.mean(pooled_scores, dim=1)
                pooled_probs = torch.softmax(pooled_scores, dim=1)
            return pooled_scores, pooled_probs
        
    def AttentionPooling(self, representation, mask=None):
        """Attention-based pooling"""
        A = self.attention(representation)  # [B, N, 1]
        A = F.softmax(A, dim=1)             # [B, N, 1]
        
        # Save attention weights for visualization
        self.slice_att_weights = A.detach().cpu().squeeze(-1)  # [B, N]
        
        # Weighted average of representations
        pooled_feature = torch.sum(A * representation, dim=1)  # [B, C]
        
        return pooled_feature

    def MilPooling(self, x, mask=None, topk=25):
        """Apply MIL pooling to the input representation."""
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            x = self.TopKPooling(x, topk)
        elif self.pooling_type == "att":
            x = self.AttentionPooling(x, mask)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}")
        
        return x
    
    def fusion_features(self, pre_features, art_features, pv_features, delay_features):
        """
        Fuse features from different phases.
        
        Args:
            pre_features (torch.Tensor): Pre-contrast phase features [B, N, C]
            art_features (torch.Tensor): Arterial phase features [B, N, C]
            pv_features (torch.Tensor): Portal venous phase features [B, N, C]
            delay_features (torch.Tensor): Delayed phase features [B, N, C]
            
        Returns:
            torch.Tensor: Fused features
        """
        if self.fusion_method == "concat":
            # Concatenate along the feature dimension
            return torch.cat([pre_features, art_features, pv_features, delay_features], dim=2)
        
        elif self.fusion_method == "attention":
            # Reshape for multi-head attention
            # Stack phases along sequence dimension
            B, N, C = pre_features.shape
            stacked = torch.stack([pre_features, art_features, pv_features, delay_features], dim=2)
            stacked = stacked.view(B * N, 4, C)  # [B*N, 4, C]
            
            # Use pv_features as query and all phases as key/value
            query = pv_features.view(B * N, 1, C)  # [B*N, 1, C]
            
            # Apply multi-head attention
            attn_output, _ = self.fusion(query, stacked, stacked)
            attn_output = self.fusion_norm(attn_output)
            
            # Reshape back
            return attn_output.view(B, N, C)
        
        elif self.fusion_method == "gated":
            # Calculate weights for each phase
            B, N, C = pre_features.shape
            
            # Reshape to combine batch and instance dimensions
            pre_flat = pre_features.view(B * N, C)
            art_flat = art_features.view(B * N, C)
            pv_flat = pv_features.view(B * N, C)
            delay_flat = delay_features.view(B * N, C)
            
            # Concatenate for gate calculation
            concat_features = torch.cat([pre_flat, art_flat, pv_flat, delay_flat], dim=1)
            
            # Calculate gate weights
            gates = self.fusion(concat_features).view(B * N, 4, 1)
            
            # Apply gates to each phase
            stacked = torch.stack([pre_flat, art_flat, pv_flat, delay_flat], dim=1)  # [B*N, 4, C]
            gated = gates * stacked
            
            # Sum across phases
            fused = torch.sum(gated, dim=1).view(B, N, C)
            
            return fused

    def forward(self, inputs):
        """
        Forward pass for the multi-phase model.
        
        Args:
            inputs (tuple): Tuple of (pre_img, arterial_img, pv_img, delay_img)
                           Each with shape [B, C, H, W, D]
                           
        Returns:
            logit: Classification logits
        """
        pre_img, arterial_img, pv_img, delay_img = inputs
        
        # Process each phase through its respective ViT
        pre_features = self.pre_vit(pre_img)  # [B, N, C]
        art_features = self.arterial_vit(arterial_img)  # [B, N, C]
        pv_features = self.pv_vit(pv_img)  # [B, N, C]
        delay_features = self.delay_vit(delay_img)  # [B, N, C]
        
         # === Save individual phase features for visualization ===
        self.phase_features = {
            "pre": pre_features.detach(),
            "art": art_features.detach(),
            "pv": pv_features.detach(),
            "delay": delay_features.detach(),
        }
        
        # Fuse features from different phases
        fused_features = self.fusion_features(pre_features, art_features, pv_features, delay_features)

        fused_features = self.concat_proj(fused_features)  
        
        self.fused_multiphase_features = fused_features.detach()  # Store before pooling
        
        if self.mil_type == "instance":
            # Apply classifier to each instance
            x = self.classifier(fused_features)
            
            # Store patch probabilities
            self.save_patch_probs(torch.softmax(x, dim=2))
            
            # Apply MIL pooling
            topk = round(self.k * x.size(1))
            logit, prob = self.MilPooling(x, mask=None, topk=topk)
            
            # Store pooled features
            self.last_features = logit.detach()
            
            # Save softmax probabilities of the bag
            self.save_softmax_bag_probs(prob)
            
            return logit
        else:  
            topk = round(float(self.k) * int(fused_features.size(1)))
            x = self.MilPooling(fused_features, mask=None, topk=topk)

            self.last_features = x.detach()
            
            x = self.classifier(x)
            prob = F.softmax(x, dim=-1)
            self.save_softmax_bag_probs(prob)
            
            return x
