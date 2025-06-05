from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks import PatchEmbed
from monai.utils import deprecated_arg

from timm_3d import create_model


__all__ = ["ViT"]


class ViTMIL(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """


    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        pooling_type: str,
        mil_type: str,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        mode="train",
        k=1.0,
        multi_task=False,
        bclc_classes=4
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to
            multi_task (bool, optional): whether to use multi-task learning (ER + BCLC)
            bclc_classes (int, optional): number of BCLC classes. Defaults to 4.
        """

        super().__init__()


        self.ViT = ViT(
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

        
        self.classifier = nn.Sequential(
            # nn.Linear(hidden_size, num_classes),
            nn.Linear(hidden_size, num_classes),
        )

        self.multi_task = multi_task
        if multi_task:
            self.bclc_classifier = nn.Sequential(
                nn.Linear(hidden_size, bclc_classes),
            )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*4),
            nn.ReLU(), 
            nn.Dropout(p=0.0),  
            nn.Linear(hidden_size*4, num_classes)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Dropout(0.1)
        )


        self.pooling_type = pooling_type
        self.mil_type = mil_type
        self.num_classes = num_classes
        self.mode = mode
        self.k = k
        self.bclc_classes = bclc_classes
    
    def save_patch_probs(self, x):
        self.patch_probs = x

    def get_patch_probs(self):
        return self.patch_probs
    
    def save_softmax_bag_probs(self, x):
        self.softmax_probs = x

    def get_softmax_bag_probs(self):
        return self.softmax_probs
    
    def save_bclc_probs(self, x):
        self.bclc_probs = x
        
    def get_bclc_probs(self):
        return self.bclc_probs
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.ViT.get_activations_gradient()
    
    def get_activations(self, x):
        return self.ViT.resnet(x)
    
    def AvgPooling(self, representation):
        
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the features of all the instances in the bag.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature = torch.mean(representation, dim=1)
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ The representation of a given bag is given by the average
            of the softmax probabilities of all the instances (patches) in the bag (image). Note: for the melanoma class.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """

            pooled_represent= torch.mean(representation, dim=1)
            pooled_probs = torch.softmax(pooled_represent, dim=1)
                
            return pooled_represent, pooled_probs
    
    def TopKPooling(self, representation, topk):
        
        if self.mil_type == "embedding":
            """The representation of a given bag is given by the average value of the top k features of all the instances in the bag.
            Args:
                representation (torch.Tensor): features of each instance in the bag. Shape (Batch_size, N, embedding_size).
                topk (torch.Tensor): Number of top k representation to be pooled. Default = 25.
            Returns:
                torch.Tensor: Pooled features. Shape (Batch_size, embedding_size).
            """
            pooled_feature,_ = torch.topk(representation, k=topk, dim=1)
            pooled_feature = torch.mean(pooled_feature, dim=1)
            return pooled_feature
        
        elif self.mil_type == "instance":
            """ The representation of a given bag is given by the average of the softmax probabilities
            of the top k instances (patches) in the bag (image). Note: for the melanoma class.
            Args:
                representation (torch.Tensor): probs of each instance in the bag. Shape (Batch_size, N, num_classes).
            Returns:
                torch.Tensor: Pooled probs. Shape (Batch_size, num_classes). 
            """
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

        A = self.attention(representation)  # [B, N, 1]
        print("attention logits min/max:", A.min(), A.max())

        A = F.softmax(A, dim=1)             # [B, N, 1]

        self.slice_att_weights = A.detach().cpu().squeeze(-1)  # [B, N]

        pooled_feature = torch.sum(A * representation, dim=1)  # [B, C]
        
        return pooled_feature






    def MilPooling(self, x:torch.Tensor, mask:torch.Tensor=None, topk:int=25) -> torch.Tensor:
        """ This function applies the MIL-Pooling to the input representation.
            Note that the shape of the input representation depends on the Mil-type.
            Note that the formulation of the "MIL-Problem" is different when we are in the Multiclass case.
        Args:
            x (torch.Tensor): Input representation. The shape of this tensor depends on the Mil-type.
                                If Mil-type is 'embedding', the shape is (Batch_size, N, embedding_size).
                                If Mil-type is 'instance', the shape is (Batch_size, N, num_classes).
            mask (torch.Tensor): Binary Masks. Shape: (Batch_size, 1, 224, 224). Defaults to None.
        Raises:
            ValueError: If pooling type is not 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'. Raises ValueError.
        Returns:
            torch.Tensor: Pooled representation. Shape (Batch_size, embedding_size) or (Batch_size, num_classes).
        """
        
        if self.pooling_type == "max":
            x = self.MaxPooling(x)
        elif self.pooling_type == "avg":
            x = self.AvgPooling(x)
        elif self.pooling_type == "topk":
            # x = self.TopKPooling(x, self.args.topk)
            x = self.TopKPooling(x, topk)
        elif self.pooling_type == "att":
            x = self.AttentionPooling(x, mask)
        elif self.pooling_type == "mask_max":
            x = self.MaskMaxPooling(x, mask) if mask is not None else self.MaxPooling(x)
        elif self.pooling_type == "mask_avg":
            x = self.MaskAvgPooling(x, mask) if mask is not None else self.AvgPooling(x)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', 'avg', 'topk', 'mask_avg' or 'mask_max'.")
        
        return x
    
    def foward_features(self, x):
        """Forward features when the backbone for the MIL model is a CNN based model, such as ResNet, VGG, DenseNet, etc.
        Args:
            x (torch.Tensor): Input image. Shape (Batch_size, channels, H, W, D).
        Returns:
            torch.Tensor: Returns the features with shape (Batch_size, N, embedding_size).
        """

        x = self.ViT(x)
        
        # Register Hook to have access to the gradients
        if x.requires_grad:
            x.register_hook(self.activations_hook)


        if len(x.shape) == 2:
            # If we get a flattened tensor (B, features)
            B, C = x.shape
            return x.unsqueeze(1)  # Shape: (B, 1, C)
        
        elif len(x.shape) == 3:
            # If we already have (B, N, C) format
            return x
        
        elif len(x.shape) == 5:
            # If we have a 5D tensor (B, C, H, W, D)
            B, C, H, W, D = x.shape
            return x.reshape(B, C, -1).permute(0, 2, 1)  # Shape: (B, H*W*D, C)
        
        else:
            # For any other shape (including 3D tensors from some models)
            B = x.shape[0]
            return x.reshape(B, -1, x.shape[1])  # Try best effort reshape
    


    def forward(self, x_in):

        if self.mil_type == "instance": 
            x = self.foward_features(x_in)

            x = self.classifier(x)
        
            self.save_patch_probs(torch.softmax(x, dim=2)) # Save the softmax probabilities of the patches (instances)

            topk = round(self.k * x.size(1)) # 25% of the instances in the bag

            logit, prob = self.MilPooling(x, mask=None, topk=topk)
    
        
            self.last_features = logit.detach() 

            self.save_softmax_bag_probs(prob) # Save the softmax probabilities of the bag

            
            return logit
        else:
            x = self.foward_features(x_in)
            topk = round(float(self.k) * int(x.size(1)))
            self.instance_level_features = x.detach()
            x = self.MilPooling(x, mask=None, topk=topk)
            self.last_features = x.detach()
            x = self.classifier(x)
            prob = F.softmax(x, dim=-1)
            self.save_softmax_bag_probs(prob)
            
            return x



class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """


    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        k=1.0
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.feature_size = hidden_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.classification = classification
        self.proj_type = proj_type
        

        if proj_type == "conv":
            self.patch_embedding = PatchEmbeddingBlock(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                proj_type=proj_type,
                pos_embed_type=pos_embed_type,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            )
        elif proj_type == "resnet10":
            # useless patch_embedding, only for code running.
            if k == 1.0:
                self.patch_embedding = PatchEmbeddingBlock(
                    in_channels=in_channels,
                    img_size=img_size,
                    patch_size=patch_size,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    proj_type="conv",
                    pos_embed_type=pos_embed_type,
                    dropout_rate=dropout_rate,
                    spatial_dims=spatial_dims,
                )
            self.resnet = create_model(
                'resnet10t.c3_in1k', pretrained=False, num_classes=0, in_chans=1, global_pool='', drop_path_rate=0.0
            )

            hidden_size = 512
            

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore


    def save_patch_probs(self, x):
        self.patch_probs = x

    def get_patch_probs(self):
        return self.patch_probs
    
    def save_softmax_bag_probs(self, x):
        self.softmax_probs = x

    def get_softmax_bag_probs(self):
        return self.softmax_probs
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x



    def forward(self, x):
        if self.proj_type == "conv":
            x = self.patch_embedding(x)
        elif self.proj_type == "resnet10":
            x = self.resnet(x)  
    
            x = F.adaptive_avg_pool3d(x, (6, 6, 6))  
            
            if x.requires_grad:
                x.register_hook(self.activations_hook)

            B, C, D, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1)  

        
        elif self.proj_type == "resnet50":
            x = self.resnet(x)  # Now returns a 5D tensor [B, 2048, H/32, W/32, D/32]
            x = F.adaptive_avg_pool3d(x, (6, 6, 6))  # Reduce to 6x6x6 spatial dims
            if x.requires_grad == True:
                x.register_hook(self.activations_hook)
            B, C, H, W, D = x.size()
            x = x.reshape(B, C, -1)  # [B, 2048, H*W*D]
            x = x.permute(0, 2, 1)  # [B, H*W*D, 2048]
            
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            # x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # x.register_hook(self.activations_hook(x[:,1:].reshape(1, -1, 6, 6, 6)))
        # x.register_hook(self.activations_hook(x.reshape(1, -1, 6, 6, 6)))

        if hasattr(self, "classification_head"):
            # x = self.classification_head(x[:, 0])
            x = self.classification_head(x)
            # prob = F.softmax(x, dim=-1)
            # self.save_softmax_bag_probs(prob)
        else:
            # (1, C, N, N, N)
            # N = self.img_size[0] // self.patch_size[0]
            # if self.proj_type == "conv":
            #     x = x.reshape(-1, self.feature_size, N, N, N)
            # elif self.proj_type == "resnet10":
            #     x = x.reshape(-1, C, H, W, D)
            
            return x
        
        return x

