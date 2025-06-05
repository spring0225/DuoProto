import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multiphase_vit import MultiphaseViTMIL
from models.ViT import ViTMIL


class PrototypeFusionModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_size=(192, 192, 192),
        patch_size=(32, 32, 32),
        hidden_size=512,
        mlp_dim=3072,
        num_layers=12,
        num_heads=16,
        pooling_type="avg",
        mil_type="embedding",
        num_classes=2,
        dropout_rate=0.0,
        spatial_dims=3,
        prototype_dim=512,  
        prototype_update_momentum=0.9,  
        fusion_method="attention",  
        proj_type="resnet10",  
        shared_encoder=False,  
        shared_proj_head=False,  
        no_proj_head=False,  
        er_ce_lambda=1.0, 
        proto_lambda=0.5, 
        align_lambda=0.1, 
        multiphase_proto_momentum=0.9,  
        singlephase_proto_momentum=0.9,  
        init_prototype="zero",
        er_rank_lambda=0.3,
        align_type="l2",
        proto_sep_lambda=0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.prototype_update_momentum = prototype_update_momentum
        self.shared_encoder = shared_encoder
        self.shared_proj_head = shared_proj_head
        self.no_proj_head = no_proj_head
        self.loss_weights = {
            "er_ce_loss": er_ce_lambda,
            "proto_loss": proto_lambda,
            "align_loss": align_lambda,
            "rank_loss": er_rank_lambda,
            "proto_sep_loss": proto_sep_lambda 
        }
        self.multiphase_proto_momentum = multiphase_proto_momentum
        self.singlephase_proto_momentum = singlephase_proto_momentum
        self.init_prototype = init_prototype
        self.align_type = align_type
        
        # Create Multiphase model (multi-phase)
        self.multiphase = MultiphaseViTMIL(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            pooling_type=pooling_type,
            mil_type=mil_type,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed="conv",
            proj_type=proj_type,  
            pos_embed_type="learnable",
            classification=False,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation="NONE",
            qkv_bias=False,
            save_attn=False,
            mode="train",
            fusion_method=fusion_method
        )
        
        # Create Singlephase model (single-phase PV)
        if shared_encoder:
            # Use the multiphase's PV encoder for the singlephase
            self.singlephase = self.multiphase.pv_vit
        else:
            # Create a separate encoder for the singlephase
            self.singlephase = ViTMIL(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                pooling_type=pooling_type,
                mil_type=mil_type,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                pos_embed="conv",
                proj_type=proj_type,
                pos_embed_type="learnable",
                classification=False,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
                post_activation="NONE",
                qkv_bias=False,
                save_attn=False,
                mode="train",
                multi_task=False,  # Not using multi-task, just BCLC for ranking loss
                bclc_classes=4
            )

        
        # Projection heads
        if no_proj_head:
            self.multiphase_proj_head = nn.Identity()
            self.singlephase_proj_head = nn.Identity()
            assert prototype_dim == hidden_size, "Without projection head, prototype_dim must equal hidden_size"
        elif shared_proj_head:
            self.multiphase_proj_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.BatchNorm1d(hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size * 2, prototype_dim)
            )
            self.singlephase_proj_head = self.multiphase_proj_head
            print("Using shared projection head")
        else:
            self.multiphase_proj_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.BatchNorm1d(hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size * 2, prototype_dim)
            )
            self.singlephase_proj_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.BatchNorm1d(hidden_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size * 2, prototype_dim)
            )
        
        # Prototype banks (multiphase and singlephase)
        if init_prototype == "zero":
            self.register_buffer("multiphase_prototypes", torch.zeros(num_classes, prototype_dim))
            self.register_buffer("singlephase_prototypes", torch.zeros(num_classes, prototype_dim))
            
        elif init_prototype == "batch_mean":   
            self.register_buffer("multiphase_prototypes", torch.empty(num_classes, prototype_dim).fill_(float("nan")))
            self.register_buffer("singlephase_prototypes", torch.empty(num_classes, prototype_dim).fill_(float("nan")))

        self.initialized_prototypes = set()
        
        # ER classification heads
        self.multiphase_er_classifier = nn.Linear(prototype_dim, num_classes)
        self.singlephase_er_classifier = nn.Linear(prototype_dim, num_classes)
        
    
    def update_prototypes(self, features, labels, prototype_bank, momentum, use_batch_update=False, is_multiphase=False):
        """
        Update prototypes using either EMA or batch mean.

        Args:
            features (Tensor): [B, D] feature embeddings
            labels (Tensor): [B] class labels
            prototype_bank (Tensor): [num_classes, D] buffer to be updated
            momentum (float): momentum for EMA
            use_batch_update (bool): if True, override momentum and use batch mean directly
            is_multiphase (bool): whether this is for multiphase model

        Returns:
            updated prototype_bank
        """

        prototype_bank = prototype_bank.to(features.device)
        unique_classes = labels.unique()
        for c in unique_classes:
            c = c.item()
            mask = (labels == c)
            if mask.sum() == 0:
                print(f"No samples for class {c} in this batch, cannot update prototype.")
                continue

            class_features = features[mask]
            if torch.isnan(class_features).any():
                print(f"NaN in features for class {c}: {class_features}")
                continue

            batch_mean = class_features.mean(dim=0)

            if torch.isnan(batch_mean).any():
                print(f"NaN in batch mean for class {c}, skipping prototype update.")
                continue

            if use_batch_update:
                prototype_bank[c] = batch_mean
                print(f"Prototype {c} updated with batch mean")
            else:
                if torch.isnan(prototype_bank[c]).any():
                    print(f"Prototype for class {c} is NaN, resetting with batch mean.")
                    prototype_bank[c] = batch_mean
                    self.initialized_prototypes.add(c)
                    print(f"Prototype {c} initialized with batch mean: {batch_mean}")
                    continue

                old_proto = prototype_bank[c].clone()
                new_proto = momentum * old_proto + (1 - momentum) * batch_mean
                prototype_bank[c] = new_proto

        return prototype_bank

    
    def contrastive_prototype_loss(self, features, labels, prototypes, temperature=0.1):
        """
        Compute contrastive prototype loss with prototype separation
        
        Args:
            features: Feature vectors of shape [B, C]
            labels: Class labels of shape [B]
            prototypes: Prototype bank of shape [num_classes, C]
            temperature: Temperature for scaling similarity
            
        Returns:
            Contrastive loss
        """
        loss = 0.0
        batch_size = features.size(0)

        for i in range(batch_size):
            z_i = features[i]
            y_i = labels[i]
            
            # Compute similarities to all prototypes
            similarities = torch.nn.functional.cosine_similarity(
                z_i.unsqueeze(0), prototypes, dim=1
            ) / temperature
            
            if torch.isnan(similarities).any() or torch.isinf(similarities).any():
                print(f"NaN or Inf in similarity at sample {i}")

            positive_sim = similarities[y_i]
            negative_mask = torch.ones_like(similarities)
            negative_mask[y_i] = 0
            
            exp_pos = torch.exp(positive_sim)
            exp_neg_sum = torch.sum(torch.exp(similarities) * negative_mask)

            weight = 1.0

            loss += weight * (-torch.log(exp_pos / (exp_pos + exp_neg_sum + 1e-10)))
            loss = loss + 0.0 * features.sum()  

        contrastive_loss = loss / batch_size

        return contrastive_loss
    
    def prototype_separation_loss(self, prototypes, separation_margin=0.8):
        """
        Compute prototype separation loss to ensure prototypes are sufficiently apart.

        Args:
            prototypes: Prototype bank of shape [num_classes, D]
            separation_margin: Minimum distance between prototypes

        Returns:
            Separation loss
        """
        num_classes = prototypes.size(0)
        separation_loss = torch.tensor(0.0, device=prototypes.device, dtype=prototypes.dtype)
        count = 0
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    sim = F.cosine_similarity(prototypes[i].view(1, -1), prototypes[j].view(1, -1))
                    penalty = F.relu(sim - separation_margin) **2  
                    separation_loss += penalty.squeeze()
                    count += 1
        if count > 0:
            separation_loss = separation_loss / count
            print(f"Average separation loss: {separation_loss.item():.4f}")
            
        return separation_loss
    
    def prototype_alignment_loss(self, multiphase_prototypes, singlephase_prototypes,
                                  multiphase_patch_features=None, singlephase_patch_features=None,
                                  align_type="l2"):
        """
        Compute alignment loss between multiphase and singlephase prototypes.

        Args:
            multiphase_prototypes (Tensor): [K, D]
            singlephase_prototypes (Tensor): [K, D]
            multiphase_patch_features (Tensor): [B, N, D]
            singlephase_patch_features (Tensor): [B, N, D]
            align_type (str): "l2" | "cosine" 

        Returns:
            loss (Tensor): alignment loss
        """
        if align_type == "cosine":
            cos_sim = F.cosine_similarity(multiphase_prototypes, singlephase_prototypes, dim=1)
            return torch.mean(1.0 - cos_sim)

        else:  # default: L2
            return torch.mean(torch.sum((multiphase_prototypes - singlephase_prototypes) ** 2, dim=1))
    
    
    def er_proto_ranking_loss(self, features, bclc_labels, prototypes):
        """
        Compute ranking loss based on BCLC labels for ER prediction.
        Higher BCLC stage should have higher similarity to positive ER prototype.
        """
        loss = 0.0
        count = 0
        for i in range(features.size(0)):
            for j in range(features.size(0)):
                if i == j:
                    continue
                if bclc_labels[i] > bclc_labels[j]:  # BCLC_i > BCLC_j
                    sim_i = F.cosine_similarity(features[i], prototypes[1], dim=0)
                    sim_j = F.cosine_similarity(features[j], prototypes[1], dim=0)
                    margin = 0.05
                    loss += F.relu(sim_i - sim_j + margin)
                    count += 1

        if count == 0:
            print("Validation rank_loss skipped, bclc labels:", bclc_labels.tolist())
            return torch.tensor(0.0, device=features.device)

        if count > 0:
            return loss / count
        

    
    def forward_multiphase(self, multi_phase_input, er_labels=None, bclc_labels=None, training=True):
        """
        Forward pass for the multiphase model
        
        Args:
            multi_phase_input: Tuple of (pre, arterial, pv, delay) tensors
            er_labels: Early recurrence labels
            bclc_labels: BCLC classification labels (used for ranking loss only)
            training: Whether in training mode
            
        Returns:
            Dictionary of outputs
        """
        # Process inputs through multiphase model
        er_logits = self.multiphase(multi_phase_input)
        
        # Get features from the multiphase model
        multiphase_features = self.multiphase.last_features
        
        # Project features to prototype space
        z_t = self.multiphase_proj_head(multiphase_features)
        z_t = F.normalize(z_t, dim=1)
        
        self.multiphase_fused_proj_features = z_t.detach()  # [B, D]

        
        # Update multiphase prototype bank during training
        if training and er_labels is not None:
            with torch.no_grad():
                self.multiphase_prototypes = self.update_prototypes(
                    z_t.detach(),
                    er_labels,
                    self.multiphase_prototypes,
                    self.prototype_update_momentum,
                    is_multiphase=True
                )
                self.multiphase_prototypes = F.normalize(self.multiphase_prototypes, dim=1)
        
        # Calculate losses
        losses = {}
        if er_labels is not None:
            losses["er_ce_loss"] = F.cross_entropy(er_logits, er_labels)
            losses["proto_loss"] = self.contrastive_prototype_loss(z_t, er_labels, self.multiphase_prototypes)
            losses["proto_sep_loss"] = self.prototype_separation_loss(self.multiphase_prototypes)
            if bclc_labels is not None:
                bclc_labels = bclc_labels.to(er_logits.device) 
                losses["rank_loss"] = self.er_proto_ranking_loss(z_t, bclc_labels, self.multiphase_prototypes)
        return {
            "er_logits": er_logits,
            "features": multiphase_features,
            "proj_features": z_t,
            "losses": losses
        }
    
    def forward_singlephase(self, pv_input, er_labels=None, bclc_labels=None, training=True):
        """
        Forward pass for the singlephase model
        
        Args:
            pv_input: PV phase input tensor
            er_labels: Early recurrence labels
            bclc_labels: BCLC classification labels (used for ranking loss only)
            training: Whether in training mode
            
        Returns:
            Dictionary of outputs
        """
        if torch.isnan(pv_input).any():
            print("NaN detected in PV input")

        er_logits = self.singlephase(pv_input)
        
        singlephase_features = self.singlephase.last_features

        z_s = self.singlephase_proj_head(singlephase_features)
        if torch.isnan(z_s).any():
            print("NaN in singlephase projected features")
        z_s = F.normalize(z_s, dim=1)
        
        
        # Update singlephase prototype bank during training
        if training and er_labels is not None:
            with torch.no_grad():
                self.singlephase_prototypes = self.update_prototypes(
                    z_s.detach(),
                    er_labels,
                    self.singlephase_prototypes,
                    self.prototype_update_momentum,
                    is_multiphase=False
                )
                self.singlephase_prototypes = F.normalize(self.singlephase_prototypes, dim=1)
        
        # Calculate losses
        losses = {}
        if er_labels is not None:
            losses["er_ce_loss"] = F.cross_entropy(er_logits, er_labels)
            losses["proto_loss"] = self.contrastive_prototype_loss(z_s, er_labels, self.singlephase_prototypes)
            losses["proto_sep_loss"] = self.prototype_separation_loss(self.singlephase_prototypes)
            if bclc_labels is not None:
                bclc_labels = bclc_labels.to(er_logits.device)
                losses["rank_loss"] = self.er_proto_ranking_loss(z_s, bclc_labels, self.singlephase_prototypes)
     
            align_kwargs = {}
            losses["align_loss"] = self.prototype_alignment_loss(
                self.multiphase_prototypes,
                self.singlephase_prototypes,
                align_type=self.align_type,
                **align_kwargs
            )

        return {
            "er_logits": er_logits,
            "features": singlephase_features,
            "proj_features": z_s,
            "losses": losses
        }
    
    def forward(self, inputs, er_labels=None, bclc_labels=None, training=True):

        multi_phase_input = inputs["multi_phase"]
        pv_input = inputs["pv"]

        if multi_phase_input is not None:
            multiphase_outputs = self.forward_multiphase(multi_phase_input, er_labels, bclc_labels, training)
            
            if not hasattr(self, "multiphase_features_dict"):
                self.multiphase_features_dict = {}
                
            # Save per-phase ViT patch-level activations into multiphase_features_dict
            with torch.no_grad():
                pre_img, art_img, pv_img, delay_img = multi_phase_input

            self.multiphase_features_dict.update({
                "bag_feature": self.multiphase.last_features.detach().cpu(),                    # [B, C]
                "fused_patch_features": self.multiphase.fused_multiphase_features.detach().cpu(),     # [B, N, C]
                "pre": self.multiphase.phase_features["pre"].detach().cpu(),
                "art": self.multiphase.phase_features["art"].detach().cpu(),
                "pv": self.multiphase.phase_features["pv"].detach().cpu(),
                "delay": self.multiphase.phase_features["delay"].detach().cpu(),
            })
            
        else:
            multiphase_outputs = {
                "er_logits": None,
                "losses": {}
            }
            self.multiphase_features_dict = {
                "bag_feature": torch.full((1, self.prototype_dim), float('nan'), device=pv_input.device),
                "fused_patch_features": torch.full((1, 1, self.prototype_dim), float('nan'), device=pv_input.device),
                "pre": torch.full((1, self.prototype_dim), float('nan'), device=pv_input.device),
                "art": torch.full((1, self.prototype_dim), float('nan'), device=pv_input.device),
                "pv": torch.full((1, self.prototype_dim), float('nan'), device=pv_input.device),
                "delay": torch.full((1, self.prototype_dim), float('nan'), device=pv_input.device),
            }

        
        if pv_input is not None:
            singlephase_outputs = self.forward_singlephase(pv_input, er_labels, bclc_labels, training)
            

            if not hasattr(self, "singlephase_features_dict"):
                self.singlephase_features_dict = {}
                

            self.singlephase_features_dict.update({
            "bag_feature": self.singlephase.last_features.detach().cpu(),               # [B, C]
            "patch_features": self.singlephase.instance_level_features.detach().cpu(),     # [B, N, C]
        })
        else:
            singlephase_outputs = {
                "er_logits": None,
                "losses": {}
            }
        

        compute_loss = er_labels is not None or bclc_labels is not None
        
        combined_loss = 0.0
        if compute_loss:
            for key, weight in self.loss_weights.items():
                for output in [multiphase_outputs["losses"], singlephase_outputs["losses"]]:
                    if key in output:
                        loss = output[key]
                        if isinstance(loss, float):
                            loss = torch.tensor(loss, device=next(self.parameters()).device)
                        combined_loss += weight * loss

        
        return {
            "multiphase": multiphase_outputs,
            "singlephase": singlephase_outputs,
            "combined_loss": combined_loss
        }
