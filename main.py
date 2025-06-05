import argparse
import torch
import torch, random, numpy as np, os
from utils.dataloader import get_singlephase_loader
from inference.evaluation import get_metrics
from utils.dataloader import get_multiphase_loader
from models.proto_model import PrototypeFusionModel
from trainer.training import train_prototype_fusion_model

def fix_seed(seed=8):
    """Fix random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="DuoProto")

parser.add_argument("--mode", default="train", type=str, help="train or test")
parser.add_argument("--log_name", default="log.txt", type=str, help="log file name")

# Data paths - will be handled by custom data loader
parser.add_argument("--image_path", type=str, help="raw image directory")
parser.add_argument("--mask_path", type=str, help="mask directory")
parser.add_argument("--clinical_data_path", type=str, help="clinical data directory")

parser.add_argument("--test_size", default=0.3, type=float, help="train test ratio")
parser.add_argument("--random_state", default=8, type=int, help="random state for train test split")
parser.add_argument("--img_size", default=128, type=int, help="resize image size")

parser.add_argument("--balanced_sampler", default=True, type=bool, help="whether to use balanced sampler")
parser.add_argument("--batch_size", default=2, type=int, help="batch size")
parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size')
parser.add_argument("--n_batches", default=60, type=int, help="number of batches")
parser.add_argument("--n_batches_multiphase", default=5, type=int, help="number of multiphase batches")
parser.add_argument("--n_batches_singlephase", default=30, type=int, help="number of singlephase batches")
parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
parser.add_argument("--epochs", default=150, type=int, help="number of epochs")

parser.add_argument("--model", default="SwinUNEC", type=str, help="model name")
parser.add_argument("--phase", default="PVimg", type=str, help="phase name") 
parser.add_argument("--stage", default=1, type=int, help="Swin Transformer stage")
parser.add_argument("--patch_size", default=128, type=int, help="patch size")
parser.add_argument("--window_size", default=7, type=int, help="window size")
parser.add_argument("--k", default=0.25, type=float, help="top k pooling")
parser.add_argument("--mil_type", default="embedding", type=str, help="MIL type")
parser.add_argument("--pooling_type", default="avg", type=str, choices=["avg", "topk", "att", "max"], help="MIL pooling type: avg, topk, att, max")
parser.add_argument("--proj_type", type=str, default="resnet10", choices=["resnet10", "conv"], help="Backbone projection type: resnet10 for ReViT, conv for pure ViT")

parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=1e-2, type=float, help="weight decay")

parser.add_argument("--settings", default="B", type=str, help="settings")
parser.add_argument("--device", default="cuda:1", type=str, help="device")

parser.add_argument("--early_stopping", action="store_true", help="whether to use early stopping")
parser.add_argument("--patience", default=10, type=int, help="patience for early stopping")
parser.add_argument("--monitor_metric", default="accuracy", type=str, choices=["loss", "accuracy", "auroc", "auprc"], help="metric to monitor for early stopping")

parser.add_argument("--loss_type", default="ce", type=str, choices=["ce", "focal", "weighted_ce"], help="loss function type")
parser.add_argument("--focal_gamma", default=2, type=float, help="gamma for focal loss")
parser.add_argument("--class_weights", nargs="+", type=float, default=None, help="manual class weights for loss function (one per class)") 

parser.add_argument("--seed", default=8, type=int, help="global random seed")

# Dataset selection
parser.add_argument("--dataset", default="CUSTOM_DATASET", type=str, help="dataset name")

parser.add_argument("--prototype_dim", type=int, default=512, help="Dimension of prototype vectors")
parser.add_argument("--prototype_momentum", type=float, default=0.9, help="Momentum for EMA update of prototypes")
parser.add_argument("--shared_encoder", action="store_true", help="Whether to share encoder between multiphase and singlephase")
parser.add_argument("--shared_proj_head", action="store_true", help="Whether to share projection head between multiphase and singlephase")
parser.add_argument("--fusion_method", type=str, default="attention", choices=["attention", "concat", "gated"], help="Method for multi-phase feature fusion in multiphase model")
parser.add_argument("--er_ce_lambda", type=float, default=1.0, help="Weight for ER CE loss")
parser.add_argument("--proto_lambda", type=float, default=0.5, help="Weight for prototype contrastive loss")
parser.add_argument("--proto_sep_lambda", type=float, default=0.5, help="Weight for prototype separation loss")
parser.add_argument("--er_rank_lambda", type=float, default=0.3,help="Weight for BCLC-aware ER prototype ranking loss")
parser.add_argument("--align_lambda", type=float, default=0.1, help="Weight for prototype alignment loss")
parser.add_argument('--multiphase_proto_momentum', type=float, default=0.9)
parser.add_argument('--singlephase_proto_momentum', type=float, default=0.9)
parser.add_argument('--no_proj_head', action='store_true', help="Disable projection head (default: use it)")
parser.add_argument("--init_prototype", type=str, default="zero", choices=["zero", "batch_mean"],help="Prototype initialization method: zero or batch_mean")
parser.add_argument('--align_type', type=str, default='l2', choices=['l2', 'cosine'], help='Alignment loss type: l2 | cosine')

parser.add_argument("--cache_data", action="store_true", help="Use cached prepared data if available")
parser.add_argument("--cache_split", action="store_true", help="Use cached train/test split if available")




def main():
    args = parser.parse_args()
    fix_seed(args.seed)
    args.balanced_sampler = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load data loaders - these should be implemented based on your specific data format
    train_mp_loader, val_mp_loader, test_mp_loader = get_multiphase_loader(args)
    train_sp_loader, val_sp_loader, test_sp_loader = get_singlephase_loader(args)
    
    # Set model parameters
    in_channels = 1
    img_size = (192, 192, 192)
    patch_size = args.patch_size
    hidden_size = 512  # Using hidden size of 512 as in document
    mlp_dim = 3072
    num_layers = 12
    num_heads = 16
    spatial_dims = 3
    dropout_rate = 0.1
    
    prototype_model = PrototypeFusionModel(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=(patch_size, patch_size, patch_size),
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pooling_type=args.pooling_type,
        mil_type=args.mil_type,
        num_classes=args.num_classes,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
        prototype_dim=args.prototype_dim,
        no_proj_head=args.no_proj_head,
        prototype_update_momentum=args.prototype_momentum,
        fusion_method=args.fusion_method,
        proj_type=args.proj_type,  
        shared_encoder=args.shared_encoder,
        shared_proj_head=args.shared_proj_head,
        er_ce_lambda=args.er_ce_lambda,
        proto_lambda=args.proto_lambda,
        align_lambda=args.align_lambda,
        multiphase_proto_momentum=args.multiphase_proto_momentum,
        singlephase_proto_momentum=args.singlephase_proto_momentum,
        init_prototype=args.init_prototype,
        er_rank_lambda=args.er_rank_lambda,
        align_type=args.align_type,
        proto_sep_lambda=args.proto_sep_lambda,
    ).to(device)
    
    multiphase_params = []
    singlephase_params = []
    for name, param in prototype_model.named_parameters():
        if "multiphase" in name:
            multiphase_params.append(param)
        else:
            singlephase_params.append(param)

    all_params = set(prototype_model.parameters())
    assert set(multiphase_params) | set(singlephase_params) == all_params
    assert set(multiphase_params).isdisjoint(set(singlephase_params))

    optimizer = torch.optim.AdamW([
        {
            "params": multiphase_params,
            "lr": 5e-4,
            "weight_decay": 1e-4
        },
        {
            "params": singlephase_params,
            "lr": 3e-4,
            "weight_decay": 1e-2
        }
    ])

    trained_model = train_prototype_fusion_model(
        prototype_model,
        train_mp_loader, 
        train_sp_loader,
        val_mp_loader, 
        val_sp_loader,
        optimizer, 
        device, 
        args,
        multiphase_params
    )
    
    # Extract singlephase model for evaluation
    singlephase_model = prototype_model.singlephase
    multiphase_model = prototype_model.multiphase

    get_metrics(singlephase_model, test_sp_loader, device, args)
    
    return

if __name__ == "__main__":
    main()
