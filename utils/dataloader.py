from monai import data, transforms
from sklearn.model_selection import train_test_split
from utils.balanced_sampler import SamplerFactory, get_class_idxs_bc
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
import torch
import numpy as np
import random
import os
import pickle


def get_custom_data():
    """
    Replace this function with your own data loading logic.
    Should return a list of dictionaries with the following structure:
    
    For single-phase data:
    [
        {
            "PVimg": "/path/to/pv_image.nii.gz",
            "PVmask": "/path/to/pv_mask.nii.gz", 
            "label": 0 or 1,  # ER label
            "bclc": 0, 1, 2, or 3,  # BCLC label for ranking loss
            "PID": "patient_id"
        },
        ...
    ]
    
    For multi-phase data:
    [
        {
            "preimg": "/path/to/pre_image.nii.gz",
            "premask": "/path/to/pre_mask.nii.gz",
            "Aimg": "/path/to/arterial_image.nii.gz", 
            "Amask": "/path/to/arterial_mask.nii.gz",
            "PVimg": "/path/to/pv_image.nii.gz",
            "PVmask": "/path/to/pv_mask.nii.gz",
            "Delayimg": "/path/to/delay_image.nii.gz",
            "Delaymask": "/path/to/delay_mask.nii.gz",
            "label": 0 or 1,  # ER label
            "bclc": 0, 1, 2, or 3,  # BCLC label for ranking loss
            "PID": "patient_id"
        },
        ...
    ]
    
    Returns:
        tuple: (files_list, labels_list) 
    """
    # TODO: Implement your custom data loading here
    # This is a placeholder - replace with your actual data loading logic
    
    # Example structure (replace with your implementation):
    files = []
    labels = []
    
    # Your data loading logic here
    # ...
    
    return files, labels


def get_singlephase_loader(args):
    """
    Get data loaders for single-phase learning
    
    Args:
        args: Command line arguments
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Get transforms from regular get_loader function
    train_transforms, val_transforms = get_transforms(args)
    
    # Get data - replace this with your custom data loading
    files, labels = get_custom_data()
    
    # Split data
    X_train, X_test, _, _ = train_test_split(
        files, labels, 
        shuffle=True, 
        test_size=args.test_size,
        random_state=args.random_state, 
        stratify=labels
    )
    
    # Create data loaders
    if args.balanced_sampler:
        print("Using balanced sampler for single-phase data")
        class_idxs = get_class_idxs_bc(X_train)
        batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=args.batch_size,
            n_batches=args.n_batches_singlephase,
            alpha=1,
            kind='fixed'
        )

        if args.mode == "train":
            train_ds = CacheDataset(data=X_train, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
            train_loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=torch.cuda.is_available(),
                collate_fn=pad_list_data_collate, 
                worker_init_fn=seed_worker, 
                generator=g
            )
            
        val_ds = CacheDataset(data=X_test, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
            worker_init_fn=seed_worker,
            generator=g
        )

        test_ds = val_ds
        test_loader = DataLoader(
            test_ds, 
            batch_size=1, 
            num_workers=args.num_workers, 
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate, 
            worker_init_fn=seed_worker, 
            generator=g
        )
    else:
        if args.mode == "train":
            # Show the proportion of each class
            print("Single-phase training data class distributions:")
            er_labels = [item['label'] for item in X_train]
            bclc_labels = [item['bclc'] for item in X_train]
            
            print("ER labels:")
            for i in range(2):  # Binary classification for ER
                print(f"ER class {i}: {er_labels.count(i)}")
                
            print("BCLC labels:")
            for i in range(4):  # 4 classes for BCLC (0, A, B, C)
                print(f"BCLC class {i}: {bclc_labels.count(i)}")
                
            train_ds = CacheDataset(data=X_train, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
            train_loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, 
                pin_memory=torch.cuda.is_available(),
                collate_fn=pad_list_data_collate, 
                worker_init_fn=seed_worker, 
                generator=g
            )
        else:
            train_loader = None

        test_ds = CacheDataset(data=X_test, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
        test_loader = DataLoader(
            test_ds, 
            batch_size=1, 
            num_workers=args.num_workers, 
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate, 
            worker_init_fn=seed_worker, 
            generator=g
        )

    return train_loader, val_loader, test_loader


def get_transforms(args):
    """Get transforms for data preprocessing and augmentation"""
    from monai import transforms
    import numpy as np
    
    train_transforms = transforms.Compose([
        ### Preprocessing
        transforms.LoadImaged(keys=["PVimg","PVmask"]),
        transforms.EnsureChannelFirstd(keys=["PVimg","PVmask"]), 
        transforms.Orientationd(keys=["PVimg","PVmask"], axcodes="PLI"), 
        transforms.ScaleIntensityRanged(keys=["PVimg"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True), 
        
        transforms.MaskIntensityd(keys=["PVimg"], mask_key="PVmask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)), 
        transforms.CropForegroundd(keys=["PVimg","PVmask"], source_key="PVmask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)), 
        
        transforms.Spacingd(keys=["PVimg"], pixdim=(1, 1, 1), mode=("bilinear")),
        transforms.Spacingd(keys=["PVmask"], pixdim=(1, 1, 1), mode=("nearest")),
        transforms.SpatialPadd(keys=["PVimg","PVmask"], spatial_size=(192, 192, 192),mode="constant", method='symmetric'),
        transforms.CenterSpatialCropd(keys=["PVimg","PVmask"], roi_size=(192, 192, 192)),
        
        ### Augmentation (only for training)
        transforms.RandAffined(keys=["PVimg", "PVmask"], prob=0.2,
                    shear_range=(0.1, 0.1),
                    translate_range=(5, 5, 5),
                    rotate_range=(np.pi/36, np.pi/36, np.pi/36),
                    scale_range=(0.05, 0.05, 0.05),
                    mode=['bilinear', 'nearest'],
                    padding_mode='border'),
        transforms.RandAdjustContrastd(keys=["PVimg"], prob=0.3, gamma=(0.6, 1.4)),
        transforms.RandGaussianNoised(keys=["PVimg"], prob=0.3, mean=0.0, std=0.05),
        transforms.RandCoarseDropoutd(keys=["PVimg"], holes=1, max_holes=3, spatial_size=(20, 20, 10), prob=0.2),
        transforms.RandGaussianSmoothd(keys=["PVimg"], prob=0.2, sigma_x=(0.5, 1.5)),

        transforms.EnsureTyped(keys=["PVimg"]),
    ])

    val_transforms = transforms.Compose([
        ### Preprocessing
        transforms.LoadImaged(keys=["PVimg","PVmask"]),
        transforms.EnsureChannelFirstd(keys=["PVimg","PVmask"]),
        transforms.Orientationd(keys=["PVimg","PVmask"], axcodes="PLI"),
        transforms.ScaleIntensityRanged(keys=["PVimg"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
        
        transforms.MaskIntensityd(keys=["PVimg"], mask_key="PVmask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["PVimg","PVmask"], source_key="PVmask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        
        transforms.Spacingd(keys=["PVimg"], pixdim=(1, 1, 1), mode=("bilinear")),
        transforms.Spacingd(keys=["PVmask"], pixdim=(1, 1, 1), mode=("nearest")),
        transforms.SpatialPadd(keys=["PVimg","PVmask"], spatial_size=(192, 192, 192),mode="constant", method='symmetric'),
        transforms.CenterSpatialCropd(keys=["PVimg","PVmask"], roi_size=(192, 192, 192)),

        transforms.EnsureTyped(keys=["PVimg"]),
    ])
    
    return train_transforms, val_transforms


def get_multiphase_loader(args):
    """
    Get data loaders for multi-phase learning (pre, A, PV, Delay phases)
    
    Args:
        args: Command line arguments
        
    Returns:
        train_loader, val_loader, test_loader: Data loaders with multi-phase images and masks
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Multi-phase transforms
    train_transforms = transforms.Compose([
        ### Preprocessing - load all phases
        transforms.LoadImaged(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"]),
        transforms.EnsureChannelFirstd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"]),
        transforms.Orientationd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], axcodes="PLI"),
        
        # Scale intensity for each phase
        transforms.ScaleIntensityRanged(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
        
        # Mask intensity and crop foreground 
        transforms.MaskIntensityd(keys=["preimg"], mask_key="premask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["Aimg"], mask_key="Amask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["PVimg"], mask_key="PVmask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["Delayimg"], mask_key="Delaymask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["preimg","premask"], source_key="premask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["Aimg","Amask"], source_key="Amask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["PVimg","PVmask"], source_key="PVmask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["Delayimg","Delaymask"], source_key="Delaymask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        
        # Resample all phases to consistent spacing
        transforms.Spacingd(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                        pixdim=(1, 1, 1), mode=("bilinear")),
        transforms.Spacingd(keys=["premask", "Amask", "PVmask", "Delaymask"], 
                        pixdim=(1, 1, 1), mode=("nearest")),
        
        # Pad and crop to consistent size
        transforms.SpatialPadd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], 
                            spatial_size=(192, 192, 192), mode="constant", method='symmetric'),
        transforms.CenterSpatialCropd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], 
                                roi_size=(192, 192, 192)),
        
        # Data augmentation (applied to all phases consistently)
        transforms.RandAffined(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], 
                            prob=0.2, 
                            shear_range=(0.1, 0.1),
                            translate_range=(5, 5, 5),
                            rotate_range=(np.pi/36, np.pi/36, np.pi/36),
                            scale_range=(0.05, 0.05, 0.05),
                            mode=['bilinear', 'nearest', 'bilinear', 'nearest', 'bilinear', 'nearest', 'bilinear', 'nearest'],
                            padding_mode='border'),
        
        # Contrast, noise, and other intensity augmentations (apply to images only)
        transforms.RandAdjustContrastd(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    prob=0.3, gamma=(0.6, 1.4)),
        transforms.RandGaussianNoised(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    prob=0.3, mean=0.0, std=0.05),
        transforms.RandCoarseDropoutd(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    holes=1, max_holes=3, spatial_size=(20, 20, 10), prob=0.2),
        transforms.RandGaussianSmoothd(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    prob=0.2, sigma_x=(0.5, 1.5)),
        
        # Ensure all outputs are PyTorch tensors
        transforms.EnsureTyped(keys=["preimg", "Aimg", "PVimg", "Delayimg"]),
    ])

    val_transforms = transforms.Compose([
        ### Preprocessing - without augmentation
        transforms.LoadImaged(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"]),
        transforms.EnsureChannelFirstd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"]),
        transforms.Orientationd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], axcodes="PLI"),
        
        transforms.ScaleIntensityRanged(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                                    a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
        
        transforms.MaskIntensityd(keys=["preimg"], mask_key="premask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["Aimg"], mask_key="Amask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["PVimg"], mask_key="PVmask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.MaskIntensityd(keys=["Delayimg"], mask_key="Delaymask", select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["preimg","premask"], source_key="premask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["Aimg","Amask"], source_key="Amask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["PVimg","PVmask"], source_key="PVmask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        transforms.CropForegroundd(keys=["Delayimg","Delaymask"], source_key="Delaymask" , select_fn=lambda x: np.isclose(x, 1.0) | np.isclose(x, 2.0)),
        
        transforms.Spacingd(keys=["preimg", "Aimg", "PVimg", "Delayimg"], 
                        pixdim=(1, 1, 1), mode=("bilinear")),
        transforms.Spacingd(keys=["premask", "Amask", "PVmask", "Delaymask"], 
                        pixdim=(1, 1, 1), mode=("nearest")),
        
        transforms.SpatialPadd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], 
                            spatial_size=(192, 192, 192), mode="constant", method='symmetric'),
        transforms.CenterSpatialCropd(keys=["preimg", "premask", "Aimg", "Amask", "PVimg", "PVmask", "Delayimg", "Delaymask"], 
                                    roi_size=(192, 192, 192)),
        
        transforms.EnsureTyped(keys=["preimg", "Aimg", "PVimg", "Delayimg"]),
    ])
    
    # Get data with multi-phase images - replace this with your custom data loading
    files, labels = get_custom_data()
    
    X_train, X_test, _, _ = train_test_split(
        files, labels, 
        shuffle=True, 
        test_size=args.test_size,
        random_state=args.random_state, 
        stratify=labels
    )
    
    # Create data loaders
    if args.balanced_sampler:
        print("Using balanced sampler for multi-phase data")
        class_idxs = get_class_idxs_bc(X_train)
        batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=args.batch_size,
            n_batches=args.n_batches_multiphase,
            alpha=1,
            kind='fixed'
        )

        if args.mode == "train":
            train_ds = CacheDataset(data=X_train, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
            train_loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=torch.cuda.is_available(),
                collate_fn=pad_list_data_collate, 
                worker_init_fn=seed_worker, 
                generator=g
            )
        else:
            train_loader = None
        val_ds = CacheDataset(data=X_test, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
            worker_init_fn=seed_worker,
            generator=g
        )

        test_ds = val_ds
        test_loader = DataLoader(
            test_ds, 
            batch_size=1, 
            num_workers=args.num_workers, 
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate, 
            worker_init_fn=seed_worker, 
            generator=g
        )
    else:
        if args.mode == "train":
            # Show class distribution
            er_labels = [item['label'] for item in X_train]
            print("ER labels in training set:")
            for i in range(2):  # Binary classification for ER
                print(f"ER class {i}: {er_labels.count(i)}")
                
            train_ds = CacheDataset(data=X_train, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
            train_loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, 
                pin_memory=torch.cuda.is_available(),
                collate_fn=pad_list_data_collate, 
                worker_init_fn=seed_worker, 
                generator=g
            )
        else:
            train_loader = None

        test_ds = CacheDataset(data=X_test, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
        test_loader = DataLoader(
            test_ds, 
            batch_size=1, 
            num_workers=args.num_workers, 
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate, 
            worker_init_fn=seed_worker, 
            generator=g
        )

    return train_loader, val_loader, test_loader
