import torch
from pathlib import Path
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.data.build import build_dataloader
from ultralytics.utils.torch_utils import is_parallel, LOGGER, torch_distributed_zero_first
from torch.utils.data import DataLoader, WeightedRandomSampler

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "yolo_single_binary"
RUN_NAME = "single_binary_yolo11l_cls_siamese_balanced"


class BalancedClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)

        if not dataset.samples:
            raise FileNotFoundError(f"No images found in '{mode}' split of {dataset_path}.")

        nc = self.data.get("nc", 0)
        dataset_nc = len(dataset.base.classes)
        if nc and dataset_nc > nc:
            extra_classes = dataset.base.classes[nc:]
            original_count = len(dataset.samples)
            dataset.samples = [s for s in dataset.samples if s[1] < nc]
            skipped = original_count - len(dataset.samples)
            LOGGER.warning(
                f"{mode} split has {dataset_nc} classes but model expects {nc}. "
                f"Skipping {skipped} samples from extra classes: {extra_classes}"
            )
            if not dataset.samples:
                raise RuntimeError(f"All {original_count} samples filtered out.")

        if mode == "train":
            labels = [s[1] for s in dataset.samples]
            num_classes = len(dataset.base.classes)
            
            class_sample_counts = torch.tensor([labels.count(c) for c in range(num_classes)], dtype=torch.float)
            class_sample_counts[class_sample_counts == 0] = 1.0
            
            per_sample_weight = 1.0 / torch.sqrt(class_sample_counts[labels])
            train_sampler = WeightedRandomSampler(
                weights=per_sample_weight,
                num_samples=len(per_sample_weight),
                replacement=True
            )
            
            LOGGER.info(f"[*] Balanced sampler active — inverse class counts: {class_sample_counts.tolist()}")
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=self.args.workers,
                pin_memory=True,
                drop_last=self.args.compile,
            )
        else:
            loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)
            
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        
        class FocalLossCriterion:
            def __init__(self, gamma=2.0):
                self.gamma = gamma
                
            def __call__(self, preds, batch):
                preds = preds[1] if isinstance(preds, (list, tuple)) else preds
                ce_loss = F.cross_entropy(preds, batch["cls"], reduction="none")
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean(), focal_loss.mean().detach()

        def custom_init_criterion():
            LOGGER.info("[*] Initializing Custom Focal Loss (gamma=2.0)")
            return FocalLossCriterion(gamma=2.0)
            
        model.init_criterion = custom_init_criterion
        model.criterion = model.init_criterion()
        return model

class BalancedYOLO(YOLO):
    @property
    def task_map(self):
        task_map = super().task_map
        task_map["classify"]["train"] = BalancedClassificationTrainer
        return task_map

if __name__ == '__main__':

    print("PyTorch version:", torch.__version__)
    print("CUDA IS:", torch.cuda.is_available())
    
    # 1. Используем кастомный класс YOLO с балансировкой
    model = BalancedYOLO("yolo11l-cls.pt") 
    
    # 2. ТВОИ идеальные сиамские параметры
    train_params = {
        "data": str(DATA_DIR.resolve()), 
        "epochs": 100,             
        "patience": 20,            
        "imgsz": 320,              
        "batch": 256,              
        
        "project": str((PROJECT_ROOT / "results" / "res_yolo").resolve()), 
        "name": RUN_NAME,          
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 2,              # Безопасность Windows!
        "deterministic": True,
        
        "optimizer": "AdamW",       
        "lr0": 0.0005,        
        "weight_decay": 0.05, 
        "warmup_epochs": 5.0,      
        "cos_lr": True,            
        
        "freeze": 10,        
        "label_smoothing": 0.1, 

        "degrees": 10.0,           
        "flipud": 0.5,              
        "fliplr": 0.0,             
        
        "hsv_h": 0.0,        
        "hsv_s": 0.2,         
        "hsv_v": 0.2,       
        
        "erasing": 0.0,     
        "mixup": 0.0,         
        "copy_paste": 0.0,    
        "scale": 0.0,        
        "translate": 0.0,     
        "auto_augment": False
    }

    torch.cuda.empty_cache()
    print(f"\n[*] Starting training on: {DATA_DIR.name}")
    model.train(**train_params)
    print(f"[+] Training complete. Artifacts saved to results/res_yolo/{RUN_NAME}")