import os
from tqdm import tqdm
import math
import json
import numpy as np
import SimpleITK as sitk
from ImplantData.data_utils import *
from monai.losses.dice import DiceLoss

import torch
import torch.nn as nn 
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Engine:
    def __init__(
        self,
        model: nn.Module,
        max_epochs: int = 100,
        batch_size: int = 2,
        val_every: int = 5,
        results_dir: str = "./results",
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        num_workers: int = 4,
        amp: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        # Core setup
        self.max_epochs = int(max_epochs)
        self.val_every = int(val_every)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        
        # Model
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = DiceLoss(sigmoid=True)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # I/O
        self.results_dir = results_dir
        self.log_dir = os.path.join(results_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.weight_dir = os.path.join(results_dir, "weight")
        os.makedirs(self.weight_dir, exist_ok=True)
        self.predict_dir = os.path.join(results_dir, "predict")
        os.makedirs(self.predict_dir, exist_ok=True)
        
        # Tracking
        self.best_dice = 0.0
        self.global_step = 0
    
    
    def train(self, train_dataset, val_dataset):
        self.tb = SummaryWriter(self.log_dir)
        
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers, drop_last=False)
        
        self.scheduler = self._build_scheduler(steps_per_epoch=len(train_loader))
        
        for epoch in range(self.max_epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            if self.tb is not None:
                self.tb.add_scalar("loss/train", train_loss, epoch)
                self.tb.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            
            if (epoch + 1) % self.val_every == 0 :
                val_loss, val_dice = self.validate(val_loader, epoch)
                if self.tb is not None:
                    self.tb.add_scalar("loss/val", val_loss, epoch)
                    self.tb.add_scalar("dice/val", val_dice, epoch)

                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.save_model_weights(os.path.join(self.weight_dir, "best.pt"))
            
                self.save_model_weights(os.path.join(self.weight_dir, "last.pt"))
        
        if self.tb is not None:
            self.tb.close()


    def train_one_epoch(self, data_loader, epoch: int):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(data_loader, leave=True)
        for step, batch in enumerate(pbar):
            images, labels, _ = self._extract_batch_tensors(batch)

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.loss(outputs, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            running_loss += loss.detach().item()
            self.global_step += 1

            pbar.set_description(f"[train e{epoch}] loss={running_loss/(step+1):.4f}")

        return running_loss / max(1, (step + 1))


    @torch.no_grad()
    def validate(self, data_loader, epoch: int):
        self.model.eval()

        running_loss = 0.0
        running_dice = 0.0
        n_batches = 0

        pbar = tqdm(data_loader, leave=True)
        for step, batch in enumerate(pbar):
            images, labels, _ = self._extract_batch_tensors(batch)

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss(outputs, labels)
            dice = self._dice_coefficient(outputs, labels)

            running_loss += loss.detach().item()
            running_dice += dice.detach().item()
            n_batches += 1

            pbar.set_description(
                f"[valid e{epoch}] loss={running_loss/n_batches:.4f} dice={running_dice/n_batches:.4f}"
            )

        mean_loss = running_loss / len(data_loader)
        mean_dice = running_dice / len(data_loader)

        return mean_loss, mean_dice


    @torch.no_grad()
    def inference_generation(self, test_dataset):
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers, drop_last=False)
        self.model.eval()

        dice_all = 0.0

        for step, batch in enumerate(test_loader):
            images, labels, infos = self._extract_batch_tensors(batch)

            preds = self.model(images)
            dice = self._dice_coefficient(preds, labels)
            dice_all += dice.detach().item()

            save_output = torch.sigmoid(preds).cpu().numpy()[0][0]
            save_output = sitk.GetImageFromArray(save_output.astype(np.uint8))
            sitk.WriteImage(save_output, f"{self.predict_dir}/{infos['name'][0]}.nii.gz")

            print(f"[test {infos['name'][0]}] dice={dice:.4f}")

        mean_dice = dice_all / len(test_loader)
        print(f"Average dice={mean_dice:.4f}")


    @torch.no_grad()
    def inference_location(self, test_dataset):
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers, drop_last=False)
        self.model.eval()

        dice_all = 0.0
        dice_all_c = 0.0
        centroids = {}

        for step, batch in enumerate(test_loader):
            images, labels, infos = self._extract_batch_tensors(batch)

            preds = self.model(images)
            dice = self._dice_coefficient(preds, labels)
            dice_all += dice.detach().item()

            preds = (torch.sigmoid(preds) > 0.5).cpu().numpy()[0][0]
            labeled, num_features = ndimage.label(preds)
            region = torch.cat((infos["region"])).numpy()
            data_name = infos['name'][0]

            center_points = []
            predict_area = []
            all_pred_clean = torch.zeros_like(labels)
            for label in range(1, num_features + 1):
                region_sum = ndimage.sum(preds, labeled, label)
                # TODO: Instead of region area only, we should also consider if the shape is alike to cylinder 
                if region_sum > 200:
                    all_pred_clean |= torch.from_numpy((labeled == label)).to(all_pred_clean.device)
                    predict_area.append(region_sum)
                    center_point = ndimage.center_of_mass(preds, labeled, label)
                    center_points.append([
                        center_point[0] * 2 + region[0],
                        center_point[1] * 2 + region[2],
                        center_point[2] * 2 + region[4],
                    ])
            
            dice_c = self._dice_coefficient(all_pred_clean, labels)
            dice_all_c += dice_c.detach().item()

            centroids[data_name] = center_points
            save_output = sitk.GetImageFromArray(all_pred_clean.astype(np.uint8))
            sitk.WriteImage(save_output, f"{self.predict_dir}/{data_name}.nii.gz")

            print(
                f"[test {data_name}] dice={dice:.4f}, dice_c={dice_c:.4f}, "
                f"filter {len(predict_area)} areas from {num_features}: {predict_area}"
            )

        mean_dice = dice_all / len(test_loader)
        mean_dice_c = dice_all_c / len(test_loader)
        print(f"Average dice={mean_dice:.4f}, dice_c={mean_dice_c:.4f}")

        with open(os.path.join(self.results_dir, "location.json"), 'w', encoding='utf-8') as sf:
            json.dump(centroids, sf, indent=4)


    @staticmethod
    def _dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Binary Dice computed on thresholded predictions."""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        targets = (targets > 0.5).float()

        # flatten per-batch
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice.mean()


    def _extract_batch_tensors(self, batch):
        if isinstance(batch, list) and isinstance(batch[0], dict):
            images = torch.stack([item["image"] for item in batch]).flatten(0, 1)
            labels = torch.stack([item["label"] for item in batch]).flatten(0, 1)
            infos = [item["info"] for item in batch]
        elif isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
            infos = batch["info"]
        elif isinstance(batch, (list, tuple)):
            images, labels, infos = batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return images, labels, infos


    def _build_scheduler(self, steps_per_epoch: int):
        """Cosine schedule with linear warmup over first 10% epochs."""
        warmup_epochs = max(1, self.max_epochs // 10)
        total_steps = self.max_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            # cosine from warmup to total
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)


    def save_model_weights(self, weight_path: str):
        to_save = self.model
        state = to_save.state_dict()
        torch.save(state, weight_path)


    def load_state_dict(self, weight_path: str, strict: bool = True):
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd, strict=strict)

