import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, HorizontalFlip,
    VerticalFlip, ShiftScaleRotate, Transpose,
)
from albumentations.pytorch import ToTensorV2

from src.utils import get_logger, seed_everything
from src.modules import myModel, myDataset, CrossEntropyLossWithLabelSmooth



class BaseRunner:
    def __init__(self, settings, config):
        self.settings = settings
        self.config = config
        self.logger = get_logger()

        seed_everything(seed=settings.SEED)


    def split(self, data, target=None, groups=None):
        if self.settings.KFOLD == "StratifiedKFold":
            skf = StratifiedKFold(
                n_splits=self.settings.N_SPLITS, shuffle=True, random_state=self.settings.SEED
            )
            for trn_idx, val_idx in skf.split(data, target):
                yield trn_idx, val_idx
        else:
            ValueError(f"invalid settings.KFOLD '{self.settings.KFOLD}'")


class Runner(BaseRunner):
    def __init__(self, settings, config):
        super().__init__(settings, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def scoring(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def run(self, is_debug, multi_gpu):
        """ Run Cross-Validation
        """
        df = pd.read_csv(f"{self.settings.DATA_PATH}/merged.csv")
        if is_debug:
            df = df.iloc[:500]

        df2020 = df[df["source"]==2020]
        oof = np.zeros((df2020.shape[0], self.settings.N_CLASS))
        for fold, (trn_idx, val_idx) in enumerate(
            self.split(df2020, df2020["label"].values), start=1
        ):
            self.logger.info(f"[TRAIN] Fold {fold}")

            # set modules
            self.criterion = CrossEntropyLoss().to(
                self.device
            ) if not self.config.label_smooth else CrossEntropyLossWithLabelSmooth(
                epsilon=self.config.label_smooth_alpha
            ).to(self.device)
            self.model = myModel(
                arch_name=self.config.arch_name,
                pretrained=True,
                img_size=self.config.image_size,
                multi_drop=self.config.multi_dropout,
                multi_drop_rate=0.5,
                att_layer=self.config.att_layer,
                att_pattern=self.config.att_pattern,
            )
            if multi_gpu:
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            self.optimizer = Adam(self.model.parameters(), lr=self.settings.LR)
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + epoch)
            )

            # split
            train = df2020.iloc[trn_idx] if not self.config.use_external else pd.concat([
                df2020.iloc[trn_idx], df[df["source"]==2019]
            ], axis=0)
            valid = df2020.iloc[val_idx]

            # set data_loader
            self.train_loader = DataLoader(
                myDataset(self.settings, train, transform=self.get_transform(is_train=True)),
                batch_size=self.settings.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=20,
            )
            self.valid_loader = DataLoader(
                myDataset(self.settings, valid, transform=self.get_transform(is_train=False)),
                batch_size=self.settings.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=20,
            )

            # training
            self.train(fold)

            oof[val_idx, :] = self.predict(self.valid_loader)
            fold_score = self.scoring(
                valid["label"].values, oof[val_idx, :].argmax(1)
            )
            self.logger.info(f"[RESULT] fold score: {fold_score}")

        cv_score = self.scoring(
            df2020["label"].values, oof.argmax(1)
        )
        self.logger.info(f"[RESULT] cv score: {cv_score}")
        self.logger.handlers.clear()
        return

    def train(self, fold):
        """ Run one fold
        """
        def _train_loop():
            self.model.train()
            total_loss = 0
            for images, labels in tqdm(
                self.train_loader, desc="[TRAIN] train loop", leave=False
            ):
                self.optimizer.zero_grad()
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().cpu()
            total_loss /= len(self.train_loader)
            return total_loss

        def _valid_loop():
            self.model.eval()
            total_loss = 0
            outputs, targets = [], []
            with torch.no_grad():
                for images, labels in tqdm(
                    self.valid_loader, desc="[TRAIN] valid loop", leave=False
                ):
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.model(images)
                    output = self.model(images)
                    loss = self.criterion(output, labels)
                total_loss += loss.detach().cpu()
                outputs.append(output); targets.append(labels)
            total_loss /= len(self.valid_loader)
            score = self.scoring(
                torch.cat(targets, dim=0).cpu(),
                torch.cat(outputs, dim=0).cpu().argmax(1)
            )
            return total_loss, score

        best_eval = 0
        model_path = f"{self.settings.OUTPUT_PATH}/{self.config.model_name}_{fold}.pth"
        with tqdm(range(1, self.settings.EPOCH)) as pbar:
            for epoch in pbar:
                pbar.set_description("[TRAIN] Epoch %d" % epoch)
                # train and valid loop
                trn_loss = _train_loop()
                val_loss, score = _valid_loop()
                self.scheduler.step()

                if score > best_eval:
                    patience, best_eval = 0, score
                    torch.save(self.model.state_dict(), model_path)
                else:
                    patience += 1
                    if patience > self.settings.MAX_PATIENCE:
                        break

                pbar.set_postfix(OrderedDict(
                    trn_loss=round(float(trn_loss), 4),
                    val_loss=round(float(val_loss), 4), val_score=score,
                    best_eval=best_eval
                ))

        self.model.load_state_dict(torch.load(model_path))


    def predict(self, data_loader):
        self.model.eval()
        outputs = []
        func = nn.Softmax(dim=1)
        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                images = images.to(self.device)
                output = self.model(images)
                outputs.append(output)
        return func(torch.cat(outputs, dim=0)).cpu().numpy()


    def get_transform(self, is_train=True):
        if is_train:
            return Compose([
                RandomResizedCrop(self.config.image_size, self.config.image_size),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(
                    mean=self.settings.MEAN,
                    std=self.settings.STD,
                ),
                ToTensorV2(),
            ])
        else:
            return Compose([
                Resize(self.config.image_size, self.config.image_size),
                Normalize(
                    mean=self.settings.MEAN,
                    std=self.settings.STD,
                ),
                ToTensorV2(),
            ])
