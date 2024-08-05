import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from loss.loss import loss_center, loss_object, loss_size  
from utils.decode_result import decode_batch  # функция для декодирования результатов
from utils.logging import initialize_wandb  # функция для инициализации WandB логирования

class PLModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.MAP_train = MeanAveragePrecision()  # метрика для тренировки
        self.MAP_valid = MeanAveragePrecision()  # метрика для валидации

    def forward(self, input):
        return self.model.forward(input)  

    def training_step(self, batch, batch_idx):
        X = batch['img']
        targ = batch['targ']
        
        # Предполагаем, что метки (target) для оружия занимают все 5 каналов
        targ_gun = targ[:, :5, :, :]

        pred = self.forward(X)

        # Предсказания также содержат 5 каналов для оружия
        pred_gun = pred[:, :5, :, :]

        # Вычисляем функции потерь для объектов, центров и размеров
        loss_gun_obj = loss_object(pred_gun, targ_gun)
        loss_gun_center = loss_center(pred_gun, targ_gun)
        loss_gun_size = loss_size(pred_gun, targ_gun)

        # Суммируем все значения потерь
        loss_value = loss_gun_obj + loss_gun_center + loss_gun_size

        # Применяем сигмоиду к определенным компонентам предсказаний
        pred = pred.clone().detach()
        pred[:, [0, 1, 2]] = torch.sigmoid(pred[:, [0, 1, 2]])

        # Декодируем предсказанные и целевые боксы
        pred_boxes = decode_batch(pred, threshold=0.4, iou_threshold=0.5)
        targ_boxes = decode_batch(targ, threshold=1.0, iou_threshold=1.0)

        # Обновляем метрику mAP для тренировочного набора данных
        self.MAP_train.update(pred_boxes, targ_boxes)

        # Логируем значения потерь
        self.log('loss_gun_obj-train', loss_gun_obj, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss_gun_center-train', loss_gun_center, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss_gun_size-train', loss_gun_size, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss-train', loss_value, on_epoch=True, prog_bar=True, on_step=True)

        return loss_value

    def validation_step(self, batch, batch_idx):
        X = batch['img']
        targ = batch['targ']
        
        # То же самое для валидации
        targ_gun = targ[:, :5, :, :]

        pred = self.forward(X)

        pred_gun = pred[:, :5, :, :]

        loss_gun_obj = loss_object(pred_gun, targ_gun)
        loss_gun_center = loss_center(pred_gun, targ_gun)
        loss_gun_size = loss_size(pred_gun, targ_gun)

        loss_value = loss_gun_obj + loss_gun_center + loss_gun_size

        pred = pred.clone().detach()
        pred[:, [0, 1, 2]] = torch.sigmoid(pred[:, [0, 1, 2]])

        pred_boxes = decode_batch(pred, threshold=0.4, iou_threshold=0.5)
        targ_boxes = decode_batch(targ, threshold=1.0, iou_threshold=1.0)

        self.MAP_valid.update(pred_boxes, targ_boxes)

        self.log('loss_gun_obj-valid', loss_gun_obj, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss_gun_center-valid', loss_gun_center, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss_gun_size-valid', loss_gun_size, on_epoch=True, prog_bar=True, on_step=True)
        self.log('loss-valid', loss_value, on_epoch=True, prog_bar=True, on_step=True)

        return loss_value

    def on_train_epoch_start(self, *args, **kwargs):
        self.MAP_train.reset()

    def on_validation_epoch_start(self, *args, **kwargs):
        self.MAP_valid.reset()

    def on_train_epoch_end(self, *args, **kwargs):
        if self.current_epoch % 10 == 0:
            result = self.MAP_train.compute()
            if 'map_50' in result:
                self.log('mAP-train', result['map_50'].item())
            self.MAP_train.reset()

    def on_validation_epoch_end(self, *args, **kwargs):
        if self.current_epoch % 10 == 0:
            result = self.MAP_valid.compute()
            if 'map_50' in result:
                self.log('mAP-valid', result['map_50'].item())
            self.MAP_valid.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.0e-4, weight_decay=1.0e-8)
     
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='min', 
                                                                    factor=0.1, 
                                                                    verbose=True),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "loss-valid",
            "strict": True,
            "name": None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


def build_trainer(
        callbacks=[],
        max_epochs=50,
        accelerator='auto',
        project='gun-detection',
        name='Gun-Detector',
        deterministic=True,
        wandb_key=None):
    
    seed_everything(42, workers=True)

    logger = None
    if wandb_key is not None:
        initialize_wandb(wandb_key)
        logger = pl.loggers.WandbLogger(
            project=project,
            name=name
        )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        benchmark=False,
        accelerator=accelerator,
        log_every_n_steps=1,
        deterministic=deterministic,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1)
    
    return trainer