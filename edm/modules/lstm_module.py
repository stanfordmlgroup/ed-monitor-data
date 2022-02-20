from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
import csv  
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle
import pandas as pd
import torch
import csv
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

from edm.utils.measures import perf_measure, calculate_output_statistics, calculate_confidence_intervals
from edm.models.lstm_model import LstmModel
from edm.dataloaders.lstm_dataloader import LstmDataLoader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
class LstmModule(pl.LightningModule):

    def __init__(self, train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids, 
                 df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens,
                 batch_size=64, fold_num=0, learning_rate=0.001, reg=None,
                 num_inner_layers=2, dropout_rate=0.2, 
                 dropout=True, inner_dim=64, balanced_training=False,
                 verbose=0):

        super().__init__()

        self.train_mat = train_mat
        self.val_mat = val_mat
        self.test_mat = test_mat
        
        self.df_train_ids = df_train_ids
        self.df_val_ids = df_val_ids
        self.df_test_ids = df_test_ids
        
        self.df_train_y = df_train_y
        self.df_val_y = df_val_y
        self.df_test_y = df_test_y
        self.train_seq_lens = train_seq_lens
        self.val_seq_lens = val_seq_lens
        self.test_seq_lens = test_seq_lens
        
        self.fold_num = fold_num
        self.learning_rate = learning_rate
        self.num_inner_layers = num_inner_layers
        self.dropout_rate = dropout_rate
        self.dropout = dropout
        self.inner_dim = inner_dim
        self.loss = nn.BCEWithLogitsLoss()

        self.training_losses = []
        self.cumulative_training_losses = []
        self.patient_ids = []
        self.training_preds = []
        self.training_y = []
        self.training_x_ids = []
        self.validation_losses = []
        self.validation_aurocs = []
        self.cumulative_validation_losses = []
        self.validation_preds = []
        self.validation_y = []
        self.batch_size = batch_size
        self.epoch = 0
        self.reg = reg
        self.balanced_training = balanced_training
        self.verbose = verbose
        self.reset_final_preds_and_y()
        self.model = LstmModel(train_mat.shape[-1], inner_dim)
        
    def forward(self, x, lens):
        x = self.model(x, lens)
        return x

    def training_step(self, batch, batch_idx):
        x, seq_lens, y, patient_id = batch
        x = x.to(device=device, dtype=torch.float)
        self.training_x_ids.append(patient_id)
        logits = self(x, seq_lens.cpu().numpy())
        loss = self.loss(logits, y.unsqueeze(1).float())
        preds = torch.sigmoid(logits)

        self.training_preds.extend(preds.cpu().detach().numpy().tolist())
        self.training_y.extend(y.cpu().detach().numpy().tolist())
        self.training_losses.append(loss.cpu().detach().numpy())

        return loss

    def training_epoch_end(self, outputs):
        auroc = roc_auc_score(np.array(self.training_y), np.array(self.training_preds))
        if self.verbose >= 2:
            print(
                f"[TRAIN]: epoch={self.epoch}, loss={np.mean(self.training_losses)}, num_samples={len(self.training_y)}, auroc={auroc}")

        self.training_preds = []
        self.training_y = []
        self.training_losses = []
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().cpu()
        self.cumulative_training_losses.append(avg_loss)
        if self.logger is not None:
            self.logger.experiment.add_scalars("losses", {"train_loss": avg_loss}, global_step=self.current_epoch)
        self.training_x_ids = []

    def validation_step(self, batch, batch_idx):
        x, seq_lens, y, patient_id = batch
        x = x.to(device=device, dtype=torch.float)
        logits = self(x, seq_lens.cpu().numpy())
        loss = self.loss(logits, y.unsqueeze(1).float())
        preds = torch.sigmoid(logits)
        preds = torch.squeeze(preds)
        preds = preds.cpu()
        y = y.cpu().int()

        # For our binary classification, we only have the sigmoid probabilities for the ACS class.
        # Here, we add a new column for the 1-preds non-ACS class because the metrics below requires it.
        #
        preds_probs = torch.cat(((1 - preds.unsqueeze(0)), preds.unsqueeze(0)), dim=0).transpose(1, 0)

        # Uncomment if you want early stopping to be measured based on steps
        # self.log('val_loss', loss, prog_bar=True)
        
        self.validation_preds.extend(preds.cpu().detach().numpy().tolist())
        self.validation_y.extend(y.cpu().detach().numpy().tolist())
        self.patient_ids.extend(patient_id.cpu().detach().numpy().tolist())
        self.validation_losses.append(loss.cpu().detach().numpy())

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean().cpu()
        self.cumulative_validation_losses.append(avg_loss)
        if self.logger is not None:
            self.logger.experiment.add_scalars("losses", {"val_loss": avg_loss}, global_step=self.current_epoch)
            
        auroc = roc_auc_score(np.array(self.validation_y), np.array(self.validation_preds))
        self.validation_aurocs.append(auroc)
        if self.verbose >= 2:
            print(
                f"[VALIDATION]: epoch = {self.epoch}, loss={np.mean(self.validation_losses)}, num_samples={len(self.validation_y)}, auroc={auroc}")
        if self.logger is not None:
            self.logger.experiment.add_scalars("aurocs", {"val_auroc": auroc}, global_step=self.current_epoch)

        # Uncomment if we want to have early stopping operate on loss per epoch 
        self.log('val_loss', np.mean(self.validation_losses), prog_bar=True)

        if self.verbose >= 2:
            print("-" * 40)
        self.epoch += 1
        
        self.validation_preds = []
        self.validation_y = []
        self.validation_losses = []
        self.patient_ids = []

    def test_step(self, batch, batch_idx):
        x, seq_lens, y, patient_id = batch
        x = x.to(device=device, dtype=torch.float)
        logits = self(x, seq_lens.cpu().numpy())
        loss = self.loss(logits, y.unsqueeze(1).float())
        preds = torch.sigmoid(logits)
        preds = torch.squeeze(preds)
        preds = preds.cpu()
        y = y.cpu().int()
        patient_id = patient_id.cpu()
        patient_id_list = [patient_id[i].item() for i in range(len(patient_id))]

        # For our binary classification, we only have the sigmoid probabilities for the ACS class.
        # Here, we add a new column for the 1-preds non-ACS class because the metrics below requires it.
        #
        preds_probs = torch.cat(((1 - preds.unsqueeze(0)), preds.unsqueeze(0)), dim=0).transpose(1, 0)

        self.final_patient_ids.extend(patient_id_list)
        self.final_preds.extend(preds.clone().detach().cpu().numpy().tolist())
        self.final_y.extend(y.clone().detach().cpu().numpy().tolist())

        return loss

    def configure_optimizers(self):
        if self.reg is not None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def reset_final_preds_and_y(self):
        self.final_preds = []
        self.final_y = []
        self.final_patient_ids = []

    def get_final_preds_and_y(self):
        return self.final_preds, self.final_y
    
    def get_final_patient_ids(self):
        return self.final_patient_ids

    def get_losses(self):
        return self.cumulative_training_losses, self.cumulative_validation_losses
    
    def get_val_aurocs(self):
        return self.validation_aurocs

    def prepare_data(self):
        # Nothing needs to happen here
        pass

    def train_dataloader(self):
        train_dats = LstmDataLoader(self.train_mat, self.df_train_ids, self.train_seq_lens, self.df_train_y)

        if self.balanced_training:
            trainratio = np.bincount(train_dats.get_labels())
            classcount = trainratio.tolist()
            train_weights = 1. / torch.tensor(classcount, dtype=torch.float)
            train_sampleweights = train_weights[train_dats.get_labels()]
            # Note: Cannot shuffle when using WeightedRandomSampler
            train_sampler = WeightedRandomSampler(weights=train_sampleweights, num_samples=len(train_sampleweights))
            train_loader = DataLoader(train_dats, sampler=train_sampler, batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            train_loader = DataLoader(train_dats, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader

    def val_dataloader(self):
        if self.verbose >= 2:
            print("val_dataloader")
        val_dats = LstmDataLoader(self.val_mat, self.df_val_ids, self.val_seq_lens, self.df_val_y)
        val_loader = DataLoader(val_dats, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return val_loader

    def test_dataloader(self):
        if self.verbose >= 2:
            print("test_dataloader")
        test_dats = LstmDataLoader(self.test_mat, self.df_test_ids, self.test_seq_lens, self.df_test_y)
        test_loader = DataLoader(test_dats, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return test_loader


def train_lstm(train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids,
               df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens,
               patience=10, dropout=True, inner_dim=64, 
               dropout_rate=0.2, learning_rate=0.001, start_from=0,
               num_inner_layers=2, batch_size=64, reg=None, epochs=200,
               verbose=0, save_model=False, save_predictions_path=None, run_bootstrap_ci=True, show_df_preview=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pl.seed_everything(1234)

    all_training_losses = []
    all_val_losses = []
    all_val_aurocs = []
    train_aurocs = []
    val_aurocs = []
    test_aurocs = []
    test_y, test_y_preds = [], []

    model = LstmModule(train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids,
                       df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens, 
                       batch_size=batch_size, inner_dim=inner_dim, dropout=dropout, learning_rate=learning_rate,
                       dropout_rate=dropout_rate, num_inner_layers=num_inner_layers, reg=reg, verbose=verbose)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    callbacks = []
    if patience is not None:
        early_stopping = EarlyStopping(
            'val_loss',
            patience=patience,
            verbose=0
        )
        callbacks.append(early_stopping)
    if save_model:
        callbacks.append(checkpoint_callback)

    if verbose >= 1:
        trainer = pl.Trainer(logger=True, gpus=1, max_epochs=epochs, progress_bar_refresh_rate=1,
                             callbacks=callbacks)
    else:
        trainer = pl.Trainer(logger=False, gpus=1, max_epochs=epochs, progress_bar_refresh_rate=0,
                             callbacks=callbacks, weights_summary=None)

    trainer.fit(model)
    training_losses, validation_losses = model.get_losses()
    validation_aurocs = model.get_val_aurocs()
    all_training_losses.append(training_losses)
    all_val_losses.append(validation_losses)
    all_val_aurocs.append(validation_aurocs)

    if verbose >= 1:
        print(f"------------------------------------")
        print()
        print()
        print("============= TRAIN ROC CURVE ===============")
    model.reset_final_preds_and_y()
    trainer.test(test_dataloaders=model.train_dataloader(), verbose=(verbose >= 2), ckpt_path='best')
    final_preds, final_y = model.get_final_preds_and_y()
    precision, recall, _ = precision_recall_curve(final_y, final_preds)
    auprc_alt = auc(recall, precision)
    pt_ids = model.get_final_patient_ids()
    
    if verbose >= 1:
        auroc_train = calculate_output_statistics(final_y, final_preds)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
    else:
        auroc_train = calculate_output_statistics(final_y, final_preds, show_plots=False)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
    
    if save_predictions_path is not None:
        Path(f"{save_predictions_path}").mkdir(parents=True, exist_ok=True)
        pt_ids = model.get_final_patient_ids()
        with open(f"{save_predictions_path}/train.csv", "w") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["patient_id", "preds", "actual"])
            for ind in range(len(pt_ids)):
                writer.writerow([pt_ids[ind], final_preds[ind], final_y[ind]])
        
    train_aurocs.append(auroc_train)
    
    if verbose >= 1:
        print(f"TRAIN AUROC = {auroc_train} AUPRC = {auprc_alt} using data size {len(final_preds)} with {sum(final_y)} ACS")
        if checkpoint_callback is not None:
            print(f"Best Checkpoint = {checkpoint_callback.best_model_path}")

    if verbose >= 1:
        print()
        print()
        print("============= VAL ROC CURVE ===============")
    
    model.reset_final_preds_and_y()
    trainer.test(test_dataloaders=model.val_dataloader(), verbose=(verbose >= 2), ckpt_path='best')
    final_preds, final_y = model.get_final_preds_and_y()

    precision, recall, _ = precision_recall_curve(final_y, final_preds)
    auprc_alt = auc(recall, precision)
    if verbose >= 1:
        auroc_val = calculate_output_statistics(final_y, final_preds)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
    else:
        auroc_val = calculate_output_statistics(final_y, final_preds, show_plots=False)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
    val_aurocs.append(auroc_val)
    if verbose >= 1:
        print(f"VAL AUROC = {auroc_val} AUPRC = {auprc_alt} using data size {len(final_preds)} with {sum(final_y)} pos")

    if save_predictions_path is not None:
        Path(f"{save_predictions_path}").mkdir(parents=True, exist_ok=True)
        pt_ids = model.get_final_patient_ids()
        with open(f"{save_predictions_path}/val.csv", "w") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["patient_id", "preds", "actual"])
            for ind in range(len(pt_ids)):
                writer.writerow([pt_ids[ind], final_preds[ind], final_y[ind]])

    if verbose >= 1:
        print()
        print()
        print("============= TEST ROC CURVE ===============")
    model.reset_final_preds_and_y()

    trainer.test(test_dataloaders=model.test_dataloader(), verbose=(verbose >= 2), ckpt_path='best')
    final_preds, final_y = model.get_final_preds_and_y()
    if verbose >= 1:
        auroc_test = calculate_output_statistics(final_y, final_preds)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
        if run_bootstrap_ci:
            calculate_confidence_intervals(final_y, final_preds, ci_type="bootstrap")
    else:
        auroc_test = calculate_output_statistics(final_y, final_preds, show_plots=False)
        calculate_confidence_intervals(final_y, final_preds, ci_type="delong")
        if run_bootstrap_ci:
            calculate_confidence_intervals(final_y, final_preds, ci_type="bootstrap")
    precision, recall, _ = precision_recall_curve(final_y, final_preds)
    auprc_alt = auc(recall, precision)
    test_aurocs.append(auroc_test)
    if verbose >= 1:
        print(f"TEST AUROC = {auroc_test} AUPRC = {auprc_alt} using data size {len(final_preds)} with {sum(final_y)} pos")

    if save_predictions_path is not None:
        Path(f"{save_predictions_path}").mkdir(parents=True, exist_ok=True)
        pt_ids = model.get_final_patient_ids()
        with open(f"{save_predictions_path}/test.csv", "w") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["patient_id", "preds", "actual"])
            for ind in range(len(pt_ids)):
                writer.writerow([pt_ids[ind], final_preds[ind], final_y[ind]])
        
    if verbose >= 1:
        print()
        print()
        print("============= TRAIN/VAL LOSS CURVE ===============")
        plt.plot(training_losses)
        plt.plot(validation_losses)
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    return auroc_train, auroc_val, auroc_test
