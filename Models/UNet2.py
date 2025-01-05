import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import mir_eval
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


# chemin vers l'enregistrement des modèles entrainés/scores obtenus
PATH_2 = "Paths/UNet2/"
SCORES_PATH_2 = "Scores/UNet2/"

# Recherche du GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

in_channels=1
out_channels=2

features=[16, 32, 64, 128, 256] # nombre des blocks


# ****************************************
# *                                      *
# *           SOUS MODULES               *
# *                                      *
# ****************************************


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size = 5, stride = 2,padding =  2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=True):
        super(DecodeBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if drop_out: 
            layers.append(nn.Dropout(0.5))
            
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)
    
# ****************************************
# *                                      *
# *             UNET2                    *
# *                                      *
# ****************************************

class UNet2(nn.Module):
    def __init__(
            self, in_channels=in_channels, out_channels=out_channels, features=features,
    ):
        super(UNet2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Down part of UNET
        _in_channels = in_channels
        for feature in features:
            self.downs.append(EncodeBlock(_in_channels, feature))
            _in_channels = feature

        self.bottleneck = EncodeBlock(features[-1], features[-1]*2)
        
        # Up part of UNET
        self.ups.append(DecodeBlock(features[-1]*2, features[-1], drop_out = (features[-1] >= 64)))
        
        for feature in reversed(features[:-1]):
            self.ups.append(DecodeBlock(feature*4, feature, drop_out = (feature >= 64)))

        self.final_conv = DecodeBlock(features[0]*2, out_channels)

    def forward(self, x):
        input_x = x
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        

        for idx, up in enumerate(self.ups):
            x = up(x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            x = torch.cat((skip_connection, x), dim=1)
        
        x = self.final_conv(x)
        if x.shape != input_x.shape:
                x = TF.resize(x, size=input_x.shape[2:])

        mask = torch.sigmoid(x)
        return mask 

    @staticmethod
    def trainModel(dataset, n_epochs=20, batch_size=16, learning_rate=0.001, valid_dataset=None):
        """
        Train the UNet model with optional resumption and validation.

        Args:
            dataset: Training dataset.
            n_epochs: Number of epochs to train (default: 20).
            batch_size: Batch size (default: 16).
            learning_rate: Initial learning rate (default: 0.001).
            valid_dataset: Validation dataset (optional).

        Returns:
            model: Trained UNet model.
            losses: List of training losses.
        """
        os.makedirs(PATH_2, exist_ok=True)
        os.makedirs(SCORES_PATH_2, exist_ok=True)
        score_path = os.path.join(SCORES_PATH_2)
        
        os.makedirs(score_path, exist_ok=True)
        model_path = os.path.join(PATH_2)
        os.makedirs(model_path, exist_ok=True)
        
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        model = UNet2().to(device)
        last_saved_epoch = 0

        if len(model_files) > 0:
            saved_epochs = [int(i.split('.')[0].split('_')[-1]) for i in model_files]
            last_saved_epoch = max(saved_epochs)
            model.load_state_dict(torch.load(os.path.join(PATH_2, f'model_{last_saved_epoch}.pth')))

        csv_path = os.path.join(SCORES_PATH_2, 'UNet2MAE.csv')
        if os.path.exists(csv_path):
            scores = pd.read_csv(csv_path)
            scores = scores.loc[scores.index < last_saved_epoch]
        else:
            scores = pd.DataFrame(columns=['train', 'valid'])

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss(reduction='mean').to(device)

        losses, valid_losses = [], []
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True) if valid_dataset else None

        for epoch in range(last_saved_epoch, last_saved_epoch + n_epochs):
            model.train()
            running_loss = 0.0

            for X, voice, noise, _, _ in tqdm(dataloader):
      
                X, voice, noise = X.to(device), voice.to(device), noise.to(device)
                optimizer.zero_grad()
                output = model(X.unsqueeze(1))
                
                voice_ = voice.unsqueeze(1)
                noise_ = noise.unsqueeze(1)
                Y = torch.cat((voice_, noise_), dim=1)  # Concatenate along the channel dimension

                loss = criterion(output * Y, Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{last_saved_epoch + n_epochs} - Train Loss: {epoch_loss:.4f}')

            # Validation loss
            if valid_dataloader:
                model.eval()
                running_valid_loss = 0.0
                with torch.no_grad():
                    for X, voice, noise, _, _  in valid_dataloader:

                        X, voice,noise = X.to(device), voice.to(device), noise.to(device)
                        optimizer.zero_grad()
                        output = model(X.unsqueeze(1))
                        
                        voice_ = voice.unsqueeze(1)
                        noise_ = noise.unsqueeze(1)
                        Y = torch.cat((voice_, noise_), dim=1)  # Concatenate along the channel dimension

                        loss = criterion(output * Y, Y)  # Add channel dimension for target
                        running_valid_loss += loss.item()

                valid_loss = running_valid_loss / len(valid_dataloader)
                valid_losses.append(valid_loss)
                print(f'Valid Loss: {valid_loss:.4f}')


            # Save model
            if (epoch + 1) % 10 == 0 or (epoch + 1) == (last_saved_epoch + n_epochs):
                torch.save(model.state_dict(), os.path.join(PATH_2, f'model_{epoch + 1}.pth'))

        valid_losses = valid_losses if valid_losses != [] else np.nan
        scores = pd.concat([scores, pd.DataFrame({
            'train': losses,
            'valid': valid_losses
        })])

        csv_path = os.path.join(SCORES_PATH_2, 'UNet2MAE.csv')
        scores.to_csv(csv_path, index=False)

        return model, losses
    
    
    @staticmethod
    def compute_metrics(target_waveform, reconstructed_waveform, mixture_waveform):
        """
        Compute SDR, SIR, SAR, and NSDR metrics using mir_eval.
        
        Args:
            target_waveform (np.ndarray): The target signal.
            reconstructed_waveform (np.ndarray): The output signal from the model.
            mixture_waveform (np.ndarray): The original mixture.
        
        Returns:
            dict: A dictionary with 'SDR', 'SIR', 'SAR', 'NSDR' metrics.
        """
        # Compute SDR, SIR, SAR using mir_eval
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            target_waveform,  # Targets (2D)
            reconstructed_waveform  # Estimated signals (2D)
        )
    
        # Compute NSDR (Normalized SDR)
        original_sdr = mir_eval.separation.bss_eval_sources(
            target_waveform,
            mixture_waveform
        )[0]
        nsdr = sdr - original_sdr
    
        return {'SDR': sdr[0], 'SIR': sir[0], 'SAR': sar[0], 'NSDR': nsdr[0]}
    
    @staticmethod
    def validateModel(model, valid_dataset,subset_size=200):
        """
        Validate the models and return raw metrics data for visualization (box plot).
    
        Args:
            voice_model: Trained voice separation model.
            noise_model: Trained noise separation model.
            valid_dataset: Validation dataset.
    
        Returns:
            metrics_data: Dictionary containing lists of metric values for each sample in the validation set.
        """
        model.eval()
    
        sdr_list, sir_list, sar_list, nsdr_list = [], [], [], []
    
        dataset_indices = random.sample(range(len(valid_dataset)), min(subset_size, len(valid_dataset)))
    
        with torch.no_grad():
            for dataset_idx in tqdm(range(len(dataset_indices)), desc="Validation Progress"):
                X, voice, noise, _, _ = valid_dataset[dataset_idx]
                X, voice, noise  = X.to(device), voice.to(device), noise.to(device)
                X_ = X.unsqueeze(0).unsqueeze(0)
                voice_ = voice.unsqueeze(0).unsqueeze(0)
                noise_ = noise.unsqueeze(0).unsqueeze(0)

                Y = torch.cat((voice_, noise_), dim=1)  # Concatenate along the channel dimension
                
                pred = (voice_model(X_) * Y).squeeze()
                
                voice_pred = pred[0]
                noise_pred = pred[1]
    
                reconstructed_voice = valid_dataset.reconstruct(
                    voice_pred.cpu(), id0=dataset_idx, reference="voice"
                ).numpy().squeeze()
                target_voice = valid_dataset.reconstruct(
                    voice.cpu(), id0=dataset_idx, reference="voice"
                ).numpy().squeeze()
    
                reconstructed_noise = valid_dataset.reconstruct(
                    noise_pred.cpu(), id0=dataset_idx, reference="noise"
                ).numpy().squeeze()
                target_noise = valid_dataset.reconstruct(
                    noise.cpu(), id0=dataset_idx, reference="noise"
                ).numpy().squeeze()
    
                target_waveform = np.array([target_voice, target_noise])
                reconstructed_waveform = np.array([reconstructed_voice, reconstructed_noise])
                
                mixture_waveform = valid_dataset.reconstruct(
                    X.cpu(), id0=dataset_idx, reference='input'
                ).numpy().squeeze()
                
                mixture_waveform_ = np.array([mixture_waveform, mixture_waveform])
    
                metrics = compute_metrics(
                    target_waveform,
                    reconstructed_waveform,
                    mixture_waveform_
                )
                
                sdr_list.append(metrics['SDR'])
                nsdr_list.append(metrics['NSDR'])
                sir_list.append(metrics['SIR'])
                sar_list.append(metrics['SAR'])
    
        metrics_data = pd.DataFrame({
            'SDR': sdr_list,
            'NSDR': nsdr_list,
            'SIR': sir_list,
            'SAR': sar_list,
        })
    
        csv_path = os.path.join(SCORES_PATH_2, 'val_metrics.csv')
        metrics_data.to_csv(csv_path, index=False)
    
        return metrics_data
