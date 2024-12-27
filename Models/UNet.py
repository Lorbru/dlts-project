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



# chemin vers l'enregistrement des modèles entrainés/scores obtenus
PATH = "Paths/UNet/"
SCORES_PATH = "Scores/UNet/"

# Recherche du GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

in_channels=1

out_channels=1

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
# *             UNET                     *
# *                                      *
# ****************************************

class UNet(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256],
    ):
        super(UNet, self).__init__()
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
        return x * input_x

    @staticmethod
    def trainModel(dataset, data_type='voice', n_epochs=20, batch_size=16, learning_rate=0.0001, valid_dataset=None):
        """
        Train the UNet model with optional resumption and validation.

        Args:
            dataset: Training dataset.
            data_type: type of signal to learn (default: 'voice').
            n_epochs: Number of epochs to train (default: 20).
            batch_size: Batch size (default: 16).
            learning_rate: Initial learning rate (default: 0.0001).
            valid_dataset: Validation dataset (optional).

        Returns:
            model: Trained UNet model.
            losses: List of training losses.
        """
        os.makedirs(PATH, exist_ok=True)
        model_path = os.path.join(PATH, data_type)
        os.makedirs(model_path, exist_ok=True)
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        model = UNet().to(device)
        last_saved_epoch = 0

        print(model_files)

        if len(model_files) > 0:
            saved_epochs = [int(i.split('.')[0].split('_')[-1]) for i in model_files]
            last_saved_epoch = max(saved_epochs)
            model.load_state_dict(torch.load(os.path.join(PATH, data_type, f'model_{last_saved_epoch}.pth')))

        csv_path = os.path.join(SCORES_PATH, data_type, 'UNetSAE.csv')
        if os.path.exists(csv_path):
            scores = pd.read_csv(csv_path)
            scores = scores.loc[scores.index < last_saved_epoch]
        else:
            scores = pd.DataFrame(columns=['train', 'valid'])

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss(reduction='sum').to(device)

        losses, valid_losses = [], []
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True) if valid_dataset else None

        for epoch in range(last_saved_epoch, last_saved_epoch + n_epochs):
            model.train()
            running_loss = 0.0

            for X, Y in tqdm(dataloader):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                output = model(X.unsqueeze(1))
                loss = criterion(Y.unsqueeze(1), output)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            losses.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{last_saved_epoch + n_epochs} - Train Loss: {epoch_loss:.4f}')

            # Validation loss
            if valid_dataloader:
                model.eval()
                running_loss = sum(criterion(Y.to(device).unsqueeze(1), model(X.to(device).unsqueeze(1))).item() for X, Y in valid_dataloader)
                valid_loss = running_loss / len(valid_dataloader)
                valid_losses.append(valid_loss)
                print(f'Valid Loss: {valid_loss:.4f}')

            # Save model
            if (epoch + 1) % 10 == 0 or (epoch + 1) == (last_saved_epoch + n_epochs):
                torch.save(model.state_dict(), os.path.join(PATH, data_type, f'model_{epoch + 1}.pth'))

        valid_losses = valid_losses if valid_losses != [] else np.nan
        scores = pd.concat([scores, pd.DataFrame({
            'train': losses,
            'valid': valid_losses
        })])

        csv_path = os.path.join(SCORES_PATH, data_type, 'UNetSAE.csv')
        scores.to_csv(csv_path, index=False)

        return model, losses

    @staticmethod
    def compute_metrics(target_waveform, reconstructed_waveform, mixture_waveform):
        """
        Compute SDR, SIR, SAR, and NSDR metrics using mir_eval.
        
        Args:
            target_waveform (np.ndarray): The target signal (clean voice).
            reconstructed_waveform (np.ndarray): The output signal from the model (estimated voice).
            mixture_waveform (np.ndarray): The original mixture (input signal).
        
        Returns:
            dict: A dictionary with 'SDR', 'SIR', 'SAR', 'NSDR' metrics.
        """
        
        # Convert signals to numpy arrays if they're not already
        target_waveform = np.asarray(target_waveform)
        reconstructed_waveform = np.asarray(reconstructed_waveform)
        mixture_waveform = np.asarray(mixture_waveform)

        # Ensure the input arrays are 2D
        # Check if the shape is 1D and reshape if needed
        if len(target_waveform.shape) == 1:
            target_waveform = np.expand_dims(target_waveform, axis=0)

        if len(target_waveform.shape) == 1:
            target_waveform = np.expand_dims(target_waveform, axis=0)

        if len(mixture_waveform.shape) == 1:
            mixture_waveform = np.expand_dims(mixture_waveform, axis=0)

        # Compute SDR, SIR, SAR using mir_eval
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            target_waveform,  # Targets (2D)
            reconstructed_waveform  # Estimated signals (2D)
        )

        # Compute NSDR (Normalized SDR)
        nsdr = sdr - mir_eval.separation.bss_eval_sources(
            target_waveform,
            mixture_waveform
        )[0][0]

        return {'SDR': sdr[0], 'SIR': sir[0], 'SAR': sar[0], 'NSDR': nsdr[0]}

    @staticmethod
    def validateModel(model, valid_dataset, data_type):
        """
        Validate the model and return raw metrics data for visualization (box plot).

        Args:
            valid_dataset: Validation dataset.
            model: Trained model.

        Returns:
            metrics_data: Dictionary containing lists of metric values for each sample in the validation set.
        """
        model.eval()

        sdr_list, sir_list, sar_list, nsdr_list = [], [], [], []

        with torch.no_grad():
            for dataset_idx in range(len(valid_dataset)):

                X, Y = X.to(device), Y.to(device)
                X, Y = valid_dataset[dataset_idx]
                output = model(X.unsqueeze(0).unsqueeze(0))

                output_squeezed = output.squeeze()

                reconstructed_output = valid_dataset.reconstruct(output_squeezed.cpu(), id0=dataset_idx, reference = data_type.lower())
                target_waveform = valid_dataset.reconstruct(Y.cpu(), id0=dataset_idx, reference= data_type.lower())
                mixture_waveform = valid_dataset.reconstruct(X.cpu(), id0=dataset_idx, reference='input')

                metrics = UNet.compute_metrics(
                    target_waveform.numpy(),
                    reconstructed_output.numpy(),
                    mixture_waveform.numpy()
                )

                sdr_list.append(metrics['SDR'])
                sir_list.append(metrics['SIR'])
                sar_list.append(metrics['SAR'])
                nsdr_list.append(metrics['NSDR'])


        metrics_data = {
            'SDR': sdr_list,
            'SIR': sir_list,
            'SAR': sar_list,
            'NSDR': nsdr_list
        }

        return metrics_data
