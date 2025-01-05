import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import mir_eval

# chemin vers l'enregistrement des modèles entrainés/scores obtenus
PATH = "Paths/WaveUNet/"
SCORES_PATH = "Scores/WaveUNet/"

# Recherche du GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

L = 12     # Nombre de blocs encodeur/decodeur
Fc = 24    # Filtres/layer
fd = 15    # Noyau encodeur
fu = 5     # Noyau decodeur


# ****************************************
# *                                      *
# *           SOUS MODULES               *
# *                                      *
# ****************************************

class Downsampler(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Downsampler, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.activate(self.conv(x)) # Convolution
        return x

class Upsampler(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Upsampler, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.activate = nn.LeakyReLU()

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='linear') # Interpolation (Upsample)
        x = torch.cat([x, y], dim=1)  # Concatenation
        return self.activate(self.conv(x))  # Convolution

class FinalUpsampler(nn.Module):

    def __init__(self, in_channels, K, kernel_size):
        super(FinalUpsampler, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=K, kernel_size=kernel_size, padding='same')
        self.activate = nn.Tanh()

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)  # Concatenation
        return self.activate(self.conv(x))  # Convolution


# ****************************************
# *                                      *
# *             WAVE UNET                *
# *                                      *
# ****************************************

class WaveUNet(nn.Module):

    def __init__(self, L=12, Fc=24, fd=15, fu=5, K=2):
        
        super(WaveUNet, self).__init__()
        
        self.L = L         # nombre de layers de l'encodeur

        # encoding
        self.DS = nn.ModuleList()
        in_channels = 1                                                                 
        for i in range(1, L+1):                                                         
            out_channels = Fc * i                                                       
            self.DS.append(Downsampler(in_channels, out_channels, fd))                 
            in_channels = out_channels                                                  

        # latent space
        self.latentConv = nn.Conv1d(in_channels, Fc*(L+1), fd, padding='same')          
        self.latentLeakRelu = nn.LeakyReLU()

        # decoding
        self.US = nn.ModuleList()
        for i in range(L, 0, -1):      
            in_channels = Fc * (2*i + 1)                                                
            out_channels = Fc * i
            self.US.append(Upsampler(in_channels, out_channels, fu))
            in_channels = out_channels

        # final output
        self.final = FinalUpsampler(Fc + 1, K, 1)


    def forward(self, x):

        DSoutputs = [x.clone()]                    # x:[B, 1, T]

        # encoding
        for i in range(self.L):                    # for i in range(1...L=12) :
            x = self.DS[i](x)                      #     x:[B, 24*i, T_i]
            DSoutputs.append(x.clone())            #     DS Block i:[B, 24*i, T_i]
            x = x[:, :, ::2]                       #     x:[N, 24*i, T_i/2]  (Decimation)

        # latent space
        x = self.latentConv(x)                     # x:[B, 24*13, 4]
        x = self.latentLeakRelu(x)

        # decoding
        for i in range(self.L):                    # for i in range(L=12...1) :
            y = DSoutputs[-1-i]                    #    DS Block i:[B, 24*i, T_i]   
            x = self.US[i](x, y)                   #    x:[B, 24*i, 2*T_i]  (interp(x:[B, 24*i, 2*T_i]) + concat x, y:[[B, 24*(i+1), 2*T_i], [B, 24*i, 2*T_i] + Conv1d)
        
        # final output
        return self.final(x, DSoutputs[0])         #    x:[B, K, T]
    

    @staticmethod
    def trainModel(dataset, n_epochs=20, batch_size=16, learning_rate=0.0001, valid_dataset=None):
        """
        -- Entrainement du réseau. Reprend l'apprentissage
        depuis le modèle le plus avancé retrouvé dans Paths/WaveUNet/

        >> IN : 
            * dataset : jeu d'entrainement
            * n_epochs : nombre d'epoques à effectuer
            * batch_size : taille de batch des données
            * learning_rate : taux d'apprentissage initial
            * valid_dataset (optionnel) : jeu de validation pour évaluer le surapprentissage simultanément

        << OUT : 
            * référence vers le modèle à l'état final
            * pertes calculées sur l'entrainement
            + Ecriture des états du modèles toutes les dix epoques dans Paths/WaveUNet
            + Mise à jour des scores, Scores/WaveUNet/WaveUNetMSE.csv
        """

        # gestion des fichiers de modèles enregistrés
        os.makedirs(PATH, exist_ok=True)

        model_files = os.listdir(PATH)
        model = WaveUNet().to(device)
        last_saved_epoch = 0

        if len(model_files) > 0 : 
            saved_epochs = [int(i.split('.')[0].split('_')[-1]) for i in model_files]
            last_saved_epoch = max(saved_epochs)
            model.load_state_dict(torch.load(os.path.join(PATH, f'model_{last_saved_epoch}.pth')))

        if os.path.exists(SCORES_PATH + 'WaveUNetMSE.csv') : 
            scores = pd.read_csv(SCORES_PATH + 'WaveUNetMSE.csv')
            scores = scores.loc[scores.index < last_saved_epoch]
        else : 
            scores = pd.DataFrame(columns=['train', 'valid'])

        # entrainement
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        criterion = nn.MSELoss().to(device)
        
        losses = []
        valid_losses = []
        
        if valid_dataset != None : valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

        for epoch in range(last_saved_epoch, last_saved_epoch + n_epochs):
            
            running_loss = 0.0
            model.train()

            for X, Y in tqdm(dataloader) : 

                X, Y = X.to(device), Y.to(device)

                optimizer.zero_grad() 

                output = model(X)

                loss = criterion(Y, output)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            # Affichage de la perte
            epoch_loss = running_loss/len(dataloader)
            losses.append(epoch_loss)
            print(f'Epoch {epoch+1}/{last_saved_epoch + n_epochs} - Train Loss : {epoch_loss:4f}')

            # Enregistrement de l'état du modèle toutes les dix époques ou sur la dernière époque
            if (epoch+1)%10 == 0 or (epoch+1) == n_epochs : torch.save(model.state_dict(), f'Paths/WaveUNet/model_{epoch+1}.pth')

            if valid_dataset != None :
                
                running_loss = 0.0

                model.eval()
                for X, Y in valid_dataloader:

                    X, Y = X.to(device), Y.to(device)

                    output = model(X)

                    loss = criterion(Y, output)

                    running_loss += loss.item()

                valid_loss = running_loss/len(valid_dataloader)
                valid_losses.append(valid_loss)
                print(f'Valid Loss : {valid_loss:4f}')

        if valid_losses == [] : valid_losses = np.nan
        scores = pd.concat([scores, pd.DataFrame({
            'train':losses,
            'valid':valid_losses
        })])

        scores.to_csv(SCORES_PATH + 'WaveUNetMSE.csv', index=False)

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