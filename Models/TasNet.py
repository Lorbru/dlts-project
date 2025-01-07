import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# paramètres par défaut : 

# encode_dim = 512    
# feature_dim = 128
# hidden_dim = 512
# encode_kernel = 32
# depth_conv_kernel = 3
# stacks = 3
# layers = 4

encode_dim=256 
feature_dim=64
hidden_dim=256
encode_kernel=32
depth_conv_kernel=3
stacks=3
layers=4

PATH = 'Paths/TasNet/'
SCORES_PATH = 'Scores/TasNet/'

# *********************************
# *                               *
# *          SNR LOSS             *
# *                               *
# *********************************

class SNRLoss(nn.Module):

    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, s_target, s_pred):
        noise = s_target - s_pred
        snr = 10 * torch.log10(torch.sum(s_target ** 2) / torch.sum(noise ** 2))
        return -snr  

# *********************************
# *                               *
# *        SOUS MODULES           *
# *                               *
# *********************************

class C1D(nn.Module):

    def __init__(self, sep_dim, hidden_dim, depth_conv_kernel, dilation, padding):
        
        super(C1D, self).__init__()
        
        self.conv11 = nn.Conv1d(in_channels=sep_dim, out_channels=hidden_dim, kernel_size=1)
        self.prelu1  = nn.PReLU()
        self.norm1  = nn.GroupNorm(1, hidden_dim, eps=1e-8)
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=depth_conv_kernel, dilation=dilation, padding=padding)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, hidden_dim, eps=1e-8)
        self.res_out = nn.Conv1d(in_channels=hidden_dim, out_channels=sep_dim, kernel_size=1)
        self.skip_out = nn.Conv1d(in_channels=hidden_dim, out_channels=sep_dim, kernel_size=1)

    def forward(self, x):
        out = self.norm1(self.prelu1(self.conv11(x)))
        out = self.norm2(self.prelu2(self.conv1d(out)))
        return self.res_out(out), self.skip_out(out)


class Separation(nn.Module):

    def __init__(self, input_dim, output_dim, sep_dim, hidden_dim, layers, stacks, kernel):
        
        super(Separation, self).__init__()

        self.layerNorm = nn.GroupNorm(1, input_dim, eps=1e-8)
        self.firstConv = nn.Conv1d(in_channels=input_dim, out_channels=sep_dim, kernel_size=1)

        self.ConvStack = nn.ModuleList([])
        for s in range(stacks) :
            for l in range(layers):
                self.ConvStack.append(C1D(sep_dim, hidden_dim, depth_conv_kernel=kernel, dilation=2**l, padding=2**l))

        self.out_prelu = nn.PReLU()
        self.out_conv = nn.Conv1d(sep_dim, output_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        
        out = self.firstConv(self.layerNorm(x))
        skip_connect = 0.0
        for i in range(len(self.ConvStack)):
            res, skip = self.ConvStack[i](out)
            out = out + res
            skip_connect = skip_connect + skip

        out = self.out_prelu(skip_connect)
        out = self.out_conv(out)
        return self.activation(out)


# **********************************
# *                                *
# *           TAS NET              *
# *                                *
# **********************************

class ConvTasNet(nn.Module):

    def __init__(self, 
                 output_dim, 
                 encode_dim=encode_dim, 
                 sep_dim=feature_dim, 
                 hidden_dim=hidden_dim, 
                 encode_kernel=encode_kernel,
                 layers=layers,
                 stacks=stacks, 
                 depth_conv_kernel=depth_conv_kernel):
        
        super(ConvTasNet, self).__init__()
        stride = encode_kernel//2

        self.encode_kernel = encode_kernel
        self.encode_dim = encode_dim
        self.stride = stride
        self.output_dim = output_dim

        # Encoder 
        self.encoder = nn.Conv1d(in_channels=1, out_channels=encode_dim, kernel_size=encode_kernel, bias=False, stride=stride)

        # Separation
        self.separation = Separation(encode_dim, encode_dim*output_dim, sep_dim, hidden_dim, layers, stacks, depth_conv_kernel)

        # Decoder
        self.decoder = nn.ConvTranspose1d(in_channels=encode_dim, out_channels=1, kernel_size=encode_kernel, bias=False, stride=stride)

    def pad(self, x):
        batch_size = x.size(0)
        nsample = x.size(2)
        rest = self.encode_kernel - (self.stride + nsample % self.encode_kernel) % self.encode_kernel
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(x.type())
            x = torch.cat([x, pad], 2) 
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(x.type())
        x = torch.cat([pad_aux, x, pad_aux], 2)
        return x, rest

    def forward(self, x):
        
        # padding + batch_size
        out, r = self.pad(x)
        B = out.size(0)

        # encoder
        enc_out = self.encoder(out)
     
        # masks generation
        masks = self.separation(enc_out).view(B, self.output_dim, self.encode_dim, -1)

        # decoder
        output = enc_out.unsqueeze(1) * masks

        # decoder
        output = self.decoder(output.view(B*self.output_dim, self.encode_dim, -1))
        return output[:,:,self.stride:-(r+self.stride)].contiguous().view(B, self.output_dim, -1)
    
    def encode_viz(self, x):
        
        out, r = self.pad(x)
        B = out.size(0)
        return self.encoder(out)
    
    def masks_viz(self, x):

        # padding + batch_size
        out, r = self.pad(x)
        B = out.size(0)

        # encoder
        enc_out = self.encoder(out)
     
        # masks generation
        return self.separation(enc_out).view(B, self.output_dim, self.encode_dim, -1)


    
    @staticmethod
    def trainModel(dataset, n_epochs=20, batch_size=16, learning_rate=0.001, valid_dataset=None):
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

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # gestion des fichiers de modèles enregistrés
        os.makedirs(PATH, exist_ok=True)

        model_files = os.listdir(PATH)
        model = ConvTasNet(2).to(device)
        last_saved_epoch = 0

        if len(model_files) > 0 : 
            saved_epochs = [int(i.split('.')[0].split('_')[-1]) for i in model_files]
            last_saved_epoch = max(saved_epochs)
            model.load_state_dict(torch.load(os.path.join(PATH, f'model_{last_saved_epoch}.pth')))

        if os.path.exists(SCORES_PATH + 'TasNetSNRLoss.csv') : 
            scores = pd.read_csv(SCORES_PATH + 'TasNetSNRLoss.csv')
            scores = scores.loc[scores.index < last_saved_epoch]
        else : 
            scores = pd.DataFrame(columns=['train', 'valid'])

        # entrainement
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        criterion = SNRLoss().to(device)
        
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
            if (epoch+1)%10 == 0 or (epoch+1) == n_epochs : torch.save(model.state_dict(), f'Paths/TasNet/model_{epoch+1}.pth')

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

        scores.to_csv(SCORES_PATH + 'TasNetSNRLoss.csv', index=False)

        return model, losses