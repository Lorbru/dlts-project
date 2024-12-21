import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt

# **************************************
# *                                    *
# *       Chargement des données       *
# *                                    *
# **************************************


### Frequence d'échantillonage des signaux par défaut : 8kHz
DEFAULT_NFFT = 1000    # taille de la fenêtre temps/fréquence
DEFAULT_HL   = 750     # intervalle d'analyse temps frequence

### Padding pour obtenir des tailles de convolution en puissance de 2 
### (identiques à ceux de l'article pour UNet ~ 11 secondes d'enregistrement)
FREQ_SIZE = 512
TIME_SIZE = 128

class Audiodataset(Dataset):


    """
    -- Récupération d'un jeu de données audio sous la forme de tenseur, (sous forme d'onde ou spectrogramme)
    """

    def __init__(self, set='train_small', spectrogram=False, snr_filter=None, resample_n_points=None, padding=False, getitem='all'):
        
        # Chemin vers le jeu de données chargé
        self.root_dir = 'Data/' + set
        samples = os.listdir(self.root_dir)

        # Propriétés
        self.__data = []                          # noms des fichiers collectés
        self.__resample = resample_n_points       # resample des audios pour obtenir un échantillon de n points (10sec * fe, par défaut 10*8000Hz)
        self.__return_spectrogram = spectrogram   # retourne les données : True = en image temps/fréquence (spectrogramme) / False = en forme d'onde (wave)
        self.__n_fft = DEFAULT_NFFT               # taille de la fenêtre de transformation temps/fréquence
        self.__hop_length = DEFAULT_HL            # pas de temps entre deux fenêtre successives
        self.__padding = padding                  # padding sur les images de spectrogramme
        self.__snr_filter = snr_filter            # pour filtrer la catégorie de signaux selon le snr spécifié (None = conservation de tous les signaux)
        self.__getitem = getitem                  # indique ce que la méthode __getitem__ doit retourner : 
                                                  #         - 'all'   : (entrée, voix, bruit, snr, sample rate)
                                                  #         - 'voice' : (entrée, voix)
                                                  #         - 'noise' : (entrée, bruit)

        # Collecte des chemins de fichiers
        for sample in samples :
            sample_path = os.path.join(self.root_dir, sample)
            for file in os.listdir(sample_path) :
                if file.startswith('mix_snr_') : 
                    input_file = str(os.path.join(sample_path, file))
                    snr = int(file.split('.')[0].split('_')[-1])
                elif file == "noise.wav" : noise_file = str(os.path.join(sample_path, file))
                elif file == "voice.wav" : voice_file = str(os.path.join(sample_path, file))

            if self.__snr_filter == None or self.__snr_filter == snr :
                sample_dic = {
                    "input":input_file,
                    "snr":snr,
                    "noise":noise_file,
                    "voice":voice_file,
                }
                self.__data.append(sample_dic)

    def __len__(self):
        """
        -- taille du dataset
        """
        return len(self.__data)
    
    def __getitem__(self, idx):
        """
        -- récupération d'un item
        """

        # récupération d'un item (chargement des fichiers audio)
        sample_dic = self.__data[idx]
        input, sr_input = torchaudio.load(sample_dic['input'])
        if self.__getitem != 'voice' : noise, _ = torchaudio.load(sample_dic['noise'])
        if self.__getitem != 'noise' : voice, _ = torchaudio.load(sample_dic['voice'])
        sample = None

        # rééchantillonage éventuel
        if self.__resample != None and self.__resample > 0 :
            resampler = torchaudio.transforms.Resample(orig_freq=10*sr_input, new_freq=self.__resample) # x10 (audio duration)
            input = resampler(input)
            if self.__getitem != 'voice' : noise = resampler(noise)
            if self.__getitem != 'noise' : voice = resampler(voice)
            sr_input = self.__resample

        # récupération du spectrogramme éventuel
        if self.__return_spectrogram :

            # STFT + magnitudes
            input = torch.abs(torch.stft(input, n_fft=self.__n_fft, hop_length=self.__hop_length, return_complex=True)).squeeze()
            if self.__getitem != 'voice' : noise = torch.abs(torch.stft(noise, n_fft=self.__n_fft, hop_length=self.__hop_length, return_complex=True)).squeeze()
            if self.__getitem != 'noise' : voice = torch.abs(torch.stft(voice, n_fft=self.__n_fft, hop_length=self.__hop_length, return_complex=True)).squeeze()

            # padding éventuel
            if self.__padding :
                s_freq, s_time = input.shape
                padding = (0, TIME_SIZE - s_time, 0, FREQ_SIZE - s_freq)
                input = F.pad(input, padding, mode='constant', value=0)
                if self.__getitem != 'voice' : noise = F.pad(noise, padding, mode='constant', value=0)
                if self.__getitem != 'noise' : voice = F.pad(voice, padding, mode='constant', value=0)

            # normalisation des magnitudes dans [0, 1]
            vmin, vmax = input.min(), input.max()
            input = (input - vmin)/(vmax - vmin)

            if self.__getitem != 'voice' :
                vmin, vmax = noise.min(), noise.max()
                noise = (noise - vmin)/(vmax - vmin)
            
            if self.__getitem != 'noise' :
                vmin, vmax = voice.min(), voice.max()
                voice = (voice - vmin)/(vmax - vmin)

        snr = sample_dic['snr']
        
        if self.__getitem == 'all' :     sample = (input, voice, noise, snr, sr_input)
        elif self.__getitem == 'concatVN' : sample = (input, torch.cat([voice, noise], dim=0))
        elif self.__getitem == 'voice' : sample = (input, voice)
        elif self.__getitem == 'noise' : sample = (input, noise)
        
        return sample
    
    # ******************** #
    # ----- Méthodes ----- #
    # ******************** #

    # ---- propriétés concernant les éléments à retourner par getitem ----

    def take_spectrogram(self, n_fft=DEFAULT_NFFT, hop_length=DEFAULT_HL):
        """
        -- Pour retourner le spectrogramme des entrées/voix/bruits

        >> :
            - n_fft : taille de fenêtre
            - hop_length : pas de fenêtre
        """
        self.__return_spectrogram = True
        self.__n_fft = n_fft
        self.__hop_length = hop_length

    def take_wave(self):
        """
        -- Pour retourner la forme d'onde des entrées/voix/bruits
        """
        self.__return_spectrogram = False

    def return_only_noise(self):
        """
        -- pour que __getitem__ retourne (entrée, bruit)
        """
        self.__getitem = 'noise'

    def return_only_voice(self):
        """
        -- pour que __getitem__ retourne (entrée, voix)
        """
        self.__getitem = 'voice'

    def return_all(self):
        """
        -- pour que __getitem__ retourne (entrée, voix, bruit, snr, sample_rate)
        """
        self.__getitem = 'all'
    
    # ---- Reconstruction de la forme d'onde à partir d'un spectrogramme ----
    
    def reconstruct(self, X, id0, reference='input'):
        """
        -- Reconstruit la forme d'onde du spectrogramme X selon les caractéristiques (magnitude, phase) de la donnée id0 associée
        """
        # donnée originale
        original, _ = torchaudio.load(self.__data[id0][reference])
        original_stft = torch.stft(original, n_fft=self.__n_fft, hop_length=self.__hop_length, return_complex=True)
        X = X.squeeze()

        # magnitude avant normalisation
        original_magnitude = torch.abs(original_stft).squeeze()

        # phase du signal
        original_phase = torch.angle(original_stft)
        
        # échelle de magnitude et section du signal 
        vmin, vmax = original_magnitude.min(), original_magnitude.max()
        s_freq, s_time = original_magnitude.shape

        # suppression du padding
        reconstruct  = X[:s_freq, :s_time]

        # denormalisation
        reconstruct =  reconstruct*(vmax - vmin) + vmin

        # domaine complexe
        reconstruct = reconstruct * torch.exp(1j * original_phase)

        # forme d'onde 
        return torch.istft(reconstruct, n_fft=self.__n_fft, hop_length=self.__hop_length, length=original.size(1))

    
    # ---- fonctions de visualisation ----

    def plot(self, idx):

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        return_mod = self.__getitem
        self.return_all()
        input, voice, noise, _, _ = self[idx]
        self.__getitem = return_mod

        input = input.squeeze().numpy()
        voice = voice.squeeze().numpy()
        noise = noise.squeeze().numpy()

        if self.__return_spectrogram : 
            
            axs[0].imshow(input, aspect='auto', origin='lower', cmap='hot')
            axs[0].set_title('Input')
            axs[0].set_xlabel('time')
            axs[0].set_ylabel('frequency')

            axs[1].imshow(voice, aspect='auto', origin='lower', cmap='hot')
            axs[1].set_title('Voice')
            axs[1].set_xlabel('time')
            axs[1].set_ylabel('frequency')

            axs[2].imshow(noise, aspect='auto', origin='lower', cmap='hot')
            axs[2].set_title('Noise')
            axs[2].set_xlabel('time')
            axs[2].set_ylabel('frequency')

        else :

            axs[0].plot(input, color='blue')
            axs[0].grid()
            axs[0].set_title('Input')
            axs[0].set_xlabel('time')
            axs[0].set_ylabel('wave')

            axs[1].plot(voice, color='green')
            axs[1].grid()
            axs[1].set_title('Voice')
            axs[1].set_xlabel('time')
            axs[1].set_ylabel('wave')

            axs[2].plot(noise, color='red')
            axs[2].grid()
            axs[2].set_title('Noise')
            axs[2].set_xlabel('time')
            axs[2].set_ylabel('wave')

        plt.tight_layout()
        



