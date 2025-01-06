# Séparation de sources audio par apprentissage profond
___

Comparaison de plusieurs architectures de réseaux de neurones profonds pour la séparation de sources audio (composante vocale/fond sonore).

On dispose de :
- 4000 données d'entrainement avec un mix snr base entre -4dB et +4dB (rapport signal sur bruit) et les composantes réelles de voix et de fond sonore (deux sources à estimer)
- 1000 données pour la validation des modèles 

## Références 

- Jansson et al. (2017) > https://archives.ismir.net/ismir2017/paper/000171.pdf  (UNet)
- Stoller et al. (2018) > https://arxiv.org/pdf/1806.03185 (WaveUNet)
- Yi Luo et al. (2019) > https://arxiv.org/pdf/1809.07454 (Conv-TasNet)