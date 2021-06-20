import matplotlib.pyplot as plt
from scipy.fftpack import dct
from datetime import datetime
import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa
import torch
import os

def stereo_to_mono(data=None):

    """
        Descripció: Funció per convertir una senyal en stereo a mono fent la mitja entre els dos canals.
        
        Arguments ->
            data: Senyal a convertir a mono
        
        Retorn ->
            signal: senyal convertit a mono

    """

    # Es calcula la mitja entre els dos canals
    signal = (data[:,0] + data[:,1]) / 2

    return signal # Retorna el senyal convertit a mono

def normalitzar(data=None):
    
    """
        Descripció: Funció per normalizar un senyal d'àudio entre els valors 1 i -1.
                    El procés divideix els valors del senyal pel valor absolut màxim.
        
        Arguments ->
            data: Senyal a normalitzar
        
        Retorn ->
            data_norm: Senyal normalitzat entre els valors 1 i -1
    """
    
    # Es divideixen els valors pel valor absolut més elevat
    data_norm = data / np.max(np.abs(data))
    
    return data_norm # Retorna els valors normalitzats entre -1 i 1

def create_folder(path=None):
    
    """
        Descripció: Funció per crear directoris.
        
        Arguments ->
            path: ruta al directori a crear
    """

    # Es comprova si existeix el directorio
    if not os.path.exists(path):

        # Si no existeix, es crea
        os.mkdir(path) 

def configuracio():
    
    """
        Descripció: Funció per recuperar la configuració del programa.
        
        Retorn ->
            parametres: diccionari amb els paràmetres de configuració del programa

    """

    # Diccionari per recueprar la configuració
    parametres = dict()

    # Varaible per guardar la línia llegide
    linia = ""

    # Variable per descomposar la línia
    valors = []

    # S'obra el fitxer de configuració
    with open('CONFIG.txt', 'r') as configFile:

        # Es llegeix línia a línia
        for linia in configFile:  

            # Es comprova que no hi hagi el caràcter #      
            if '#' not in linia:

                # Es descomposa la línia
                valors = linia.replace('\n', '').split('=')

                # Es guarda el paràmetre i el valor al diccionari
                parametres[valors[0]] = valors[1]
    
    return parametres # Retorna el diccionari amb els paràmetres

def extract_mfcc(y=None, fm=44100, n_fft=2048, win_length=2048, hop_length=512, n_mels=40, n_mfcc=20):

    """
        Descripció: Funció per extreure calcular els coeficients cepstrals de mel

        Arguments ->
            y: senyal
            fm: freqüència de mostratge
            n_fft: punts de la trasformada
            win_length: mida de la finestra
            hop_length: mida del salt entre trames
            n_mels: número de bandes de mel
            n_mfcc: número de coeficients a retornar
    """

    spectrum = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))**2

    mel_basis = librosa.filters.mel(sr=fm, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=22050)

    mel_spectrum = np.dot(mel_basis, spectrum)

    mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(mel_spectrum), n_mfcc=n_mfcc)

    features_vector = mfcc
    
    deltas = librosa.feature.delta(data=mfcc, order=1)

    features_vector = np.vstack((features_vector, deltas))

    deltas2 = librosa.feature.delta(data=mfcc, order=2)

    features_vector = np.vstack((features_vector, deltas))
    
    return features_vector.T # Retorna la matriu transposada

def split_into_sequences(data=None, frame_padding=None):

    if frame_padding is not None:
        fstart = frame_padding
        fend = len(data)-1

        padding = np.zeros((frame_padding, data.shape[1]))

        data = np.concatenate((padding, data, padding))

        sequences = np.zeros((data.shape[0], data.shape[1]*(frame_padding*2 + 1)))

        for target_frame in range(fstart,fend):
            sequences[target_frame-frame_padding] = np.concatenate((data[target_frame-frame_padding:target_frame], data[target_frame], data[target_frame:target_frame+frame_padding]), axis=None)
    else:
        sequences = data

    return sequences

def extract_melspectrogram(y=None, fm=44100, n_fft=2048, n_mels=128, hop_length=512):

    """
        Descripció: Funció per calcular l'espectrograma de mel

        Arguments ->
            y: Senyal
            fm: Freqüència de mostratge
            n_fft: Punts de la transformada
            n_mels: Nombre de bandes de mel
            hop_length: Tamny del salt
            win_length: Tamany de al finestra
            window: Finestra a aplicar
            power: Potència
        
        Retorna ->
            L'espectrograma de mel en dB
    """

    spec = librosa.feature.melspectrogram(y=y, sr=fm, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)

    spec = librosa.power_to_db(spec).T

    features = np.zeros((spec.shape[0], 200))

    t = spec.shape[0]

    spec = np.concatenate((spec, (np.zeros([4,n_mels]))))

    for i in range(t):
        a = spec[i]
        for j in range(4):
            a = np.concatenate((a, spec[i+j]))
        
        features[i] = a

    return features