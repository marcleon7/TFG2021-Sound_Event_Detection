from sklearn import preprocessing
import soundfile as sf
from tqdm import tqdm
from model import *
from utils import *
import numpy as np
import os

# Diccionari per convertir les classes a números
class_labels = {
    'brakes squeaking'  : 0,
    'car'               : 1,
    'children'          : 2,
    'large vehicle'     : 3,
    'people speaking'   : 4,
    'people walking'    : 5
}

def load_setup(folder='train', fold_number=1):

    """
        Descripció: Funció per recuperar la carpeta de configuració.
        
        Arguments ->
            folder: Nom de la carpeta
            fold_number: Número de carpeta
        
        Retorn ->
            setup_dic: Retorna un diccionari amb les dades de cada fitxer d'àudio

    """

    # Diccionari per guardar les dades
    setup_dic = dict() 

    # Ruta al directori de fitxers de configuració
    test_setup = 'dataset/TUT-sound-events-2017-development/evaluation_setup'

    if folder=='train':
        # Fitxer d'entrenament
        path = os.path.join(test_setup, 'street_fold{}_train.txt'.format(fold_number))
    elif folder=='test':
        # Fitxer de test
        path = os.path.join(test_setup, 'street_fold{}_evaluate.txt'.format(fold_number))

    # Es recorren les línies del fitxer
    for line in open(path):
        words = line.strip().split('\t') # Es guarden les paraules en una llista
        audio_name = words[0].split('/')[-1] # Nom del fitxer d'àudio

        # Si el nom no existeix al diccionari, es crea una entrada nova
        if audio_name not in setup_dic:
            setup_dic[audio_name] = list()
            
        # Es guarden el temps d'inici, el temps de fi i la etiqueta (en format de número)
        setup_dic[audio_name].append([float(words[2]), float(words[3]), int(class_labels[words[-1]])])
    
    return setup_dic # Retorna el diccionari

def list_unique_files(folder='train', fold_number=1):

    """
        Descripció: Funció per llistar els fitxers totals que formen el fitxer d'enumeracions

        Arguments ->
            folder: Nom de la carpeta
            fold_number: Número de la carpeta
        
        Retorna ->
            unique_files: Llista amb els noms dels arxius que formen el fitxer d'enumeracions
    """

    setup_file = 'dataset/TUT-sound-events-2017-development/evaluation_setup/street_fold{}_{}.txt'.format(fold_number, folder)

    # Llista amb els fitxers únics
    unique_files = []

    # Variable per guardar l'últim fitxer llegit
    last_file = ""

    with open(setup_file, 'r') as file:
        for line in file:
            # Fitxer actual
            this_file = line.strip().split('\t')[0].split('/')[-1]

            # Si el fitxer actual és diferent a l'anterior, es guarda a la llista
            if this_file != last_file:
                last_file = this_file
                unique_files.append(last_file)

    return unique_files # Retorna la llista dels fitxers existentes en l'arxiu

def load_features_data(features_folder=None, fold=1):
    
    """
        Descripció: Funció per recuperar les dades d'entrenament i de test.

        Arguments ->
            features_folder: Carpeta on es troben les característiques per cada fitxer d'àudio
            fold: Número de carpeta
        
        Retorna ->
            X_train: Dades d'entrada de la xarxa neuronal per l'entrenament
            X_test: Dades d'entrada de la xarxa neuronal per la validació
            Y_train: Dades de sortida de la xarxa neuronal per l'entrenament
            Y_test: Dades de sortida de la xarxa neuronal per la validació
    """

    # Llistes on es guarden les dades recuperades
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None

    # Es recuperen els arxius únics de la carpeta de setup d'entrenament i avaluació
    train_files = list_unique_files(folder='train', fold_number=fold)
    test_files = list_unique_files(folder='test', fold_number=fold)

    # Es recuperen les matrius d'entrada i sortida pels fitxers d'entrenament
    for file in train_files:
        npz_file = np.load(os.path.join(features_folder, file.replace(".wav", ".npz")), allow_pickle=True)
        mfcc = npz_file['arr_0']
        labels = npz_file['arr_1']

        if X_train is None and Y_train is None:
            X_train = mfcc
            Y_train = labels
        else:
            X_train = np.concatenate((X_train, mfcc), 0)
            Y_train = np.concatenate((Y_train, labels), 0)
            
    # Es recuperen les matrius d'entrada i sortida pels fitxers d'avaluació
    for file in test_files:
        npz_file = np.load(os.path.join(features_folder, file.replace(".wav", ".npz")))
        mfcc = npz_file['arr_0']
        labels = npz_file['arr_1']

        if X_test is None and Y_test is None:
            X_test = mfcc
            Y_test = labels
        else:
            X_test = np.concatenate((X_test, mfcc), 0)
            Y_test = np.concatenate((Y_test, labels), 0)
    
    # Es normalitzen les dades
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def extract_features(audio_folder=None, features_folder=None, fm=44100, nfft=2048, window_length=2048, hop_size=1024, n_mels=40, n_mfcc=12, frame_padding=None):

    """
        Descripció: Funció per calcular els coeficients de mel dels senyals d'àudio de la base de dades

        Arguments ->
            audio_folder: Carpeta que conté els arxius d'àudio
            features_folder: Carpeta on es guardaran les dades de característiques
            fm: Freqüència de mostratge dels senyals d'àudio
            nftt: Número de punts de la transformada
            window_length: Mida de la finestra
            hop_size: Mida del salt entre trames
            n_mels: Número de filtres de Mel
            n_mfcc: Número de coeficients de Mel a calcular
            frame_padding: Número de trames a afegir
    """

    # Es llegeix el fitxer de setup i s'importen els arxius amb les seves etiquetes
    data_dict = load_setup(folder='train', fold_number=1)
    data_dict.update(load_setup(folder='test', fold_number=1))

    # Es crea la carpeta on es guarden els Mel Frequency Cepstral Coefficients
    create_folder(path=features_folder)

    # S'importen tots els arxius d'àudio
    audio_files = os.listdir(audio_folder)
           
    for audiofile in tqdm(audio_files):

        # Es llegeix el fitxer d'àudio
        data, sr = sf.read(os.path.join(audio_folder, audiofile))

        # Es converteix de stereo a mono
        data = stereo_to_mono(data=data)

        # Es normalitza entre els valors 1 i -1
        data = normalitzar(data=data)

        # S'extreuen els mel frequency cepstral coefficients del senyal d'àudio
        mfcc = extract_mfcc(y=data, fm=fm, n_fft=nfft, win_length=window_length, hop_length=hop_size, n_mels=n_mels, n_mfcc=n_mfcc)

        # Es generen els vectors de característiques
        mfcc = split_into_sequences(data=mfcc, frame_padding=frame_padding)

        # Matriu de labels per cada fitxer d'àudio
        labels = np.zeros((mfcc.shape[0], len(class_labels))) # Tamany: [número de frames, número de categories]
            
        # Es recuperen els valors del diccionari corresponents al fitxer d'àudio
        tmp_data = np.array(data_dict[audiofile]) # [Temps inici, Temps final, Categoria]

        # Frame de inici on es troba l'esdeveniment acústic
        frame_start = np.floor(tmp_data[:,0] * fm / hop_size).astype(int) # Int

        # Frame final on acaba l'esdeveniment acústic
        frame_end = np.ceil(tmp_data[:,1] * fm / hop_size).astype(int) # Int

        # Categoria de l'esdeveniment acústic
        se_class = tmp_data[:,2].astype(int) # Int

        # Es construeix la matriu de nivells indicant per cada frame, la categoria present 
        for i, valor in enumerate(se_class):
            # S'indica 1 si la categoria està present, 0 si no està present
            labels[frame_start[i]:frame_end[i], valor] = 1
            
        # Nom de l'arxiu d'àudio on es guarda el seu pre-processament de dades (mfcc i labels)
        file_name = os.path.join(audiofile.split('/')[-1].replace(".wav", "")) + ".npz"

        # Directori on es guarden les dades pre-processades
        path = os.path.join(features_folder, file_name)

        # Es guarden les dades en un arxiu .npz per cada fitxer d'àudio
        np.savez(path, mfcc, labels)

def tagging_test_files(fold=None, audio_folder=None, setup_folder=None, fm=44100, nfft=2048, window_length=2048, hop_size=1024, n_mels=40, n_mfcc=12, frame_padding=None):
    test_file = os.path.join(setup_folder, "street_fold{}_test.txt".format(fold))

    with open(test_file, 'r') as test:
        for line in test:
            fileName = line.split('\t')[0].split('/')[-1]
            audioPath = os.path.join(audio_folder, fileName)

            predict(file=audioPath, tagFile="street_fold{}_{}_detected.txt".format(fold, fileName.split('.')[0]), fold=fold, nfft=nfft, window_length=window_length, hop_size=hop_size, n_mels=n_mels, n_mfcc=n_mfcc, frame_padding=frame_padding)