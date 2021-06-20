from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import soundfile as sf
from utils import *
from dataset import *
import numpy as np
import os

class_labels = {
    'brakes squeaking'  : 0,
    'car'               : 1,
    'children'          : 2,
    'large vehicle'     : 3,
    'people speaking'   : 4,
    'people walking'    : 5
}

def build_model(input=None, output=None, lr=0.001, show=False):

    """
        Descripció: Funció per configurar el model.
        
        Arguments ->
            input: Tamany de la entrada al sistema
            output: Tamany de la sortida del sistema
            lr: Pas d'aprenentatge en cada iteració
            show: Mostrar la configuració del model
        
        Retorn ->
            model: model generat

    """

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(input))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(output, activation='sigmoid'))
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    if(show): model.summary()

    return model

def save_model(model=None, fold=None):

    """
        Descripció: Funció per guardar el model generat.
        
        Arguments ->
            model: Model a guardar
            fold: Número de la carpeta de referència del model
    """

    # Es crea la carpeta de models si no existeix
    create_folder('models')

    # Es guarda el model
    model.save('models/Sistema_Acustic_Marc_Leon_fold{}.h5'.format(fold))

def load_model(fold=None):

    """
        Descripció: Funció per carregar un model guardat.
        
        Arguments ->
            fold: Número de la carpeta de referència del model a recuperar
        
        Retorn ->
            model: Retorna el model carregat

    """
    # Es carrega el model
    model = keras.models.load_model('models/Sistema_Acustic_Marc_Leon_fold{}.h5'.format(fold))

    return model

def save_results(metrics=None):

    """
        Descripció: Funció per generar les gràfiques del model.
        
        Arguments ->
            history: Llista de valors a graficar
            fold: Número de la carpeta de referència del model
    """

    create_folder('resultats')

    accuracy_mean = np.mean(metrics['accuracy_train'], axis=0)
    val_accuracy_mean = np.mean(metrics['val_accuracy_train'], axis=0)

    plt.figure("Precisió")
    plt.title("Precisió")
    plt.plot(accuracy_mean, label='Entrenament')
    plt.plot(val_accuracy_mean, label='Validació')
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    plt.legend(loc='lower right')
    plt.xlabel('Iteracions')
    plt.ylabel('Precisió')
    plt.savefig('resultats/accuracy.png')

    loss_mean = np.mean(metrics['loss_train'], axis=0)
    val_loss_mean = np.mean(metrics['val_loss_train'], axis=0)

    plt.figure("Funció de pèrdua")
    plt.title("Funció de pèrdua")
    plt.plot(loss_mean, label='Entrenament')
    plt.plot(val_loss_mean, label='Validació')
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    plt.legend(loc='lower right')
    plt.xlabel('Iteracions')
    plt.ylabel('Pèrdua')
    plt.savefig('resultats/loss.png')

def train_model(model=None, X=None, Y=None, X_test=None, Y_test=None, EPOCHS=None, BATCH_SIZE=None):

    """
        Descripció: Funció per entrenar el model

        Arguments ->
            features_folder: Directori de característiques
            input: Nombre de neurones de la capa d'entrada
            output: Nombre de neurones de la última capa
            lr: Learning rate
            EPOCHS: Nombre d'iteracions d'entrenament
            BATCH_SIZE: Tamany del lot d'entrenament

        Retorna ->
            model: model entrenat
            history: resultats de l'entrenament
    """
          
    # S'entrena i es recuperen els resultats del model
    history = model.fit(x=X, y=Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), shuffle=True, verbose=1)

    return history # Es retorna el model i els resultats

def evaluate_model(model=None, X_test=None, Y_test=None):

    """
        Descripció: Funció per avaluar el sistema

        Arguments ->
            model: model a avaluar
            X_test: dades d'entrada per generar les prediccions
            Y_test: prediccions reals

        Retorna -> 
            Retorna les puntuacions obtingudes pel model avaluat
            
    """

    # Predicció
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred)
    
    # True Positives
    m = keras.metrics.TruePositives()
    m.update_state(Y_test, Y_pred)
    TP = m.result().numpy()

    # True Negatives
    m = keras.metrics.TrueNegatives()
    m.update_state(Y_test, Y_pred)
    TN = m.result().numpy()

    # False Positives
    m = keras.metrics.FalsePositives()
    m.update_state(Y_test, Y_pred)
    FP = m.result().numpy()

    # False Negatives
    m = keras.metrics.FalseNegatives()
    m.update_state(Y_test, Y_pred)
    FN = m.result().numpy()

    # Error Rate ER
    S = np.min([FN, FP])
    D = np.max([0, FN-FP])
    I = np.max([0, FP-FN])
    N = Y_test.sum()

    ER = (S + D + I) / N

    # Accuracy
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    # Recall
    recall = TP / (TP + FN)

    # Precision
    precision = TP / (TP + FP)

    # F1 score
    F1 = 2 * ((precision * recall) / (precision + recall))

    return accuracy, recall, precision, F1, ER, TP, TN, FP, FN

def predict(file=None, tagFile='etiquetes.txt', fold=1, fm=44100, nfft=2048, window_length=2048, hop_size=1024, n_mels=40, n_mfcc=12, frame_padding=None):

    """
        Descripció: Funció per generar les etiquetes d'un senyal acústic.
        
        Arguments ->
            file: Ruta al directori on es troba el arxiu d'àudio
            tagFile: Nome del fitxer d'etiquetes
            fm: Freqüència de mostratge
            nfft: Nombre de punts de la transformada de fourier
            hop_size: Tamany del salt per calcular les característiques del arxiu d'àudio
            n_mels: Nombre de filtres de mel a utilitzar
            n_mfcc: Nombre de coeficients a calcular
    """

    segments_predits = {
        'brakes squeaking'  : [],
        'car'               : [],
        'children'          : [],
        'large vehicle'     : [],
        'people speaking'   : [],
        'people walking'    : []
    }

    segments_postprocessats = {
        'brakes squeaking'  : [],
        'car'               : [],
        'children'          : [],
        'large vehicle'     : [],
        'people speaking'   : [],
        'people walking'    : []
    }

    nomArxiu = file.split('/')[-1]

    # Es carrega el model    
    model = load_model(fold=fold)
    
    # Es llegeix el senyal
    samples, sr = sf.read(file)

    # Es converteix de stereo a mono
    data = stereo_to_mono(data=samples)

    # Es normalitza entre els valors 1 i -1
    data = normalitzar(data=data)

    # S'extreuen els mel frequency cepstral coefficients del senyal d'àudio
    mfcc = extract_mfcc(y=data, fm=fm, n_fft=nfft, hop_length=hop_size, n_mels=n_mels, n_mfcc=n_mfcc)

    # Es generen les seqüències
    mfcc = split_into_sequences(data=mfcc, frame_padding=frame_padding)

    # Es normalitzen les mostres
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    mfcc = scaler.fit_transform(mfcc)

    # Es realitza la predicció del model
    prediction = model.predict(mfcc)

    prediction = np.round(prediction)

    # Es crea la carpeta on guardar les etiquetes en el cas que no existeixi
    create_folder('etiquetes')
            
    for classe in range(np.shape(prediction)[1]):
            
        # Segment actual
        frame = 0

        # Control de classes actives
        active_class = False
            
        # Es recorren tots els segments
        while frame < np.shape(prediction)[0]: 

            # Es comprova si la classe està activa
            if prediction[frame][classe] == 1:    

                # Es recupera el segment d'inici
                frame_start = frame

                # S'estableix que al classe està activa
                active_class = True
                
                # Es comprova el segment de finalització
                while prediction[frame][classe] != 0:
                    frame += 1

                    # Es controla que no se superi el màxim de segments del senyal
                    if frame >= np.shape(prediction)[0]:
                        break
                    
                # Es guarda el segment de finalització
                frame_end = frame

            if active_class:
                # Es calcula el temps d'inici i fi i la etiqueta de l'esdeveniment
                time_start = frame_start * hop_size / fm
                time_end = frame_end * hop_size / fm
                etiqueta = list(class_labels.keys())[list(class_labels.values()).index(classe)]

                duracio = time_end - time_start

                segments_predits[etiqueta].append([time_start, time_end])

            active_class = False
            frame+=1

    for classe in segments_predits:

        # Es mira si hi ha etiquetes generades per la classe a processar
        if(len(segments_predits[classe])):
            # Es guarden els temps d'inici i fi del primer segment
            buffer_onset = segments_predits[classe][0][0]
            buffer_offset = segments_predits[classe][0][1]

            for i in range(1, len(segments_predits[classe])):
                # Si la distància entre dos segments és major a 0.5s, es guarda com una etiqueta nova
                if segments_predits[classe][i][0] - buffer_offset > 0.5:
                    # Es guarda el temps d'inici i fi com una etiqueta
                    segments_postprocessats[classe].append([buffer_onset, buffer_offset])

                    # Es guarden els nous temps d'inici i fi del següent segment
                    buffer_onset = segments_predits[classe][i][0]
                    buffer_offset = segments_predits[classe][i][1]

                # Si és menor a 100ms, es fusionen els segments
                else:
                    buffer_offset = segments_predits[classe][i][1]

            # Es genera la etiqueta per l'últim segment
            segments_postprocessats[classe].append([buffer_onset, buffer_offset])

    with open(os.path.join('etiquetes/', tagFile), 'w') as tagFile:

        for classe in segments_postprocessats.keys():

            for time_tag in segments_postprocessats[classe]:
                
                if time_tag[1] - time_tag[0] >= 1.5:
                    # Es genera la etiqueta
                    tagFile.write("audio/street/{}\t{}\t{}\t{}\n".format(nomArxiu, time_tag[0], time_tag[1], classe))   
