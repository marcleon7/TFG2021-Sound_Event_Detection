from datetime import datetime
from dataset import *
from model import *
from utils import *
from params import *

if __name__ == "__main__":
    
    print("\n---------------------------------------------\n")
    print("SISTEMA ACÚSTIC\n")
    print("Autor: Marc León Gimeno\n")
    print("Treball de Fi de Grau 2021")
    print("\n---------------------------------------------\n")
    
    # ********** VARIABLES DEL SISTEMA **********

    # Directoris de la base de dades
    features_folder = parameters['features_folder']
    audio_folder = parameters['audio_folder']
    setup_folder = parameters['setup_folder']

    # Configuració del model
    fm = parameters['sample_rate']
    nfft = parameters['nfft']
    win_size = int(parameters['window_length'] * fm)
    hop_size = int(parameters['hop_length'] * fm)
    n_mels = parameters['mels']
    n_mfcc = parameters['mfcc']
    frame_padding = parameters['frame_padding']

    EPOCHS = parameters['epochs']
    lr = parameters['learning_rate']
    BATCH_SIZE = parameters['batch_size']

    # COnfiguració dels processos a executar
    feature_extraction_process = parameters['extract_features']
    train_process = parameters['train_model']
    tagging_process = parameters['tagging']

    audio_for_tagging = parameters['audio_file_for_tagging']
    tagFile = parameters['tags_file_name']

    INPUT = (n_mfcc*3*(frame_padding*2 + 1),)
    OUTPUT = 6

    folds = parameters['folds']

    binary_accuracy = []
    val_binary_accuracy = []
    loss = []
    val_loss = []

    train_metrics = {
        'accuracy_train' : [],
        'loss_train' : [],
        'val_accuracy_train' : [],
        'val_loss_train' : []
    }

    metrics = {
        'accuracy' : [],
        'recall' : [],
        'precision' : [],
        'F1' : [],
        'ER' : [],
        'TP' : [],
        'TN' : [],
        'FP' : [],
        'FN' : []
    }

    # S'indica el temps d'inici de l'execució
    timeStart = datetime.now()

    # ********** Inici de la execució del programa principal **********

    print("\nINICIANT EXECUCIÓ\n")

    if feature_extraction_process:        
        print("\n---------------------------------------------\n")
        print("Pre-processant els arxius d'àudio")
        print("\n---------------------------------------------\n")
        
        # Funció per extreure les característiques dels arxius de la base de dades
        extract_features(audio_folder=audio_folder, features_folder=features_folder, fm=fm, nfft=nfft, window_length=win_size, hop_size=hop_size, n_mels=n_mels, n_mfcc=n_mfcc, frame_padding=frame_padding)

    if train_process:
        print("\n---------------------------------------------\n")
        print("Iniciant entrenament")
        print("\n---------------------------------------------\n")

        
        # Es construeix el model
        model = build_model(input=INPUT, output=OUTPUT, lr=lr, show=False)

        for fold in folds:
            
            print("\n---------------------------------------------\n")
            print("Executant l'entrenament amb el Set d'Entrenament: {}".format(fold))
            print("\n---------------------------------------------\n")

            # Es recuperen les dades d'entrenament i de test
            X_train, X_test, Y_train, Y_test = load_features_data(features_folder=features_folder, fold=fold)

            # S'entrena el model
            history = train_model(model=model, X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE)

            # Puntuacions per fer la mitja de l'entrenament dels 4 models
            binary_accuracy.append(history.history['binary_accuracy'][-1])
            val_binary_accuracy.append(history.history['val_binary_accuracy'][-1])
            loss.append(history.history['loss'][-1])
            val_loss.append(history.history['val_loss'][-1])
            
            # Llistes de valores de precisió i de funció de cost al llarg de l'entrenament
            train_metrics['accuracy_train'].append(history.history['binary_accuracy'])
            train_metrics['val_accuracy_train'].append(history.history['val_binary_accuracy'])
            train_metrics['loss_train'].append(history.history['loss'])
            train_metrics['val_loss_train'].append(history.history['val_loss'])
            
            # Es guarda el model generat
            save_model(model=model, fold=fold)

            print("\n---------------------------------------------\n")
            print("Avaluant l'entrenament amb el set de test: {}".format(fold))
            print("\n---------------------------------------------\n")

            # S'avalua el mdoel
            accuracy, recall, precision, F1, ER, TP, TN, FP, FN = evaluate_model(model=model, X_test=X_test, Y_test=Y_test)

            # Puntuacions obtingudes pel model
            metrics['accuracy'].append(accuracy)
            metrics['recall'].append(recall)
            metrics['precision'].append(precision)
            metrics['F1'].append(F1)
            metrics['ER'].append(ER)
            metrics['TP'].append(TP)
            metrics['TN'].append(TN)
            metrics['FP'].append(FP)
            metrics['FN'].append(FN)
            
            print("\n---------------------------------------------\n")
            print("Generant les etiquetes del set de test: {}".format(fold))
            print("\n---------------------------------------------\n")

            tagging_test_files(fold=fold, audio_folder=audio_folder, setup_folder=setup_folder, nfft=nfft, window_length=win_size, hop_size=hop_size, n_mels=n_mels, n_mfcc=n_mfcc, frame_padding=frame_padding)

        # Gràfiques del model
        save_results(metrics=train_metrics)

        binary_accuracy_mean = np.round(np.mean(binary_accuracy)*100, 2)
        error = np.round(100 - binary_accuracy_mean, 2)
        val_binary_accuracy_mean = np.round(np.mean(val_binary_accuracy)*100, 2)
        val_error = np.round(100 - val_binary_accuracy_mean, 2)

        accuracy_mean = np.round(np.mean(metrics['accuracy'])*100, 2)
        recall_mean = np.round(np.mean(metrics['recall'])*100, 2)
        precision_mean = np.round(np.mean(metrics['precision'])*100, 2)
        f1_mean = np.round(np.mean(metrics['F1'])*100, 2)
        ER_mean = np.round(np.mean(metrics['ER']), 2)
        TP_mean = np.round(np.mean(metrics['TP']))
        TN_mean = np.round(np.mean(metrics['TN']))
        FP_mean = np.round(np.mean(metrics['FP']))
        FN_mean = np.round(np.mean(metrics['FN']))

    if tagging_process:
        print("\n---------------------------------------------\n")
        print("Generant prediccions")
        print("\n---------------------------------------------\n")

        # Funció per generar les etiquetes
        predict(file=audio_for_tagging, tagFile=tagFile, fm=fm, nfft=nfft, window_length=win_size, hop_size=hop_size, n_mels=n_mels, n_mfcc=n_mfcc, frame_padding=frame_padding)
    

    # S'indica el temps de fi de l'execució
    timeEnd = datetime.now()

    # S'indica el missatge de fi de programa segons l'estat del programa final
    print("\nEXECUCIÓ FINALIZADA\n")

    # ********** Generar arxiu LOG.txt **********

    with open('LOG.txt', 'w') as logFile:
        if train_process:
            logFile.write("\n-------------------------------------\n")
            logFile.write("Puntuacions del model generat")
            logFile.write("\n-------------------------------------\n\n")
            logFile.write("Binary Accuracy: {}%\n".format(binary_accuracy_mean))
            logFile.write("Error: {}%\n".format(error))
            logFile.write("\nBinary Accuracy (Dades de validació): {}%\n".format(val_binary_accuracy_mean))
            logFile.write("Error (Dades de validació): {}%\n".format(val_error))
        
            logFile.write("\n-------------------------------------\n")
            logFile.write("Avaluació del model")
            logFile.write("\n-------------------------------------\n\n")
            logFile.write("Accuracy: {}%\n".format(accuracy_mean))
            logFile.write("Recall: {}%\n".format(recall_mean))
            logFile.write("Precisió: {}%\n".format(precision_mean))
            logFile.write("F-Score: {}%\n".format(f1_mean))
            logFile.write("ER: {}\n".format(ER_mean))
            logFile.write("TP: {}\n".format(TP_mean))
            logFile.write("TN: {}\n".format(TN_mean))
            logFile.write("FP: {}\n".format(FP_mean))
            logFile.write("FN: {}\n".format(FN_mean))

        
        logFile.write("\n-------------------------------------\n")
        logFile.write("TEMPS D'EXECUCIÓ")
        logFile.write("\n-------------------------------------\n\n")
        logFile.write("Execució iniciada el {} a les {}\n\n".format(timeStart.date().strftime("%d/%m/%Y"), timeStart.time().strftime("%H:%M:%S")))
        logFile.write("Execució finalizada el {} a les {}\n\n".format(timeEnd.date().strftime("%d/%m/%Y"), timeEnd.time().strftime("%H:%M:%S")))
        logFile.write("Temps d'execució: {}\n\n".format((timeEnd - timeStart)))
        logFile.write("-------------------------------------\n\n")

    print("\nS'ha generat el fitxer LOG.txt amb els resultats de l'execució.\n")