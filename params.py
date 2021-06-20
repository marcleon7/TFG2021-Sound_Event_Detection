
parameters = {

    # Directoris necessaris per tractar amb la base de dades
    'features_folder' : 'dataset/TUT-sound-events-2017-development/pre-processed/',
    'audio_folder' : 'dataset/TUT-sound-events-2017-development/audio/street/',
    'setup_folder' : 'dataset/TUT-sound-events-2017-development/evaluation_setup/',

    # Divisions de la base de dades per la validació creuada
    'folds' : [1, 2, 3, 4],

    'sample_rate': 44100, # Freqüència de mostreig (Hz)
    'nfft' : 2048,# Nombre de punts de la transformada
    'window_length' : 0.04, # Tamany de la finestra (s)
    'hop_length' : 0.02, # Mida del solapament entre finestres (s)
    'mels' : 40, # Nombre de bandes freqüencials de mel
    'mfcc' : 20, # Nombre de coeficients a calcular
    'frame_padding' : 1, # Nombre de trames a adjuntar

    'epochs' : 50, # Nombre d'iteracions en l'entrenament
    'learning_rate' : 0.001, # Rang 'aprenentatge
    'batch_size' : 256, # Mida del lot
 
    'extract_features' : True, # Procés d'extracció de característiques (True o False)
    'train_model' : True, # Procés d'entrenament (True o False)
    'tagging' : True, # Procés d'etiquetatge (True o False)

    # Directori del senyal a etiquetar
    'audio_file_for_tagging' : 'dataset/TUT-sound-events-2017-development/audio/street/b093.wav',
    'tags_file_name' : 'b093.txt' # Nom del fitxer d'etiquetes a realitzar
}