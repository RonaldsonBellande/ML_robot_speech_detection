from header_imports import *

class pitch(object):
    def __init__(self):

        self.
        
    
    def librosa_pitch_shift(self):

        self.path  = "voice_data/"
        if self.data_type == "commands":
            self.folder = "commands/" 
            self.true_path = self.path + self.folder
        elif self.data_type == "utensils":
            self.folder =  "utensils/" 
            self.true_path = self.path + self.folder
        elif self.data_type == "fruits":
            self.folder = "fruits/"
            self.true_path = self.path + self.folder
        elif self.data_type == "objects":
            self.folder = "objects/"
            self.true_path = self.path + self.folder

        self.category_names =  os.listdir(self.true_path)
        
        for label in self.category_names:
            self.wav_files = [self.true_path + label + '/' + i for i in os.listdir(self.true_path + '/' + label)]
            for wavfile in self.wav_files:
                wave, sr = librosa.load(wavfile)
                y_shifted = librosa.effects.pitch_shift(wave, sr, n_steps=4, bins_per_octave=24)
                librosa.output.write_wav(directory, y_shifted, sr=sr, norm=False)
