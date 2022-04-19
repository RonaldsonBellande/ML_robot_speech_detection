from header_imports import *

class speech_building(models):
    def __init__(self, model_type):

        self.label_name = []
        self.mfcc_vectors = [] 
        self.path  = "voice_data/"
        self.true_path = self.path
        self.channel = 1
        self.number_mfcc = 22050 
        self.mfcc_vectors = np.empty([320, 120, 100]) 
        
        self.labelencoder = LabelEncoder()
        self.valid_sound = [".wav"]
        self.model = None

        self.model_summary = "model_summary/"
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.create_model_type = model_type
        
        self.data_to_array_label_sound()
        self.splitting_data_normalize()

        if self.create_model_type == "model1":
            self.model = self.create_models_1()
        elif self.create_model_type == "model2":
            self.model = self.create_models_2()
        elif self.create_model_type == "model3":
            self.model = self.create_model_3()
        elif self.create_model_type == "model4":
            self.model = self.create_models_4()

        self.save_model_summary()


    def data_to_array_label_sound(self):
        
        self.category_names =  os.listdir(self.true_path)
        folder = next(os.walk(self.true_path))[1]
        self.number_classes = len(folder)
        
        for label in self.category_names:
            self.wav_files = [self.path + label + '/' + i for i in os.listdir(self.path + '/' + label)]

            for wavfile in self.wav_files:
                wave, sr = librosa.load(wavfile)
                mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=self.number_mfcc)
                np.append(self.mfcc_vectors, mfcc)
                self.label_name.append(label)


        if self.create_model_type == "model4":
            self.mfcc_vectors =  self.mfcc_vectors.reshape(self.mfcc_vectors.shape[0], self.mfcc_vectors.shape[1], self.mfcc_vectors.shape[2])
        else:
            self.mfcc_vectors =  self.mfcc_vectors.reshape(self.mfcc_vectors.shape[0], self.mfcc_vectors.shape[1], self.mfcc_vectors.shape[2], self.channel)

        self.label_name = self.labelencoder.fit_transform(self.label_name)
        self.label_name = np.array(self.label_name)
        self.label_name = tf.keras.utils.to_categorical(self.label_name , num_classes=self.number_classes)



    def splitting_data_normalize(self):
        
        self.X_train, self.X_test, self.Y_train_vec, self.Y_test_vec = train_test_split(self.mfcc_vectors, self.label_name, test_size = 0.1, random_state=42)
        self.input_shape = self.X_train.shape[1:]
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train_vec, self.number_classes)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test_vec, self.number_classes)
    


    def save_model_summary(self):
        with open(self.model_summary + self.create_model_type +"_summary_architecture_" + str(self.number_classes) +".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()


    



    
