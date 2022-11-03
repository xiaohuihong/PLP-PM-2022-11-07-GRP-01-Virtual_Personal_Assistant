import numpy as np
import pickle
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


class classification_model():
    def __init__(self):
        with open(dir_path + '/model/one_hot_rnn/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(dir_path + '/model/one_hot_rnn/classes.npy', allow_pickle=True)
        # load model
        self.model = load_model(dir_path + '/model/one_hot_rnn/one_hot_rnn.h5')
        # summarize model.
        # model.summary()

    def predict(self, context_list):
        sequences = self.tokenizer.texts_to_sequences(context_list)
        sample_train_padseq = pad_sequences(sequences, maxlen=100)
        prediction = self.model.predict(np.stack(sample_train_padseq))
        output = np.argmax(prediction, axis=1)
        cate_list = self.encoder.inverse_transform(output)
        return cate_list

if __name__ == '__main__':
    context_list = []
    test_text1 = "Biden won the election."
    context_list.append(test_text1)
    model = classification_model()
    cate_list = model.predict(context_list)
    print('Output:\n', cate_list)
