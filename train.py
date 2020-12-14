from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from keras.utils import np_utils
from hyperas.distributions import choice, uniform
from hyperas import optim

def data():
    """
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    import os
    import config
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from keras import backend as K
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Add
    from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Input
    
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 'PADword')
        vector[-pad_width[1]:] = pad_value
        return vector
    
    path = config.SPACY_PATH
    n = config.TRAIN_SIZE
    
    files = os.listdir(path)[:n]
    data = pd.DataFrame({'Document #':[],'Word':[], 'Tag':[]})

    for docnum in range(1,len(files)+1):
        file = files[docnum-1]
        f = open(path+file)
        temp = f.readlines()
        d = eval(temp[0])[1]
        temp = eval(temp[0])[0]

        lst = [[i,j] for i,j,_ in list(d.values())[0]]
        lst = [item for sublist in lst for item in sublist]
        lst = [0]+lst+[len(temp)]

        isOdd = 1
        words = []
        tags = []
        for v, w in zip(lst[:-1],lst[1:]):
            grp = temp[v:w].split(' ')
            if isOdd%2==0:
                words.extend(grp)
                tags.extend([1]*len(grp))
            else:
                words.extend(grp)
                tags.extend([0]*len(grp))
            isOdd+=1

        data = data.append(pd.DataFrame({'Document #':[docnum]*len(words),'Word':words, 'Tag':tags}), ignore_index=True)
        data['Document #'] = data['Document #'].astype('int')
        data.dropna(inplace=True)

    max_len = max(data.groupby('Document #').count()['Word'])

    sentences_list = data.groupby('Document #')['Word'].apply(list)
    X_new = [np.pad(x, (0,max_len-len(x)), pad_with).tolist() if len(x)<max_len else x[:max_len] for x in sentences_list.values]

    y = data.groupby('Document #')['Tag'].apply(list)
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)

    print("Splitting data into train and test")
    test_size = config.VALIDATION_SIZE
    X_tr = X_new[:-test_size]
    X_val = X_new[-test_size:]
    y_tr = y[:-test_size]
    y_val = y[-test_size:]
    y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
    print("The train, validation and test subsets are ready")

    print("Original Training Addresses")
    for n in range(5):
        print([i for i,j in zip(X_tr[n],y_tr[n]) if j==1])
    
    print("Number of training samples:",len(X_tr))
    print("Number of testing samples:",len(X_val))
    
    return X_tr, X_val, y_tr, y_val, max_len

def model(X_tr, X_val, y_tr, y_val, max_len):
    """
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def ElmoEmbedding(x):
        return elmo_model(inputs={
                                "tokens": tf.squeeze(tf.cast(x, tf.string)),
                                "sequence_len": tf.constant(batch_size*[max_len])
                          })["elmo"]

    def scores(actual,predicted):
        tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1,0]).reshape(-1)
        if (tp+fp)==0:
            precision = 0.0
        else:
            precision = round(tp/(tp+fp), 2)
        recall = round(tp/(tp+fn), 2)
        return [precision, recall]
    
    def evaluate(X_te,y_te,batch_size):
        lst = []
        for n in range(len(X_te)//batch_size):
            test_pred = model.predict(np.array(X_te[n*batch_size:batch_size+n*batch_size]))
            for i in range(len(test_pred)):
                p = np.argmax(test_pred[i], axis=-1)
                score = scores(y_te[i+(n*batch_size)], p)
                lst.append(score)
        return [round(sum(i)/len(lst),2) for i in zip(*lst)]

    batch_size = config.TRAIN_BATCH_SIZE
    elmo_model = hub.KerasLayer(config.MODEL_NAME , signature="tokens", 
                                trainable=False, signature_outputs_as_dict=True)

    print("Defining model architecture..")
    units = {{choice([64,128,256,512])}}

    input_text = Input(shape=(max_len,), dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
    x = Bidirectional(LSTM(units=units, return_sequences=True,
                           recurrent_dropout=0.2, dropout={{uniform(0, .5)}}))(embedding)
    x_rnn = Bidirectional(LSTM(units=units, return_sequences=True,
                               recurrent_dropout=0.2, dropout={{uniform(0, .5)}}))(x)
    x = Add()([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(2, activation="softmax"))(x)

    model = Model(input_text, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[f1_m,precision_m, recall_m])

    print("Starting model training..")

    history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                        batch_size=batch_size, epochs=config.EPOCHS, verbose=1)
    
    #train_rmse = model.evaluate(X_train, y_train, verbose=0)[1]
    precision = evaluate(X_val,y_val,batch_size)[0]

    return {'loss': -precision, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    
    import os
    import numpy as np
    import pickle
    import config
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import gc; gc.collect()

    import time
    stime = time.time()
    
    print("Starting hyperparameter tuning")

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=Trials(),
                                          notebook_name='__notebook_source__')
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Total time taken:",time.time()-stime)
    
    filename = config.BEST_MODEL
    best_model.save_weights('ner_model.h5')

    with open(config.BEST_PARAMS, 'wb') as fp:
        pickle.dump(best_run, fp, protocol=pickle.HIGHEST_PROTOCOL)