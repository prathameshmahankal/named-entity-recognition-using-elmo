import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import config
import pickle
import tensorflow as tf
from keras import backend as K
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Input
from sklearn.metrics import confusion_matrix

def scores(actual,predicted):
    tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1,0]).reshape(-1)
    if (tp+fp)==0:
        precision = 0.0
    else:
        precision = round(tp/(tp+fp), 2)
    recall = round(tp/(tp+fn), 2)
    return [precision, recall]

def evaluate(X_te,y_te,batch_size,model):
    lst = []
    for n in range(len(X_te)//batch_size):
        test_pred = model.predict(np.array(X_te[n*batch_size:batch_size+n*batch_size]))
        for i in range(len(test_pred)):
            p = np.argmax(test_pred[i], axis=-1)
            score = scores(y_te[i+(n*batch_size)], p)
            lst.append(score)
    return [round(sum(i)/len(lst),2) for i in zip(*lst)]

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
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 'PADword')
        vector[-pad_width[1]:] = pad_value
        return vector
    
    path = config.SPACY_PATH
    n = config.TEST_SIZE
    
    files = os.listdir(path)[-n:]
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

    print("Number of testing samples:",len(X_new))
    
    return X_new, y, max_len

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

batch_size = config.TEST_BATCH_SIZE
elmo_model = hub.KerasLayer(config.MODEL_NAME , signature="tokens", 
                                trainable=False, signature_outputs_as_dict=True)

X_val, y_val, max_len = data()

with open(config.BEST_PARAMS, 'rb') as fp:
    best_params = pickle.load(fp)

units = [64,128,256,512][best_params.get('units')]

input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
x = Bidirectional(LSTM(units=units, return_sequences=True,
                       recurrent_dropout=0.2, dropout=best_params.get('dropout') ))(embedding)
x_rnn = Bidirectional(LSTM(units=units, return_sequences=True,
                           recurrent_dropout=0.2, dropout=best_params.get('dropout_1') ))(x)
x = Add()([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(2, activation="softmax"))(x)

loaded_model = Model(input_text, out)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

loaded_model.load_weights(config.BEST_MODEL)

print("Length of X_val:",len(X_val))
test_pred = loaded_model.predict(np.array(X_val))
print(test_pred)
# for n in range(len(X_val)//batch_size):
#     # print(np.array(X_val[n*batch_size:batch_size+n*batch_size]))
#     test_pred = loaded_model.predict(np.array(X_val[n*batch_size:batch_size+n*batch_size]))
#     for i in range(len(test_pred)):
#         p = np.argmax(test_pred[i], axis=-1)
#         print("Predicted Address:")
#         print([i for i,j in zip(X_val[i+(n*batch_size)],p) if j==1])

# if config.SHOW_ONLY_SCORE:
#     evaluate(X_val, y_val, batch_size, loaded_model)
# else:
#     lst = []

#     for n in range(len(X_val)//batch_size):

#         test_pred = loaded_model.predict(np.array(X_val[n*batch_size:batch_size+n*batch_size]))

#         for i in range(len(test_pred)):
#             p = np.argmax(test_pred[i], axis=-1)
#             print("Original Address:")
#             print([i for i,j in zip(X_val[i+(n*batch_size)],y_val[i+(n*batch_size)]) if j==1])
#             print("Predicted Address:")
#             print([i for i,j in zip(X_val[i+(n*batch_size)],p) if j==1])
#             score = scores(y_val[i+(n*batch_size)], p)
#             lst.append(score)
#             print("Precision:{} Recall:{}".format(*score))
#             print(" ")

#     print("*****************************************")
#     print("Final values - Precision:{} Recall:{}".format(*[round(sum(i)/len(lst),2) for i in zip(*lst)]))
#     print("*****************************************")