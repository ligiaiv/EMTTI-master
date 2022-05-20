
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Flatten,Input, Dropout, LSTM
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.utils import plot_model
def LSTM_model(config,embedding_matrix):
    MAXLEN = config["maxlen"]
    EMBEDDING_DIM = config["emb_dim"]
    LSTM_DIM = config["hidden_dim"]
    EMBEDDING_DIM = config["emb_dim"]
    OUTPUT_SIZE = config["num_labels"]


    inputs_ = Input(shape = MAXLEN,name="input_layer")
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAXLEN,
                                trainable=True,
                                name="embedding_layer")(inputs_)
    lstm_output= LSTM(LSTM_DIM,name="encoder_lstm")(embedding_layer)
    outputs_ = Dense(OUTPUT_SIZE,activation='softmax',name = "dense_layer")(lstm_output)
    model = Model(inputs_, outputs_)

    # plot_model(model, to_file='model1.png', show_shapes=True)

    return model
