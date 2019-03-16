from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, model_from_json
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from tensorflow.python.lib.io import file_io
import sys
import io

def run(dir_name, job_dir, gcp):
    # Read a file and return a string
    def load_doc(filename):
        with file_io.FileIO(filename, mode='r') as file:
            text = file_io.FileIO.read(file)
            return text

    def load_data(data_dir):
        text = []
        images = []
        # Load all the files and order them
        # all_filenames = listdir(data_dir)
        print("DIR: "+ data_dir)
        all_filenames = file_io.list_directory(data_dir)
        all_filenames.sort()
        for filename in (all_filenames):
            print(filename) #Check if images are in correct order
            if filename[-3:] == "npz":
                # Load the images already prepared in arrays
                # with file_io.FileIO(data_dir+filename, mode='r') as file:
                test = io.BytesIO(file_io.read_file_to_string(data_dir+filename, binary_mode=True))
                image = np.load(test)
                images.append(image['features'])
            else:
                # Load the boostrap tokens and rap them in a start and end tag
                syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
                # Seperate all the words with a single space
                syntax = ' '.join(syntax.split())
                # Add a space after each comma
                syntax = syntax.replace(',', ' ,')
                text.append(syntax)
        images = np.array(images, dtype=float)
        return images, text

    train_features, texts = load_data(dir_name)

    # Initialize the function to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    if gcp:
        tokenizer.fit_on_texts([load_doc('gs://capita/Screenshot-to-code/Bootstrap/resources/bootstrap.vocab')])
    else:
        tokenizer.fit_on_texts([load_doc('resources/bootstrap.vocab')])

    # Add one spot for the empty word in the vocabulary
    vocab_size = len(tokenizer.word_index) + 1
    # Map the input sentences into the vocabulary indexes
    train_sequences = tokenizer.texts_to_sequences(texts)
    # The longest set of boostrap tokens
    max_sequence = max(len(s) for s in train_sequences)
    # Specify how many tokens to have in each input sentence
    max_length = 48

    def preprocess_data(sequences, features):
        X, y, image_data = list(), list(), list()
        for img_no, seq in enumerate(sequences):
            for i in range(1, len(seq)):
                # Add the sentence until the current count(i) and add the current count to the output
                in_seq, out_seq = seq[:i], seq[i]
                # Pad all the input token sentences to max_sequence
                in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
                # Turn the output into one-hot encoding
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # Add the corresponding image to the boostrap token file
                image_data.append(features[img_no])
                # Cap the input sentence to 48 tokens and add it
                X.append(in_seq[-48:])
                y.append(out_seq)
        return np.array(X), np.array(y), np.array(image_data)

    X, y, image_data = preprocess_data(train_sequences, train_features)

    #Create the encoder
    image_model = Sequential()
    image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
    image_model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    image_model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    image_model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
    image_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

    image_model.add(Flatten())
    image_model.add(Dense(1024, activation='relu'))
    image_model.add(Dropout(0.3))
    image_model.add(Dense(1024, activation='relu'))
    image_model.add(Dropout(0.3))

    image_model.add(RepeatVector(max_length))

    visual_input = Input(shape=(256, 256, 3,))
    encoded_image = image_model(visual_input)

    language_input = Input(shape=(max_length,))
    language_model = Embedding(vocab_size, 50, input_length=max_length, mask_zero=True)(language_input)
    language_model = LSTM(128, return_sequences=True)(language_model)
    language_model = LSTM(128, return_sequences=True)(language_model)

    #Create the decoder
    decoder = concatenate([encoded_image, language_model])
    decoder = LSTM(512, return_sequences=True)(decoder)
    decoder = LSTM(512, return_sequences=False)(decoder)
    decoder = Dense(vocab_size, activation='softmax')(decoder)

    # Compile the model
    model = Model(inputs=[visual_input, language_input], outputs=decoder)
    optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #Save the model for every 2nd epoch
    # filepath="gs://capita/Screenshot-to-code/models/org-weights-epoch-{epoch:04d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=2)
    # callbacks_list = [checkpoint]

    # model.fit([image_data, X], y, batch_size=1, shuffle=False, validation_split=0.1, callbacks=callbacks_list, verbose=1, epochs=20)
    model.fit([image_data, X], y, batch_size=1, shuffle=False, validation_split=0.1, verbose=1, epochs=20)

    # Save model.h5 on to google storage
    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 1:
        dir_name = 'resources/data/'
        job_dir = "/"
        gcp = False
    else:
        print("ARGS: ", argv)
        dir_name = argv[0]
        if argv[1] == '--job-dir':
            job_dir = argv[2]
        gcp = True

    print("passed")
    run(dir_name, job_dir, gcp)
