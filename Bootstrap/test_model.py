from os import listdir
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from compiler.classes.Compiler import *
from keras.models import Model, Sequential
from keras.layers.core import Dropout, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers import Embedding, RepeatVector, LSTM, concatenate , Input, Dense, GRU

def load_model(model_name, vocab_size, max_length):
    if model_name == 'LSTM':
        # Create the encoder
        image_model = Sequential()
        image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
        image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

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

        # Create the decoder
        decoder = concatenate([encoded_image, language_model])
        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(vocab_size, activation='softmax')(decoder)

        # Compile the model
        model = Model(inputs=[visual_input, language_input], outputs=decoder)
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    elif model_name == 'GRU':
        image_model = Sequential()
        image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
        image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

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
        language_model = GRU(128, return_sequences=True)(language_model)
        language_model = GRU(128, return_sequences=True)(language_model)

        # Create the decoder
        decoder = concatenate([encoded_image, language_model])
        decoder = GRU(512, return_sequences=True)(decoder)
        decoder = GRU(512, return_sequences=False)(decoder)
        decoder = Dense(vocab_size, activation='softmax')(decoder)

        # Compile the model
        model = Model(inputs=[visual_input, language_input], outputs=decoder)
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    else:
        return None

# Read a file and return a string
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_data(data_dir):
    text = []
    images = []
    # Load all the files and order them
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames)[-2:]:
        if filename[-3:] == "npz":
            # Load the images already prepared in arrays
            image = np.load(data_dir+filename)
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

# Initialize the function to create the vocabulary
print("Creating vocabulary")
tokenizer = Tokenizer(filters='', split=" ", lower=False)
# Create the vocabulary
tokenizer.fit_on_texts([load_doc('resources/bootstrap.vocab')])
vocab_size = len(tokenizer.word_index) + 1
max_length = 48

print("Loading test data")
dir_name = 'resources/custom_test/'
train_features, texts = load_data(dir_name)

print("Loading model")
#load model and weights
# json_file = open('models/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
loaded_model = load_model('LSTM', vocab_size, max_length)
print(loaded_model.summary())
# load weights into new model
print("Loading weights")
loaded_model.load_weights("models/LSTM_model.h5")
print("Loaded model from disk")

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
print(word_for_id(17, tokenizer))

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = loaded_model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        print(word + ' ', end='')
        if word == '<END>':
            break
    return in_text


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for i in range(len(texts)):
        yhat = generate_desc(model, tokenizer, photos[i], max_length)
        # store actual and predicted
        print('\n\nReal---->\n\n' + texts[i])
        actual.append([texts[i].split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    print("XXXXX: ", bleu)
    return bleu, actual, predicted

print("Evaluating model")
bleu, actual, predicted = evaluate_model(loaded_model, texts, train_features, tokenizer, max_length)

print("Compiling the first prediction to HTML/CSS")
#Compile the tokens into HTML and css
dsl_path = "compiler/assets/web-dsl-mapping.json"
compiler = Compiler(dsl_path)
compiled_website = compiler.compile(predicted[0], 'index.html')

print(compiled_website )

print(bleu)