#we use music21 toolkit from MIT to extract information from midi files and then use them to train the network
#import libraries
import glob
import pickle
import numpy
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    notes = get_notes()
    n_vocab = len(set(notes)) #size of the vocab list
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)

def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        try: 
            s2 = instrument.partitionByInstrument(midi) #if instruments involve
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note): #extract notes
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord): #extract chords
                notes.append('.'.join(str(n) for n in element.normalOrder))
    #save the notes into a file
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 100 #use 100 notes to train and predict 101st
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames)) #transform notes to integers
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
        
    n_patterns = len(network_input)
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1)) #reshape
    network_input = network_input / float(n_vocab) #normalization
    network_output = np_utils.to_categorical(network_output) #converts a class vector to binary class matrix
    return (network_input, network_output) 

def create_network(network_input, n_vocab):
    #a 4-LSTM-layer RNN model
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam') #change the optimizer to 'adam' 
    #model.summary()
    return model

def train(model, network_input, network_output):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint] #might use 'early stopping' as well

    #model.fit(network_input, network_output, epochs=50, batch_size=256, callbacks=callbacks_list)
    #change num of epoch and batch size to speed up
    model_fit = model.fit(network_input, network_output, epochs=100, batch_size=256, callbacks=callbacks_list)
    #monitor loss with plot
    plt.plot(model_fit.history["loss"])
    plt.title("loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

if __name__ == '__main__':
    train_network()




