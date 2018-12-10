import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    network_input, network_output = prepare_sequences(notes)

    model = create_network(network_input)

    train(model, network_input, network_output)
    
    
def merge_notes(notes_dict, max_offset):
            
    ret = np.array([])
    for i in np.arange(0, max_offset, 0.5):
        pitches = np.zeros(88)
        if i in notes_dict:            
            for element in notes_dict[i]:            
                if isinstance(element, note.Note):
                    pitches[element.pitch.midi-21] = 1
                else:
                    for p in element.pitches:
                        pitches[p.midi-21] = 1    
        ret = np.append(ret, pitches)
    return ret
    

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = np.array([])

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        notes_dict = {}
        max_offset = 0
        for element in notes_to_parse:        
            if isinstance(element, note.Note) or isinstance(element, chord.Chord):  
                if element.offset not in notes_dict:
                    notes_dict[element.offset] = []
                notes_dict[element.offset].append(element)
                max_offset = element.offset    
        ret = merge_notes(notes_dict, max_offset)
        #print(len(ret))
        notes = np.append(notes, ret)
        #print(len(notes))
    row = notes.size / 88
    notes = notes.reshape(int(row), 88)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    return notes

def prepare_sequences(notes):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    n_patterns = len(network_input)
    #print(n_patterns)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 88))
    network_output = np.reshape(network_output, (n_patterns, 88))
    return (network_input, network_output)

def create_network(network_input):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(88))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    #model.add(Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    #print(network_input.shape)
    #print(network_output.shape)
    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)
    
if __name__ == '__main__':
    train_network()
