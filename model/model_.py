import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np

'''
        Max length -> 60 , containing 60 words per input
'''
MAX_LENGTH = 60
BATCH_SIZE = 64
BUFFER_SIZE = 20000

def tokenize_sentence(token_processor, inputs, outputs):
    t_inputs, t_outputs = [],[]

    for (s1,s2) in zip(inputs, outputs):
        s1 = START_TOKEN + token_processor.encode(s1) + END_TOKEN
        s2 = START_TOKEN + token_processor.encode(s2) + END_TOKEN

        if len(s1) <= MAX_LENGTH and len(s2) <= MAX_LENGTH:
            t_inputs.append(s1)
            t_outputs.append(s2)
    
    t_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        t_inputs, maxlen=MAX_LENGTH, padding='post'
    )
    t_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        t_outputs, maxlen=MAX_LENGTH, padding='post'
    )

    return t_inputs, t_outputs

def preprocess_sentence(sentence):
    formated_sentence = sentence.lower().strip()
    formated_sentence = re.sub(r"([?.!,])",r" \1 ", formated_sentence)
    formated_sentence = re.sub(r'[" "]', " ", formated_sentence)

    formated_sentence = re.sub(r"[^a-zA-Z?.!,]+"," ", formated_sentence)
    formated_sentence = formated_sentence.strip()

    return formated_sentence

'''
    Load Conversation
'''

def load_conversations(path_to_movie_lines, path_to_movie_conversations):
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]
    
    inputs, outputs = [],[]
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation)-1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i+1]]))
    return inputs, outputs

def run():
    global START_TOKEN, END_TOKEN
    path_to_zip = tf.keras.utils.get_file(
    'cornell_movie_dialogs.zip',
    origin=
    'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

    path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
    path_to_movie_conversations = os.path.join(path_to_dataset,
                                            'movie_conversations.txt')
    
    questions, answers = load_conversations(path_to_movie_lines, path_to_movie_conversations)
    
    '''
        Creating a tokenizer which would break the questions and answers
        and create a vocabulary
    '''
    token_processor = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions+answers, target_vocab_size=2**14)
    START_TOKEN, END_TOKEN = [token_processor.vocab_size], [token_processor.vocab_size+1]
    VOCAB_SIZE = token_processor.vocab_size+2

    '''
        Tokenizing the Inputs and Outputs using the vocab
    '''
    questions, answers = tokenize_sentence(token_processor, questions, answers)
    
    '''
        Dataset prefetching and stuff
    '''
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:,:-1]
        },
        {
            'outputs': answers[:,1:]
        }
    )).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
