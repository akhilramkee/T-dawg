import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
from model import transformer
import h5py

'''
        Max length -> 60 , containing 60 words per input
'''
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model,tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self,step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)
    
    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps,

        }
        return config


class Model:

    def __init__(self):
        self.token_processor = None
        self.model = None

    def tokenize_sentence(self,token_processor, inputs, outputs):
        t_inputs, t_outputs = [],[]

        for (s1,s2) in zip(inputs, outputs):
            s1 = self.START_TOKEN + token_processor.encode(s1) + self.END_TOKEN
            s2 = self.START_TOKEN + token_processor.encode(s2) + self.END_TOKEN

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

    def preprocess_sentence(self,sentence):
        formated_sentence = sentence.lower().strip()
        formated_sentence = re.sub(r"([?.!,])",r" \1 ", formated_sentence)
        formated_sentence = re.sub(r'[" "]', " ", formated_sentence)

        formated_sentence = re.sub(r"[^a-zA-Z?.!,]+"," ", formated_sentence)
        formated_sentence = formated_sentence.strip()

        return formated_sentence

    '''
        Load Conversation
    '''

    def load_conversations(self,path_to_movie_lines, path_to_movie_conversations):
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
                inputs.append(self.preprocess_sentence(id2line[conversation[i]]))
                outputs.append(self.preprocess_sentence(id2line[conversation[i+1]]))
        return inputs, outputs

    def loss_function(self,y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


    def accuracy(self, y_true,y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    def create_model(self):
        '''
            Initializing Hyper-parameters
        '''
        NUM_LAYERS = 2
        D_MODEL = 256
        NUM_HEADS = 8
        UNITS = 512
        DROPOUT = 0.1

        model = transformer.transformer(
            vocab_size = self.VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units = UNITS,
            d_model = D_MODEL,
            num_heads = NUM_HEADS,
            dropout = DROPOUT
        )

        learning_rate = CustomSchedule(D_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])
        
        return model

    def evaluate(self,model, sentence):
        
        sentence = self.preprocess_sentence(sentence)

        sentence = tf.expand_dims(
            self.START_TOKEN + self.token_processor.encode(sentence) + self.END_TOKEN, axis=0)

        output = tf.expand_dims(self.START_TOKEN, 0)

        for i in range(MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)
            
        # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)


    def predict(self, model, sentence):
        prediction = self.evaluate(model, sentence)

        predicted_sentence = self.token_processor.decode(
            [i for i in prediction if i < self.token_processor.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence



    def model_run(self, status = "train", predict_w=""):
        
        path_to_zip = tf.keras.utils.get_file(
        'cornell_movie_dialogs.zip',
        origin=
        'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
        extract=True)

        if self.token_processor == None:
            path_to_dataset = os.path.join(
                os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

            path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
            path_to_movie_conversations = os.path.join(path_to_dataset,
                                                    'movie_conversations.txt')
            
            questions, answers = self.load_conversations(path_to_movie_lines, path_to_movie_conversations)
            
            '''
                Creating a tokenizer which would break the questions and answers
                and create a vocabulary
            '''
            self.token_processor = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions+answers, target_vocab_size=2**13)
            self.START_TOKEN, self.END_TOKEN = [self.token_processor.vocab_size], [self.token_processor.vocab_size+1]
            self.VOCAB_SIZE = self.token_processor.vocab_size+2
            '''
                Tokenizing the Inputs and Outputs using the vocab
            '''
            questions, answers = self.tokenize_sentence(self.token_processor, questions, answers)
        
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
            
        checkpoint_path = "training_1/cp.ckpt"

        if self.model == None:
            self.model = self.create_model()
            self.model.load_weights(checkpoint_path)

        if status=="train":
            checkpoint_dir = os.path.dirname(checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)

            EPOCHS = 10
            self.model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])

            self.predict(self.model, "Who are you")

        else:
            self.predict(self.model,predict_w)
        
