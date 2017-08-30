"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import helper
import numpy as np
import problem_unittests as tests
from distutils.version import LooseVersion
import warnings
import tensorflow as tf



data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
print(text[:50])


view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""


print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

import numpy as np
import problem_unittests as tests
from collections import Counter


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    words_list = list(set(word for word in text))
    word_counts = Counter(words_list)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {i: word for i, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: i for i, word in enumerate(sorted_vocab)}

    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """

    tokens = {}
    tokens['.'] = '||Period||'
    tokens[','] = '||Comma||'
    tokens['\"'] = '||QuotaionMark||'
    tokens[';'] = '||Semicolon||'
    tokens['!'] = '||ExclamationMark||'
    tokens['?'] = '||QuestionMark||'
    tokens['('] = '||LeftParentheses||'
    tokens[')'] = '||RightParentheses||'
    tokens['--'] = '||Dash||'
    tokens['\n'] = '||Return||'

    # TODO: Implement Function
    return tokens


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""


int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))




def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(dtype=tf.int32,shape=(None,None),name="input")
    targets = tf.placeholder(dtype=tf.int32,shape=(None,None),name="targets")
    learning_rate = tf.placeholder(dtype=tf.float32,name="learning_rate")
    return input, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import BasicLSTMCell

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm_cell = BasicLSTMCell(rnn_size)
    multi_rnn_cell = MultiRNNCell([lstm_cell])



    init_state = multi_rnn_cell.zero_state(batch_size,tf.float32)
    init_state = tf.identity(init_state,"initial_state")
    return multi_rnn_cell, init_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)



def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    #download embedding vector from vol?
    #apply data extend

    #embedding = np.zeros((input_data.shape[0],input_data.shape[1],embed_dim),dtype=np.float32)

    input_data_shape = input_data.get_shape().as_list()
    embedding = np.zeros((input_data.shape[0],input_data.shape[1],embed_dim),dtype=np.float32)


    input_array = tf.get_default_session().run(input_data)

    tf.Tensor.get_shape().as_list()




    fname = 'data/glove.6B.%dd.txt' % embed_dim
    glove_index_dict = {}
    with open(fname, 'r') as fp:
        glove_symbols = len(fp.readlines())

    glove_embedding_weights = np.empty((glove_symbols, embed_dim))

    with open(fname, 'r') as fp:
        i = 0
        for ls in fp:
            ls = ls.strip().split()
            w = ls[0]
            glove_index_dict[w] = i
            glove_embedding_weights[i, :] = np.asarray(ls[1:], dtype=np.float32)
            i += 1



    j=0
    for i in range(vocab_size):
        vec = glove_embedding_weights[i]
        embedding[j,:]= vec
        j+=1
    return embedding

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    final_state = tf.identity(state, "final_state")

    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function'
    embedding = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embedding)

    logits = tf.layers.dense(outputs, vocab_size)

    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function

    big_serial_size = batch_size * seq_length
    big_serial_count = len(int_text) // big_serial_size
    int_text_input = int_text[0:big_serial_count * big_serial_size]
    int_text_target = int_text[1:big_serial_count * big_serial_size + 1]
    int_text_input = np.asarray(int_text_input, np.int32)
    int_text_target = np.asarray(int_text_target, np.int32)
    serial_data = np.empty((big_serial_count, 2, batch_size, seq_length))

    index = 0
    for x in range(0, len(int_text_input), big_serial_size):
        inner_input = np.reshape(int_text_input[x:x + big_serial_size], (batch_size, seq_length))
        inner_target = np.reshape(int_text_target[x:x + big_serial_size], (batch_size, seq_length))
        serial_data[index, :, :, :] = [inner_input, inner_target]
    return serial_data



tests.test_get_batches(get_batches)

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 32
# RNN Size
rnn_size = 128
# Embedding Dimension Size
embed_dim = 200
# Sequence Length
seq_length = 100
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = True

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    return None, None, None, None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    return None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)

gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)