import pickle
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras import initializers

from glove import load_glove_embedding
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


TOKENIZER_STATE_PATH = 'saved_models/tokenizer.p'
GLOVE_EMBEDDING_PATH = 'saved_models/glove.6B.100d.txt'


class Metrics(Callback):
    def __init__(self, platform):
        self.log_file = open('./Log_Defend_' + platform + '.txt', 'a')

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict_onehot = (
            np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        val_targ_onehot = self.validation_data[2]
        val_predict = np.argmax(val_predict_onehot, axis=1)
        val_targ = np.argmax(val_targ_onehot, axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_auc = roc_auc_score(val_targ, val_predict)
        _val_acc = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        self.val_acc.append(_val_acc)
        print "Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f auc: %f" % (
            epoch, _val_acc, _val_precision, _val_recall, _val_f1, _val_auc)
        self.log_file.write(
            "Epoch: %d - val_accuracy: % f - val_precision: % f - val_recall % f val_f1: %f auc: %f\n" % (epoch,
                                                                                                          _val_acc,
                                                                                                          _val_precision,
                                                                                                          _val_recall,
                                                                                                          _val_f1,
                                                                                                          _val_auc))
        return


class LLayer(Layer):
    """
    Co-attention layer which accept content and comment states and computes co-attention between them and returns the
     weighted sum of the content and the comment states
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.latent_dim = 200
        self.k = 80
        super(LLayer, self).__init__(**kwargs)

    def build(self, input_shape, mask=None):
        self.Wl = K.variable(self.init((self.latent_dim, self.latent_dim)))

        self.Wc = K.variable(self.init((self.k, self.latent_dim)))
        self.Ws = K.variable(self.init((self.k, self.latent_dim)))

        self.whs = K.variable(self.init((1, self.k)))
        self.whc = K.variable(self.init((1, self.k)))
        self.trainable_weights = [self.Wl, self.Wc, self.Ws, self.whs, self.whc]

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        comment_rep = x[0]
        sentence_rep = x[1]
        sentence_rep_trans = K.permute_dimensions(sentence_rep, (0, 2, 1))
        comment_rep_trans = K.permute_dimensions(comment_rep, (0, 2, 1))
        L = K.tanh(tf.einsum('btd,dD,bDn->btn', comment_rep, self.Wl, sentence_rep_trans))
        L_trans = K.permute_dimensions(L, (0, 2, 1))

        Hs = K.tanh(tf.einsum('kd,bdn->bkn', self.Ws, sentence_rep_trans) + tf.einsum('kd,bdt,btn->bkn', self.Wc,
                                                                                      comment_rep_trans, L))
        Hc = K.tanh(tf.einsum('kd,bdt->bkt', self.Wc, comment_rep_trans) + tf.einsum('kd,bdn,bnt->bkt', self.Ws,
                                                                                     sentence_rep_trans, L_trans))
        As = K.softmax(tf.einsum('yk,bkn->bn', self.whs, Hs))
        Ac = K.softmax(tf.einsum('yk,bkt->bt', self.whc, Hc))
        co_s = tf.einsum('bdn,bn->bd', sentence_rep_trans, As)
        co_c = tf.einsum('bdt,bt->bd', comment_rep_trans, Ac)
        co_sc = K.concatenate([co_s, co_c], axis=1)

        return co_sc

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.latent_dim + self.latent_dim)


class AttLayer(Layer):
    """
    Attention layer used for the calcualting attention in word and sentence levels
    """

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = 100

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        print ait
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Defend():
    def __init__(self, platform):
        self.model = None
        self.MAX_SENTENCE_LENGTH = 120
        self.MAX_SENTENCE_COUNT = 50
        self.MAX_COMS_COUNT = 150
        self.MAX_COMS_LENGTH = 120
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.sentence_comment_co_model = None
        self.tokenizer = None
        self.class_count = 2
        self.metrics = Metrics(platform)

        # Variables for calculating attention weights
        self.news_content_word_level_encoder = None
        self.comment_word_level_encoder = None
        self.news_content_sentence_level_encoder = None
        self.comment_sequence_encoder = None
        self.co_attention_model = None

    def _generate_embedding(self, path, dim):
        return load_glove_embedding(path, dim, self.tokenizer.word_index)

    def _build_model(self, n_classes=2, embedding_dim=100, embeddings_path=False, aff_dim=80):
        GLOVE_DIR = "."
        embeddings_index = {}
        f = open(os.path.join('./', 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        word_index = self.tokenizer.word_index
        embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SENTENCE_LENGTH,
                                    trainable=True,
                                    mask_zero=True)

        com_embedding_layer = Embedding(len(word_index) + 1,
                                        embedding_dim,
                                        weights=[embedding_matrix],
                                        input_length=self.MAX_COMS_LENGTH,
                                        trainable=True,
                                        mask_zero=True)

        sentence_input = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True), name='word_lstm')(embedded_sequences)
        l_att = AttLayer(name='word_attention')(l_lstm)
        sentEncoder = Model(sentence_input, l_att)
        # plot_model(sentEncoder, to_file='SentenceEncoder.png', show_shapes=True)

        self.news_content_word_level_encoder = sentEncoder

        content_input = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32')
        content_encoder = TimeDistributed(sentEncoder)(content_input)
        l_lstm_sent = Bidirectional(GRU(100, return_sequences=True), name='sentence_lstm')(content_encoder)

        self.news_content_sentence_level_encoder = Model(content_input, l_lstm_sent)

        # learn comments representations
        comment_input = Input(shape=(self.MAX_COMS_LENGTH,), dtype='int32')
        com_embedded_sequences = com_embedding_layer(comment_input)
        c_lstm = Bidirectional(GRU(100, return_sequences=True), name='comment_lstm')(com_embedded_sequences)
        c_att = AttLayer(name='comment_word_attention')(c_lstm)
        comEncoder = Model(comment_input, c_att, name='comment_word_level_encoder')
        # plot_model(comEncoder, to_file='CommentEncoder.png', show_shapes=True)

        self.comment_word_level_encoder = comEncoder
        comEncoder.summary()

        all_comment_input = Input(shape=(self.MAX_COMS_COUNT, self.MAX_COMS_LENGTH), dtype='int32')
        all_comment_encoder = TimeDistributed(comEncoder, name='comment_sequence_encoder')(all_comment_input)

        self.comment_sequence_encoder = Model(all_comment_input, all_comment_encoder)

        # Co-attention

        L = LLayer(name="co-attention")([all_comment_encoder, l_lstm_sent])
        L_Model = Model([all_comment_input, content_input], L)

        self.co_attention_model = L_Model

        # plot_model(L_Model, to_file='l_representation.png', show_shapes=True)

        preds = Dense(2, activation='softmax')(L)
        model = Model(inputs=[all_comment_input, content_input], outputs=preds)
        model.summary()
        # plot_model(model, to_file='CHATT.png', show_shapes=True)

        optimize = RMSprop(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimize)

        return model

    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'AttLayer': AttLayer, 'LLayer': LLayer}):
            self.model = load_model(str(os.path.join(saved_model_dir, saved_model_filename)))
            self.news_sequence_encoder = self.model.get_layer('time_distributed_1')
            self.comment_sequence_encoder = self.model.get_layer('comment_sequence_encoder')
            self.co_attention_model = self.model.get_layer('co-attention')
            tokenizer_path = os.path.join(
                saved_model_dir, self._get_tokenizer_filename(saved_model_filename))
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state['tokenizer']
            self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
            self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
            self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
            self._create_reverse_word_index()

    def _get_tokenizer_filename(self, saved_model_filename):
        return saved_model_filename + '.tokenizer'

    def _fit_on_texts_and_comments(self, train_x, train_c, val_x, val_c):
        """
        Creates vocabulary set from the news content and the comments
        """
        texts = []
        texts.extend(train_x)
        texts.extend(val_x)
        comments = []
        comments.extend(train_c)
        comments.extend(val_c)
        self.tokenizer = Tokenizer(num_words=20000)
        all_text = []

        all_sentences = []
        for text in texts:
            for sentence in text:
                all_sentences.append(sentence)

        all_comments = []
        for com in comments:
            for sentence in com:
                all_comments.append(sentence)

        all_text.extend(all_comments)
        all_text.extend(all_sentences)
        self.tokenizer.fit_on_texts(all_text)
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()

    def _create_reverse_word_index(self):
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def _encode_texts(self, texts):
        """
        Pre process the news content sentences to equal length for feeding to GRU
        :param texts:
        :return:
        """
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32')
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_SENTENCE_LENGTH, padding='post', truncating='post', value=0))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def _encode_comments(self, comments):
        """
        Pre process the comments to equal length for feeding to GRU
        """
        encoded_texts = np.zeros((len(comments), self.MAX_COMS_COUNT, self.MAX_COMS_LENGTH), dtype='int32')
        for i, text in enumerate(comments):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_COMS_LENGTH, padding='post', truncating='post', value=0))[:self.MAX_COMS_COUNT]
            encoded_texts[i][:len(encoded_text)] = encoded_text

        return encoded_texts

    def _save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {
                'tokenizer': self.tokenizer,
                'maxSentenceCount': self.MAX_SENTENCE_COUNT,
                'maxSentenceLength': self.MAX_SENTENCE_LENGTH,
                'vocabularySize': self.VOCABULARY_SIZE
            }
            pickle.dump(tokenizer_state, open(path, "wb"))

    def train(self, train_x, train_y, train_c, val_c, val_x, val_y,
              batch_size=20, epochs=10,
              embeddings_path=False,
              saved_model_dir='saved_models', saved_model_filename=None, ):
        # Fit the vocabulary set on the content and comments
        self._fit_on_texts_and_comments(train_x, train_c, val_x, val_c)
        self.model = self._build_model(
            n_classes=train_y.shape[-1],
            embedding_dim=100,
            embeddings_path=embeddings_path)

        # Create encoded input for content and comments
        encoded_train_x = self._encode_texts(train_x)
        encoded_val_x = self._encode_texts(val_x)
        encoded_train_c = self._encode_comments(train_c)
        encoded_val_c = self._encode_comments(val_c)
        callbacks = [
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(
                    os.path.join(saved_model_dir,
                                 self._get_tokenizer_filename(saved_model_filename)), epoch))
        ]

        if saved_model_filename:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(saved_model_dir, saved_model_filename),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                )
            )
        callbacks.append(self.metrics)
        self.model.fit([encoded_train_c, encoded_train_x], y=train_y,
                       validation_data=([encoded_val_c, encoded_val_x], val_y),
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self, x, c):
        encoded_x = self._encode_texts(x)
        encoded_c = self._encode_comments(c)
        return self.model.predict([encoded_c, encoded_x])

    def process_atten_weight(self, encoded_text, content_word_level_attentions, sentence_co_attention):
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append((wd, content_word_level_attentions[k][i][j]))

                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        # Normalize without padding tokens
        no_pad_text_att_normalize = None
        for npta in no_pad_text_att:
            if len(npta) == 0:
                continue
            sen_att, sen_weight = list(zip(*npta))
            new_sen_weight = [float(i) / sum(sen_weight) for i in sen_weight]
            new_sen_att = []
            for sw in sen_att:
                word_list, att_list = list(zip(*sw))
                att_list = [float(i) / sum(att_list) for i in att_list]
                new_wd_att = list(zip(word_list, att_list))
                new_sen_att.append(new_wd_att)
            no_pad_text_att_normalize = list(zip(new_sen_att, new_sen_weight))

        return no_pad_text_att_normalize

    def process_atten_weight_com(self, encoded_text, sentence_co_attention):
        no_pad_text_att = []
        for k in range(len(encoded_text)):
            tmp_no_pad_text_att = []
            cur_text = encoded_text[k]
            for i in range(len(cur_text)):
                sen = cur_text[i]
                no_pad_sen_att = []
                if sum(sen) == 0:
                    continue
                for j in range(len(sen)):
                    wd_idx = sen[j]
                    if wd_idx == 0:
                        continue
                    wd = self.reverse_word_index[wd_idx]
                    no_pad_sen_att.append(wd)
                tmp_no_pad_text_att.append((no_pad_sen_att, sentence_co_attention[k][i]))

            no_pad_text_att.append(tmp_no_pad_text_att)

        return no_pad_text_att

    def activation_maps(self, news_article_sentence_list, news_article_comment_list, websafe=False):
        """
        :param news_article_sentence_list: List of sequence of sentences for each sample
        :param news_article_comment_list: List of sequences
        :param websafe: parameter to indicate if the interface is used in multithreaded web environment
        :return: Return attention weights of the sentences and comments for the samples passed
        """
        encoded_text = self._encode_texts(news_article_sentence_list)
        encoded_comment = self._encode_comments(news_article_comment_list)
        content_word_level_attentions = []

        # Get the word level attentions for news document
        W, b, u = self.news_sequence_encoder.layer.get_layer('word_attention').get_weights()

        content_word_encoder = Model(inputs=self.news_sequence_encoder.layer.input,
                                     outputs=self.news_sequence_encoder.layer.get_layer('word_lstm').output)
        for sent_text in encoded_text:
            word_level_weights = content_word_encoder.predict(sent_text)

            uit = np.tanh(np.matmul(word_level_weights, W) + b)
            ait = np.matmul(uit, u)
            ait = np.squeeze(ait, -1)
            # ait = np.exp(ait)
            content_word_wattention = (np.exp(ait) / np.sum(np.exp(ait), axis=1)[:, np.newaxis])
            content_word_level_attentions.append(content_word_wattention)

        # Get the word level attention for comments
        comment_word_level_attentions = []
        comment_word_encoder = Model(inputs=self.comment_sequence_encoder.layer.input,
                                     outputs=self.comment_sequence_encoder.layer.get_layer('comment_lstm').output)

        W, b, u = self.comment_sequence_encoder.layer.get_layer('comment_word_attention').get_weights()

        for comment_text in encoded_comment:
            comment_word_level_weights = comment_word_encoder.predict(comment_text)
            uit = np.tanh(np.matmul(comment_word_level_weights, W) + b)
            ait = np.matmul(uit, u)
            ait = np.squeeze(ait, -1)
            comment_word_level_attention = (np.exp(ait) / np.sum(np.exp(ait), axis=1)[:, np.newaxis])
            comment_word_level_attentions.append(comment_word_level_attention)

        # Get the co attention between document sentences and comments

        comment_level_encoder = Model(inputs=self.comment_sequence_encoder.input,
                                      outputs=self.comment_sequence_encoder.output)
        comment_level_weights = comment_level_encoder.predict(encoded_comment)

        sentence_level_encoder = Model(inputs=self.news_sequence_encoder.input,
                                       outputs=self.news_sequence_encoder.output)

        sentence_level_weights = sentence_level_encoder.predict(encoded_text)

        [Wl, Wc, Ws, whs, whc] = self.co_attention_model.get_weights()

        ### Calculate the co attention
        sentence_rep = sentence_level_weights
        comment_rep = comment_level_weights
        sentence_rep_trans = np.transpose(sentence_rep, axes=(0, 2, 1))
        comment_rep_trans = np.transpose(comment_rep, axes=(0, 2, 1))

        L = np.tanh(np.einsum('btd,dD,bDn->btn', comment_rep, Wl, sentence_rep_trans))
        L_trans = np.transpose(L, axes=(0, 2, 1))

        Hs = np.tanh(np.einsum('kd,bdn->bkn', Ws, sentence_rep_trans) + np.einsum('kd,bdt,btn->bkn', Wc,
                                                                                  comment_rep_trans, L))
        Hc = np.tanh(np.einsum('kd,bdt->bkt', Wc, comment_rep_trans) + np.einsum('kd,bdn,bnt->bkt', Ws,
                                                                                 sentence_rep_trans, L_trans))
        sent_unnorm_attn = np.einsum('yk,bkn->bn', whs, Hs)
        comment_unnorm_attn = np.einsum('yk,bkt->bt', whc, Hc)
        sentence_co_attention = (np.exp(sent_unnorm_attn) / np.sum(np.exp(sent_unnorm_attn), axis=1)[:, np.newaxis])
        comment_co_attention = (
                np.exp(comment_unnorm_attn) / np.sum(np.exp(comment_unnorm_attn), axis=1)[:, np.newaxis])
        if websafe:
            sentence_co_attention = sentence_co_attention.astype(float)
            comment_co_attention = comment_co_attention.astype(float)
            comment_word_level_attentions = np.array(comment_word_level_attentions).astype(float)
            content_word_level_attentions = np.array(content_word_level_attentions).astype(float)

        # res_content = self.process_atten_weight(encoded_text,content_word_level_attentions,sentence_co_attention)
        # res_comment = self.process_atten_weight(encoded_comment,comment_word_level_attentions,comment_co_attention)
        res_comment_weight = self.process_atten_weight_com(encoded_comment, comment_co_attention)

        res_sentence_weight = self.process_atten_weight_com(encoded_text, sentence_co_attention)

        return res_comment_weight, res_sentence_weight
