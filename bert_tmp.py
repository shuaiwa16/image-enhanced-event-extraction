#! usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import numpy as np

import modeling
import optimization
import tokenization
from modules import multihead_attention
from modules import feedforward
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
import lstm_layer
import sys
sys.path = sys.path[5:]
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

# from tensorflow_large_model_support import LMSSessionRunHook
#tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 48,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.2,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("image_feature_dir", 'picture_resnet',
                    "The directory of training images")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, image, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.image = image


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, image_features):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.image_features = image_features
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            doc_id = '23r434rf2'
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                #print(word)
                #print(label)
                if contends.startswith("-DOCSTART-"):
                    #print('dadadadadadadad')
                    words.append('')
                    doc_id = contends[10:].strip().split()[0]
                    image_dir = os.path.join(FLAGS.image_feature_dir, doc_id + '.src')
                    print(image_dir)
                    image_names = os.listdir(image_dir)
                    continue
                if len(contends) == 0:
                    #print('dadadadadadadad')
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    cname = os.path.join(image_dir, image_names[0])
                    with open(cname, 'rb') as fin:
                        image = np.reshape(pickle.load(fin, encoding='bytes'), [-1]) / 100.0
                    lines.append([l, w, image])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            #print(lines)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_v2.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test_v2.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test_v2.txt")), "test")


    #def get_labels(self):
        #return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
    def get_labels(self):
        return ['Conflict-Demonstrate', 'Justice-Charge-Indict', 'Life-Die', 'Justice-Sue', 'Justice-Trial-Hearing', 'Justice-Arrest-Jail', 'Justice-Acquit', 'Justice-Fine', 'Business-Merge-Org', 'Justice-Pardon', 'Life-Marry', 'Conflict-Attack', 'Transaction-Transfer-Ownership', 'Business-Start-Org', 'Personnel-Start-Position', 'Life-Injure', 'Contact-Phone-Write', 'Movement-Transport', 'Business-End-Org', 'Justice-Convict', 'Personnel-End-Position', 'Business-Declare-Bankruptcy', 'Personnel-Nominate', 'Life-Divorce', 'Life-Be-Born', 'Transaction-Transfer-Money', 'Justice-Release-Parole', 'Justice-Appeal', 'Justice-Sentence', 'Personnel-Elect', 'Justice-Extradite', 'Justice-Execute', 'Contact-Meet', 'O', 'X', '[CLS]', '[SEP]']
        #return ['Transport', 'Pardon', 'Release-Parole', 'Phone-Write', 'Acquit', 'Convict', 'Injure', 'End-Position', 'Appeal', 'Fine', 'Transfer-Ownership', 'Start-Position', 'Elect', 'Extradite', 'Arrest-Jail', 'Sentence', 'Be-Born', 'Declare-Bankruptcy', 'Charge-Indict', 'End-Org', 'Marry', 'Attack', 'Sue', 'Trial-Hearing', 'Die', 'Demonstrate', 'Execute', 'Start-Org', 'Merge-Org', 'Divorce', 'Transfer-Money', 'Meet', 'Nominate', 'O', 'X', '[CLS]', '[SEP]']
        #return ['B-Transfer-Ownership', 'I-Sentence', 'B-End-Org', 'B-Transport', 'B-Extradite', 'B-Convict', 'I-Start-Position', 'I-Start-Org', 'B-Declare-Bankruptcy', 'B-Trial-Hearing', 'B-Pardon', 'I-Injure', 'B-Elect', 'B-Die', 'I-Divorce', 'B-Transfer-Money', 'I-Extradite', 'I-Acquit', 'I-Release-Parole', 'B-Arrest-Jail', 'I-Demonstrate', 'B-Divorce', 'I-Trial-Hearing', 'I-Meet', 'B-Nominate', 'I-Marry', 'I-Convict', 'I-Execute', 'B-Acquit', 'B-Injure', 'I-Fine', 'I-Arrest-Jail', 'B-Execute', 'B-Charge-Indict', 'B-Sentence', 'I-End-Position', 'I-Transfer-Ownership', 'I-Merge-Org', 'B-Release-Parole', 'B-Fine', 'B-Attack', 'I-Transport', 'I-End-Org', 'B-Start-Position', 'I-Sue', 'B-Merge-Org', 'I-Appeal', 'I-Declare-Bankruptcy', 'I-Pardon', 'I-Elect', 'I-Die', 'I-Attack', 'B-Demonstrate', 'B-Marry', 'B-End-Position', 'I-Phone-Write', 'B-Sue', 'B-Meet', 'I-Transfer-Money', 'B-Appeal', 'B-Be-Born', 'I-Be-Born', 'I-Nominate', 'I-Charge-Indict', 'B-Start-Org', 'B-Phone-Write', 'O', 'X', '[CLS]', '[SEP]']
    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            image = line[2]
            examples.append(InputExample(guid=guid, text=text, label=label, image = image))
        return examples


def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode):
    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        image_features = example.image
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f
        def bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["image_features"] = create_float_feature(feature.image_features)
        
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "image_features": tf.FixedLenFeature([49, 2048], tf.float32)
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings, image_features):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    #*************** edited by Shuai in 2019.08.16 **********
    #mask = tf.constant([[1 for _ in range(i + 1)] + [0 for _ in range(output_layer.shape[1].value - i - 1)] for i in range(output_layer.shape[1].value)], dtype = tf.float32)
    #print(mask.shape)
    #mask = tf.concat([tf.expand_dims(mask, -1), tf.expand_dims(1 - mask, -1)], axis = -1) #[len, len, 2]
    #print(mask.shape)
    #mask = tf.expand_dims(mask, axis = 0) #[1, len, len, 2]
    #mask = tf.zeros(shape = [tf.shape(output_layer)[0], output_layer.shape[1].value, output_layer.shape[1].value, 2], dtype = tf.float32) + tf.expand_dims(mask, axis = 0)
    #mask = tf.concat([tf.expand_dims(mask, 0) for _ in range(output_layer.shape[1])], axis = 0) #[batch, len, len, 2]
    #print(mask.shape)
    #mask = tf.expand_dims(mask, axis = 3) #[1, len, len, 1, 2]
    # mask = tf.constant(mask,dtype = tf.float32)
    #print(mask.shape)
    #output_layer = tf.expand_dims(output_layer, axis = -1) #[batch, len, dim, 1]
    #output_layer = tf.expand_dims(output_layer, axis = 2) #[batch, len, 1, dim, 1]
    #masked_output = mask * output_layer #[batch, len, len, dim, 2]
    #masked_output = tf.reduce_max(masked_output, axis = 2) #[batch, len, dim, 2]
    #pooled = tf.reshape(masked_output, shape = [-1, output_layer.shape[1].value, 768 * 2]) #[batch, len, dim*2]
    #output_layer = pooled
    #print('****************')
    #print(output_layers.shape)
    #print('****************')
    
    #*************** end ************************************

    # output_layer = LSTM_layer(input_ids, input_mask, is_training)
    #output_base = multihead_attention(output_layer, image_features, is_training = is_training, num_units = 768, num_heads = 8, dropout_rate = 0.2)
    print('**********************************')
    print(image_features)
    #image_features = tf.layers.dense(image_features, units = 768)
    #output_base = output_layer
    #image_features = multihead_attention(image_features, image_features, is_training = is_training, num_units = 512, num_heads = 8, dropout_rate = 0.2, scope = 'decoder')
    #output_layer = multihead_attention(output_layer, image_features, is_training = is_training, num_units = 768, num_heads = 8, dropout_rate = 0.2)
    #output_base = tf.concat([output_layer, output_base], axis = 2)
    #print('******************')
    #print(output_layer)
    #print('******************')
    #output_layer = output_layer + multihead_attention(output_layer, image_features, is_training = is_training, num_units = 1024, num_heads = 8, dropout_rate = 0.2)
    
    #output_layer_res = output_layer
    #for i in range(1):
    #    with tf.variable_scope("num_blocks_{}".format(i)):
    #        output_layer_res = multihead_attention(output_layer_res, output_layer_res, is_training = is_training, num_units = 768, num_heads = 8)
            #output_layer_res = multihead_attention(output_layer_res, image_features, is_training = is_training, num_units = 768, num_heads = 8, scope = 'decoder')
    #        output_layer_res = feedforward(output_layer_res, num_units=[4*768, 768])
    
    #output_layer = output_layer_res
    #f_gate = tf.layers.dense(output_base, units = 1, activation = tf.nn.sigmoid)
    #r_gate = tf.layers.dense(output_base, units = 1, activation = tf.nn.sigmoid)
    #z_val = tf.layers.dense(output_base, units = 768)
    #output_layer = r_gate * output_layer + f_gate * z_val
    hidden_size = output_layer.shape[-1].value
    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 38])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits,predict)
        ##########################################################################
        
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        image_features = features["image_features"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,  per_example_loss,logits,predicts) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, image_features)
        logging_hook = tf.train.LoggingTensorHook({"loss" : total_loss}, every_n_iter=100)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        #tf.logging.info("**** Trainable Variables ****")

        #for var in tvars:
        #    init_string = ""
        #    if var.name in initialized_variable_names:
        #        init_string = ", *INIT_FROM_CKPT*"
        #    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                    init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("optimizer_hook"):
                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def metric_fn(per_example_loss, label_ids, logits):
            # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,38, list(range(1, 33)),average="micro")
                recall = tf_metrics.recall(label_ids,predictions,38,list(range(1, 33)),average="micro")
                f = tf_metrics.f1(label_ids,predictions,38,list(range(1, 33)),average="micro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions= {'predictions':predicts,'gold_label':features["label_ids"],'origin_id':features["input_ids"]},scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        #lms_hook = LMSSessionRunHook({'optimizer_hook'})
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        list_of_monitors_and_hooks = [tf.contrib.learn.monitors.ValidationMonitor(input_fn = eval_input_fn, eval_steps=eval_steps, every_n_steps=500)]
        hooks = monitor_lib.replace_monitors_with_hooks(list_of_monitors_and_hooks, estimator)        
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks = hooks)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value:key for key,value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")
                            
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        #result = estimator.predict(input_fn=predict_input_fn)
        #output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        #with open(output_predict_file,'w') as writer:
        #    for prediction in result:
        #        output_line = "\n".join(id2label[id] for id in prediction if id!=0) + "\n"
        #        writer.write(output_line)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        with open(output_predict_file,'w') as writer:
            writer.write("word gold_label predict_label"+"\n")
            for ppi,rr in enumerate(result):
                tf.logging.info(rr)
                prediction=rr["predictions"]
                gold_label=rr["gold_label"]
                origin_id=rr["origin_id"]
                origin_words=tokenizer.convert_ids_to_tokens(origin_id)
        #origin_words=[]
        #for x in predict_examples[ppi].text.split(' '):
        #    origin_words.extend(tokenizer.tokenize(x))
                output_line=predict_examples[ppi].text+"\n"
        # print(len(origin_words),len(prediction[[i for i, e in enumerate(prediction) if e != 0]]))
                for idi,id in enumerate(prediction[[i for i, e in enumerate(prediction) if e != 0]][1:-1]):
                    output_line = output_line+origin_words[idi+1]+"__"+id2label[gold_label[idi+1]]+"__"+id2label[id]+"\n"
                writer.write(output_line+"\n")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


