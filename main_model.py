# inbuilt lib imports:
from typing import List, Dict, Tuple
import os

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models

# project imports
from sequence_to_vector import DanSequenceToVector, GruSequenceToVector


class MainClassifier(models.Model):
    def __init__(self,
                 seq2vec_choice: str,
                 vocab_size: int,
                 embedding_dim: int,
                 num_layers: int = 2,
                 num_classes: int = 2) -> 'MainClassifier':
        """
        It is a wrapper model for DAN or GRU sentence encoder.
        The initializer typically stores configurations in private/public
        variables, which need to accessed during the call (forward pass).
        We also define the trainable variables (Parameters in TF1.0)
        in the initializer.

        Parameters
        ----------
        seq2vec_choice : ``str``
            Name of sentence encoder: "dan" or "gru".
        vocab_size : ``int``
            Vocabulary size used to index the data instances.
        embedding_dim : ``int``
            Embedding matrix dimension
        num_layers : ``int``
            Number of layers of sentence encoder to build.
        num_classes : ``int``
            Number of classes that this Classifier chooses from.
        """
        super(MainClassifier, self).__init__()
        # Construct and setup sequence_to_vector model

        if seq2vec_choice == "dan":
            self._seq2vec_layer = DanSequenceToVector(embedding_dim, num_layers)
        else:
            self._seq2vec_layer = GruSequenceToVector(embedding_dim, num_layers)

        # Trainable Variables
        self._embeddings = tf.Variable(tf.random.normal((vocab_size, embedding_dim)),
                                       trainable=True)
        self._classification_layer = layers.Dense(units=num_classes)

    def call(self,
             inputs: tf.Tensor,
             training=False):
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        embedded_tokens = tf.nn.embedding_lookup(self._embeddings, inputs)
        tokens_mask = tf.cast(inputs!=0, tf.float32)
        outputs = self._seq2vec_layer(embedded_tokens, tokens_mask, training)
        classification_vector = outputs["combined_vector"]
        layer_representations = outputs["layer_representations"]
        logits = self._classification_layer(classification_vector)
        return {"logits": logits, "layer_representations": layer_representations}
