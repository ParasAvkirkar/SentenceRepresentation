# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GRU


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.dense_layers_list = []
        for i in range(num_layers):
            self.dense_layers_list.append(tf.keras.layers.Dense(input_dim, activation='relu'))


        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...

        batch_size = vector_sequence.shape[0]
        max_token_size = vector_sequence.shape[1]
        final_mask = sequence_mask
        if training:
            drop_out_mask = tf.random.uniform([batch_size, max_token_size])
            drop_out_mask = tf.where(drop_out_mask < 0.2, 0.0, 1.0)
            final_mask = tf.multiply(drop_out_mask, sequence_mask)

        # print(str(sequence_mask.shape))
        # print("final mask shape " + str(final_mask.shape))

        filtered_vector_sequence = tf.multiply(vector_sequence, tf.reshape(final_mask, [batch_size, max_token_size, 1]))
        # print("filtered vec seq " + str(filtered_vector_sequence.shape))
        vectors_considered = tf.reshape(tf.reduce_sum(final_mask, axis=1), [batch_size, 1])

        combined_vector = tf.reduce_sum(filtered_vector_sequence, 1)
        # num_words = vector_sequence.shape[1]
        combined_vector = tf.math.divide_no_nan(combined_vector, vectors_considered * 1.0)

        # print(str("combined vector shape ") + str(combined_vector.shape))

        layer_representations = []
        prev_output = combined_vector
        for i in range(len(self.dense_layers_list)):
            representation = self.dense_layers_list[i](prev_output)
            prev_output = representation
            layer_representations.append(prev_output)


        # TODO(students): end
        return {"combined_vector": prev_output,
                "layer_representations": tf.stack(layer_representations, axis=1)}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...

        self.gru_encoders = []
        for i in range(num_layers):
            self.gru_encoders.append(GRU(input_dim, return_sequences=True, return_state=True))

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...

        batch_size = vector_sequence.shape[0]
        max_token_size = vector_sequence.shape[1]

        prev_output = vector_sequence
        layer_representations = []
        # print("vector_seq " + str(prev_output.shape))
        for i in range(len(self.gru_encoders)):
            prev_output, last_state = self.gru_encoders[i](prev_output, mask=sequence_mask)
            layer_representations.append(last_state)
            # print("prev_output " + str(prev_output.shape) + " last state " + str(last_state.shape))

        # TODO(students): end
        return {"combined_vector": last_state,
                "layer_representations": tf.stack(layer_representations, axis=1)}
