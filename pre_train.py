from model import Understandable_Embedder
from data import find_all_occurence_and_replace,load_pretrain_ds
import os
import pickle
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


if __name__ == "__main__":
    batch_size = 4
    
 
    from transformers import BertTokenizer, glue_convert_examples_to_features
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    import tensorflow_datasets as tfds