from model import Understandable_Embedder
from data import find_all_occurence_and_replace,load_pretrain_ds,get_understanding_set
import os
import pickle
from transformers import DataCollatorForLanguageModeling


if __name__ == "__main__":
    batch_size = 4
    
 
    from transformers import BertTokenizer
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    import tensorflow_datasets as tfds

    epochs = 1

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)   
    
    pre_training_set = load_pretrain_ds( batch_size = batch_size)
    
    
    def train_understandable(train_dataset, understanding_dataset,only_dense = False, save_path = "model",debias_freq = 10):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model = Understandable_Embedder(batch_size, train_only_dense = only_dense, debias_freq = debias_freq)
          
        model.compile(optimizer=optimizer, loss=loss,metrics=["sparse_categorical_accuracy"])
         
         
        history = model.fit_pretrain(train_dataset,understanding_dataset, epochs=epochs, steps_per_epoch=int(5000), tokenizer = tokenizer) #3600 datapoints for cola
        path = "results/" + save_path
        os.makedirs(path,exist_ok=True)
        with open(path + "/history.txt", "wb") as fp:   
            pickle.dump(history, fp)
        model.save_weights(path +"/model")
        
        return history

    
#     definition_pairs = [[[" good ", " positive "],[" bad ", " negative "]],
#                         [[" women ", " girl "],[" men ", " boy "]],                       
#                         [[" asian ", "  Asian "],[" caucasian "," Caucasian "],[" african "," African "],[" european "," European "],[" american ", " American "]],
#                         [[" Muslim ", " muslims "],[" Jew ", " jews "], [" Christian ", " christians "]]] 
#                           
#     tokenized_definition_train_all = get_understanding_set(definition_pairs,tokenizer)
                                                                        
    
#     definition_pairs = [ [[" asian ", "  Asian "],[" caucasian "," Caucasian "],[" african "," African "],[" european "," European "],[" american ", " American "]]]
# 
#     tokenized_definition_train_race = get_understanding_set(definition_pairs,tokenizer)
#     
#     definition_pairs = [[[" women ", " girl "],[" men ", " boy "]]]
#        
#     tokenized_definition_train_gender = get_understanding_set(definition_pairs,tokenizer)

    definition_pairs = [[[" women ", " girl ", " female ", " she ", " actress ", " heroine ", " queen "," sister ", " mother ", " lady ", " her " ],[" men ", " boy ", " male ", " he ", " actor ", " hero ", " king ", " brother ", " father ", " gentleman ", " him "]]]
       
    tokenized_definition_train_gender_large = get_understanding_set(definition_pairs,tokenizer)
    
    
    for i in range(7):
        train_understandable(pre_training_set,tokenized_definition_train_gender_large, save_path = ("pre_gender_5000/train" + str(2**i)), debias_freq = 2**i)


