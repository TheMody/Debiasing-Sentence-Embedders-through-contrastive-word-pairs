
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


    
    #create pretrain ds
#     pretrain_ds = load_pretrain_ds(tokenizer)
#     pretrain_ds =  pretrain_ds.shuffle(100)#.batch(batch_size)
#     for output in pretrain_ds:
#         print(list(output.keys()))
     #   print(output["document"])
    
    # create additional ds
    
    def get_understanding_set(definition_pairs):
        definiton_train = find_all_occurence_and_replace(definition_pairs, max_examples = 10000)
        tokenized_definition_train=[]
        for definition_set in definiton_train:
            new_pair = []
            for pair in definition_set:
                new_def_set_pair = tokenizer(pair, max_length=128, padding=True, truncation=True, return_tensors='tf')
                new_pair.append(new_def_set_pair)
            tokenized_definition_train.append(new_pair)
        return tokenized_definition_train
    
    def train_understandable(train_dataset, understanding_dataset,only_dense = False, save_path = "model"):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model = Understandable_Embedder(batch_size, train_only_dense = only_dense)
          
        model.compile(optimizer=optimizer, loss=loss,metrics=["sparse_categorical_accuracy"])
         
         
        history = model.fit_classify_understandable(train_dataset,understanding_dataset, epochs=2, steps_per_epoch=int(3600/batch_size)) #3600 datapoints
        path = save_path + "/understandable"
        os.makedirs(path,exist_ok=True)
        with open(path + "/understandable_history.txt", "wb") as fp:   
            pickle.dump(history, fp)
        model.save_weights(path +"/model")
        
        return history
        
    def train_normal(train_dataset,only_dense = False, save_path = "normal"):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model_normal = Understandable_Embedder(batch_size, train_only_dense = only_dense)
         
        model_normal.compile(optimizer=optimizer, loss=loss,metrics=["sparse_categorical_accuracy"])
        
      #  model.fit(train_dataset, epochs=2, steps_per_epoch=1840/batch_size) 
        history = model_normal.fit_pretrain(train_dataset, epochs=2, steps_per_epoch=int(3600/batch_size))
        path = save_path + "/normal"
        os.makedirs(path,exist_ok=True)
        with open(path +"/normal_history.txt", "wb") as fp:   
            pickle.dump(history, fp)
        model_normal.save_weights(path +"/model")
        
        return history
    
#     definition_pairs = [[[" good ", " positive "],[" bad ", " negative "]],
#                         [[" women ", " girl "],[" men ", " boy "]],                       
#                         [[" asian ", "  Asian "],[" caucasian "," Caucasian "],[" african "," African "],[" european "," European "],[" american ", " American "]],
#                         [[" muslim ", " muslims "],[" jew ", " jews "], [" christian ", " christians "]]]  [[[" women ", " girl "],[" men ", " boy "]]]
#                         
#     tokenized_definition_train_all = get_understanding_set(definition_pairs)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)                                                                        
    
    definition_pairs = [ [[" asian ", "  Asian "],[" caucasian "," Caucasian "],[" african "," African "],[" european "," European "],[" american ", " American "]]]

    tokenized_definition_train_race = get_understanding_set(definition_pairs)
    
    definition_pairs = [[[" women ", " girl "],[" men ", " boy "]]]
    
    tokenized_definition_train_gender = get_understanding_set(definition_pairs)
        
    #iterate over glue tasks
    task_list = ["mrpc" , "cola"]
    for task in task_list:
        
        data = tfds.load('glue/'+task)
        train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128,  task=task)
        train_dataset = train_dataset.shuffle(100).batch(batch_size).repeat(2)
        
       # train_understandable(train_dataset,tokenized_definition_train_gender, save_path = "gender_contrastive")
        train_understandable(train_dataset,tokenized_definition_train_race, save_path = "race_contrastive")
        
        #train_normal(train_dataset, save_path = "normal")
