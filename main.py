
from model import Understandable_Embedder
from data import find_all_occurence_and_replace
import os
import pickle
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


if __name__ == "__main__":
    batch_size = 4
    
 
    from transformers import BertTokenizer, glue_convert_examples_to_features
    import tensorflow as tf
    import tensorflow_datasets as tfds
#     definition_pairs = [[[" good ", " positive "],[" bad ", " negative "]],
#                         [[" women ", " girl "],[" men ", " boy "]],                       
#                         [[" asian "],[" caucasian "],[" african "],[" european "],[" american "]],
#                         [[" muslim ", " muslims "],[" jew ", " jews "], [" christian ", " christians "]]] 
    definition_pairs =  [[[" women ", " girl "],[" men ", " boy "]]]
    definiton_train = find_all_occurence_and_replace(definition_pairs)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenized_definition_train=[]
    for definition_set in definiton_train:
        new_pair = []
        for pair in definition_set:
            new_def_set_pair = tokenizer(pair, max_length=128, padding=True, truncation=True, return_tensors='tf')
            new_pair.append(new_def_set_pair)
        tokenized_definition_train.append(new_pair)
    
    
    #iterate over glue tasks
    task_list = ["mrpc" , "cola"]
    for task in task_list:
        
        data = tfds.load('glue/'+task)
        train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128,  task='mrpc')
        train_dataset = train_dataset.shuffle(100).batch(batch_size).repeat(2)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model = Understandable_Embedder(batch_size)
         
        model.compile(optimizer=optimizer, loss=loss,metrics=["sparse_categorical_accuracy"])
        
        
        history_extra = model.fit_classify_understandable(train_dataset,tokenized_definition_train, epochs=2, steps_per_epoch=int(3600/batch_size)) #3600 datapoints
        path = "model"+task + "/understandable"
        os.makedirs(path,exist_ok=True)
        with open(path + "/understandable_history.txt", "wb") as fp:   
            pickle.dump(history_extra, fp)
        model.save_weights(path +"/model")# cant save because model.build not called
        
        model_normal = Understandable_Embedder(batch_size)
         
        model_normal.compile(optimizer=optimizer, loss=loss,metrics=["sparse_categorical_accuracy"])
        
      #  model.fit(train_dataset, epochs=2, steps_per_epoch=1840/batch_size) 
        history = model_normal.fit_classify(train_dataset, epochs=2, steps_per_epoch=int(3600/batch_size))
        with open(path +"/normal_history.txt", "wb") as fp:   
            pickle.dump(history, fp)
        path = "model"+task + "/normal"
        model_normal.save_weights(path +"/model")
