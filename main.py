
from model import Understandable_Embedder

from data import find_all_occurence_and_replace

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


if __name__ == "__main__":
    batch_size = 8
    model = Understandable_Embedder(batch_size)
 
    from transformers import BertTokenizer, glue_convert_examples_to_features
    import tensorflow as tf
    import tensorflow_datasets as tfds
    definition_pairs = [[" good "," bad "],[" women "," men "]] 
    definiton_train = find_all_occurence_and_replace(definition_pairs)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    data = tfds.load('glue/mrpc')
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128,  task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(batch_size).repeat(2)
    
    
    tokenized_definition_train=[]
    for definition_set in definiton_train:
        new_pair = []
        for pair in definition_set:
            new_def_set_pair = tokenizer(pair, max_length=128, padding=True, truncation=True, return_tensors='tf')
            new_pair.append(new_def_set_pair)
        tokenized_definition_train.append(new_pair)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
     
    model.compile(optimizer=optimizer, loss=loss)
    
  #  model.fit(train_dataset, epochs=2, steps_per_epoch=1840/batch_size) 
    model.custom_fit(train_dataset,tokenized_definition_train, epochs=4, steps_per_epoch=int(1840/batch_size))
    model.save("model")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    training_args = TrainingArguments(
        output_dir="model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset)
    
    trainer.train()
    
    trainer.save_model("model")