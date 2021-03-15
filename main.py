
from transformers import TFBertModel
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

def find_all_occurence_and_replace(definition_pairs, max_examples = 1000, max_length = 128):
    
    dataset = tfds.load('multi_news', split="train", shuffle_files=False)
    definition_sentences = []
    for definition_words in definition_pairs:
        bias_sentence = []
        for newsarticle in dataset:
            newsarticle = str(newsarticle["document"])
            for sentence in newsarticle.split("."):
             #   print(sentence)
              if len(sentence)>10 & len(sentence)<max_length:
                for word in definition_words:
                    sentence_pair = []
                    if word in sentence:
                        for word2 in definition_words:
                            sentence_pair.append(sentence.replace(word,word2))
                        bias_sentence.append(sentence_pair)  
            if ((max_examples != 0) & (len(bias_sentence)>max_examples)):
                break
        print("found ",len(bias_sentence), "of word", definition_words)
        definition_sentences.append(bias_sentence)
    return definition_sentences


class MyModel(tf.keras.Model):

  def __init__(self, target_units=768):
    super(MyModel, self).__init__()
    self.bert = TFBertModel.from_pretrained('bert-base-uncased')
    self.dropout = layers.Dropout(0.2)
    self.dense = layers.Dense(units = 2, activation = "softmax")
   # self.dense_headless = layers.Dense(units = target_units, activation = "Relu")
    self.compare_loss = keras.losses.MeanSquaredError()

  def call(self, inputs, training = False):
    x = self.bert(inputs,training=training)[1]
    x = self.dropout(x,training=training)
    outputs = self.dense(x)
    return outputs

  def call_headless(self, inputs, training = False):
    x = self.bert(inputs,training=training)[1]
  #  x = self.dense_headless(x)
    return x
  
  
  #todo abwechselndes fit von dataset und def_pairs mehr vermischen
  def custom_fit(self, dataset,definition_pairs,epochs,steps_per_epoch ):
      for e in range(epochs):
        step = 0
        for x,y in dataset:
            step = step +1
            if steps_per_epoch > step:
                break
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute our own loss
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
    
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
            # Compute our own metrics
            self.compiled_metrics.update_state(y, y_pred)
            
        for i,attribute_set in enumerate(definition_pairs):
            for pair in attribute_set:
                with tf.GradientTape() as tape:
                    def delete_dim(i,vec):
                        split1,split2,split3 = tf.split(vec,[i,1,(target_units-(i+1))],1)
                        short_vec = tf.concat([split1,split3],1)
                        return short_vec 
                    y_pred_1 = self.call_headless(pair[0], training=True)  # Forward pass                    
                    y_pred_2 = self.call_headless(pair[1], training=True)
                    y_pred_1 = delete_dim(i,y_pred_1)
                    y_pred_2 = delete_dim(i,y_pred_2)
                    loss = compare_loss(y_pred_1,y_pred_2, regularization_losses=self.losses)
                    
                        # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
    
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
            # Compute our own metrics
           # self.compiled_metrics.update_state(y, y_pred)
            
        return {m.name: m.result() for m in self.metrics}

if __name__ == "__main__":
    
    
    model = MyModel()
 
    from transformers import BertTokenizer, glue_convert_examples_to_features
    import tensorflow as tf
    import tensorflow_datasets as tfds
    definition_pairs = [[" good "," bad "],[" women "," men "]] 
    definiton_train = find_all_occurence_and_replace(definition_pairs)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    data = tfds.load('glue/mrpc')
    train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    train_dataset = train_dataset.shuffle(100).batch(16).repeat(2)
    
    tokenized_definition_train=[]
    for definition_set in definiton_train:
        new_def_set = []
        for pair in definition_set:
            new_pair = tokenizer(pair, max_length=128)
            new_def_set.append(new_pair)
        tokenized_definition_train.append(new_def_set)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
     
    model.compile(optimizer=optimizer, loss=loss)
     
    model.custom_fit(train_dataset,tokenized_definition_train, epochs=2, steps_per_epoch=115)
    model.summary()