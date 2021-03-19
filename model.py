from transformers import TFBertModel,TFBertForPreTraining
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class Understandable_Embedder(tf.keras.Model):
    
    def __init__(self, target_units=768, batch_size=8):
      super(MyModel, self).__init__()
      self.batch_size = batch_size
      self.bert = TFBertForPreTraining.from_pretrained('bert-base-uncased')
      self.dropout = layers.Dropout(0.2)
      self.dense = layers.Dense(units = 2, activation = "softmax")
    #  self.dense_headless = layers.Dense(units = target_units, activation = "Relu")
      self.compare_loss = keras.losses.MeanSquaredError()
      
    
    
    def call(self, inputs, training = False):
      x = self.bert.bert(inputs,training=training)[1]
    #  x = self.dense_headless(x)
      x = self.dropout(x,training=training)
      outputs = self.dense(x)
      return outputs
  
    def call_pre_training(self, inputs, training = False):
        prediction_scores,seq_relationship_score = self.bert(inputs,training=training)
        return prediction_scores,seq_relationship_score
    
    def call_headless(self, inputs, training = False):
      x = self.bert.bert(inputs,training=training)[1]
   #   x = self.dense_headless(x)
      return x
    
    @tf.function
    def normal_train_step(self,x,y):
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
      return loss
    
    @tf.function
    def delete_dim(self,i,vec, axis = 1):
        split1,split2,split3 = tf.split(vec,[i,1,(vec.shape[axis]-(i+1))],axis)
        short_vec = tf.concat([split1,split3],1)
        return short_vec 
    
    @tf.function
    def compare_train_step(self,x1,x2,i):
      with tf.GradientTape() as tape:
    
          y_pred_1 = self.call_headless(x1, training=True)  # Forward pass                    
          y_pred_2 = self.call_headless(x2, training=True)
          y_pred_1 = self.delete_dim(i,y_pred_1)
          y_pred_2 = self.delete_dim(i,y_pred_2)
          loss = self.compare_loss(y_pred_1,y_pred_2)
       #   tf.multiply(loss[0,i], 0) 
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      return loss
        
    def custom_fit(self, dataset,definition_pairs,epochs,steps_per_epoch ):
        self.bert.nsp.trainable = False
        self.bert.mlm.trainable = False
        for e in range(epochs):
            print("At epoch", e+1, "of", epochs)
            step = 0
            avg_loss = 0.0
            avg_loss_compare = 0.0
            for x,y in dataset:
                
                if step > steps_per_epoch :
                    break
                loss = self.normal_train_step(x,y)
                avg_loss = avg_loss+float(loss)
                
                self.dense.trainable = False
                for current_attribute_id,attribute_set in enumerate(definition_pairs):
                    start = ((e*steps_per_epoch +step)*self.batch_size) % len(attribute_set)
                    stop = start + self.batch_size
                    
                    #one dictionary for each word
                    feed_dict_1 = {}
                    feed_dict_1["input_ids"] = attribute_set[0]["input_ids"][start:stop,:]
                    feed_dict_1["token_type_ids"] = attribute_set[0]["token_type_ids"][start:stop,:]
                    feed_dict_1["attention_mask"] = attribute_set[0]["attention_mask"][start:stop,:]
                    
                    #one dictionary for each word
                    feed_dict_2 = {}
                    feed_dict_2["input_ids"] = attribute_set[1]["input_ids"][start:stop,:]
                    feed_dict_2["token_type_ids"] = attribute_set[1]["token_type_ids"][start:stop,:]
                    feed_dict_2["attention_mask"] = attribute_set[1]["attention_mask"][start:stop,:]
                    
                   # print(feed_dict_1)    
                    loss = self.compare_train_step(feed_dict_1,feed_dict_2,tf.constant(current_attribute_id))
                    avg_loss_compare = avg_loss_compare+float(loss)  
                self.dense.trainable = True        
                        
                step = step +1
                if step % 20 == 0: 
                    print("Training loss (for one batch) at step",step, "/", steps_per_epoch,":", avg_loss/step)
                    print("Training loss_compare (for one batch) at step",step, "/", steps_per_epoch,":", avg_loss_compare/step)
                
        return {m.name: m.result() for m in self.metrics}
