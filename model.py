from transformers import TFBertModel,TFBertForPreTraining, BertTokenizer
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Understandable_Embedder(tf.keras.Model):
    
    def __init__(self, batch_size = 8, target_units=768, train_only_dense=False, contrastive_scale = 0.01):
      super(Understandable_Embedder, self).__init__()
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
      self.batch_size = batch_size
      self.bert = TFBertForPreTraining.from_pretrained('bert-base-uncased')
      self.dropout = layers.Dropout(0.2)
      self.train_only_dense = train_only_dense
      if self.train_only_dense:
          self.dense_projection = layers.Dense(units = target_units)
      self.dense = layers.Dense(units = 2, activation = "softmax")
    #  self.dense_headless = layers.Dense(units = target_units, activation = "Relu")
      self.compare_loss = keras.losses.MeanSquaredError()
      self.contrastive_scale = tf.constant(contrastive_scale)
      
    
    
    def __call__(self, inputs, training = False):
      x = self.bert.bert(inputs,training=training)[1]
    #  x = self.dense_headless(x)
      x = self.dropout(x,training=training)
      if self.train_only_dense:
          x = self.dense_projection(x)
      outputs = self.dense(x)
      return outputs
  
    def call_pre_training(self, inputs, training = True):
        prediction_scores,seq_relationship_score = self.bert(inputs,training=training)
        return prediction_scores,seq_relationship_score
    
    def call_headless(self, inputs, training = False):
      x = self.bert.bert(inputs,training=training)[1]
      if self.train_only_dense:
          x = self.dense_projection(x)
   #   x = self.dense_headless(x)
      return x
  
    def predict_simple(self,inputs):
        tokenized_inputs = self.tokenizer(inputs, max_length=128, padding=True, truncation=True, return_tensors='tf')
        x = self.call_headless(tokenized_inputs,training=False)
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
    def compare_train_step(self,x1,x2,i,loss_factor):
        with tf.GradientTape() as tape:
        
            y_pred_1 = self.call_headless(x1, training=True)  # Forward pass                    
            y_pred_2 = self.call_headless(x2, training=True)
            
            contrastive_loss = (1.0/self.compare_loss(y_pred_1[:,i],y_pred_2[:,i]))* self.contrastive_scale
            
            y_pred_1 = self.delete_dim(i,y_pred_1)
            y_pred_2 = self.delete_dim(i,y_pred_2)
            if contrastive_loss < self.contrastive_scale*self.batch_size:
                contrastive_loss = 0.0
            loss = (self.compare_loss(y_pred_1,y_pred_2) + contrastive_loss)*loss_factor #scale loss by 1/number of set meaning dimensions
       #   tf.multiply(loss[0,i], 0) 
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return loss
  

        
    def fit_classify_understandable(self, dataset,definition_pairs,epochs,steps_per_epoch , report_intervall = 20):
        self.bert.nsp.trainable = False
        self.bert.mlm.trainable = False
        self.bert.bert.trainable = not self.train_only_dense
        history = {}
        history["loss"] = []
        history["loss_compare"] = []     
        for e in range(epochs):
            print("At epoch", e+1, "of", epochs)
            step = 0
            avg_loss = 0.0
            avg_loss_compare = 0.0
            loss_report = 0.0
            loss_report_compare = 0.0
            for x,y in dataset:
                
                if step > steps_per_epoch :
                    break
                loss = self.normal_train_step(x,y)
                avg_loss = avg_loss+float(loss)
                loss_report = loss_report + +float(loss)
                
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
                    loss_compare = self.compare_train_step(feed_dict_1,feed_dict_2,tf.constant(current_attribute_id),tf.constant(1.0/len(definition_pairs)))
                    avg_loss_compare = avg_loss_compare+float(loss_compare)  
                    loss_report_compare = loss_report_compare + float(loss_compare) 
                self.dense.trainable = True        
                        
                step = step +1
                if step % report_intervall == 0: 
                    print("Average Training loss at step",step, "/", steps_per_epoch,":", loss_report/report_intervall)
                    print("Average Training loss_compare at step",step, "/", steps_per_epoch,":", loss_report_compare/report_intervall)
                    history["loss"].append(loss_report/report_intervall)
                    history["loss_compare"].append(loss_report_compare/report_intervall)
                    loss_report = 0.0
                    loss_report_compare = 0.0
 
        return history
    
    def fit_classify(self, dataset,epochs,steps_per_epoch , report_intervall = 20):
        self.bert.nsp.trainable = False
        self.bert.mlm.trainable = False
        self.bert.bert.trainable = not self.train_only_dense
        history = {}
        history["loss"] = []
        history["loss_compare"] = []     
        for e in range(epochs):
            print("At epoch", e+1, "of", epochs)
            step = 0
            avg_loss = 0.0
            avg_loss_compare = 0.0
            loss_report = 0.0
            for x,y in dataset:
                
                if step > steps_per_epoch :
                    break
                loss = self.normal_train_step(x,y)
                avg_loss = avg_loss+float(loss)
                loss_report = loss_report + +float(loss)    
                        
                step = step +1
                if step % report_intervall == 0: 
                    print("Average Training loss at step",step, "/", steps_per_epoch,":", loss_report/report_intervall)
                    history["loss"].append(loss_report/report_intervall)
                    loss_report = 0.0
 
        return history
    
    def evaluate(self, dataset, batch_size, dataset_length, verbose = False):
        accuracy = 0.0
        step = 0
        
        for x,y in dataset:
            step = step + batch_size
            if step > dataset_length :
                break
            y_pred = self(x)
            if verbose:
                if step % np.round(dataset_length/5) < batch_size: 
                    print("at", step, "of", dataset_length)
            for i,pred in enumerate(y):
                if np.argmax(y_pred[i].numpy()) == pred.numpy():
                    accuracy = accuracy +1.0
        accuracy = accuracy /step
        return accuracy
    
#     def compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#         )
#         # make sure only labels that are not equal to -100
#         # are taken into account as loss
#         masked_lm_active_loss = tf.not_equal(tf.reshape(tensor=labels["labels"], shape=(-1,)), -100)
#         masked_lm_reduced_logits = tf.boolean_mask(
#             tensor=tf.reshape(tensor=logits[0], shape=(-1, shape_list(logits[0])[2])),
#             mask=masked_lm_active_loss,
#         )
#         masked_lm_labels = tf.boolean_mask(
#             tensor=tf.reshape(tensor=labels["labels"], shape=(-1,)), mask=masked_lm_active_loss
#         )
#         next_sentence_active_loss = tf.not_equal(tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), -100)
#         next_sentence_reduced_logits = tf.boolean_mask(
#             tensor=tf.reshape(tensor=logits[1], shape=(-1, 2)), mask=next_sentence_active_loss
#         )
#         next_sentence_label = tf.boolean_mask(
#             tensor=tf.reshape(tensor=labels["next_sentence_label"], shape=(-1,)), mask=next_sentence_active_loss
#         )
#         masked_lm_loss = loss_fn(y_true=masked_lm_labels, y_pred=masked_lm_reduced_logits)
#         next_sentence_loss = loss_fn(y_true=next_sentence_label, y_pred=next_sentence_reduced_logits)
#         masked_lm_loss = tf.reshape(tensor=masked_lm_loss, shape=(-1, shape_list(next_sentence_loss)[0]))
#         masked_lm_loss = tf.reduce_mean(input_tensor=masked_lm_loss, axis=0)
# 
#         return masked_lm_loss + next_sentence_loss
    
    
   # @tf.function
    def pretrain_train_step(self,x,y):
      with tf.GradientTape() as tape:
          trainingoutput = self.call_pre_training(x, training=True)  # Forward pass
          # Compute our own loss
          loss = trainingoutput[0]
      
      # Compute gradients
      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))

      # Compute our own metrics
      self.compiled_metrics.update_state(y, y_pred)
      return loss
  
    def mask_tokens(self, inputs):
        labels = inputs.clone()
        
       # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = tf.fill(labels.shape, self.mlm_probability)
      #  if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = tf.constant(special_tokens_mask, dtype=tf.bool)
#         else:
#             special_tokens_mask = special_tokens_mask.bool()
        
        special_token_indices = tf.where(special_tokens_mask)
        probability_matrix[special_token_indices] = 0.0
        masked_indices = tfp.distributions.Bernoulli(probs = probability_matrix).sample(probability_matrix.shape)
        labels[~masked_indices] = -100
        
        
         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs,labels
  
    def fit_pretrain(self, dataset,definition_pairs,epochs,steps_per_epoch,tokenizer, mlm_prob = 0.15 ):
        self.tokenizer = tokenizer
        self.bert.nsp.trainable = False
        self.dense.trainable = False
       # self.bert.mlm.trainable = False
        for e in range(epochs):
            print("At epoch", e+1, "of", epochs)
            step = 0
            avg_loss = 0.0
            avg_loss_compare = 0.0
            for x,y in dataset:
                
                if step > steps_per_epoch :
                    break
                loss = self.pretrain_train_step(self.mask_tokens(x))
                avg_loss = avg_loss+float(loss)
                
                self.bert.mlm.trainable = False
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
                self.bert.mlm.trainable = True        
                        
                step = step +1
                if step % 20 == 0: 
                    print("Training loss (for one batch) at step",step, "/", steps_per_epoch,":", avg_loss/step)
                    print("Training loss_compare (for one batch) at step",step, "/", steps_per_epoch,":", avg_loss_compare/step)
                
        return {m.name: m.result() for m in self.metrics}
        
