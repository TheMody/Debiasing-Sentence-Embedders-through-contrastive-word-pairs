
import pickle
import matplotlib.pyplot as plt
from model import Understandable_Embedder
import numpy as np
import math
from sklearn.decomposition import PCA
import tensorflow_datasets as tfds

def load_dataset(split = "train"):
    ds = tfds.load('multi_news', split=split, shuffle_files=False)
    #ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

class Sent_Debias:
    
    def find_all_occurence_and_replace(self,bias_words,dataset, max_words = 0):
        bias_sentence = []
        for newsarticle in dataset:
            newsarticle = str(newsarticle["document"])
            for sentence in newsarticle.split("."):
             #   print(sentence)
              if len(sentence)>20 & len(sentence)<300:
                for word in bias_words:
                    sentence_pair = []
                    if word in sentence:
                        for word2 in bias_words:
                            sentence_pair.append(sentence.replace(word,word2))
                        bias_sentence.append(sentence_pair)  
            if ((max_words != 0) & (len(bias_sentence)>max_words)):
                break
        print("found ",len(bias_sentence), "of word", bias_words)
        return bias_sentence
    
    def encode_sentence_pairs(self,encoder, pairs):
        encoded_pairs = []
        for sentence_pair in pairs:
            encoded_sentences = encoder.predict_simple(sentence_pair)
            encoded_pairs.append(encoded_sentences)
        return np.asarray(encoded_pairs)
    
    def sub_mean(self,pairs):
        means = np.mean(pairs, axis = 1) 
        for i,_ in enumerate(means):
            for vec in pairs[i]:
                vec = vec - means[i]
        return pairs

    def flatten(self,pairs):
        flattened = []
        for pair in pairs:
            for vec in pair:
                flattened.append(vec)
        return flattened
    
    def fit(self,encoder,bias_words, dataset = load_dataset(),k = 3, max_words = 0):
        bias_sentence_pair = self.find_all_occurence_and_replace(bias_words,dataset,max_words = max_words)
        encoded_pairs = self.encode_sentence_pairs(encoder,bias_sentence_pair)
        encoded_pairs = self.sub_mean(encoded_pairs)
        flattened = self.flatten(encoded_pairs)
        self.pca = PCA(n_components=k)
        self.pca.fit(flattened)
        
    def predict(self,X):
        V = self.pca.components_
        debiased = []
        for u in X:
            norm_sqrd = np.sum(V*V, axis=-1)
            vecs = np.divide(V@u, norm_sqrd)[:, None] * V
            subspace = np.sum(vecs, axis=0)
            debiased_sample =  u - subspace
            debiased.append(debiased_sample)
        
        
#         for sample in X:
#             dot_prod = []
#             for comp in v:
#                 dot_prod.append(np.dot(sample,comp))
#             dot_prod = np.asarray(dot_prod)
#             debiased_sample = sample - np.matmul(dot_prod,v)
#             debiased.append(debiased_sample)
        return np.asarray(debiased)
    
    def save(self, name = 'data/sent_debias_estimator.pkl'):
        file = open(name, 'wb')
        pickle.dump(self.pca, file)
        file.close()
        return
        
    def load(self, name = 'data/sent_debias_estimator.pkl'):
        file = open(name, 'rb')
        self.pca = pickle.load(file)   
        file.close()
        return
        
        
if __name__ == "__main__":        
    encoder = Understandable_Embedder() 
    bias_words = ["girl", "boy"]### # #["good","bad"]##["amazon", "facebook"]   #["amazon", "facebook","google","blizzard"] #["positive","negative"]
    print(bias_words)
    Sent_Deb = Sent_Debias()
    
    number_dim = 3
    global load 
    load = False
    max_words = 5000
    if load == False:
        print("fitting sent debiasing")
        Sent_Deb.fit(encoder,bias_words ,k= number_dim, max_words = max_words)
        Sent_Deb.save()
    else:
        Sent_Deb.load()