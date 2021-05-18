import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


def find_all_occurence_and_replace(definition_pairs, max_examples = 5000, max_length = 512, min_length = 8):
    
    dataset = tfds.load('multi_news', split="train", shuffle_files=False)
    definition_sentences = []
    rng = np.random.default_rng()
    for definition_words in definition_pairs:
        bias_sentence_pair1 = []
        bias_sentence_pair2 = []
        for newsarticle in dataset:
            newsarticle = str(newsarticle["document"])
            for sentence in newsarticle.split("."):
              if len(sentence)>min_length & len(sentence)<max_length:
                  for i,equalword in enumerate(definition_words):
                    for a,word in enumerate(equalword):
                        if word in sentence:
                            listwithout_word = definition_words[:i] +definition_words[i+1:]
                            replace_word = rng.choice(listwithout_word, axis = 0)[a]
                            bias_sentence_pair1.append(sentence.replace(word,replace_word))
                            bias_sentence_pair2.append(sentence)
#                             print(sentence.replace(word,replace_word))
#                             print(sentence)
            if ((max_examples != 0) & (len(bias_sentence_pair1)>max_examples)):
                break
        print("found ",len(bias_sentence_pair1), "of word", definition_words)
        definition_sentences.append([bias_sentence_pair1,bias_sentence_pair2])
    return definition_sentences






def load_pretrain_ds(tokenizer, batch_size, max_length = 1024, min_length = 8):
    dataset = tfds.load('multi_news', split="train", shuffle_files=False)
    allsentences = []
    for newsarticle in dataset:
        text = str(newsarticle["document"])
        for sentence in newsarticle.split("."):
            if len(sentence)>min_length & len(sentence)<max_length:
                allsentences.append(sentence)

    tokenizer(list, max_length=128, padding=True, truncation=True, return_tensors='tf')
    
    return new_ds

