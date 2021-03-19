import tensorflow_datasets as tfds


def find_all_occurence_and_replace(definition_pairs, max_examples = 5000, max_length = 128):
    
    dataset = tfds.load('multi_news', split="train", shuffle_files=False)
    definition_sentences = []
    for definition_words in definition_pairs:
        bias_sentence_pair1 = []
        bias_sentence_pair2 = []
        for newsarticle in dataset:
            newsarticle = str(newsarticle["document"])
            for sentence in newsarticle.split("."):
              if len(sentence)>10 & len(sentence)<max_length:
                    if definition_words[0] in sentence:
                        bias_sentence_pair1.append(sentence.replace(definition_words[0],definition_words[1]))
                        bias_sentence_pair2.append(sentence)
                    else:
                        if definition_words[1] in sentence:
                            bias_sentence_pair1.append(sentence.replace(definition_words[1],definition_words[0]))
                            bias_sentence_pair2.append(sentence)
            if ((max_examples != 0) & (len(bias_sentence_pair1)>max_examples)):
                break
        print("found ",len(bias_sentence_pair1), "of word", definition_words)
        definition_sentences.append([bias_sentence_pair1,bias_sentence_pair2])
    return definition_sentences