
import pickle
import matplotlib.pyplot as plt
from model import Understandable_Embedder
from utils import DataLoader, SentGenerator
from bias_eval import BiasMetrics, WEATMetrics, WEAT, DistanceTests, ImprovedBiasMetrics
import copy
import numpy as np
from numpy import linalg as LA
import math
from data import get_understanding_set
from transformers import BertTokenizer, glue_convert_examples_to_features
from nltk.corpus.reader import lin


def plot_history(path):
    with open(path + "/understandable/understandable_history.txt", "rb") as fp:   # Unpickling
        understandable_history = pickle.load(fp)
    with open(path +"/understandable/normal_history.txt", "rb") as fp:   # Unpickling
        normal_history = pickle.load(fp)
  #  plt.plot(understandable_history["loss_compare"], label = "loss_compare")
    plt.plot(understandable_history["loss"], label = "loss")
    plt.plot(normal_history['loss'], label = "loss_original")
    plt.legend()
   # plt.plot(normal_history['sparse_categorical_accuracy'], label = "accuracy")
    plt.savefig("History_race_gender.eps")
    plt.show()
    
    
def plot_average_history(path, intervall):

#     history_list_gender=[]
#     for i in range(intervall):
#         with open(path +"cola_gender_large_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
#             understandable_history = pickle.load(fp)
#             history_list_gender.append(understandable_history)
#     generate_error_bound_plot(history_list_gender)
    
    history_list_race =[]
    for i in range(intervall):
        with open(path +"Genderlarge/sst2_gender_large_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
            understandable_history = pickle.load(fp)
            history_list_race.append(understandable_history)
    generate_error_bound_plot(history_list_race, color=(0.1,0.6,0.1), label = "debiased")
#     
    history_list_normal=[]
    for i in range(intervall):
        with open(path +"without/sst2_normal_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
            understandable_history = pickle.load(fp)
            history_list_normal.append(understandable_history)        
    generate_error_bound_plot(history_list_normal, color=(0.6,0.1,0.1), label = "original")
    
    plt.savefig("average_history_sst2_both.png")
    
def generate_error_bound_plot(history_list, color=(0.1,0.1,0.6), label = ""):
    history_matrix=np.zeros((len(history_list),len(history_list[0]["loss"])))
    for i,history in enumerate(history_list):
        history_matrix[i,:]=history["loss"]
    mean_history = np.mean(history_matrix,axis=0)
    var_history = np.var(history_matrix,axis=0)
    confidence_intervall = 1.96*np.sqrt(var_history)/np.sqrt(len(history_list))
    confidence_intervall_upper = mean_history + confidence_intervall
    confidence_intervall_lower =   mean_history - confidence_intervall
    
    boundcolor = (color[0]*1.5,color[1]*1.5,color[2]*1.5)
    line, = plt.plot(mean_history,c = color)
    if not label == "":
        line.set_label(label)
        plt.legend()
    plt.plot(confidence_intervall_upper,c = boundcolor)
    plt.plot(confidence_intervall_lower,c = boundcolor)
    
def load_model(modelpath):
    model = Understandable_Embedder()
    model.load_weights(modelpath)
    return model

def evaluate_model_accuracy(modelpath, eval_ds, dataset_length):    
    model = Understandable_Embedder()
    if not modelpath == "":
        model.load_weights(modelpath)

    eval_ds = eval_ds.batch(batch_size)
    return model.evaluate(eval_ds, batch_size, dataset_length)

def evaluate_model_accuracy_project(modelpath, eval_ds, dataset_length, P):    
    model = Understandable_Embedder()
    if not modelpath == "":
        model.load_weights(modelpath)

    eval_ds = eval_ds.batch(batch_size)
    return model.eval_simple_project(eval_ds, batch_size, dataset_length,P)

def evaluate_average_model_accuracy(path,eval_ds, dataset_length, intervall):
    average = 0.0
    for i in range(intervall):
        average = average + evaluate_model_accuracy(path + str(i)+"/model", eval_ds, dataset_length)
    average = average / intervall
    return average
    
def evaluate_model_bias(modelpath, savepath, delete_dim_bool=False, pca_deb= False):

    
    # load the tf news dataset
    data_loader = DataLoader()#news_sent = data_loader.load_news_dataset()
    
    
    # professions and company names as target words
    prof = data_loader.read_target_words_from_json('exp_data/target_words/professions.json', index=0)
    comp = data_loader.read_target_words_from_json('exp_data/target_words/companies.json')
    sent_test = data_loader.read_target_words_from_json('exp_data/sentiment/test_words.json')
    
    # bias attributes for gender, sentiment and religion
    gender_attr = data_loader.read_bias_attributes_from_json('exp_data/gender/definitional_pairs.json')
    religion_attr = data_loader.read_bias_attributes_from_json('exp_data/religion/definitional_pairs.json')
    sentiment_attr = data_loader.read_bias_attributes_from_json('exp_data/sentiment/definitional_pairs.json')
    race_attr = data_loader.read_bias_attributes_from_json('exp_data/race/definitional_pairs.json')
    
    weats = {}
    for test in ['weat3', 'weat3b', 'weat4', 'weat5', 'weat5b', 'weat6', 'weat6b', 'weat7', 'weat7b', 'weat8', 'weat8b']:
        file = 'exp_data/seat/'+test+'.jsonl'
        print("read: "+file)
        weats.update({test: data_loader.read_dict_from_json(file)})
    weats.update({'angry_black_woman': data_loader.read_dict_from_json('exp_data/seat/angry_black_woman_stereotype.jsonl')})
    weats.update({'angry_black_woman2': data_loader.read_dict_from_json('exp_data/seat/angry_black_woman_stereotype_b.jsonl')})
    weats.update({'double_bind_competent': data_loader.read_dict_from_json('exp_data/seat/heilman_double_bind_competent_one_word.jsonl')})
    weats.update({'double_bind_likable': data_loader.read_dict_from_json('exp_data/seat/heilman_double_bind_likable_one_word.jsonl')})
    
    # read sentence templates and replacement rules
    template_dict = data_loader.read_dict_from_json('exp_data/templates.json')
    rules_dict = data_loader.read_dict_from_json('exp_data/template_rules.json')
   
    sent_generator = SentGenerator()

    print("test sentences (target words):")
    test_sent_sentiment = sent_generator.create_sent_from_template_dict(sent_test, template_dict, rules_dict, keys=['adjectives', 'things_singular', 'things_no_pronoun'])
    test_sent_prof = sent_generator.insert_to_templates(prof, template_dict['people_singular'], template_dict['token'])
    test_sent_comp = sent_generator.insert_to_templates(comp, template_dict['things_no_pronoun'], template_dict['token'])
    
    print("bias attribute sentences:")
    btest_sent_gender = sent_generator.create_attr_sent_from_template_dict(list(zip(*gender_attr)), template_dict, rules_dict)
    #print(btest_sent_gender)
    btest_sent_religion = sent_generator.create_attr_sent_from_template_dict(list(zip(*religion_attr)), template_dict, rules_dict)
    btest_sent_sentiment = sent_generator.create_attr_sent_from_template_dict(list(zip(*sentiment_attr)), template_dict, rules_dict)
    btest_sent_race = sent_generator.create_attr_sent_from_template_dict(list(zip(*race_attr)), template_dict, rules_dict)
    
    
    # create sentences for SEATs
    def create_seat_sent(weat_dict, template_dict, rules_dict):
        seat_dict = copy.deepcopy(weat_dict)
        for test in weat_dict.keys():
            for k in weat_dict[test].keys():
                category = weat_dict[test][k]['category']
                if 'Names' in category or category in ['Male', 'Female']:
                    seat_dict[test][k]['examples'] = sent_generator.insert_to_templates(weat_dict[test][k]['examples'], template_dict['names'], template_dict['token'])
                elif category in ['EuropeanAmericanTerms', 'AfricanAmericanTerms']:
                    seat_dict[test][k]['examples'] = sent_generator.insert_to_templates(weat_dict[test][k]['examples'], template_dict['people_description'], template_dict['token'])
                else:
                    seat_dict[test][k]['examples'] = sent_generator.create_sent_from_template_dict(weat_dict[test][k]['examples'], template_dict, rules_dict)
        return seat_dict
    
    seats = create_seat_sent(weats, template_dict, rules_dict)
    
    def embed_sent(embed_function):
        print("embed gender...")
        emb_gender = [embed_function(tup) for tup in btest_sent_gender]
        print("embed religion...")
        emb_religion = [embed_function(tup) for tup in btest_sent_religion]
        print("embed race...")
        emb_race = [embed_function(tup) for tup in btest_sent_race]
        print("embed sentiment...")
        emb_sentiment_tup = [embed_function(tup) for tup in btest_sent_sentiment]
    
        for k, v in seats.items():
            print("embed "+k+"...")
            for v2 in v.values():
                v2['embeddings'] = embed_function(v2['examples'])
        
        return {'gender': emb_gender, 'religion': emb_religion, 'race': emb_race, 'sentiment': emb_sentiment_tup}
    
    def cossim(x, y):
        return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))
    
    def mac(T, As):
        return np.mean([1-cossim(t,a) for t in T for A in As for a in A])
    
    def normalize(vectors: np.ndarray):
        norms = np.apply_along_axis(LA.norm, 1, vectors)
        vectors = vectors / norms[:, np.newaxis]
        return np.asarray(vectors)
    
    def delete_dim(x,dims=[0]):
        for dim in dims:
           # print(x.shape)
            x = np.concatenate((x[:,:dim],x[:,dim+1:]), axis = 1)
           # print(x.shape)
        return x
    
    def compare_metrics(seats,log_file): # embeddings
        f = open(log_file,"w")
        print("opened file", log_file)
        f.write("test;eff_weat;pval_weat;eff_own;own_bias_mean;own_std;mac;cluster\n")
        result_dict = {}
        for test in seats.keys():
            if not ('embeddings' in seats[test]['attr1'].keys() and 'embeddings' in seats[test]['targ1'].keys() and 'embeddings' in seats[test]['attr2'].keys() and 'embeddings' in seats[test]['targ2'].keys()):
                print("skip "+test)
                continue
                 
                
            # normalize embeddings
            seats[test]['attr1']['embeddings'] = normalize(seats[test]['attr1']['embeddings'])
            seats[test]['attr2']['embeddings'] = normalize(seats[test]['attr2']['embeddings'])
            seats[test]['targ1']['embeddings'] = normalize(seats[test]['targ1']['embeddings'])
            seats[test]['targ2']['embeddings'] = normalize(seats[test]['targ2']['embeddings'])
    
            print(test)
    
            pval, esize = weat.run_tests(seats[test])
            
            esize2, bias_mean, bias_std = weat.run_tests_own(seats[test])
    
            mac_score = mac(seats[test]['targ1']['embeddings']+seats[test]['targ2']['embeddings'], [seats[test]['attr1']['embeddings'],seats[test]['attr2']['embeddings']])
    
            X_bef = np.vstack([seats[test]['targ1']['embeddings'],seats[test]['targ2']['embeddings']])
            y = np.asarray([0]*len(seats[test]['targ1']['embeddings'])+[1]*len(seats[test]['targ2']['embeddings']))
          #  cluster_score = distance_tests.cluster_test(X_bef, y, num=2)
            
            f.write(test+"&"+str(esize)+"&"+str(pval)+";&"+str(esize2)+"&"+str(bias_mean)+"&"+str(bias_std)+"&"+str(mac_score)+"\\"+"\n")
            
            test_list = []
            test_list.append(esize)
            test_list.append(esize2)
            test_list.append(mac_score)
            test_list.append(pval)
        
            
            result_dict[test] = test_list
            # get single word biases
            #W = np.vstack([weats[test]['targ1']['embeddings'],weats[test]['targ2']['embeddings']])
            #word_biases = [weat.own_metric.bias_w(w, A) for w in W]
    
        # TODO run other tests
            
        f.close()
        
        return result_dict
        
    def delete_dim_seats(seats,dims=[0]):
        for test in seats.keys():
            if not ('embeddings' in seats[test]['attr1'].keys() and 'embeddings' in seats[test]['targ1'].keys() and 'embeddings' in seats[test]['attr2'].keys() and 'embeddings' in seats[test]['targ2'].keys()):
                print("skip "+test)
                continue
                
            seats[test]['attr1']['embeddings'] = delete_dim(seats[test]['attr1']['embeddings'],dims)
            seats[test]['attr2']['embeddings'] = delete_dim(seats[test]['attr2']['embeddings'],dims)
            seats[test]['targ1']['embeddings'] = delete_dim(seats[test]['targ1']['embeddings'],dims)
            seats[test]['targ2']['embeddings'] = delete_dim(seats[test]['targ2']['embeddings'],dims)  
            
        return seats
        
        
    weat_metrics = WEATMetrics()
    own_metrics = ImprovedBiasMetrics()
    weat = WEAT(weat_metrics, own_metrics)

    model = Understandable_Embedder()
    if not modelpath == "original":
        model.load_weights(modelpath)
    #model.call_headless(inputs)
    
  #  print(model.predict_simple(["hallo du da", "ich bin hier"]))
    def emb_func(input):
      x =   model.predict_simple(input)
      return x
    
    if pca_deb:
        from SentDebias import Sent_Debias
        Sent_Deb = Sent_Debias()
        Sent_Deb.load()
        def emb_func(input):
            x =   model.predict_simple(input)
            x = Sent_Deb.predict(x)
            return x
    emb = embed_sent(emb_func)
  #  print(emb)
    
   # print(delete_dim)
    if delete_dim_bool:
        seats = delete_dim_seats(seats)
    
    result_dict = compare_metrics(seats,savepath)
    
    
    # 3-5b european american vs african american pleasant vs unpleasant
    # 6-8b  gender
    # angry black woman racial
    # double bind gender
    return result_dict

def word_correlation(word_pair, modelpath, dim = 0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)   
    dataset = get_understanding_set(word_pair,tokenizer, max_examples = 1000)
    
    model = Understandable_Embedder()
    model.load_weights(modelpath)
    
    batch_size = 4
    steps = round(len(dataset[0][0]["input_ids"]) / batch_size)
    
    for i in range(0,steps):
        feed_dict_1 = {}
        feed_dict_1["input_ids"] = dataset[0][0]["input_ids"][i*batch_size:(i+1)*batch_size]
        feed_dict_1["token_type_ids"] = dataset[0][0]["token_type_ids"][i*batch_size:(i+1)*batch_size]
        feed_dict_1["attention_mask"] = dataset[0][0]["attention_mask"][i*batch_size:(i+1)*batch_size]
        
        feed_dict_2 = {}
        feed_dict_2["input_ids"] = dataset[0][1]["input_ids"][i*batch_size:(i+1)*batch_size]
        feed_dict_2["token_type_ids"] = dataset[0][1]["token_type_ids"][i*batch_size:(i+1)*batch_size]
        feed_dict_2["attention_mask"] = dataset[0][1]["attention_mask"][i*batch_size:(i+1)*batch_size]
        
        if i == 0 :
            original_emb = model.call_headless(feed_dict_1)
            changed_emb = model.call_headless(feed_dict_2)
        original_emb = np.concatenate((original_emb,model.call_headless(feed_dict_1)))
        changed_emb = np.concatenate((changed_emb,model.call_headless(feed_dict_2)))
    
    
#     print(original_emb[:,dim])
#     print(changed_emb[:,dim])
#     print(original_emb[:,dim]- changed_emb[:,dim])
    change = np.mean(np.abs(original_emb[:,dim]- changed_emb[:,dim]))
    change_other = np.sum(np.mean(np.abs(original_emb[:,dim:]- changed_emb[:,dim:]), axis = 0), axis = 0)
    
    print("change:",change)
    print("change in other dimension:",change_other)
    print("percent in right dim:", change/(change+change_other))
    
def evaluate_model_set(model_path, intervall_range, save_path = "results/finetuned.txt", delete_dim = False):
    
    if intervall_range == 0:
        result_dict = {}
        dict = evaluate_model_bias(model_path +"/model", "results/finetuned.txt", delete_dim_bool= delete_dim)
        for test in dict:
            if not test in result_dict:
                result_dict[test] =[]
            result_dict[test].append(dict[test])
    else:
        result_dict = {}
        for i in range(intervall_range):
            dict = evaluate_model_bias(model_path+str(i) +"/model", "results/finetuned.txt", delete_dim_bool= delete_dim)
            for test in dict:
                if not test in result_dict:
                    result_dict[test] =[]
                result_dict[test].append(dict[test])
            
    log_file = save_path
    f = open(log_file,"w")
    print("opened file", log_file)
    f.write("test & weat & own & mac\\\n")
    
    average_dict = {}
    for test in result_dict:
        result_dict[test] = np.asarray(result_dict[test])
        variance = np.var(result_dict[test], axis = 0 )
        average_dict[test + "intervall"] = 1.96*np.sqrt(variance)/np.sqrt(result_dict[test].shape[0])
        average_dict[test] = np.mean(np.abs(result_dict[test]), axis = 0)
    #    print(test, average_dict[test], average_dict[test + "intervall"] )
        f.write(test+" & "+str(round(average_dict[test][0], 2))+ " +/- " + str(round(average_dict[test + "intervall"][0], 2)) +" & "
                +str(round(average_dict[test][1],2))+ " +/- " + str(round(average_dict[test + "intervall"][1], 2)) +" & "
                +str(round(average_dict[test][2],2))+ " +/- " + str(round(average_dict[test + "intervall"][2], 2)) +" & "
                +str(round(average_dict[test][3],2))+ " +/- " + str(round(average_dict[test + "intervall"][3], 2)) +"\\"+"\n")
        

    
  
        
    race_test = ["weat3", "weat3b", "weat4", "weat5", "weat5b", "angry_black_woman", "angry_black_woman2"]
    gender_test = ["weat6", "weat6b", "weat7", "weat7b", "weat8b", "weat8"]#, "double_bind_competent", "double_bind_likable"]
    
    race_list =[]
    for test in race_test:
        race_list.append(average_dict[test])
    race_score = np.mean(race_list,axis = 0)
    
    gender_list =[]
    for test in gender_test:
        gender_list.append(average_dict[test])
    gender_score = np.mean(gender_list,axis = 0)
    
    f.write("gender_score: "+ str(gender_score) + "\n")
    f.write("race_score: "+ str(race_score) + "\n")
    
    
    f.close()  
    
def generate_sentences(words):
    templates =[
        "I am a ",
        "I work as a ",
        "My occupation is ",
        "My dream job is ",
        "works as a ",
        "wants to become a ",
        "Overall i am quite happy with being a ",
        "I am depressed working as a ",
        "In conclusion i am neutral towards my job as a ",
        "I am working as a ",
        "This is my job: ",
        "I am conflicted about being a "
        ]
    sentences =[]
    for word in words:
        new_sentences = []
        for template in templates:
            new_sentences.append(template + word)
        sentences.append(new_sentences)
    return sentences

def plot_tsne(model):
    import json
#     with open("professions.json") as f:
#         list = json.load(f)
#     
#     x = []
#     y = []
#     for element in list:
#         x.append(element[0])
#         y.append(element[2])    
    # x = ["women", "man", "boy", "girl"]
    # y = [-1,1,1,-1]
    # y = np.asarray(y)
    # y = y*0.5+0.5
    # X = generate_sentences(x)  
    # new_X = []
    # for batch in X:
    #     new_X.append(model.predict_simple(batch))
    # X = np.asarray(new_X)
    # X = np.mean(X, axis = 1)
#     y = np.asarray(y)
#     y = y*0.5+0.5 #scale to 0-1
    #X = generate_sentences(x)  
    x = [" women ", " girl ", " female ", " she ", " actress ", " heroine ", " queen "," sister ", " mother ", " lady ", " her " ," men ", " boy ", " male ", " he ", " actor ", " hero ", " king ", " brother ", " father ", " gentleman ", " him "]
    y = [0]*11
    y2 = [1]*11
    y = y + y2
    X = model.predict_simple(x)
    print(X)
#     
#     new_X = []
#     for batch in X:
#         new_X.append(model.predict_simple(batch))
#     X = np.asarray(new_X)
#     X = np.mean(X, axis = 1)

    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2).fit_transform(X.astype(np.float64))
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = y, cmap = plt.get_cmap("winter"))
    plt.show()
    return
    
def gender_bias_test(model,pca_deb=False, null = False):
    import json
    with open("professions.json") as f:
        list = json.load(f)
    
    x = []
    y = []
    for element in list:
        x.append(element[0])
        y.append(element[2])
    


    X = generate_sentences(x)  

    new_y = []
    new_x =[]
    for i,batch in enumerate(X):
        for i in range(len(batch)):
            new_y.append(y[i])
        for sentence in batch:
            new_x.append(sentence)
    
    y = new_y
    y = np.asarray(y)
    y = y*0.5+0.5
    X = new_x
    
    print("base classifier: mse loss: " , np.var(y))
    
#     X_test = generate_sentences(X_test)  
#     new_y = []
#     new_x = []
#     for i,batch in enumerate(X_test):
#         for i in range(len(batch)):
#             new_y.append(y_test[i])
#         for sentence in batch:
#             new_x.append(sentence)
#     
#     y_test = new_y
#     y_test = np.asarray(y_test)
#     y_test = y_test*0.5+0.5
#     X_test = new_x
    
    X = model.predict_simple(X)

    if null:
   #     y_class = np.asarray([np.round(label) for label in y])
        P = nullspace(model)
        X = np.transpose(np.matmul(P,np.transpose(X)))
 #   X_test = model.predict_simple(X_test)
  #  print(len(y))
    
    if pca_deb:
        from SentDebias import Sent_Debias
        Sent_Deb = Sent_Debias()
        Sent_Deb.load()
        X = Sent_Deb.predict(X)
     #   X_test = Sent_Deb.predict(X_test)
    
    from sklearn.model_selection import train_test_split        
    X, X_test,y, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

    from tensorflow import keras
    simple_model = keras.Sequential(
    [
        keras.Input(shape=X[0].shape),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    simple_model.compile(optimizer =tf.keras.optimizers.SGD() ,loss = tf.keras.losses.MeanSquaredError())
    simple_model.fit(X,y, batch_size = 8, epochs = 50, validation_data = (X_test,y_test), verbose = 0)#, callbacks = keras.callbacks.EarlyStopping())
    simple_result = simple_model.evaluate(X_test,y_test)
    print("linear model:",simple_result)
    
    
    
    medium_model = keras.Sequential(
    [
        keras.Input(shape=X[0].shape),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    medium_model.compile(optimizer =tf.keras.optimizers.SGD() ,loss = tf.keras.losses.MeanSquaredError())
    medium_model.fit(X,y, batch_size = 8, epochs = 50, validation_data = (X_test,y_test), verbose = 0)#, callbacks = keras.callbacks.EarlyStopping())
    medium_result = medium_model.evaluate(X_test,y_test)
    print("not as simple model:",medium_result)
    
    return simple_result, medium_result
    
    
def gender_bias_test_classification(model,pca_deb=False):
    import json
    with open("professions.json") as f:
        list = json.load(f)
    
    x = []
    y = []
    for element in list:
        x.append(element[0])
        y.append(element[2])
    


    X = generate_sentences(x)  

    new_y = []
    new_x =[]
    for i,batch in enumerate(X):
        for i in range(len(batch)):
            new_y.append(y[i])
        for sentence in batch:
            new_x.append(sentence)
    
    y = new_y
    y = np.asarray(y)
    y = y*0.5+0.5
    y = np.asarray([np.round(label) for label in y])
    
    X = new_x
    
    print("base classifier: mse loss: " , np.var(y))
    
#     X_test = generate_sentences(X_test)  
#     new_y = []
#     new_x = []
#     for i,batch in enumerate(X_test):
#         for i in range(len(batch)):
#             new_y.append(y_test[i])
#         for sentence in batch:
#             new_x.append(sentence)
#     
#     y_test = new_y
#     y_test = np.asarray(y_test)
#     y_test = y_test*0.5+0.5
#     X_test = new_x
    
    X = model.predict_simple(X)
 #   X_test = model.predict_simple(X_test)
  #  print(len(y))
    
    if pca_deb:
        from SentDebias import Sent_Debias
        Sent_Deb = Sent_Debias()
        bias_words = ["girl", "boy"]
        Sent_Deb.fit(model,bias_words ,k= 3, max_words = 1000)
        X = Sent_Deb.predict(X)
     #   X_test = Sent_Deb.predict(X_test)
    
    from sklearn.model_selection import train_test_split        
    X, X_test,y, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    
    print(X_test.shape)
    print(y_test.shape)

    from tensorflow import keras
    simple_model = keras.Sequential(
    [
        keras.Input(shape=X[0].shape),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    simple_model.compile(optimizer =tf.keras.optimizers.SGD() ,loss = tf.keras.losses.BinaryCrossentropy())
    simple_model.fit(X,y, batch_size = 8, epochs = 50, validation_data = (X_test,y_test))#, callbacks = keras.callbacks.EarlyStopping())
    print("linear model:",simple_model.evaluate(X_test,y_test))
    
    
    # --- Deep View Parameters ----#
    from deepview import DeepView
    pred_wrapper = DeepView.create_simple_wrapper(simple_model.predict)
    
    classes = np.arange(2)
    batch_size = 32
    max_samples = 500
    data_shape = (768,)
    resolution = 100
    N = 10
    lam = 0.64
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'linear model - occupations'
     
    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape, 
        N, lam, resolution, cmap, interactive, title)
     
    deepview.add_samples(X_test[:200], y_test[:200])
    deepview.show()
    
    
    
    medium_model = keras.Sequential(
    [
        keras.Input(shape=X[0].shape),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    medium_model.compile(optimizer =tf.keras.optimizers.SGD() ,loss = tf.keras.losses.BinaryCrossentropy())
    medium_model.fit(X,y, batch_size = 8, epochs = 50, validation_data = (X_test,y_test))#, callbacks = keras.callbacks.EarlyStopping())
    print("not as simple model:",medium_model.evaluate(X_test,y_test))
    
    pred_wrapper = DeepView.create_simple_wrapper(medium_model.predict)
    
    classes = np.arange(2)
    batch_size = 32
    max_samples = 500
    data_shape = (768,)
    resolution = 100
    N = 10
    lam = 0.64
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'nonlinear model - occupations'
     
    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape, 
        N, lam, resolution, cmap, interactive, title)
     
    deepview.add_samples(X_test[:200], y_test[:200])
    deepview.show()

from nullspaceproject import get_debiasing_projection
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
def nullspace(model):
    import json
    with open("professions.json") as f:
        list = json.load(f)
    
    x = []
    y = []
    for element in list:
        x.append(element[0])
        y.append(element[2])
    


    X = generate_sentences(x)  

    new_y = []
    new_x =[]
    for i,batch in enumerate(X):
        for i in range(len(batch)):
            new_y.append(y[i])
        for sentence in batch:
            new_x.append(sentence)
    
    y = new_y
    y = np.asarray(y)
    y = y*0.5+0.5
    X = new_x
    
    
#     X_test = generate_sentences(X_test)  
#     new_y = []
#     new_x = []
#     for i,batch in enumerate(X_test):
#         for i in range(len(batch)):
#             new_y.append(y_test[i])
#         for sentence in batch:
#             new_x.append(sentence)
#     
#     y_test = new_y
#     y_test = np.asarray(y_test)
#     y_test = y_test*0.5+0.5
#     X_test = new_x
    
    X = model.predict_simple(X)

    y = np.asarray([np.round(label) for label in y])

    num_classifiers = 20
    classifier_class = SGDClassifier #Perceptron
    input_dim = 768
    is_autoregressive = True
    min_accuracy = 0.0

    P, rowspace_projections, Ws = get_debiasing_projection(classifier_class, {}, num_classifiers, input_dim, is_autoregressive, min_accuracy, X, y, X, y, by_class = False)

    return P
    
if __name__ == "__main__":

    #plot_average_history("results/", 5)
    

    
    #plot_history("race_modelmrpc")
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    import tensorflow_datasets as tfds
    from transformers import BertTokenizer, glue_convert_examples_to_features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    batch_size = 4
    
    plot_tsne(model = Understandable_Embedder())
   # plot_average_history("results/", 2)
    
#     model = Understandable_Embedder()
#     tokenized_inputs = tokenizer(["hallo du da", "ich bin hier"], max_length=128, padding=True, truncation=True, return_tensors='tf')
#     print(model.call_pre_training(tokenized_inputs))
#      
#    evaluate_model_bias("results/pre_gender_5000/train1/model", savepath= "bertpre.txt" )
#     evaluate_model_set("results/Genderlarge/qnli_gender_large_", 5, save_path = "qnlifine.txt" )
#     evaluate_model_set("results/Genderlarge/sst2_gender_large_", 5, save_path = "sst2fine.txt" )
# #     evaluate_model_set("results/Genderlarge/cola_gender_large_", 5, save_path = "colafine.txt" )
#     model = Understandable_Embedder()
# #     # gender_bias_test(model = model, pca_deb= True)
#     plot_tsne(model)
#     model = load_model("results/qnli_prefine_gender_0/model")
#     plot_tsne(model)




    task = "cola"  
    if task == "sst2":
        task_name = "sst-2"
    else:
        task_name = task
    data = tfds.load('glue/'+task)
    dataset_length = data['validation'].cardinality().numpy()
    dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128,  task=task_name)
  #  acc = evaluate_model_accuracy("results/colafine/train3/model", dataset, dataset_length)
   # P = np.identity(768)
    
    accs =[]
    lin_biases = []
    nonlin_biases = []
    for i in range(1):
        model_load_path = "results/"+task+"fine/train"+str(i)+ "/model"
        model = load_model(model_load_path)
       # model = Understandable_Embedder()
        P = nullspace(model)
        acc = evaluate_model_accuracy_project(model_load_path, dataset, dataset_length, P)
        accs.append(acc)
        lin_bias, nonlin_bias = gender_bias_test(model = model, null= True)
        lin_biases.append(lin_bias)
        print("lin_bias", lin_bias)
        nonlin_biases.append(nonlin_bias)
        print("nonlin_bias", nonlin_bias)
        print("acc", acc)

    def printmeanvar(list):
        mean = np.mean(list)
        std = np.std(list)
        print("mean", mean)
        print("std", std)
    print("accs")
    printmeanvar(accs)
    print("lin_biases")
    printmeanvar(lin_biases)
    print("nonlin_biases")
    printmeanvar(nonlin_biases)
    print("extra")
#    P = nullspace(model)
  #  P = np.identity(768)
   # acc = model.eval_simple_project(dataset, P, dataset_length)
  #  print(acc)
  #  
#     model = load_model("results/Genderlarge/cola_gender_large_0/model")
#     gender_bias_test_classification(model = model)
#     model = load_model("results/Genderlarge/cola_gender_large_1/model")
#     gender_bias_test_classification(model = model)

#     model = load_model("results/cola_prefine_gender_0/model")
#     gender_bias_test_classification(model = model, pca_deb= True)
#     model = load_model("results/without/cola_normal_1/model")
#     gender_bias_test_classification(model = model)
#     model = load_model("results/Genderlarge/sst2_gender_large_0/model")
#     gender_bias_test_classification(model = model)
#     model = load_model("results/Genderlarge/sst2_gender_large_1/model")
#     gender_bias_test_classification(model = model)
   # print("not pre trained:")
#     model = Understandable_Embedder()
#     gender_bias_test(model = model, pca_deb = True)
    
#     model = load_model("results/pre_gender_50000/model")
#     lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#     
#     

#     f =open('results_debiasfreq_search.txt', 'w')
#   
#     num = 4
#     overalllin = []
#     overallnonlin = []
#     for a in range(7):
#         linavg = []
#         nonlinavg = []
#         for i in range(5):
#             model = load_model("results/qnliDebias_freq/train"+str(i) +"_"+ str(2**a)+ "/model")
#             lin,nonlin = gender_bias_test(model = model, pca_deb = False)
#             linavg.append(lin)
#             nonlinavg.append(nonlin)
#         
#         overalllin.append(linavg)   
#         overallnonlin.append(nonlinavg)    
#         print("ratio ", 2**a, file = f)
#         print("linavg mean", np.mean(linavg), file = f)
#         print("nonlinavg mean", np.mean(nonlinavg), file = f)
#         print("linavg var", np.var(linavg), file = f)
#         print("nonlinavg var", np.var(nonlinavg), file = f)
#     
#     f.close()
#     with open('linandnonlin.npy', 'wb') as f:
#         np.save(f, overalllin)
#         np.save(f, overallnonlin)
#     linavg = 0
#     nonlinavg = 0
#     for i in range(num):
#         model = load_model("results/sst2prefine_debfreq1/train" + str(i)+ "/model") 
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/num, file = f)
#     print("nonlinavg", nonlinavg/num, file = f)
#   
#     linavg = 0
#     nonlinavg = 0
#     for i in range(num):
#         model = load_model("results/qnliprefine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/num, file = f)
#     print("nonlinavg", nonlinavg/num, file = f)
#  
#     linavg = 0
#     nonlinavg = 0
#     for i in range(num):
#         model = load_model("results/colapre_lessfine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/num, file = f)
#     print("nonlinavg", nonlinavg/num, file = f)
#       
#     linavg = 0
#     nonlinavg = 0
#     for i in range(num):
#         model = load_model("results/sst2pre_lessfine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/num, file = f)
#     print("nonlinavg", nonlinavg/num, file = f)
#      
#     linavg = 0
#     nonlinavg = 0
#     for i in range(num):
#         model = load_model("results/qnlipre_lessfine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/num, file = f)
#     print("nonlinavg", nonlinavg/num, file = f)
     

     
#     for i in range(5):
#         model = load_model("results/Genderlarge/sst2_gender_large_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/5.0)
#     print("nonlinavg", nonlinavg/5.0)
#     
#     
#     for i in range(5):
#         model = load_model("results/Genderlarge/qnli_gender_large_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         linavg += lin
#         nonlinavg += nonlin
#     print("linavg", linavg/5.0)
#     print("nonlinavg", nonlinavg/5.0)
    
    
#     task = "cola"  
#     data = tfds.load('glue/'+task)
#     dataset_length = data['test'].cardinality().numpy()
#     dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length=128,  task=task)
#              
#     gender_acc = evaluate_average_model_accuracy("results/cola_prefine_gender_", dataset, dataset_length, 2)
#     print("gender_acc cola:", gender_acc)
#     
#     gender_acc = evaluate_average_model_accuracy("results/cola_prefine_without_gender_", dataset, dataset_length, 2)
#     print("gender_acc cola:", gender_acc)
     
     
#     task = "sst2"  
#     data = tfds.load('glue/'+task)
#     task = "sst-2" 
#     dataset_length = data['validation'].cardinality().numpy()
#     dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128,  task=task)
#     gender_acc = evaluate_average_model_accuracy("results/sst2_prefine_without_gender_", dataset, dataset_length, 2)
#     print("gender_acc sst2:", gender_acc)
#     gender_acc = evaluate_average_model_accuracy("results/sst2_prefine_gender_", dataset, dataset_length, 2)
#     print("gender_acc sst2:", gender_acc)
#          

#     f =open('results.txt', 'w ')

#     dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128,  task=task)
#     gender_acc = evaluate_average_model_accuracy("results/qnli_prefine_gender_", dataset, dataset_length, 5)
#     print("gender_acc qnli:", gender_acc, file = f)
#     f =open('resultscolabias.txt', 'w')
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/colaBaselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/colaBaselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert Sent",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/colaprefine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert prefine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/Genderlarge/cola_gender_large_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert fine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#   
#     f.close()
#     
#     f =open('resultssst2bias.txt', 'w')
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/sst2Baselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/sst2Baselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert Sent",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/sst2prefine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert prefine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/Genderlarge/sst2_gender_large_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert fine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#   
#     f.close()
#     
#     f =open('resultsqnlibias.txt', 'w')
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/qnliBaselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/qnliBaselines/_baseline_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model, pca_deb = True)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert Sent",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/qnliprefine_debfreq1/train" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert prefine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#     avglin = []
#     avgnonlin = []
#     for i in range(5):
#         model = load_model("results/Genderlarge/qnli_gender_large_" + str(i)+ "/model")
#         lin,nonlin = gender_bias_test(model = model)
#         avglin.append(lin)
#         avgnonlin.append(nonlin)
#       
#     print("Original Bert fine",  file = f)    
#     print("gender_acc lin qnli:", np.mean(avglin), file = f)    
#     print("gender_acc std qnli:", np.std(avglin), file = f)   
#     print("gender_acc nonlin qnli:", np.mean(avgnonlin), file = f)    
#     print("gender_acc std qnli:", np.std(avgnonlin), file = f)  
#     
#   
#     f.close()
#     

#        


#      
# #     word_pair=[[[" man "],[" woman "]]]
# #     word_correlation(word_pair, modelpath="results/5Epochs_mae/gender_con_0/model", dim = 0)    
#     print("race_acc:", race_acc)
  #  print("acc:", acc)
 #   print(evaluate_model_accuracy("results/gender_con_0/model", dataset,dataset_length))
    
    
#     evaluate_model_accuracy("race_modelmrpc/understandable/model", dataset)
#     evaluate_model_accuracy("gender_contrastive/understandable/model", dataset)
#     evaluate_model_accuracy("race_contrastive/understandable/model", dataset)
#     evaluate_model_accuracy("race_modelmrpc/normal/model", dataset)
#     evaluate_model_accuracy("normal_5_epochs/normal/model", dataset)

#     evaluate_model_set("results/gender_con_",1,"results/gender_con_mae.txt",delete_dim=False)
#     evaluate_model_set("results/gender_con_",1,"results/gender_con_mae_del.txt",delete_dim=True)
#     evaluate_model_set("results/race_con_",1,"results/race_con_mae.txt",delete_dim=False)
#     evaluate_model_set("results/race_con_",1,"results/race_con_mae_del.txt",delete_dim=True)s
#     evaluate_model_set("results/race_con_",5,"results/race_con_del.txt",delete_dim=True)
  #  evaluate_model_set("results/Genderlarge/cola_gender_large_",5,"results/Genderlargecola_gender_large.txt",delete_dim=False)
  #  evaluate_model_set("results/SimpleLoss/sst2_gender_",5,"results/SimpleLoss/sst2_gender_.txt",delete_dim=False)
  #  evaluate_model_set("results/without/qnli_normal_",5,"results/without/qnli_normal_.txt",delete_dim=False)
   # evaluate_model_set("results/mrpc_genderlarge_con_",3,"results/mrpc_genderlarge_con_.txt",delete_dim=False)

        
        
  #  evaluate_model_bias("results/pre_gender_/model","results/pre_gender.txt")
#     evaluate_model_bias("race_modelmrpc/understandable/model","results/race_bert_finetuned_understandable_deleted_dim.txt", True)
    #plot_history("modelmrpc")