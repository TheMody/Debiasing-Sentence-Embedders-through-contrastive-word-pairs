
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
    history_list_gender=[]
    for i in range(intervall):
        with open(path +"gender_con_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
            understandable_history = pickle.load(fp)
            history_list_gender.append(understandable_history)
    generate_error_bound_plot(history_list_gender)
    
    history_list_race =[]
    for i in range(intervall):
        with open(path +"race_con_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
            understandable_history = pickle.load(fp)
            history_list_race.append(understandable_history)
    generate_error_bound_plot(history_list_race, color=(0.1,0.6,0.1))
    
    history_list_normal=[]
    for i in range(intervall):
        with open(path +"normal_" + str(i) + "/history.txt", "rb") as fp:   # Unpickling
            understandable_history = pickle.load(fp)
            history_list_normal.append(understandable_history)        
    generate_error_bound_plot(history_list_normal, color=(0.6,0.1,0.1))
    
    plt.savefig("average_history.png")
    
def generate_error_bound_plot(history_list, color=(0.1,0.1,0.6)):
    history_matrix=np.zeros((len(history_list),len(history_list[0]["loss"])))
    for i,history in enumerate(history_list):
        history_matrix[i,:]=history["loss"]
    mean_history = np.mean(history_matrix,axis=0)
    var_history = np.var(history_matrix,axis=0)
    confidence_intervall = 1.96*np.sqrt(var_history)/np.sqrt(len(history_list))
    confidence_intervall_upper = mean_history + confidence_intervall
    confidence_intervall_lower =   mean_history - confidence_intervall
    
    boundcolor = (color[0]*1.5,color[1]*1.5,color[2]*1.5)
    plt.plot(mean_history,c = color)
    plt.plot(confidence_intervall_upper,c = boundcolor)
    plt.plot(confidence_intervall_lower,c = boundcolor)
    

    
def evaluate_model_accuracy(modelpath, eval_ds, dataset_length):    
    model = Understandable_Embedder()
    model.load_weights(modelpath)

    eval_ds = eval_ds.batch(batch_size)
    return model.evaluate(eval_ds, batch_size, dataset_length)

def evaluate_average_model_accuracy(path,eval_ds, dataset_length, intervall):
    average = 0.0
    for i in range(intervall):
        average = average + evaluate_model_accuracy(path + str(i)+"/model", eval_ds, dataset_length)
    average = average / intervall
    return average
    
def evaluate_model_bias(modelpath, savepath, delete_dim_bool=False):

    
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
    model.load_weights(modelpath)
    #model.call_headless(inputs)
    
  #  print(model.predict_simple(["hallo du da", "ich bin hier"]))
    
    emb = embed_sent(model.predict_simple)
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
        f.write(test+"&"+str(round(average_dict[test][0], 2))+ " +/- " + str(round(average_dict[test + "intervall"][0], 2)) +"&"
                +str(round(average_dict[test][1],2))+ " +/- " + str(round(average_dict[test + "intervall"][1], 2)) +"&"
                +str(round(average_dict[test][2],2))+ " +/- " + str(round(average_dict[test + "intervall"][2], 2)) +"\\"+"\n")
        

    
  
        
    race_test = ["weat3", "weat3b", "weat4", "weat5", "weat5b", "angry_black_woman", "angry_black_woman2"]
    gender_test = ["weat6", "weat6b", "weat7", "weat7b", "weat8b", "weat8", "double_bind_competent", "double_bind_likable"]
    
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
    
if __name__ == "__main__":
    #plot_average_history("results/", 5)
    
    
    #plot_history("race_modelmrpc")
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    import tensorflow_datasets as tfds
    from transformers import BertTokenizer, glue_convert_examples_to_features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', output_hidden_states=True)
    batch_size = 4
#     task = "cola"  
#     data = tfds.load('glue/'+task)
#     dataset_length = data['test'].cardinality().numpy()
#     dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length=128,  task=task)
#          
#     gender_acc = evaluate_average_model_accuracy("results/without/cola_normal_", dataset, dataset_length, 5)
#     print("gender_acc cola:", gender_acc)
#     task = "sst2"  
#     data = tfds.load('glue/'+task)
#     task = "sst-2" 
#     dataset_length = data['validation'].cardinality().numpy()
#     dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128,  task=task)
#     gender_acc = evaluate_average_model_accuracy("results/SimpleLoss/sst2_gender_", dataset, dataset_length, 5)
#     print("gender_acc sst2:", gender_acc)
#        
#     task = "qnli"  
#     data = tfds.load('glue/'+task)
#     dataset_length = data['validation'].cardinality().numpy()
#     dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128,  task=task)
#     gender_acc = evaluate_average_model_accuracy("results/SimpleLoss/qnli_gender_", dataset, dataset_length, 5)
#     print("gender_acc qnli:", gender_acc)

#        


#      
# #     word_pair=[[[" man "],[" woman "]]]
# #     word_correlation(word_pair, modelpath="results/5Epochs_mae/gender_con_0/model", dim = 0)    
#     print("race_acc:", race_acc)
  #  print("acc:", acc)
 #   print(evaluate_model_accuracy("results/gender_con_0/model", dataset,dataset_length))
    
  #  evaluate_model_accuracy("race_only_dense/understandable/model", dataset)
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
    evaluate_model_set("results/gender_",5,"results/gender_.txt",delete_dim=False)
  #  evaluate_model_set("results/SimpleLoss/sst2_gender_",5,"results/SimpleLoss/sst2_gender_.txt",delete_dim=False)
  #  evaluate_model_set("results/without/qnli_normal_",5,"results/without/qnli_normal_.txt",delete_dim=False)
   # evaluate_model_set("results/mrpc_genderlarge_con_",3,"results/mrpc_genderlarge_con_.txt",delete_dim=False)

        
        
    #evaluate_model_bias("results/gender_con_0/model","results/race_higher_loss.txt")
#     evaluate_model_bias("race_modelmrpc/understandable/model","results/race_bert_finetuned_understandable_deleted_dim.txt", True)
    #plot_history("modelmrpc")