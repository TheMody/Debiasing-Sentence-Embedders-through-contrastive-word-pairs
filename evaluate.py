
import pickle
import matplotlib.pyplot as plt

def plot_history(path):
    with open(path + "/understandable_history.txt", "rb") as fp:   # Unpickling
        understandable_history = pickle.load(fp)
    with open(path +"/normal_history.txt", "rb") as fp:   # Unpickling
        normal_history = pickle.load(fp)
    plt.plot(understandable_history["loss_compare"], label = "loss_compare")
    plt.plot(understandable_history["loss"], label = "loss")
    plt.plot(normal_history['loss'], label = "loss_original")
   # plt.plot(normal_history['sparse_categorical_accuracy'], label = "accuracy")
    plt.show()
    
def evaluate_models(path):
    return
    
if __name__ == "__main__":
    plot_history("modelmrpc")