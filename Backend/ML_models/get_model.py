import isolation_forest
import lstm 
import svm

def get_model(model):
    match model:
        case "lstm":
            lstm_instance = lstm.LSTMModel()
            return lstm_instance
            
        case "isolation_forest":
            if_instance = isolation_forest.IsolationForest()
            return if_instance
            
        case "svm":
            svm_instance = svm.SVM()
            return svm_instance
            
    