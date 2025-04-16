from ML_models import isolation_forest
from ML_models import lstm 
from ML_models import svm
from ML_models import XGBoost
def get_model(model):
    match model:
        case "lstm":
            lstm_instance = lstm.LSTMModel()
            return lstm_instance
            
        case "isolation_forest":
            if_instance = isolation_forest.IsolationForestModel()
            return if_instance
            
        case "svm":
            svm_instance = svm.SVMModel()
            return svm_instance
        
        case "XGBoost":
            XGBoost_instance = XGBoost.XGBoostModel()
            return XGBoost_instance
            
    