from src.models.train_lgbm import train_lgbm_kfold, train_lgbm
from src.models.train_knn import train_knn_kfold, train_knn
from src.models.train_mlp import train_mlp_kfold, train_mlp
from src.models.train_logres import train_logistic_kfold, train_logistic
from src.models.train_rf import train_rf_kfold, train_rf
from src.models.train_catboost import train_catboost_kfold, train_catboost


def train_kfold():
    # train_mlp_kfold()
    # train_knn_kfold()
    # train_lgbm_kfold()
    # train_logistic_kfold()
    # train_rf_kfold()
    train_catboost_kfold()

def train_all_models():
    train_mlp()
    train_knn()
    train_lgbm()
    train_logistic()
    train_rf()
    train_catboost()

if __name__ == '__main__':
    train_all_models()