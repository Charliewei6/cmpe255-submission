from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

class SVM_classifier:
    def __init__(self):
        print('data loding...')
    
    def load_data(self):
        faces = fetch_lfw_people(min_faces_per_person=60)
        print('data loaded')
        print(faces.target_names)
        target_names = faces.target_names

        return faces,target_names
    
    def split_data(self,faces):
        X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    #Use a grid search cross-validation to explore combinations of parameters to determine the best model
    def grid_search_model(self):
        comb_param = {'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005]}
        pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
        svc = SVC(kernel='rbf', class_weight='balanced')
        svc = GridSearchCV(svc, comb_param)
        model = make_pipeline(pca, svc)
        return model
    
    def cal_score(self,model,X_train, X_test, y_train, y_test,target_names):
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(y_pred)
        result = metrics.classification_report(y_test, y_pred,target_names=target_names)
        return result,y_pred

if __name__ == "__main__":
    SVM = SVM_classifier()
    faces, target_names= SVM.load_data()
    X_train, X_test, y_train, y_test = SVM.split_data(faces)
    model = SVM.grid_search_model()
    result, y_pred = SVM.cal_score(model,X_train, X_test, y_train, y_test,target_names)
    print(result)

    plt.figure()
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(24):
        plt.subplot(4, 6, i + 1)
        plt.imshow(X_test[i].reshape(faces.images[0].shape), cmap=plt.cm.gray)
        color = ('black' if y_pred[i] == y_test[i] else 'red')
        plt.title(faces.target_names[y_pred[i]],fontsize='medium',color=color)
        plt.xticks(())
        plt.yticks(())
    plt.show()

    plt.figure()
    sns.heatmap(metrics.confusion_matrix(y_pred, y_test))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()


    
