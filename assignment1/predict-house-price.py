import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
class HousePrice:

    def __init__(self):
        load = load_boston()
        self.df = pd.DataFrame(data = load.data, columns= load.feature_names)
        self.df['PRICE'] = load.target
        # corrmat = self.df.corr()
        # print(corrmat['PRICE'].sort_values(ascending=False))
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
    def validate(self):
        np.random.seed(2)

        n = len(self.df)

        n_test = int(0.2 * n)
        n_train = n - n_test

        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_test  = df_shuffled.iloc[n_train:].copy()
       

        y_train = df_train.price.values
        y_test = df_test.price.values

        del df_train['price']
        del df_test['price']

        return df_train, df_test, y_train, y_test

    def prepare_X(self,df):
        base = ['lstat']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values

        return X
    def prepare_mul_X(self,df):
        base = ['lstat','rm','ptratio','indus']
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values

        return X

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
    def linearReg(self,X_train,X_test,y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        return y_predict

def test() -> None:
    housePrice = HousePrice()
    housePrice.trim()
    df_train, df_test, y_train, y_test = housePrice.validate()
    
    # Linear regression
    X_train = housePrice.prepare_X(df_train)
    X_test = housePrice.prepare_X(df_test)
    y_predict = housePrice.linearReg(X_train,X_test,y_train)
    print('Linear Regression RMSE score:', housePrice.rmse(y_test, y_predict))
    score = r2_score(y_test, y_predict)
    print('Linear Regression  R-squared score: ',score)

    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, y_predict, color = 'blue')
    plt.title('Best fit line')
    plt.xlabel('lstat')
    plt.ylabel('Price')
    plt.show()
    
    #polynomial regression degree=2
    X_train = housePrice.prepare_X(df_train)
    polynomial_regression = PolynomialFeatures(degree = 2)
    x_polynomial = polynomial_regression.fit_transform(X_train)
    model_pol= LinearRegression()
    model_pol.fit(x_polynomial, y_train)

    x_polynomial = polynomial_regression.fit_transform(X_test)
    y_predict = model_pol.predict(x_polynomial)
    print('Polynomial regression RMSE score:', housePrice.rmse(y_test, y_predict))
    r2score = r2_score(y_test, y_predict)
    print('Polynomial regression  R-squared score: ',r2score)

    X_grid = np.arange(min(X_test), max(X_test), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_grid, model_pol.predict(polynomial_regression.fit_transform(X_grid)), color = 'blue')
    plt.title('Polynomial 2 degree')
    plt.xlabel('lstat')
    plt.ylabel('Price')
    plt.show()

    # polynomial regression degree=20
    polynomial_regression2 = PolynomialFeatures(degree = 20)
    x_polynomial2 = polynomial_regression2.fit_transform(X_train)
    model_pol2= LinearRegression()
    model_pol2.fit(x_polynomial2, y_train)
    x_polynomial2 = polynomial_regression2.fit_transform(X_test)
    y_predict = model_pol2.predict(x_polynomial2)

    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_grid, model_pol2.predict(polynomial_regression2.fit_transform(X_grid)), color = 'blue')
    plt.title('Polynomial 20 degree')
    plt.xlabel('lstat')
    plt.ylabel('Price')
    plt.show()


    #Multiple regression
    X_train = housePrice.prepare_mul_X(df_train)
    X_test = housePrice.prepare_mul_X(df_test)
    y_predict = housePrice.linearReg(X_train,X_test,y_train)
    print('Multiple Regression RMSE score:', housePrice.rmse(y_test, y_predict))
    r2score = r2_score(y_test, y_predict)
    print('Multiple Regression  R-squared score: ',r2score)

    adjusted_r_squared = 1 - (1-r2score)*(len(y_test)-1)  /  (len(y_test)- X_test.shape[1]-1)

    print('Multiple Regression adjusted R-squared score:',adjusted_r_squared)
if __name__ == "__main__":
    # execute only if run as a script
    test()