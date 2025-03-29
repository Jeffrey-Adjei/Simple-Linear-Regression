import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def load_dataset():
    data = pd.read_csv('Salary_dataset.csv')
    return data

def define_variables(data):
    x = data[['YearsExperience']]
    y = data['Salary']
    return x, y

def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

def train(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def predictions(model):
    y_pred = model.predict(x_test)
    return y_pred

def evaluate_model(y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("coefficient of determination:", r2)
    print("mean squared error: ", mse)

def data_plot():
    sns.regplot(x=x, y=y, color="red")
    plt.scatter(x, y, color="black")
    plt.title("The relationship between Years of Experience and Salary")
    plt.xlabel("Years Of Experience")
    plt.ylabel("Salary")
    plt.show()

if __name__ == "__main__":
    data = load_dataset()
    x, y = define_variables(data)
    x_train, x_test, y_train, y_test = split_dataset(x, y)
    model = train(x_train, y_train)
    y_pred = predictions(model)
    evaluate_model(y_pred)
    data_plot()




    

   

