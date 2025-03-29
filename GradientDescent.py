# Here we will construct a gradient descent algorithm from scratch

import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('Salary_dataset.csv')

def gradient_descent_optimisation(m_now, b_now, points, alpha):
    m_gradient = 0 
    b_gradient = 0

    n = len(data)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * alpha
    b = b_now - b_gradient * alpha

    return m, b

m = 0
b = 0
alpha = 0.01
epochs = 1500

for i in range(epochs):
    if i % 50 == 0:
        print("epoch:", i)
    m, b = gradient_descent_optimisation(m, b, data, alpha)

print(m, b)

plt.scatter(data.YearsExperience, data.Salary, color="blue")
plt.title("The relationship between Years of Experience and Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(list(range(1,12)), [m * x + b for x in range(1, 12)], color="red")
plt.show()
