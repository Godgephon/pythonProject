import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv(
    '数据.csv',
    encoding='utf-8',
    delimiter='\t'
)


data = data.dropna()
x = data['t']
y = data['x']
x = np.array(x)
y = np.array(y)
matplotlib.rc('font', family='SimHei')
matplotlib.rc('axes', unicode_minus=False)
plt.figure(figsize=(9, 9))
plt.scatter(x, y, color='black', linewidths=2.5, label='实验数据')
plt.grid()
plt.xlabel('时间(s)', fontsize=15)
plt.ylabel('下落高度(m/s)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


reg = LinearRegression()
reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
y_fit = reg.predict(x.reshape(-1, 1))
plt.plot(x, y_fit, color='red', label='拟合直线')
plt.legend(fontsize=15)
coef = round(reg.coef_[0][0], 2)
intercept = round(reg.intercept_[0], 2)
plt.text(2, 0.8, rf'$y={coef}x+{intercept}$'+'\n'+rf'$r^2={round(r2_score(y, y_fit), 4)}$', fontdict={'size': 30, 'color': 'red', 'weight': 'bold'})
plt.savefig('output.jpg', dpi=100)
