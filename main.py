import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'x': np.array([1, 2, 3, 4, 5]),
    'y': np.array([2, 3, 5, 7, 11])
}
df = pd.DataFrame(data)
print("--- Dữ liệu thử nghiệm Pandas ---")
print(df)

X = df[['x']]
y = df['y']
model = LinearRegression()
model.fit(X, y)
print(f"\nHệ số góc (slope): {model.coef_[0]:.2f}")


plt.scatter(df['x'], df['y'], color='blue', label='Dữ liệu')
plt.plot(df['x'], model.predict(X), color='red', label='Đường dự đoán')
plt.title("Thử nghiệm Thư viện Python")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()