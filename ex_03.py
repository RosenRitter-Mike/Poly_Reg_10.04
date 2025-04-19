import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

"""
- Explain why data scaling is important?

- Data prediction models use gradient decent to find the best model (linear regression or polynomial).
If your features have wildly different scales, the cost becomes faulty.
Example:
    a gaming company wants to survey the market to make a successful Sci-Fi RPG
    data of games:

    number of playable races 1 to 10
    number of playable factions 1 to 4
    number of usable item types 1000 to 1000000
    is part of major sci-fi universe (Star Trek, Star Wars, W40K, Dune, Stargate, ...) (bool) 1 or 0
    character customisation (1 - next to none,..., 5 - up to the last wrinkle in the character eye corner)
    product cost in usd 25 to 100

In this example the item types stat would make all others (except perhaps product cost) insignificant,
though in fact they are significant.
"""
"""
- Unite the arrays
"""
x1 = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
x2 = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])

X_united = np.column_stack((x1, x2))
print("X_united: \n", X_united)

'''
Standardisation
'''

print("\n", "="*5, "Standardisation", "="*5, "\n")
scaler_std = StandardScaler()
X_united_1 = scaler_std.fit_transform(X_united.copy())

print("X_united: \n", X_united_1)

'''
Normalisation - MinMax
'''

print("\n","="*5, "Normalisation", "="*5, "\n")
scaler_norma = MinMaxScaler()
X_united_2 = scaler_norma.fit_transform(X_united.copy())

print("X_united: \n", X_united_2)

"""
- Explain the difference between the scaling methods

- Standardization means that data is centered around 0 with a standard deviation of 1 (can include negative values).
Its most useful when data follows a Gaussian (normal) distribution or approximately does. 
And is used for algorithms that assume normally distributed data.

- Normalization (Min-Max Scaling) scales data to the range [0, 1]. 
Its most useful when you want all features to have equal weight and fit within the same scale.
And is used for algorithms that rely on distance metrics.
"""