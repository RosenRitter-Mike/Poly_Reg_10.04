import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data
fertilizer = np.array([100, 150, 200, 250, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200])
water = np.array([5, 5, 5, 5, 5, 3, 4, 6, 7, 5, 5, 5, 5, 5])
temperature = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 15, 18, 22, 25, 28])
corn = np.array([2.5, 3.2, 3.8, 4.1, 3.9, 3.0, 3.5, 3.9, 3.7, 3.2, 3.6, 3.7, 3.5, 3.1])

# Combine features
X = np.column_stack((fertilizer, water, temperature))
y = corn

# Polynomial features up to degree 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Grid for prediction
fertilizer_grid = np.linspace(100, 300, 50)
water_grid = np.linspace(2, 8, 50)
temperature_grid = np.linspace(15, 30, 30)
fertilizer_mesh, water_mesh, temperature_mesh = np.meshgrid(fertilizer_grid, water_grid, temperature_grid)

# Predict over grid
X_grid = np.column_stack((fertilizer_mesh.ravel(), water_mesh.ravel(), temperature_mesh.ravel()))
X_grid_poly = poly.transform(X_grid)
harvest_pred = model.predict(X_grid_poly).reshape(water_mesh.shape)

# Find optimal point
max_corn = np.max(harvest_pred)
max_idx = np.unravel_index(np.argmax(harvest_pred), harvest_pred.shape)
optimal_fertilizer = fertilizer_mesh[max_idx]
optimal_water = water_mesh[max_idx]
optimal_temperature = temperature_mesh[max_idx]

# Print results
print(f"Model polynomial coefficients: {model.coef_}")
print()
print(f"Optimal fertilizer amount: {optimal_fertilizer:.2f} Kg")
print(f"Optimal water amount: {optimal_water:.2f} liters per day")
print(f"Optimal temperature: {optimal_temperature:.2f} degrees celsius")
print(f"Maximum predicted corn harvest: {max_corn:.2f} ton per 900 square meters(dunam)")


