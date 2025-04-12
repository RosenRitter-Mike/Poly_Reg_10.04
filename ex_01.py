import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# Our data
fertilizer = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
potatoes = np.array([7.5, 10.2, 12.8, 14.5, 15.6, 16.0, 15.8, 15.0, 13.5, 11.2])

# missing data
print("___ missing data ___")
xi_squared = fertilizer^2
print(f"xi_squared: {xi_squared}")
print(f"xi * yi: {fertilizer*potatoes}")
print(f"xi_squared * yi: {xi_squared*potatoes}")

# coefficients
print("\n___ coefficients(sklearn) ___")
polynomial_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Fit the model
polynomial_model.fit(fertilizer.reshape(-1, 1), potatoes)

# Get the coefficients
coefficients = polynomial_model.named_steps['linear'].coef_
intercept = polynomial_model.named_steps['linear'].intercept_

print(f"Intercept (β₀): {intercept:.2f}")
print(f"Coefficient for x (β₁): {coefficients[1]:.2f}")
print(f"Coefficient for x² (β₂): {coefficients[2]:.2f}")

# Equation
equation = f"y = {intercept:.2f} + ({coefficients[1]:.2f})x + ({coefficients[2]:.2f})x²"
print(f"Polynomial equation: {equation}")

coefficients = np.polyfit(fertilizer, potatoes, 2)
print("\n___ coefficients(polyfit) ___")
# Extract the coefficients
a = coefficients[0]  # coefficient for x²
b = coefficients[1]  # coefficient for x
c = coefficients[2]  # intercept

print(f"Coefficient for x² (a): {a:.2f}")
print(f"Coefficient for x (b): {b:.2f}")
print(f"Intercept (c): {c:.2f}")

# Equation
equation = f"y = {c:.2f} + ({b:.2f})x + ({a:.2f})x²"
print(f"Polynomial equation: {equation}")

optimal_fertilizer = -b / (2 * a)
print(f"\nOptimal fertilizer amount: {optimal_fertilizer:.2f}")

# Create a polynomial function using the coefficients
poly_function = np.poly1d(coefficients)
max_potatoes = poly_function(optimal_fertilizer)
print(f"Predicted maximal potato harvest: {max_potatoes:.2f} Ton")