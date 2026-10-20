"""
Simple Linear Regression from scratch using Gradient Descent
Predicting exam score from hours studied (synthetic data).
Only NumPy is used for the maths; Matplotlib is used for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Create a simple dataset: score = 35 + 5 * hours + noise
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
m = 100                                   # number of training examples
x = rng.uniform(0, 10, m)                 # hours studied
y = 35 + 5 * x + rng.normal(0, 4, m)      # exam score (with noise)

# ---------------------------------------------------------------
# 2. Model, cost function, gradient
#    Model:  y_hat = w * x + b
#    Cost:   J(w, b) = (1 / 2m) * sum((y_hat - y)^2)
# ---------------------------------------------------------------
def predict(x, w, b):
    return w * x + b

def compute_cost(x, y, w, b):
    errors = predict(x, w, b) - y
    return np.sum(errors ** 2) / (2 * len(x))

def compute_gradients(x, y, w, b):
    errors = predict(x, w, b) - y
    dw = np.sum(errors * x) / len(x)      # dJ/dw
    db = np.sum(errors) / len(x)          # dJ/db
    return dw, db

# ---------------------------------------------------------------
# 3. Gradient descent
# ---------------------------------------------------------------
def gradient_descent(x, y, lr=0.02, n_iters=5000):
    w, b = 0.0, 0.0                       # start from zero
    cost_history = []
    for i in range(n_iters):
        dw, db = compute_gradients(x, y, w, b)
        w -= lr * dw                      # update rule
        b -= lr * db
        cost_history.append(compute_cost(x, y, w, b))
        if i % 1000 == 0:
            print(f"Iter {i:5d} | cost = {cost_history[-1]:8.4f} | w = {w:.4f} | b = {b:.4f}")
    return w, b, cost_history

w, b, cost_history = gradient_descent(x, y)

print("\n--- Results ---")
print(f"Learned  : w = {w:.3f}, b = {b:.3f}")
print(f"True     : w = 5.000, b = 35.000")

# Sanity check against NumPy's closed-form least-squares fit
w_ref, b_ref = np.polyfit(x, y, 1)
print(f"np.polyfit (check): w = {w_ref:.3f}, b = {b_ref:.3f}")

# R^2 score: how much of the variance does the model explain?
y_pred = predict(x, w, b)
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"R^2 score: {r2:.3f}")
print(f"A student who studies 8 hours is predicted to score {predict(8, w, b):.1f}")

# ---------------------------------------------------------------
# 4. Visualise: fitted line + cost curve
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(x, y, alpha=0.7, label="Data")
xs = np.linspace(0, 10, 100)
axes[0].plot(xs, predict(xs, w, b), color="red", linewidth=2,
             label=f"y = {w:.2f}x + {b:.2f}")
axes[0].set_xlabel("Hours studied")
axes[0].set_ylabel("Exam score")
axes[0].set_title("Linear regression fit")
axes[0].legend()

axes[1].plot(cost_history, color="green")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Cost J(w, b)")
axes[1].set_title("Gradient descent: cost going down")
axes[1].set_yscale("log")

plt.suptitle("Linear Regression from Scratch (NumPy + Gradient Descent)", fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/linear_regression_result.png", dpi=150)
print("\nSaved plot to linear_regression_result.png")
