# Linear Regression from Scratch

Predicting exam scores from hours studied, using only NumPy and gradient descent (no scikit-learn).

![Result](linear_regression_result.png)

## What's inside
- Hypothesis function: ŷ = wx + b
- Mean Squared Error cost function
- Gradient descent to learn w and b
- Learning-rate experiment (too small, good, too large)

## Results
- Learned w = 5.03, b = 34.79 (true values: 5 and 35)
- Matches NumPy's polyfit exactly
- R² = 0.92

## How to run
Open `linear_regression_from_scratch.ipynb` in Google Colab or Jupyter and run the cells in order.
Or run the plain script: `python linear_regression_from_scratch.py`

## Requirements
numpy, matplotlib

## What I learned
- How a straight line (y = wx + b) can be a model
- How the cost function measures how wrong a model is
- How gradient descent updates w and b step by step
- Why the learning rate matters: too small is slow, too large makes the cost explode

## Next steps
- [ ] Multiple linear regression (more than one feature)
- [ ] Feature scaling
- [ ] Train/test split
- [ ] Logistic regression
