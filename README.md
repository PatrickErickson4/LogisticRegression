# Regularized Logistic Regression with PyPlot Visualizations

The following is my custom implemented class for a numpy implementation performing logistic regression with
ElasticNet, Ridge, and LASSO capabilities, with heavy inspiration from sci-kit. The class contains different optimizers, such as Gradient Descent, mini-batch Gradient Descent (under SGD), and
Stochastic Gradient Descent. 
The class also contains visualizable capabilities, such as loss over epoch graphs, and pop-out classification
reports and confusion matrices, implemented with matplotlib's pyplot. Please see logreg.py for more details. 

Attached:
  - **exampleEDA.ipynb**: an example EDA file that can be used to take inspirations from. The dataset can be found [here](https://www.kaggle.com/datasets/marshuu/breast-cancer).
  - **tutorial.ipynb**: tutorial file to show how to use logreg.py


Feel free to use this file, for teaching and/or classification purposes.

**Note:** This model was meant to be used with smaller datasets. Due the the nature of the gradient descent functions I had hard-coded, there are numerical inaccuracies that are introduced, as well as poor convergence for bigger datasets. For best use, refrain from larger datasets.
