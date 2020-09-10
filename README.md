# Logistic-Regression-From-Scratch
Implementing linear regression from scratch on Iris data

## Introduction

Similar to linear regression, logistic regression can also be prone to overfitting if there are large number of features. If the decision boundary is overfit, the shape might be highly contorted to fit only the training data while failing to generalise for the unseen data.

So, the cost function of the logistic regression is updated to penalize high values of the parameters and is given by,

![](https://latex.codecogs.com/svg.latex?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Cleft%28%20-y%5E%7B%28i%29%7D%20%5Clog%28h_%7B%5Ctheta%7Dx%5E%7B%28i%29%7D%29%20-%20%281-y%5E%7B%28i%29%7D%29%20%5Clog%281-h_%7B%5Ctheta%7Dx%5E%7B%28i%29%7D%29%20%5Cright%29%20&plus;%20%5Cfrac%7B%5Clambda%7D%7B2m%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20%5Ctheta_j%5E2)

Where 

* ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B%5Clambda%7D%7B2m%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20%5Ctheta_j%5E2) is the *regularization term*
* ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Clambda) is the *regularization factor*

Since this cost function accounts for a new regularization term, the derivative of the cost function must also account for this. Thus we have,

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta_j%7D%20%3D%20%5Cleft%28%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%28h_%7B%5Ctheta%7Dx%5E%7B%28i%29%7D%20-%20y%5E%7B%28i%29%7D%29x_j%5E%7B%28i%29%7D%20%5Cright%29%20&plus;%20%5Cfrac%7B%5Clambda%7D%7Bm%7D%5Ctheta_j%20%5Cquad%20%5Ctext%7Bfor%20%7D%20j%20%5Cgeq%201)

We can use these functions to make predictions on our data.

## Dataset

The data set I chose to implement this regression model was the [Iris dataset](https://www.kaggle.com/uciml/iris). It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
