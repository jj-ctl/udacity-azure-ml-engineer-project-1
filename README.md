# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

The [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) is concerned with the prediction of the success rate of a telemarketing campain of a portuguese bank.

Using the AutoML feature of Azure the best model was a VotingEnsemble w/ an accuracy 91.76%.


## Scikit-learn Pipeline

First we modifed a script for training a Logistic Regression regression model, which can be parameterzied with the maximum iteration steps and the regularization parameter. This gives us the opporunity to use Azure HyperDrive to find the best combination of $C$ and *max_iter*.


**What are the benefits of the parameter sampler you chose?**
Using the random sampler enables us to quickly sample the configuration space and generate a good estimate w/o running a full grid search and therefore saving computation costs. Since the regulazation $C$ is continous we choose the uniform sampler, while for the maximum iteration parameter the choice sampler helps to find a good parameter.

**What are the benefits of the early stopping policy you chose?**
The bandit policy helps to stop bad performing runs. This saves costs and therefore spends the available compute resources to promising runs. 


## AutoML

AutoML had a timeout of 15minutes in order to find the best model. In this time it was able to check eight different models with accuracies ranging from 73% w/ an Extreme Random Tree model up to 91.77% percent w/ a VotingEnsemble model. In order to reuse the best model the VotingEnsemble was saved to the workspace model section.
![](registered_models.png)


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Both approaches are really effective in generating a good model. With the HyperDrive approach the data scientist is able to quickly generate a model based on his or her instincts. While AutoML is great for letting Azure select the best performing model. AutoML not only checks one model but
a whole list of different models and performs the hyperparameter tuning in one run. 

Therefore with Hyperdrive the data scientist has more control over the model and its hyperparameters but AutoML enables the user to quickly setup a model with comparable accuracy without a lot of investigation into the data.

In this case best performing model w/ hyperdrive was found in around 12 minutes with an accurcy of 90.89%, while AutoML took all 15min and resulted in an accuracy of 91.77%

Since AutoML can choose from many different algorithms it is not surprising that it can check for algorithms, which are better taylored for the problem.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

- Give hyperdrive more computing resources, in order to search a larger grid w/ more *max_iter* and a lager range of *C*.
- Change from random to grid search in the hyperdrive model selection
- AutoML detected *Class balancing detection*. One should take a closer look here and rerun w/ a balenced dataset

