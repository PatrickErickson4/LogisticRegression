'''
Author: Patrick Erickson
Title: Visualizable Regularized Logistic Regression
Inspiration was taken from sci-kit, with capabilities tuned to better
match personal preferences.

NOTE: Sci-kit still contains more efficiently optimized functions for logistic regression due
to its parallelizing capabilities. This is a project of to streamline and potentially teach logistic
regression and machine learning to others.
'''

import numpy as np
import matplotlib.pyplot as plt

# Ensure implementation will only show for the following functions.
__all__ = [
    "trainTestSplit",
    "standardizer",
    "standardizeCompute",
    "unNormalize",
    "fit",
    "crossfoldValidation",
    "getAccuracy",
    "getConfusionMatrix",
    "getReport",
    "displayLossOverEpochs",
    "displayConfusionMatrix",
    "displayClassificationReport"
]

class LogisticRegression:
    '''
    The following is a class for a numpy implementation performing logistic regression with
    ElasticNet, Ridge, and LASSO capabilities, with heavy inspiration from sci-kit.
    The class contains different optimizer, such as Gradient Descent, mini-batch Gradient Descent (under SGD), and
    Stochastic Gradient Descent. 
    The class also contains visualizable capabilities, such as loss over epoch graphs, and pop-out classification
    reports and confusion matrices, implemented with matplotlib's pyplot.

    NOTE: This class assumes all variables are numerical, including the the labels being binary encoded.
    Please clean the data prior to constructing a model.

    NOTE: for all dataset parameters, please ensure that the datasets are numpy datasets
    Eg. if you are in pandas, you can change the dataset by doing:
    df = df.to_numpy()
    after cleaning.
    '''

    def trainTestSplit(features, labels, trainSize=None, testSize=None, randomState=None):
        '''
        Used on a split label and feature dataset. Will automatically handle single input training and testing.
        Ensures a random shuffling of the train/test. Client can specify train size, test size, or both.
        If no train size is stated, automatically does a randomized 80/20 train/test split.

        INPUT: 
        features: the feature datset, labels removed
        labels: the labels corresponding the the feature dataset
        trainSize: allocated percentage of datset to be meant for training, represented as decimal.
        testSize: allocated percentage of datset to be meant for testing, represented as decimal.
        randomState: Can be specified by client. Will use an int to see the train-test split for reproduceability.

        OUTPUT:
        trainFeatures: the training set split as requested
        testFeatures: the training labels split as requested
        testFeatures: the testing set split as requested
        testLabels: the testing labels split as requested
        '''

        cases = len(features)

        # if else ladders to handle bad input and to correctly set train and test size based on user input.
        if randomState is not None:
            np.random.seed(randomState)

        if cases < 2:
            raise Exception("Your dataset is too small.")

        if trainSize is None and testSize is None:
            trainSize, testSize = 0.8, 0.2
        if trainSize is not None and testSize is None:
            if trainSize <= 0.0 or trainSize >= 1.0:
                raise Exception("Bad Train/Test Split")
            testSize = 1 - trainSize
        elif testSize is not None and trainSize is None:
            if testSize <= 0.0 or testSize >= 1.0:
                raise Exception("Bad Train/Test Split")
            trainSize = 1 - testSize
        #need to handle floating point errors.
        if trainSize is not None and testSize is not None:
            if abs(trainSize + testSize - 1.0) > 1e-6 or trainSize < 0.0 or trainSize > 1.0 or testSize < 0.0 or testSize > 1.0:
                raise Exception("Bad Train/Test Split")

        #randomly shuffles indices of size of feature rows, then partitions the array into a train and
        #test indices
        splitIndex = np.random.permutation(cases)
        lastTrainIndex = int(cases*trainSize)
        trainIndices = splitIndex[:lastTrainIndex]
        testIndices = splitIndex[lastTrainIndex:]

        #indices are used to reshuffle the train and test splits.
        trainFeatures = features[trainIndices]
        trainLabels = labels[trainIndices]
        testFeatures = features[testIndices]
        testLabels = labels[testIndices]

        return trainFeatures, trainLabels, testFeatures, testLabels 

    def _BinaryCrossEntropyAndLoss(x,y,w,alpha,lam):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.
        Parent function for the loss and the gradient for every epoch for gradient descent.
        Computes sigmoid.
        If using SGD, computes this for every batch, and handles conditionals if regularization
        is specified or not.

        INPUT:
        x: feature dataset, represented mathematically as x for linear transformation ease.
        y: labels for the feature dataset, represented mathematically as y for linear transformation ease.
        w: weights for the current iteration of gradient and loss.
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        lam: regularization coefficient. the greater the coefficient, the stronger the regularization.

        OUTPUT (from child function): 
        gradient: Regularized or non-regularized gradient for the iteration
        loss: Regularized or non-regularized loss for the iteration
        NOTE: These will be averaged.
        '''
        cases = len(x)

        wtx = np.dot(x,w)
        #ensure that if values converge to infinity, we clip them as to not get overflow
        sigmoid = np.where(wtx >= 0,1 / (1 + np.exp(-wtx)),np.exp(wtx) / (1 + np.exp(wtx)))        
        sigmoid = np.clip(sigmoid, 1e-8, 1 - 1e-8)
        
        #conditional for regularization/no regularization. Ensures proper regularization hyperparams
        if alpha is None:
            return LogisticRegression._lossAndGradientNoReg(sigmoid,x,y,cases)
        elif alpha < 0 or alpha > 1:
            raise Exception("Enter a valid alpha between 0 and 1 (0 for ridge, 1 for lasso)")
        else:
            if(lam < 0):
                raise Exception("Lambda can not be less than 0")
            return LogisticRegression._lossAndGradientWithReg(sigmoid,x,y,cases,alpha=alpha,lam=lam,w=w)

        
    def _lossAndGradientNoReg(sigmoid,x,y,cases):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.
        Computes loss and the gradient for every epoch for gradient descent if there 
        is no specified regularization.
        If using SGD, computes this for every batch.

        INPUT:
        wtx: the dot product of the weights and the feature dataset.
        sigmoid: Calculated from the parent functions to formulate odds.
        x: feature dataset
        y: labels for feature dataset
        cases: the number of data points

        OUTPUT:
        gradient: non-Regularized gradient for the iteration
        loss: non-Regularized loss for the iteration
        NOTE: These will be averaged.
        '''

        #loss function for base logistic regression. 
        # NOTE: if coming back to this later, @ = matrix multiply, and .T is a transpose, as offered by numpy libraries
        loss = - (y.T @ np.log(sigmoid) + (1 - y).T @ np.log(1 - sigmoid)) / cases  
        loss = loss[0] #produces a 1x1 numpy matrix. We force only 1 value to come out.
        gradient = (x.T @ (sigmoid-y)) / cases
        return gradient, loss
    
    def _lossAndGradientWithReg(sigmoid,x,y,cases, alpha, lam, w):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.
        Computes loss and the gradient for every epoch for gradient descent if there 
        is specified regularization.
        Bias is set to 0 for a deep copied weight matrix to ensure we do not regularize bias.
        If using SGD, computes this for every batch.

        INPUT:
        wtx: the dot product of the weights and the feature dataset.
        sigmoid: Calculated from the parent functions to formulate odds.
        x: feature dataset
        y: labels for feature dataset
        cases: the number of data points
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        w: weights for every feature for the current iteration of gradient and loss.

        OUTPUT:
        gradient: Regularized gradient for the iteration
        loss: Regularized loss for the iteration
        NOTE: These will be averaged.
        '''

        #deep copy for regularization calculations w/o bias
        wNoBias = np.copy(w)
        wNoBias[0,0] = 0

        #loss function for regularization
        loss = (- (y.T @ np.log(sigmoid) + (1 - y).T @ np.log(1 - sigmoid)) +
                 (1-alpha)*lam*np.dot(wNoBias.T,wNoBias) +
                 alpha*lam*np.sum(np.abs(wNoBias)))/ cases  
        loss = loss[0] #produces a 1x1 numpy matrix. We force only 1 value to come out.

        #gradient for regularization
        gradient = ((x.T @ (sigmoid-y)) +
                     (1-alpha)*2*lam*wNoBias +
                     alpha*lam*np.sign(wNoBias)) / cases
        
        return gradient, loss
    
    def standardizer(trainingDataset):
        '''
        Used to ensure that the training set and testing set will have a mean of 0 and
        standard deviation of one. This allows for better convergence.

        NOTE: use the standardizer on ONLY the training set, then use
        StandardizeCompute on both the training and testing set to avoid data leakage

        INPUT:
        trainingDataset: The training set, labels removed.

        OUTPUT:
        an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset
        '''
        mean = np.mean(trainingDataset, axis=0)
        std = np.std(trainingDataset, axis=0)
        std[std == 0] = 1.0
        return [mean, std]
    
    def standardizeCompute(toStandardize,standardizer):
        '''
        function used to ensure standardization of all of the features
        to minimize scaling differences and better convergence. to be called as such:

        standardizer = LogisticRegression.standardizer(trainingSet)
        standardizedTrainingSet = LogisticRegression.LogisticRegression.standardizeCompute(trainingSet,standardizer)
        standardizedTestingSet = LogisticRegression.LogisticRegression.standardizeCompute(trainingSet,standardizer)

        NOTE: use the standardizer on ONLY the training set, then use
        StandardizeCompute on both the training and testing set to avoid data leakage

        INPUT:
        trainingDataset: The training set, labels removed.
        standardizer: an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset

        OUTPUT:
        standardizedSet: the resulting dataset, standardized on the training dataset.
        '''
        return (toStandardize-standardizer[0]) / standardizer[1]
    

    def unNormalize(tounNormalize, standardizer):
        '''
        function used to unNormalize all standardized functions from standardizeCompute.
        This returns the data back to its original state.

        INPUT:
        dataset: The dataset, with labels at the end. NOTE: Ensure labels are at the end
        standardizer: an array of:
            mean: means of every feature in the dataset
            std: the standard deviations of every feature in the dataset

        OUTPUT:
        tounNormalize: the resulting dataset, unstandardized to revert it back to its orginal state.
        '''
        labels = tounNormalize[:,-1]
        tounNormalize = tounNormalize[:,:-1]
        tounNormalize =  (tounNormalize*standardizer[1]) + standardizer[0]
        tounNormalize = np.c_[tounNormalize,labels]
        return tounNormalize
     
    def _performGDRegression(trainingSet,trainingLabels,alpha,lam,learningRate=.01,epochs = 1000):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.
        Performs basic gradient descent. Computes the loss and gradient over the entire
        dataset (aka. corpus), and saves the loss for each epoch. Returns the final weights and the losses per epoch
        after the model has finished training.

        INPUT:
        trainingSet: The dataset you are trying to train Logistic Regression to, without the labels
        trainingLabels: the labels of the dataset corresponding to the trainingSet (binary 0/1)
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        lam: regularization coefficient. Bigger value = stronger regularization.
        learningRate: the size of the step towards the convex minima. Default set to .01.
        epochs: the number of calculations of the entire dataset. Default set to 1000.

        OUTPUT:
        weights: The final weights (INCLUDING BIAS) computed after training. is a (d+1)x1 numpy matrix.
        lossHistory: an array of losses over every epoch. Can be used for calculating loss over epoch graphs.
        '''

        #add bias feature randomly initialize weights for every feature, bias included
        cases = len(trainingSet)
        trainingSetWithBias = np.c_[np.ones((cases,1)),trainingSet]
        weights = np.random.randn(trainingSetWithBias.shape[1],1)

        lossHistory = [] 

        for num in range(epochs):

            #NOTE: We shuffle SGD because depending on the batch, we want to get the best random representation of the data
            #this doesnt matter here because we are computing the gradient over the entire dataset every time.
            
            #compute loss and gradient. Append loss to history and readjust weight after epoch
            gradient,loss = LogisticRegression._BinaryCrossEntropyAndLoss(trainingSetWithBias,trainingLabels,weights,alpha=alpha,lam=lam)
            gradient = np.mean(gradient, axis=1, keepdims=True) 
            weights -= learningRate*gradient
            lossHistory.append(loss)

        return weights, lossHistory
    
    def _performSGDRegression(trainingSet, trainingLabels, alpha, lam, learningRate=.01, epochs=1000, batchSize=1):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.

        NOTE: SGD is not accelerated, and therefore has very high loop overhead. This is more for proof of concept
        rather than a real loss function.

        Performs stochastic gradient descent and mini batch gradient descent. Computes the loss and gradient over client-specified
        batches, and saves the loss for each epoch. Returns the final weights and the losses per epoch
        after the model has finished training.

        INPUT:
        trainingSet: The dataset you are trying to train Logistic Regression to, without the labels
        trainingLabels: the labels of the dataset corresponding to the trainingSet (binary 0/1)
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        lam: regularization coefficient. Bigger value = stronger regularization.
        learningRate: the size of the step towards the convex minima. Default set to .01.
        epochs: the number of calculations of the entire dataset. Default set to 1000.
        batchSize: the size of each batch for gradient computations. default to stochastic and not minibatch, ie. batchSize = 1.

        OUTPUT:
        weights: The final weights (INCLUDING BIAS) computed after training. is a (d+1)x1 numpy matrix.
        lossHistory: an array of losses over every epoch. Can be used for calculating loss over epoch graphs.
        '''

        #add bias feature randomly initialize weights for every feature, bias included
        cases = len(trainingSet)
        trainingSetWithBias = np.c_[np.ones((cases,1)), trainingSet]
        weights = np.random.randn(trainingSetWithBias.shape[1], 1)
        lossHistory = []

        for num in range(epochs):
            #shuffle dataset per epoch
            indices = np.random.permutation(cases)
            trainingSetShuffled = trainingSetWithBias[indices]
            trainingLabelsShuffled = trainingLabels[indices]

            totalLoss = 0
            #depending on batch size (could be smaller on the last batch)
            #used for averaging
            totalSamples = 0

            for item in range(0, len(trainingSetShuffled), batchSize):

                # iterate batches over the entire set
                trainingSetBatch = trainingSetShuffled[item:item+batchSize]
                trainingLabelBatch = trainingLabelsShuffled[item:item+batchSize]

                gradient, loss = LogisticRegression._BinaryCrossEntropyAndLoss(trainingSetBatch, trainingLabelBatch, weights, alpha=alpha, lam=lam)
                gradient = np.mean(gradient, axis=1, keepdims=True) 

                weights -= learningRate * gradient
                currentBatchSize = len(trainingSetBatch)

                #ensures batched averaging is handled correctly
                totalLoss += loss * currentBatchSize
                totalSamples += currentBatchSize

            lossHistory.append(totalLoss / totalSamples)

        return weights, lossHistory

    
    def fit(X,Y,alpha=None,lam=0, batchSize=1,learningRate=.01,epochs=1000, loss="GD",randomState=None):
        '''
        Driver function for the user specified logistic regression model.
        returns the final weights and the losses per epoch
        after the model has finished training.

        INPUT:
        trainingSet: The dataset you are trying to train Logistic Regression to, without the labels
        trainingLabels: the labels of the dataset corresponding to the trainingSet (binary 0/1)
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        lam: regularization coefficient. Bigger value = stronger regularization.
        learningRate: the size of the step towards the convex minima. Default set to .01.
        epochs: the number of calculations of the entire dataset. Default set to 1000.
        batchSize: the size of each batch for gradient computations. default to stochastic and not minibatch, ie. batchSize = 1.
        randomState: ensures reproduceability if randomstate is specified. Sets a random seed with int specified by client. Does not set one otherwise

        OUTPUT:
        weights: The final weights (INCLUDING BIAS) computed after training. is a (d+1)x1 numpy matrix.
        lossHistory: an array of losses over every epoch. Can be used for calculating loss over epoch graphs.
        '''
        if randomState is not None:
            np.random.seed(randomState)

        #if gradient descent is specified 
        if loss == "GD":
            return LogisticRegression._performGDRegression(X,Y,alpha=alpha, learningRate=learningRate,epochs=epochs,lam=lam)
        #if stochastic gradient descent/batch descent is specified
        elif loss == "SGD":
            if batchSize < 0:
                raise Exception("Specify a Valid batch size.")
            return LogisticRegression._performSGDRegression(X,Y,alpha=alpha, learningRate=learningRate,epochs=epochs,lam=lam, batchSize=batchSize)
        else:
            raise Exception("Enter a Valid Loss function.")
    
    def crossfoldValidation(trainingSet, trainingLabels, alpha=None, lam=0, batchSize=1, learningRate=.01, epochs=1000, loss="GD", folds=5, randomState=None,probabilityThreshold=.5):

        '''
        Computes k-fold cross validation on the dataset. fits models for the number of folds, then averages all the weights 
        for better generalization. Returns an array of final losses and accuracies for comparison as well 

        INPUT:
        trainingSet: The dataset you are trying to train Logistic Regression to, without the labels
        trainingLabels: the labels of the dataset corresponding to the trainingSet (binary 0/1)
        alpha: ElasticNet Regularizer. 0 is Ridge Regression, 1 is LASSO, 0 < alpha < 1 is ElasticNet.
        lam: regularization coefficient. Bigger value = stronger regularization.
        learningRate: the size of the step towards the convex minima. Default set to .01.
        epochs: the number of calculations of the entire dataset. Default set to 1000.
        batchSize: the size of each batch for gradient computations. default to stochastic and not minibatch, ie. batchSize = 1.
        randomState: ensures reproduceability if randomstate is specified. Sets a random seed with int specified by client. Does not set one otherwise

        OUTPUT:
        weights: The final averaged weights (INCLUDING BIAS) computed after training. is a (d+1)x1 numpy matrix.
        finalLoss: an array of losses over every fold.

        '''

        #calculate the size of folds
        cases = len(trainingSet)
        foldSize = int(cases / folds)
        if foldSize < 1:
            raise Exception("Too many folds for the data.")

        # initialize return variables and aggregators
        finalLoss = []
        weightsMatrix = []
        accuracies = []

        #shuffle to ensure randomness
        indices = np.random.permutation(cases)
        XShuffled = trainingSet[indices]
        YShuffled = trainingLabels[indices]


        folds_X = np.array_split(XShuffled, folds)
        folds_Y = np.array_split(YShuffled, folds)

        for i in range(folds):

            X_valid = folds_X[i]
            Y_valid = folds_Y[i]
            X_train = np.concatenate(folds_X[:i] + folds_X[i+1:])
            Y_train = np.concatenate(folds_Y[:i] + folds_Y[i+1:])
            
            #run a fit model on certain split of the k-fold and append results
            weights, lossHistory = LogisticRegression.fit(X_train, Y_train, alpha=alpha, lam=lam, learningRate=learningRate, epochs=epochs, loss=loss, randomState=randomState, batchSize=batchSize)
            accuracies.append(LogisticRegression.getAccuracy(X_valid,Y_valid,weights,probabilityThreshold=probabilityThreshold))
            weightsMatrix.append(weights)
            finalLoss.append(lossHistory)

        #return weights
        return weightsMatrix, finalLoss, accuracies
    
    def _generateReport(testingSet, testingLabels, weights, probabilityThreshold):
        '''
        NOTE: NOT MEANT FOR CLIENT USE.
        generates a dictionary of sci-kit learn styled classification report, followed by the True Positives, True Negatives, False Positives, and False Negatives.
        Used as a main driver function for the getters and plots.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        dictionary of:
        'accuracy': accuracy,
            'class 0': {
                'precision': precision0,
                'recall': recall0,
                'f1-score': f1_0,
                'support': support0,
            },
            'class 1': {
                'precision': precision1,
                'recall': recall1,
                'f1-score': f1_1,
                'support': support1,
            },
            'macro avg': macroAvg,
            'weighted avg': weightedAvg
        TP: Number of True Positive Classifications
        TN: Number of True Negative Classifications
        FP: Number of False Positive Classifications
        FN: Number of False Negative Classifications
        '''

        #ensure there is bias to account for the bias weight
        testingSetWithBias = np.c_[np.ones((len(testingSet), 1)), testingSet]
        cases = len(testingSetWithBias)

        #initialize classifications
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        correct = []

        for i in range(cases):
            # calculate TP, FP, TN, FN
            # add to correct classification / true label columns
            z = np.dot(weights.T, testingSetWithBias[i])
            probability = 1 / (1 + np.exp(-z))
            predicted = 1 if probability > probabilityThreshold else 0

            label = testingLabels[i]

            if predicted == 1 and label == 1:
                TP += 1
                correct.append([1,label])
            elif predicted == 1 and label == 0:
                FP += 1
                correct.append([0,label])
            elif predicted == 0 and label == 0:
                TN += 1
                correct.append([1, label])
            else:
                FN += 1
                correct.append([0,label])

        flattened = [
            [x, y.item() if isinstance(y, np.ndarray) and y.size == 1 else y]
            for x, y in correct
        ]

        # build append the classifications and true labels to the testing set to return for another getter
        correct = np.array(flattened)
        
        classifiedSet = np.c_[testingSetWithBias,correct]

        # else 0's are added to ensure that if the denominator is 0 in rare cases, then the report value is also 0
        # perform classification report formulas
        precision0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        recall0    = TN / (TN + FP) if (TN + FP) > 0 else 0

        precision1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall1    = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1_1       = 2 * precision1 * recall1 / (precision1 + recall1) if (precision1 + recall1) > 0 else 0
        f1_0       = 2 * precision0 * recall0 / (precision0 + recall0) if (precision0 + recall0) > 0 else 0

        support0 = TN + FP
        support1 = TP + FN

        accuracy = (TP + TN) / cases

        macroAvg = {
            'precision': (precision0 + precision1) / 2,
            'recall':    (recall0 + recall1) / 2,
            'f1-score':  (f1_0 + f1_1) / 2,
            'support':   cases
        }

        weightedAvg = {
            'precision': (precision0 * support0 + precision1 * support1) / cases,
            'recall':    (recall0 * support0 + recall1 * support1) / cases,
            'f1-score':  (f1_0 * support0 + f1_1 * support1) / cases,
            'support':   cases
        }

        return {
            'accuracy': accuracy,
            'class 0': {
                'precision': precision0,
                'recall': recall0,
                'f1-score': f1_0,
                'support': support0,
            },
            'class 1': {
                'precision': precision1,
                'recall': recall1,
                'f1-score': f1_1,
                'support': support1,
            },
            'macro avg': macroAvg,
            'weighted avg': weightedAvg
        },TP,FP,TN,FN,classifiedSet

    def getAccuracy(testingSet, testingLabels, weights, probabilityThreshold=.5):
        '''
        Getter for accuracy on the testing set.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        accuracy: correct classifications percentage with given weights
        '''
        report, _, _, _, _, _ = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold=probabilityThreshold)
        return round(report['accuracy'],6)
    
    def getClassifications(testingSet, testingLabels, weights, probabilityThreshold=.5):
        '''
        Getter for a set of classified test features, bias removed

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        correct: correct classifications with given weights as matrix of features, with added classifications and true labels column appended 
        to the end, with no bias.
        incorrect: correct: correct classifications with given weights as matrix of features, with added classifications and true labels column appended 
        to the end, with no bias.
        '''
        _,_,_,_,_,classificationSet = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold)
        classificationSet = classificationSet[:,1:]

        incorrect = classificationSet[classificationSet[:, -2] == 0]
        correct = classificationSet[classificationSet[:, -2] == 1]

        incorrect = np.c_[incorrect[:,:-2],incorrect[:,-1:]]
        correct = np.c_[correct[:,:-2],correct[:,-1:]]

        return correct, incorrect

    def getConfusionMatrix(testingSet, testingLabels, weights, probabilityThreshold=.5):
        '''
        getter for the True Positives, True Negatives, False Positives, and False Negatives.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        TP: Number of True Positive Classifications
        TN: Number of True Negative Classifications
        FP: Number of False Positive Classifications
        FN: Number of False Negative Classifications
        '''
        _, TP, FP, TN, FN, _ = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold=probabilityThreshold)
        return TP, FP, FN, TN
    
    def getClassificationReport(testingSet, testingLabels, weights, probabilityThreshold=0.5):
        '''
        Getter for a dictionary of sci-kit learn styled classification report.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        dictionary of:
        'accuracy': accuracy,
            'class 0': {
                'precision': precision0,
                'recall': recall0,
                'f1-score': f1_0,
                'support': support0,
            },
            'class 1': {
                'precision': precision1,
                'recall': recall1,
                'f1-score': f1_1,
                'support': support1,
            },
            'macro avg': macroAvg,
            'weighted avg': weightedAvg'
        '''
        report, _, _, _, _, _ = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold=probabilityThreshold)
        return report
    
    def displayLossOverEpochs(lossHistories):
        '''
        Generates a plot of the loss histories for ONLY UP TO 10 specified training losses over epochs,
        as given by the .fit method. This was chosen to reduce noise in graphs.

        INPUT:
        Array of:
            Loss Histories (arrays obtained second parameter of .fit)

        OUTPUT:
        Pyplot Graph of the respective losses over epochs
        '''

        # 10 is the hard limit for histories
        if len(lossHistories) > 10:
            raise Exception("Plots can only hold up to 10 loss histories to avoid cluttered graphs. Make a new plot for more. (Did you ensure loss was encapsulated by an array?)")
        colors = plt.get_cmap("tab10").colors

        #contains all of tableau color palette
        [plt.plot(history, color=colors[i], linestyle='-', label=f'Loss History {i+1}') for i, history in enumerate(lossHistories)]
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def displayConfusionMatrix(testingSet, testingLabels, weights, probabilityThreshold=0.5):
        '''
        pop out table for the True Positives, True Negatives, False Positives, and False Negatives.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        Pyplot table of the confusion matrix
        '''

        #get confusion matrix
        _, TP, FP, TN, FN,_ = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold)
    
        # Build table data for confusion matrix
        table_data = [
            ["", "Predicted 1", "Predicted 0"],
            ["Actual 1", f"{TP}", f"{FN}"],
            ["Actual 0", f"{FP}", f"{TN}"]
        ]

        #remove graph properties for table
        _, ax = plt.subplots()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
    
        plt.suptitle("Confusion Matrix", fontsize=16)
    
        # Create and customize the table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
    
        plt.show()

    def displayClassificationReport(testingSet, testingLabels, weights, probabilityThreshold=0.5):
        '''
        pop out table for the classification report in a sci-kit style manner.

        INPUTS:
        testingSet: the test set for the dataset, without the labels
        testingLabels: the labels corresponding to the test set.
        weights: the finalized weights of some fitted model
        probabilityThreshold: Defaulted to .5 in parent functions. can be changed by the client.

        OUTPUTS:
        Pyplot table of the classification report
        '''
        #get classification report
        report, _, _, _, _, _ = LogisticRegression._generateReport(testingSet, testingLabels, weights, probabilityThreshold)
    
        # build table with all metrics from report
        table_data = []
        header = ["", "precision", "recall", "f1-score", "support"]
        table_data.append(header)
    
        for key in ['class 0', 'class 1', 'macro avg', 'weighted avg']:
            metrics = report[key]
            row = [
                key,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']}"
            ]
            table_data.append(row)
    
        total_support = report['class 0']['support'] + report['class 1']['support']
        accuracy_row = ["accuracy", "", "", f"{report['accuracy']:.3f}", f"{total_support}"]
        table_data.append(accuracy_row)
    
        #construct table
        _, ax = plt.subplots()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
    
        plt.suptitle("Classification Report", fontsize=16)
    
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
    
        plt.show()

    def displayLogisticGraph(trainingSet, testingSet, testingLabels, weights, probabilityThreshold=.5):
        '''
        Displays the logistic function (sigmoid) along with the data points (binary 0/1) 
        for a given problem, model, and testing set. Shows the misclassified points
        Based on the logistic regression model as different from the correct classifications.

        INPUT:
        trainingSet: the training features for the 
        testingSet: the testing features for the dataset. Used to compute logistic function.
        testingLabels: the binary labels (0 or 1) corresponding to the testing features
        weights: the model weights (including the bias as the first element)

        OUTPUT:
        A graph of the logistic function curve with the binary data points overlaid.
        '''

        trainingSetWithBias = np.c_[np.ones((len(trainingSet), 1)), trainingSet]
        trainingZ= np.dot(trainingSetWithBias, weights).flatten()
        function = 1 / (1 + np.exp(-trainingZ)).flatten()

        #sort datapoints to make a valid graph instead of just noise
        sort_idx = np.argsort(trainingZ)
        trainingZ = trainingZ[sort_idx]
        function = function[sort_idx]

        # grabs test set with true false classifications as binary along with true labels tacked onto the end
        _,_,_,_,_,classifiedSet = LogisticRegression._generateReport(testingSet,testingLabels,weights,probabilityThreshold)

        #split correct and incorrect indices and compute z
        incorrect = classifiedSet[classifiedSet[:, -2] == 0]
        incorrectLabels = incorrect[:,-1:]
        incorrect = incorrect[:,:-2]
        incorrectZ = np.dot(incorrect, weights)

        correct = classifiedSet[classifiedSet[:, -2] == 1]
        correctLabels = correct[:,-1:]
        correct = correct[:,:-2]
        correctZ = np.dot(correct,weights)

        plt.figure(figsize=(10, 6))
        plt.plot(trainingZ, function, label='Logistic Function', color='blue')
        plt.axhline(y=probabilityThreshold, color='red', label='Probability Threshold')  
        plt.scatter(correctZ, correctLabels, label='Correct Classifications', color='green', marker='o')
        plt.scatter(incorrectZ, incorrectLabels, label='Incorrect Classifications', color='red', marker='x')

        # legend apart from graph for better readability
        plt.xlabel("weights dotted with features")
        plt.ylabel("Probability")
        plt.title("Plot of the Logistic Function")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
