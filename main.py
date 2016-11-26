import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, chi2
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def buildModel(X, y, k, alg):
	folds = 10
	accuracy = []
	fmeasure = []
	skf = StratifiedKFold(n_splits=folds)
	
	if alg == "knn":
		knn = KNeighborsClassifier(n_neighbors=5)
	elif alg == "gau":
		#Can take the probability of classes
		knn = GaussianNB([0.61, 0.39])
	elif alg == "rfc":
		knn = RandomForestClassifier(n_estimators=k, n_jobs=-1) 

	features = k
	sp = SelectKBest(score_func=chi2,k=features) 
	Xtr = sp.fit_transform(X, y)
	
	for train, test in skf.split(Xtr,y):
		knn.fit(Xtr[train], y[train])
		pred = knn.predict(Xtr[test])
		ac = accuracy_score(y[test], pred)
		f1 = f1_score(y[test], pred)
		fmeasure.append(f1)
		accuracy.append(ac)
	

	return [accuracy, fmeasure]


def getNumberOfFeatures(X, y, alg):
	maxAcc = 0
	features = 1	
	acc = []
	fm = []
	
	ran = range(1,47)
	for i in ran:
		ac = buildModel(X, y, i, alg)
		
		avgacc = sum(ac[0])/len(ac[0])
		avgf1 = sum(ac[1])/len(ac[1])
		acc.append(avgacc)
		fm.append(avgf1)
		
		if (avgacc + avgf1) > maxAcc:
			maxAcc = (avgacc + avgf1)
			features = i	

	print "Max avg (accuracy+f-measure)/2 at " + str(features) + " neighbours with " + str(maxAcc/2) + " (accuracy+f-measure)/2"
	
	return features

def table(knn, gau, rfc):
	ran = range(0,len(knn))
	
	print "Fold\t\tKNN\t\tGAU\t\tRFC"
	for i in ran:
		print str(i+1) + "\t" + str(knn[i]) + "\t" + str(gau[i]) + "\t" + str(rfc[i])

	print "avg" + "\t" + str(sum(knn)/len(knn)) + "\t" + str(sum(gau)/len(gau)) + "\t" + str(sum(rfc)/len(rfc))

def main(argv):
	#read file
	#file = open(argv[0])
	
	
	algorithm = argv[0]

	X = np.genfromtxt('./data/spambase.data.X', delimiter=',')
	y = np.genfromtxt('./data/spambase.data.y', delimiter=',')
	
	i = getNumberOfFeatures(X, y, "knn")
	knnMeasure = buildModel(X, y, i, "knn")
	
	i = getNumberOfFeatures(X, y, "gau")
	gauMeasure = buildModel(X, y, i, "gau")

	#i = getNumberOfFeatures(X, y, "rfc")
	#returns 32 but takes long time to run...
	rfcMeasure = buildModel(X, y, 32, "rfc")

	table(knnMeasure[0], gauMeasure[0], rfcMeasure[0])

	#remove all rows containing nan values
	#seperate features and response
	#make sure data is numeric 
	#check type of array, should be numpy arrays
	#check shape of matrix, iris.data.shape
	#number of rows should match in features and response

	 

	return 0


if __name__ == "__main__":
        main(sys.argv[1:])



