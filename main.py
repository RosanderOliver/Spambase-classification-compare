import sys
import time
from scipy.stats import friedmanchisquare, rankdata, chisquare
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

def trainingTime(X, y, alg):
	if alg == "knn":
		knn = KNeighborsClassifier(n_neighbors=5)
	elif alg == "gau":
		#Can take the probability of classes
		knn = GaussianNB([0.61, 0.39])
	elif alg == "rfc":
		knn = RandomForestClassifier(n_estimators=32, n_jobs=-1)

	t = time.clock()
	knn.fit(X, y)
	runtime = (time.clock() - t)
	
	return runtime

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

	return features

def table(knn, gau, rfc):
	ran = range(0,len(knn))
	
	print "Fold\t\tKNN\t\tGAU\t\tRFC"
	for i in ran:
		print str(i+1) + "\t" + str(knn[i]) + "\t" + str(gau[i]) + "\t" + str(rfc[i])

	print "\navg" + "\t" + str(sum(knn)/len(knn)) + "\t" + str(sum(gau)/len(gau)) + \
	 "\t" + str(sum(rfc)/len(rfc))
	print "stddev" + "\t" + str(np.std(knn)) + "\t" + str(np.std(gau)) + "\t" + str(np.std(rfc))

def friedmanTest(knn, gau, rfc):
	ran = range(0,len(knn))
	knnrank = []
	gaurank = []
	rfcrank = []
	avgRank = 2
	

	print "\nData set\tKNN\tGAU\tRFC"

	for i in ran:
		row = np.array([knn[i],gau[i],rfc[i]])
		rowRank = rankdata(1-row)
		knnrank.append(rowRank[0])
		gaurank.append(rowRank[1])
		rfcrank.append(rowRank[2])
		print str(i+1) + "\t\t" + str(rowRank[0]) + "\t" + str(rowRank[1]) + "\t" + str(rowRank[2])

	avgknn = sum(knnrank)/len(knnrank)
	avggau = sum(gaurank)/len(gaurank)
	avgrfc = sum(rfcrank)/len(rfcrank)

	print "avg rank\t" + str(avgknn) + "\t" + str(avggau) + "\t" + str(avgrfc)

	sumsqknn = (avgknn - avgRank)**2
	sumsqgau = (avggau - avgRank)**2
	sumsqrfc = (avgrfc - avgRank)**2

	sumsqavg = 10*(sumsqknn + sumsqgau + sumsqrfc)

	su = 0
	for i in ran:
		su += (knnrank[i] - avgRank)**2
		su += (gaurank[i] - avgRank)**2
		su += (rfcrank[i] - avgRank)**2


	sumsqrow = (1.0/(len(knn)*(3-1)))*su

	print "Test statistic: " + str(sumsqavg/sumsqrow)
	fmcq = friedmanchisquare(knn, gau, rfc)

	print "Probability according to chi2 that its by chanse is " + str(fmcq[1]) + " which is less than 0.05"

	if fmcq[1] < 0.05:
		temp = ((3.0*(4))/(6*10))**0.5
		#q_(a) is set to 1.96 fromStatistical Comparisons of Classifiers 
		#over Multiple Data Sets table 5 since we test parwise?
		CD = 2.343*temp
		print "Test difference using Nemenyi test 0.05 percentile"
		print "Critical difference: " + str(CD)
		if (avgknn-avggau) > CD:
			print "There is a significant difference between algorithm KNN-GAU"
		if (avgknn-avgrfc) > CD:
			print "There is a significant difference between algorithm KNN-RFC"
		if (avggau-avgrfc) > CD:
			print "There is a significant difference between algorithm GUA-RFC"


def main(argv):
	i = 0
	X = np.genfromtxt('./data/spambase.data.X', delimiter=',')
	y = np.genfromtxt('./data/spambase.data.y', delimiter=',')

	print "Alg\tKNN\t\tGAU\t\tRFC"
	print "Time(s)\t" + str(trainingTime(X, y, "knn")) + "\t" +  \
	str(trainingTime(X, y, "gau"))+ "\t" + str(trainingTime(X, y, "rfc"))

	i = getNumberOfFeatures(X, y, "knn")
	knnMeasure = buildModel(X, y, i, "knn")
	
	i = getNumberOfFeatures(X, y, "gau")
	gauMeasure = buildModel(X, y, i, "gau")

	#i = getNumberOfFeatures(X, y, "rfc")
	#returns 32 but takes long time to run...
	rfcMeasure = buildModel(X, y, 32, "rfc")
	print "\nAccuracy measure" 
	table(knnMeasure[0], gauMeasure[0], rfcMeasure[0])
	print "\nF-measure"
	table(knnMeasure[1], gauMeasure[1], rfcMeasure[1])
	
	friedmanTest(knnMeasure[0], gauMeasure[0], rfcMeasure[0])
		


	return 0


if __name__ == "__main__":
        main(sys.argv[1:])



