This zip folder containg 8 files and 5 folders
Files-
1. Readme - this very file
2. Regression.py - for question-1          ----> each part(like 1(a), 1(b)... is distinguishable from each other part by using """1(a)"""" --"""1(a) done""" comments)
3. LogRegression.py - for question-2 and 3 ----> each part(like 1(a), 1(b)... is distinguishable from each other part by using """1(a)"""" --"""1(a) done""" comments)
4. Analysis - this contains the observations(plots, tables, inferences etc) made in all 3 questions.
5. Assignment_2_questions.pdf
6,7,8 - dataset files
Folders-
1. Assignment_2_datasets - contains datasets for all 3 questions
2. models_1 - contains 10 saved models(.pkl files) for question-1
	      first 5-models for Q1b [1 model per fold]
	      next 5-models for Q1c [1 model per fold]
3. models_2 - contains 10 saved models(.pkl files) for question-2
	      first 5-models for Q2c [1 model per fold]
	      next 5-models for Q2d [1 model per fold]
4. models_3 - contains 7 saved models(.pkl files) for question-3
	      first 3-models for Q3b [only for fold-1] OVO:each model for each class vs the reference class(i.e., class-4)
	      next 4-models for Q3c [only for fold-1] OVR:each model for each class vs rest of the classes
5. dump - By default all models will be saved under dump folder(if name and path of the model to be saved is not specified)

Python Codes-
	All .py files have been edited and executed on Spyder(python version - 3.7). 
Running the .py codes-
	Open a file in Spyder editor and execute the whole file.
	Or we can also execute the code on Anaconda Powershell Prompt by giving the command in python console like this
	>>>python Regression.py
Path to datasets-
        Keep the datasets in the same folder or change the path in below commands in Regression.py and LogRegression.py 
	>>pass path to the dataset as first argument in Regression class
	>>pass peth to the dataset as first argument in LogRegression class
	
Disclaimer-
	In order to use '.predict()' method of any class on saved models, change the last input argument in its call like below-
	initially it is set to --

	>>M = ob1.fit(X_train, Y_train,X_test,Y_test,m_c[0])  #m_c[0] tells the path to store the model
	>>y_pred = ob1.predict(X_train,M) 
	
	In .predict(,) change M with the path-to-saved-model
	This has to be done for each model as it is set to first fit() and then use the output model of fit in predict().














