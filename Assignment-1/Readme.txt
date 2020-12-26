This zip folder containg 5 files --
1. Readme - this very files
2. part1.py - for question-1 
3. part2.py - for question-2
4. part3.py - for question-3
5. Analysis - this contains the observations(plots, tables, inferences etc) made in all 3 questions.
6. 1a_images.pdf - this contains output images of question 1(a)

Python Codes-
	All three .py files have been edited and executed on Spyder(python version - 3.7). 
Running the .py codes-
	Open a file in Spyder editor and execute the whole file.
	Or we can also execute the code on Anaconda Powershell Prompt by giving the command in python console like this
	>>>python part1.py
Path to datasets-
        Keep the datasets in the same folder or change the path in below commands in part1.py and part2.py 
	scipy.io.loadmat('dataset_1.mat') ---> scipy.io.loadmat('<your dataset location>dataset_1.mat')
	For part3.py change the path in below command
	pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')  ----> pd.read_csv('<your dataset location>PRSA_data_2010.1.1-2014.12.31.csv');
	
Disclaimer-
	The accuracy score after each execution might be a little different from the analysis document, because data will be chosen randomly so it will result in a bit different model.