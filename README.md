README


Part A Coursework:

Google Drive Setup:
Upload these files to the folder:
TrainingDataBinary.csv
TestingDataBinary.csv

For the best model (my submisson), open the BestModel_A.ipynb file and check that the pathfile is correct in your drive, the files above are uploaded, and the file path for the destination is correct as well. 

If you are running it in google collab keep these two lines:
Line 6: from google.colab import drive
Line 7: drive.mount('/content/drive')

When you do run you will be asked for permission to access your google drive 

If you are running it locally remove them

file paths you need to change:
On line 10: data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TrainingDataBinary.csv') 
On line 65: test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TestingDataBinary.csv')
On line 78: test_data.to_csv('/content/drive/My Drive/Colab Notebooks/TestingResultsBinary.csv', index=False)

Click run (press the play button), and the statement should be output:
'The predictions have been saved successfully to 'TestingResultsBinary.csv'

The file TestingResultsBinary.csv should be found at the location you specify, which will have a column “marker” added at the end with the predictions.


Part B Coursework:

Google Drive Setup:
Upload these files to the folder:
TrainingDataMulti.csv
TestingDataMulti.csv

For the best model (my submisson), open the BestModel_B.ipynb file and check that the pathfile is correct in your drive, the files above are uploaded, and the file path for the destination is correct as well.

If you are running it in google collab keep these two lines:
Line 6: from google.colab import drive
Line 7: drive.mount('/content/drive')

When you do run you will be asked for permission to access your google drive 

If you are running it locally remove them

file paths you need to change:
On line 10: data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TrainingDataMulti.csv') 
On line 65: test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TestingDataMulti.csv')
On line 78: test_data.to_csv('/content/drive/My Drive/Colab Notebooks/TestingResultMulti.csv', index=False)


Click run (press the play button), and the statement should be output:
'The predictions have been saved successfully to 'TestingResultsMulti.csv'

The file TestingResultsMulti.csv should be found at the location you specify, which will have a column “marker” added at the end with the predictions.
