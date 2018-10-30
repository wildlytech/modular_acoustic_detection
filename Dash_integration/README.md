##  Dash for creating a UI
  

##### DASH is used to create a simple UI that aimed at doing two things basicaly :

1. Upload an audio file from the local drive to run the analyis and see the predictions.
2. Scrape through the remote FTP server for any ```.wav``` files and download it to local drive to run analysis on it.

##### To create a UI in your local host:
```
$ python dash_integrate.py 
```

 ###### UI consists of 3 pages :
 - ##### Home Page
 - ##### Input audio :
   This page lets you to upload a ```.wav``` file from your local drive to run the analysis and see the predictions.
 - ##### FTP status :
   This will scrape the remote ftp server and download if any ```.wav``` file is uploaded.
