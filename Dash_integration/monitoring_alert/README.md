# Monitoring FTP server & SMS Alert application

<br>

## Monitoring and Alerting
- Layout: We have taken the layout of the device report from one of the dash open source gallery [Dash bio Gallery - Clustergram](https://github.com/plotly/dash-bio/blob/master/tests/dashbio_demos/dash-clustergram/app.py)
- Tabs
	- **Monitor Tab** : This tab enables to select the directories from the FTP server, monitor those directories continuously (Background using Threads) for any sound of interest predictions and alert the phone numbers provided using Text SMS
	-   **Upload Tab** : This will enable to user to upload single/multiple audio wav files at a time and get the predictions for each file
	-  **FTP server Tab** : This tab unlike monitor tab doesn’t run in the background but only processes audio files which are manually selected from the ftp directories
- Input the FTP credentials and fast2sms authorisation token for sending SMS here [Line 64: app_Wildly_Acoustic_Monitoring.py](https://github.com/wildlytech/modular_acoustic_detection/blob/6c31e9a100faf3f3d26f08c7e183619f60f82e57/Dash_integration/monitoring_alert/app_Wildly_Acoustic_Monitoring.py#L64)
	- Fast2sms  : [Authorization | Signup](https://www.fast2sms.com/)
```python
####################################################################################
# FTP credentials & Message Authorisation token from fast2sms.com
####################################################################################
FTP_HOST = '34.211.117.196'
AUTHORIZATION_TOKEN = "***********"
```

- Input the ```FTP server directory``` name in the script here [Line 43: app_Wildly_Acoustic_Monitoring.py](https://github.com/wildlytech/modular_acoustic_detection/blob/6c31e9a100faf3f3d26f08c7e183619f60f82e57/Dash_integration/monitoring_alert/app_Wildly_Acoustic_Monitoring.py#L43)
```python
ftp_path = "BNP/"
```
- Follow the command below to start the application to monitor an alert via SMS
```shell
$ python -m Dash_integration.monitoring_report.app_Wildly_Acoustic_Monitoring -ftp_username FTP_USERNAME
                                           -ftp_password FTP_PASSWORD
                                           [-predictions_cfg_json PREDICTIONS_CFG_JSON]
```

**Note**: Check for Link [http://127.0.0.1:8050/](http://127.0.0.1:8050/) from your browser to view & interact with the  application

###### References
- You can check for screenshots of the application to see how it looks.
	- Monitoring Tab: [Monitoring Tab Screenshot](https://drive.google.com/open?id=1n82c_Xp3EFMQbW8ryEtc49mf0upzR0nJ)
	- File Upload Tab :
		- Layout: [Upload Tab Layout](https://drive.google.com/open?id=16yal83TZoXYJiRyfe3RcoDPnmq4ci-ZY)
		- Prediction: [Upload Tab Prediction Screenshot](https://drive.google.com/open?id=1KvrJk6qmpYCAcQlqDFzRxnXoCpbJdFDJ)
  - FTP Tab :
	  - Files Listing: [Directory Files Listing](https://drive.google.com/open?id=1qCQcgm-8oWlPhGibQtMbpMMZHhoP9b5G)
	  - Prediction: [Prediction on FTP files](https://drive.google.com/open?id=1YLJrEfTNBgu5zwcJm4O7NQtc-kkOXsAR)




