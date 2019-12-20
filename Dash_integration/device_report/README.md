# Device Report 

<br>

## Generating Device Report
-  Layout : We have taken the layout of the device report from one of the dash open source gallery - [Dash Gallery - Financial Report](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-financial-report)
-  Tabs : We have tabs for each functionality and graphs wherever necessary.
	- **Monitor Device Tab** : Displays all the device in the directory. In backend scripts will be analysing each folder parallely(Using Python Threading concepts). It will also display if the status of device as Active or Inactive based on the directory activity for last 5mins    
	-  **Battery Performance Tab** : Plots Battery (Percentage) V/s Time (Seconds) of any device (or devices) 
	-   **Location Tab** : Plots the location (Coordinate Points) on the map and renders as the graph. We are using Mapbox library for plotting coordinates. We can learn more about it here [Scatter Mapbox Plot](https://plot.ly/python/scattermapbox/).
	-   **Details Tab** : Displays all the device details like DeviceID, Operator(If GSM), Network Type, Filename(1st Filename) etc  
	-   **Review Tab** : This tab is not auto-populated, we have to feed the string / sentence in the script to generate the tab appropriately so it can be downloaded as a whole
	-   **Overview Tab** : This tab consists of all other tabs in gist / overview manner  
	-   Learn more about Tabs and its usage from -   [Dash Tabs Documentation](https://dash.plot.ly/dash-core-components/tabs)
- There are some constant values that are to be inputted in script before starting the application here [Line 48: device_report.py](https://github.com/wildlytech/modular_acoustic_detection/blob/a3ef2f58d4dde8fb25ad30fedff8515df693af4b/Dash_integration/device_report/app_device.py#L48)
	- ```PRIMARY_PATH``` : ```FTP Server``` directory path absolute path
	- ```DIR_REQ``` : ```FTP server``` directory name
	- ```MAPBOX_ACCESS_TOKEN``` : Copy the access token here. If you don't have, you can create one here [Mapbox Access Token](https://www.mapbox.com/studio)
 ```python
##########################################################################################
# Inputs Required: Path FTP
##########################################################################################
PRIMARY_PATH = "/home/user-u0xzU/BNP/"
DIR_REQ = "BNP/"
MAPBOX_ACCESS_TOKEN = "*************"
```
- FTP login credentials must be passed here [Line 58: device_report.py](https://github.com/wildlytech/modular_acoustic_detection/blob/a3ef2f58d4dde8fb25ad30fedff8515df693af4b/Dash_integration/device_report/app_device.py#L58)
	-``` FTP_PASSWORD``` : ```FTP server``` password
```python
##########################################################################################
# Inputs Required: FTP credentials
##########################################################################################
FTP_USERNAME = "user-u0xzU"
FTP_PASSWORD = "**********"
FTP_HOST = '34.211.117.196'
```
- Follow the command below to start the application 
```shell
$ python app_device.py
```

**Note**: Check for Link [http://127.0.0.1:8050/](http://127.0.0.1:8050/) from your browser to view & interact with the  application

###### References:
- You can check for screenshots of the application to see how it looks.
	- Monitor Device Tab : [Monitoring Tab Screenshot](https://drive.google.com/open?id=1LLENF4kM3dteQd8eo6Mo7cwgsFlgTkr6)
	- Transmission Performance Tab : [Transmission Tab Screenshot](https://drive.google.com/open?id=1oy4SwgnYtgKr_V3_vg5WGy338IyDw8dj)
	- Details Tab : [Details Tab Screenshot](https://drive.google.com/open?id=1gwAUdinPqV66QQdyF_4Rxeb9QqYOw7JO)
	- Battery Performance Tab: [Battery Tab Screenshot](https://drive.google.com/open?id=1UsIqs1WEmzKO7ajxMorAzrlnClGApUfl)
	- Overview Tab: [Overview Tab Screenshot](https://drive.google.com/open?id=1cEFS9p9XbtdfhaGH9GTnTOeAylSbg__h)
	- Location Tab : [Location Tab screenshot](https://drive.google.com/open?id=1U_b5Esrl6N0T2KniXhi9EPfvH0ViLZQo)



