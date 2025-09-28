import pandas as pd
import numpy as np
import math

df=pd.DataFrame()
temp={}
temp2={}
projectCoords=[[],[]]
rMoon=1737400
dataDivide=100

#Get Latitude and Longitude
dftemp=pd.read_csv("Data/Latitude.csv").iloc[::dataDivide]
dftemp=dftemp.to_numpy().flatten()
temp["Latitude"]=dftemp
temp2["Latitude"]=dftemp
dftemp=pd.read_csv("Data/Longitude.csv").iloc[::dataDivide]
dftemp=dftemp.to_numpy().flatten()
temp["Longitude"]=dftemp
temp2["Longitude"]=dftemp

#Convert to Cartesian Coordinates
for i in range(0,len(temp2["Latitude"])):
    latRad=temp2["Latitude"][i]*math.pi/180
    longRad=temp2["Longitude"][i]*math.pi/180
    scaleFactor=(2*rMoon)*math.tan(math.pi/4+latRad/2)
    x=scaleFactor*math.sin(longRad)
    y=scaleFactor*math.cos(longRad)
    projectCoords[0].append(x)
    projectCoords[1].append(y)

#Center the Coordinates
temp["X"]=np.array(projectCoords[0])
temp["Y"]=np.array(projectCoords[1])
avX=(min(temp["X"])+max(temp["X"]))/2
avY=(min(temp["Y"])+max(temp["Y"]))/2
for i in range(0,len(temp["X"])):
    temp["X"][i]=temp["X"][i]-avX
    temp["Y"][i]=temp["Y"][i]-avY

#Get Height and Slope
dftemp=pd.read_csv("Data/Height.csv").iloc[::dataDivide]
dftemp=dftemp.to_numpy().flatten()
temp["Height (meters)"]=dftemp
dftemp=pd.read_csv("Data/Slope.csv").iloc[::dataDivide]
dftemp=dftemp.to_numpy().flatten()
temp["Slope (degrees)"]=dftemp
df=pd.concat([df,pd.DataFrame(temp)],ignore_index=True)
df.to_csv("Nobile1.csv")