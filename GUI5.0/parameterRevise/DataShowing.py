import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

def getData():
    Data=pd.read_csv("PulseData.csv")
    MaxD=Data["Discharge_Capacity(Ah)"].max()
    Data["SOC"]=Data["Discharge_Capacity(Ah)"]/MaxD
    Data["Test_Time(s)"]=Data["Test_Time(s)"]-Data.iloc[0]["Test_Time(s)"]
    Data=Data.loc[Data["Test_Time(s)"] <=60000]
    return Data

def DivideData(Data):
    I=Data["Current(A)"].tolist()
    begin=0
    end=0
    SplitData=[]
    for i in range(0,len(I)):
        if I[i]!=0 and begin==0:
            begin=i
        elif I[i]==0 and begin!=0:
            end=i-1
            # print(begin,end)
            SplitData.append(Data.iloc[begin-100:end+100])
            begin=0
    SplitData=SplitData[1:]
    return SplitData


if __name__=="__main__":
    AllData=getData()
    Split=DivideData(AllData)
    # print(Split)
    AllTime=AllData["Test_Time(s)"]
    AllSOC=AllData["SOC"]
    plt.plot(AllTime,AllSOC)
    plt.show()
    # AllCurrent=AllData["Current(A)"]
    # AllVoltage=AllData["Voltage(V)"]
    # plt.plot(AllTime,AllCurrent)
    # plt.plot(AllTime,AllVoltage)


    # for PulseData in Split: 
    #     Time=PulseData["Test_Time(s)"]
    #     Current=PulseData["Current(A)"]
    #     Voltage=PulseData["Voltage(V)"]
    #     plt.plot(Time,Current)
    #     plt.plot(Time,Voltage)
    # plt.show()