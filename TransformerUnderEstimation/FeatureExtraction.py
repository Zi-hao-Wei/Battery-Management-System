import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.signal
from scipy.optimize import curve_fit 
import math
import pandas as pd 
from sklearn.model_selection import train_test_split
import json 
from pyts.image import GASF, GADF

#Read data from pkl
def GetData(train=True):
    batch1 = pickle.load(open(r'..\Data\batch1V.pkl', 'rb'))
    #remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    batch2 = pickle.load(open(r'..\Data\batch2V.pkl','rb'))
    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]
    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]
    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']
    
    for i in batch1:
        # print(i)
        idx=0
        for j in batch1[i]["cycles"]:
            if (str(idx+1)) in batch1[i]["cycles"]:
                # if(idx==0):
                    # print(batch1[i]["cycles"][str(idx)])
                batch1[i]["cycles"][str(idx)]=batch1[i]["cycles"][str(idx+1)]
            idx+=1
            # print(j,type(j))

    # batch3 = pickle.load(open(r'..\Data\batch3V.pkl','rb'))
    # # remove noisy channels from batch3
    # del batch3['b3c37']
    # del batch3['b3c2']
    # del batch3['b3c23']
    # del batch3['b3c32']
    # del batch3['b3c38']
    # del batch3['b3c39']

    bat_dict={**batch1,**batch2} 
    #,**batch3}
    Data=[]
    for i in bat_dict.keys():
        Data.append((bat_dict[i],bat_dict[i]["cycle_life"]))

    Data=sorted(Data,key=lambda x:x[1])

    Processed=[]
    k=0
    for i,j in Data:

        if i["cycle_life"]<250:
            continue 
        if k%5==0 and train:
            k+=1
            continue
        if k%5!=0 and not train:
            k+=1
            continue
        Processed.append(i)
        k+=1
    return Processed


def GetSmoothedData(width=51,degree=6,train=True):
#Denoising
    Data,Life=GetData(train)
    for i in Data:
        for j in i['summary']:
            if j != "cycle":
                original=i['summary'][j]
                smoothed=scipy.signal.savgol_filter(original,width,degree)
                i['summary'][j]=smoothed
    return Data, Life 

def f(x,y_fit,degree):
    ans=0
    for i in range(degree+1):
        ans+=y_fit[i]*pow(x,degree-i)
    return ans

class PolyFit():
    def __init__(self,x,y,degree=10):
        x=np.array(x)
        y=np.array(y)
        self.degree=degree
        self.y_fit=np.polyfit(x,y,self.degree)
    def Fitted(self,x):
        return f(x,self.y_fit,self.degree)




class FeatureExtraction:
    def __init__(self,Smoothed=False,train=True,process=True):
        if process:
            if Smoothed:
                self.Data=GetSmoothedData(train=train)
            else:
                self.Data=GetData(train)
            
        self.droplist=[]
        self.Graph=[]
        self.GraphConcat=[]
        self.batteryProcessed=[]
        self.life=[]
        self.label=[]
        if train:
            self.file="Batch12Train\\"
        else:
            self.file="Batch12Test\\"
        if train:
            self.len=66
        else:
            self.len=17

    def pipeline(self,process=True):
        if process:
            self.pipelineProcess()
        else:
            self.readData()
        
        print(self.life)

        self.TimeSeriesToGraph()
        return self.GraphConcat 

    def DataSetDisplay(self):
        battery=self.Data[5]
        print(len(battery["summary"]["IR"]))
        IR=battery["summary"]["IR"] #[1:]
        Tavg=battery["summary"]["Tavg"] #[1:]
        Tmin=battery["summary"]["Tmin"] #[1:]
        Tmax=battery["summary"]["Tmax"] #[1:]
        QC=battery["summary"]["QC"] #[1:]
        QD=battery["summary"]["QD"] #[1:]
        V=battery["Vdlin"]
        plt.cla()
        for idx in battery["cycles"]:                
            if int(idx)%50==0:
                Tmp={}
                Tmp["T"]=battery["cycles"][idx]["Tdlin"] 
                Tmp["Qd"]=battery["cycles"][idx]["Qdlin"]
                Tmp["DQDV"]=battery["cycles"][idx]["dQdV"]
                plt.subplot(321)
                plt.plot(V,Tmp["T"])

                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("Temperature(°C)")

                plt.subplot(323)
                plt.plot(V,Tmp["Qd"])
                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("Capacity(Ah)")

                plt.subplot(325)
                plt.plot(V,Tmp["DQDV"])
                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("dQdV")
       

        battery=self.Data[-5]
        print(len(battery["summary"]["IR"]))
        print("-----")
        IR=battery["summary"]["IR"] #[1:]
        Tavg=battery["summary"]["Tavg"] #[1:]
        Tmin=battery["summary"]["Tmin"] #[1:]
        Tmax=battery["summary"]["Tmax"] #[1:]
        QC=battery["summary"]["QC"] #[1:]
        QD=battery["summary"]["QD"] #[1:]
        V=battery["Vdlin"]
      
        for idx in battery["cycles"]:                
            if int(idx)%50==0:
                Tmp={}
                Tmp["T"]=battery["cycles"][idx]["Tdlin"] 
                Tmp["Qd"]=battery["cycles"][idx]["Qdlin"]
                Tmp["DQDV"]=battery["cycles"][idx]["dQdV"]
                plt.subplot(322)
                plt.plot(V,Tmp["T"])

                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("Temperature(°C)")

                plt.subplot(324)
                plt.plot(V,Tmp["Qd"])
                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("Capacity(Ah)")

                plt.subplot(326)
                plt.plot(V,Tmp["DQDV"])
                if(int(idx)==50):
                    plt.xlabel("Voltage(V)")
                    plt.ylabel("dQdV")



        plt.show()




    def pipelineProcess(self):  
        print(len(self.Data))
        w=0
        for battery in self.Data:
            V=battery["Vdlin"]
            CycleData=[]
            CycleLabel=[]


            IR=battery["summary"]["IR"] #[1:]
            Tavg=battery["summary"]["Tavg"] #[1:]
            Tmin=battery["summary"]["Tmin"] #[1:]
            Tmax=battery["summary"]["Tmax"] #[1:]
            QC=battery["summary"]["QC"] #[1:]
            QD=battery["summary"]["QD"] #[1:]
            plt.cla()
            first=0
            firste=0
            firstm=0
            # if (len(IR)!=491):
            #     continue

            for idx in battery["cycles"]:                
                # if(idx=="0"):
                #     continue 

                # Tmp[""]

                Tmp={}
                Tmp["T"]=battery["cycles"][idx]["Tdlin"] 
                
                # if (len(Tmp["T"])!=491):
                    # continue
                
                # -battery["cycles"]["0"]["Tdlin"]
                Tmp["Qd"]=battery["cycles"][idx]["Qdlin"]
                # Tmp["T"]=battery["cycles"][idx]["T"]
                # Tmp[""]
                Tmp["T"]=scipy.signal.savgol_filter(Tmp["T"],53,3)

                # -battery["cycles"]["0"]["Qdlin"]
                Tmp["V"]=battery["cycles"][idx]["V"]
                Tmp["I"]=battery["cycles"][idx]["I"]
                Tmp["QdO"]=battery["cycles"][idx]["Qd"]

                DIDQ=[0]
                DVDQ=[0]
                DQDV=[0]
                DTDV=[0]

                Q=[]                               
                I=[]
                VC=[]
                for i in range(1,len(Tmp["QdO"])):
                #    DIDQ.append( ((Tmp["I"][i]-Tmp["I"][i-1])/(Tmp["QdO"][i]-Tmp["QdO"][i-1])))
                   DVDQ.append( ((Tmp["V"][i]-Tmp["V"][i-1])/(Tmp["QdO"][i]-Tmp["QdO"][i-1])))
                   if (abs(DVDQ[i])<10):
                    #    print(DVDQ[i
                       Q.append(Tmp["QdO"][i])
                       I.append(Tmp["I"][i])
                       VC.append(Tmp["V"][i])
                if (idx=="0"):
                    first=len(Q)
                    for i in range(0,len(I)):
                        if I[i]>-3:
                            firste=i
                            firstm=np.mean(VC[0:firste])
                            break
                    continue
                else:
                    # print(firste)
                    if len(Q)<first or max(I[0:firste])>-3 or  np.mean(VC[0:firste])<firstm:
                        continue
                # # plt.show()
                plt.figure(1)
                plt.subplot(221)
                plt.plot(Q,I)
                # print(DIDQ)
                plt.subplot(222)
                plt.plot(Q,VC)
                # plt.subplot(223)
                # plt.plot(V,Tmp["DTDV"])
                # plt.plot(Q,I)
                # plt.subplot(224)
                # plt.plot(Q,VC)
                # print(I)
                # print(VC)
                # for i in range(0,len(DIDQ)):
                    # if 0
                # print(Tmp["I"].tolist())
                # print(DVDQ)
                # plt.show()
               
                for i in range(1,len(Tmp["Qd"])):
                #    DQDV.append( abs((Tmp["Qd"][i]-Tmp["Qd"][i-1])/(V[i]-V[i-1])))
                #    print(i)
                   DTDV.append( abs((Tmp["T"][i]-Tmp["T"][i-1])/(V[i]-V[i-1])))
                

                Tmp["DQDV"]=abs(battery["cycles"][idx]["dQdV"]-battery["cycles"]["0"]["dQdV"])[0:800]
                Tmp["DTDV"]=DTDV[0:600]
                
                # Tmp["DTDV"]=scipy.signal.savgol_filter(Tmp["DQDV"],53,3)
                # Tmp["DTDV"]=scipy.signal.savgol_filter(Tmp["DTDV"],53,3)


                plt.subplot(223)
                # plt.plot(V,Tmp["DTDV"])
                plt.plot(V[0:600],Tmp["DTDV"])
                plt.subplot(224)
                plt.plot(V[0:800],Tmp["DQDV"])
                # plt.show()
            plt.savefig("Datafigure2\\"+str(idx)+".png")

                # for key in Tmp:
                #     Tmp[key]=Tmp[key][30:-30]

                # w+=1

                # DataOneCycle=self.OneBattery(V, Tmp["QDQV"])+self.OneBatteryhalf(V, Tmp["QTQV"]) 
                # CycleData.append(DataOneCycle)
            
        #     CycleData=np.transpose(np.array(CycleData))
        #     plt.cla()
        #     for i in range(0,CycleData.shape[0]):
        #         meannum=np.mean(CycleData[i])
        #         for j in Crange(CycleData.shape[1]):
        #             if CycleData[i][j]>meannum*2:
        #                 CycleData[i][j]=(CycleData[i][j-1]+CycleData[i][j+1])/2
        #         CycleData[i]=scipy.signal.savgol_filter(CycleData[i],53,5)
        #         CycleData[i]=(CycleData[i]-min(CycleData[i]))/(max(CycleData[i])-min(CycleData[i]))
        #         CycleData[i]=scipy.signal.savgol_filter(CycleData[i],53,7)
        #         plt.plot(CycleData[i])
        #     plt.savefig(self.file+str(idx)+".png")


        #     np.save(self.file+str(w)+'.npy',CycleData)   # 保存为.npy格式
        #     w+=1
            
            
        #     self.batteryProcessed.append(CycleData)
        #     print(CycleData.shape[1])
        #     self.life.append(CycleData.shape[1])

        #     # plt.figure(1)
        #     # x=420
        #     # for i in range(0,6):
        #     #     plt.subplot(str(x+i))
        #     #     plt.plot(CycleData[i])
        #     #     CycleData[i]=scipy.signal.savgol_filter(CycleData[i],53,3)
        #     #     plt.plot(CycleData[i])
        #     # plt.show()

        #     # plt.figure(2)
        #     # x=420
        #     # for i in range(0,6):
        #     #     plt.subplot(str(x+i))
        #     #     plt.plot(CycleData[i+6])
        #     #     CycleData[i+6]=scipy.signal.savgol_filter(CycleData[i+6],53,3)
        #     #     plt.plot(CycleData[i+6])
        #     # plt.show()
        # np.save(self.file+'life.npy',np.array(self.life))
        # print(self.life)



    def readData(self):
        # 读取
        for i in range(0,self.len):
            a=np.load(self.file+str(i)+'.npy')
           
            self.batteryProcessed.append(a)
        self.life=np.load(self.file+'life.npy')
        

    def OneBattery(self,X,Y):
        l=len(X)/2
        return self.OneBatteryhalf(X[:l], Y[:l]) + self.OneBatteryhalf(X[-l:], Y[-l:])

    def OneBatteryhalf(self,V,Y):
        # MAXY
        maxY_idx=np.argmax(Y,axis=0)
        maxY=Y[maxY_idx]    
        maxY_V=V[maxY_idx]
        
    

        D_mean = np.mean(Y) #计算均值
        D_var = np.var(Y)  #计算方差
        D_sc = np.mean((Y - D_mean) ** 3)  #计算偏斜度
        D_ku = np.mean((Y - D_mean) ** 4) / pow(D_var, 2) #计算峰度
        return [maxY,maxY_V,D_mean,D_var,D_sc,D_ku]


    def TimeSeriesToGraph(self,img_size=224):
        gasf=GASF(img_size)
        idx=0
        # print(len(batteryProcessed),len(self.life))

        for battery,life in zip(self.batteryProcessed,self.life):
            
            print(life)
            for j in range(250,battery.shape[1],100):            
                temp=[]
                for f in range(0,battery.shape[0]):
                    x=battery[f][j-250:j].squeeze()
                    x=x.reshape(1,-1)
                    temp.append(gasf.fit_transform(x).squeeze()) 
                    self.label.append(life-j)
                temp=np.array(temp)
                self.GraphConcat.append(temp)  
            idx+=1


    def getLifeTime(self):
        return self.label

if __name__=="__main__":

    Test = FeatureExtraction(train=True,process=True)
    Test.DataSetDisplay()
    
    # Featured=Test.pipeline(process=True)
    # Test = FeatureExtraction(train=False,process=True)
    # Featured=Test.pipeline(process=True)