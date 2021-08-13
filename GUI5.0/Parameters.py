import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import random
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import ParameterEstimation
import pandas as pd
class ParameterDisplay(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        # self.master = master

        self.create_widgets()
        self.pack()
        self.run=True
        self.continueRunning = False
        self.startTask()
        self.parameterEstimation=ParameterEstimation.ParameterEstimation()
        # self.pointer=0
        self.Data=pd.DataFrame({'Voltage':[],'Current':[]})
        


    def create_widgets(self):
        
        self.DataDisplay = DataShowing(self, title="SOC/SOH Display")
        self.DataDisplay.grid(row=1, column=1, pady=(20,0), padx=(20,20), ipady=10)

        self.photo = Image.open("ECM.png")#file：t图片路径
        self.photo = self.photo.resize((280,180))
        self.photo = ImageTk.PhotoImage(self.photo)
        self.imgLabel = tk.Label(self,image=self.photo)#把图片整合到标签类中
        self.imgLabel.grid(row=0, column=1, pady=(20,0),padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0,rowspan=2,column=2, pady=(20,0), ipady=10)


    def startTask(self):
        #Prevent user from starting task a second time
        # self.inputSettingsFrame.startButton['state'] = 'disabled'

        #Shared flag to alert task if it should stop
        self.continueRunning = True
        

        self.R0=0
        self.R1=0
        self.C1=0
        self.R0L=[]
        self.R1L=[]
        self.C1L=[]



        #spin off call to check 
        self.master.after(1000, self.runTask)
        


    def runTask(self):
        #Check if task needs to update the graph
        
        
        #This part is to test Data transition across tabs.
        self.current=self.Data["Current"].tolist()
        self.voltage=self.Data["Voltage"].tolist()
        if(len(self.current)==0):
            self.R0,self.R1,self.C1= 0,0,0
        else:
            self.R0,self.R1=self.current[-1],self.voltage[-1]
            self.C1=0


        # self.R0,self.R1,self.C1=self.parameterEstimation.pipeline(self.Data)

        # self.R0=random.random()
        # self.R1=random.random()
        # self.C1=random.random()

        self.R0L.append(self.R0)
        self.R1L.append(self.R1)
        self.C1L.append(self.C1)

        self.DataDisplay.R0Data['text']=str(self.R0)[0:4]+" Ohm"
        self.DataDisplay.R1Data['text']=str(self.R1)[0:4]+" Ohm"
        self.DataDisplay.C1Data['text']=str(self.C1)[0:4]+" F"

        self.graphDataFrame.ax1.cla()
        self.graphDataFrame.ax1.set_title("R0")
        self.graphDataFrame.ax1.plot (self.R0L[-100:])
        self.graphDataFrame.ax1.set_ylabel("Ohm")
        
        self.graphDataFrame.ax2.cla()
        self.graphDataFrame.ax2.set_title("R1")
        self.graphDataFrame.ax2.plot(self.R1L[-100:])
        self.graphDataFrame.ax2.set_ylabel("Ohm")


        self.graphDataFrame.ax3.cla()
        self.graphDataFrame.ax3.set_title("C1")
        self.graphDataFrame.ax3.plot(self.C1L[-100:])
        self.graphDataFrame.ax3.set_ylabel("F")

        self.graphDataFrame.graph.draw()
            

        #check if the task should sleep or stop
        if(self.continueRunning):
            # print("Parameter",self.pointer)

            self.master.after(100, self.runTask)

class DataShowing(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (70,70)
        self.create_widgets()
    def create_widgets(self):
        self.R0label = ttk.Label(self, text="Resistor0 R0")
        self.R0label.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.R0Data = ttk.Label(self, text="--")
        self.R0Data.grid(row=1, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.R1label = ttk.Label(self, text="Resistor1 C1")
        self.R1label.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.R1Data = ttk.Label(self, text="--")
        self.R1Data.grid(row=3, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.C1label = ttk.Label(self, text="Capacitor1 C1")
        self.C1label.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.C1Data = ttk.Label(self, text="--")
        self.C1Data.grid(row=5, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        # self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(8,6))
        self.ax1 = self.fig.add_subplot(3,1,1)
        self.ax1.set_title("R0")
        self.ax1.set_ylabel("Ohm")

        self.ax2 = self.fig.add_subplot(3,1,2)
        self.ax2.set_title("R1")
        self.ax2.set_ylabel("Ohm")
        
        self.ax3 = self.fig.add_subplot(3,1,3)
        self.ax3.set_title("C1")
        self.ax3.set_ylabel("F")

        self.fig.tight_layout()
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

#Creates the tk class and primary application "voltageContinuousInput"
if __name__=="__main__":
    root = tk.Tk()
    app = SOHorSOC(root)

    #start the application
    app.mainloop()
