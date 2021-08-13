import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 


class SOHorSOC(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)
        # self.master = master

        self.create_widgets()
        self.pack()
        self.run = True
        self.startTask()

    def create_widgets(self):
        
        self.DataDisplay = DataShowing(self, title="SOC/SOH Display")
        self.DataDisplay.grid(row=0, column=1, pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0,column=2, pady=(20,0), ipady=10)


    def startTask(self):
        #Prevent user from starting task a second time
        # self.inputSettingsFrame.startButton['state'] = 'disabled'

        #Shared flag to alert task if it should stop
        self.continueRunning = True

        self.SOC=100
        self.SOH=100



        #spin off call to check 
        self.master.after(10, self.runTask)
        


    def runTask(self):
        #Check if task needs to update the graph
        # print(self.SOC)
        self.SOC=self.SOC-1
        self.SOH=self.SOH-2


        if(self.SOC<=0):
            self.SOC=100
        if(self.SOH<=0):
            self.SOH=100

        self.DataDisplay.SOCData['text']=str(self.SOC)[0:4]+"%"
        self.DataDisplay.SOHData['text']=str(self.SOH)[0:4]+"%"
        self.graphDataFrame.ax1.cla()
        self.graphDataFrame.ax1.set_title("SOC")
        self.graphDataFrame.ax1.pie([self.SOC,100-self.SOC],labels=["SOC",""])
        self.graphDataFrame.ax1.axis("equal")
        
        self.graphDataFrame.ax2.cla()
        self.graphDataFrame.ax2.set_title("SOH")
        self.graphDataFrame.ax2.axis("equal")

        self.graphDataFrame.ax2.pie([self.SOH,100-self.SOH],labels=["SOH",""])
        self.graphDataFrame.graph.draw()
            

        #check if the task should sleep or stop
        if(self.continueRunning):
            self.master.after(10, self.runTask)

class DataShowing(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (70,70)
        self.create_widgets()

    def create_widgets(self):
        self.SOClabel = ttk.Label(self, text="SOC")
        self.SOClabel.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.SOCData = ttk.Label(self, text="100%")
        self.SOCData.grid(row=1, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.SOHlabel = ttk.Label(self, text="SOH")
        self.SOHlabel.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))
        self.SOHData = ttk.Label(self, text="100%")
        self.SOHData.grid(row=3, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        # self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(8,4))
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax1.pie([0,100])
        self.ax1.set_title("SOC")

        self.ax2 = self.fig.add_subplot(1,2,2)
        self.ax2.pie([0,100])
        self.ax2.set_title("SOH")

        # self.fig.tight_layout()
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

#Creates the tk class and primary application "voltageContinuousInput"
if __name__=="__main__":
    root = tk.Tk()
    app = SOHorSOC(root)

    #start the application
    app.mainloop()
