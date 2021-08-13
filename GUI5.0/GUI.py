import nidaqmx
import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 


class voltageContinuousInput(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #Configure root tk class
        # self.master = master
        # self.master.title("Voltage - Continuous Input")
        # self.master.geometry("1200x600")
        self.create_widgets()
        self.pack()
        self.run = False
        self.continueRunning=False
        self.pointer=0
        self.voltage=[]
        self.current=[]
        self.voltageAll=[]
        self.currentAll=[]

        self.Data=pd.DataFrame({'Voltage':self.voltage,'Current':self.current})



    def create_widgets(self):
        #The main frame is made up of three subframes
        self.channelSettingsFrame = channelSettings(self, title ="Channel Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20,0), padx=(20,20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20,0), padx=(20,20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20,0), ipady=10)


    def startTask(self):
        #Prevent user from starting task a second time
        self.inputSettingsFrame.startButton['state'] = 'disabled'

        #Shared flag to alert task if it should stop
        self.continueRunning = True 

        #Get task settings from the user
        currentChannel = self.channelSettingsFrame.physicalCurrentChannelEntry.get()
        voltageChannel = self.channelSettingsFrame.physicalVoltageChannelEntry.get()

        # maxVoltage = int(self.channelSettingsFrame.maxVoltageEntry.get())
        # minVoltage = int(self.channelSettingsFrame.minVoltageEntry.get())
        sampleRate = int(self.inputSettingsFrame.sampleRateEntry.get())
        self.numberOfSamples = int(self.inputSettingsFrame.numberOfSamplesEntry.get()) #Have to share number of samples with runTask
        self.numberOfShown = int(self.inputSettingsFrame.numberOfShownEntry.get()) #Have to share number of shown with runTask


        #Create and start task
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(voltageChannel)
        self.task.ai_channels.add_ai_current_chan(currentChannel)
        self.task.timing.cfg_samp_clk_timing(sampleRate,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,samps_per_chan=self.numberOfSamples*3)
        self.task.start()

        

        #spin off call to check 
        self.master.after(10, self.runTask)

    def runTask(self):
        #Check if task needs to update the graph
        samplesAvailable = self.task._in_stream.avail_samp_per_chan
        if(samplesAvailable >= self.numberOfSamples):
            vals = self.task.read(self.numberOfSamples)
            self.voltage=self.voltage[-self.numberOfShown+len(vals[0]):]+vals[0]
            self.current=self.current[-self.numberOfShown+len(vals[1]):]+vals[1]

            self.voltageAll=self.voltageAll+vals[0]
            self.currentAll=self.currentAll+vals[1]
            self.graphDataFrame.ax1.cla()
            self.graphDataFrame.ax1.set_title("Current Data")
            self.graphDataFrame.ax1.plot(self.current)
            self.graphDataFrame.ax1.set_ylabel("A")
            
            self.graphDataFrame.ax2.cla()
            self.graphDataFrame.ax2.set_title("Voltage Data")
            self.graphDataFrame.ax2.set_ylabel("V")

            self.graphDataFrame.ax2.plot(self.voltage)
            self.graphDataFrame.graph.draw()
                        

        #check if the task should sleep or stop
        if(self.continueRunning):
            if(self.pointer==5):
                self.Data=pd.DataFrame({'Voltage':self.voltage,'Current':self.current})
                # self.Data.to_csv("Data.csv")
            else:
                print("GUI",self.pointer)
                self.pointer+=1

            voltage=self.voltage
            current=self.current
           
            self.master.after(10, self.runTask)

        
        else:
            self.task.stop()
            # with open("./voltage.txt","w+") as f:
            # 	f.write(",".join('%s' %id for id in self.voltageAll))
            # with open("./current.txt","w+") as f:
            # 	f.write(",".join('%s' %id for id in self.currentAll))
            
            self.voltageAll=[]
            self.currentAll=[]

            self.task.close()
            self.inputSettingsFrame.startButton['state'] = 'enabled'

    def stopTask(self):
        #call back for the "stop task" button
        self.continueRunning = False

class channelSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30,30)
        self.create_widgets()

    def create_widgets(self):

        self.physicalCurrentChannelLabel = ttk.Label(self, text="Current Channel")
        self.physicalCurrentChannelLabel.grid(row=0,sticky='w', padx=self.xPadding, pady=(10,0))

        self.physicalCurrentChannelEntry = ttk.Entry(self)
        self.physicalCurrentChannelEntry.insert(0, "dev1/ai0")
        self.physicalCurrentChannelEntry.grid(row=1, sticky="ew", padx=self.xPadding)

        self.physicalVoltageChannelLabel = ttk.Label(self, text="Voltage Channel")    
        self.physicalVoltageChannelLabel.grid(row=2,sticky='w', padx=self.xPadding, pady=(10,0))

        self.physicalVoltageChannelEntry = ttk.Entry(self)
        self.physicalVoltageChannelEntry.insert(0, "dev1/ai4")
        self.physicalVoltageChannelEntry.grid(row=3, sticky="ew", padx=self.xPadding)

class inputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (30,30)
        self.create_widgets()

    def create_widgets(self):
        self.sampleRateLabel = ttk.Label(self, text="Sample Rate")
        self.sampleRateLabel.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.sampleRateEntry = ttk.Entry(self)
        self.sampleRateEntry.insert(0, "1000")
        self.sampleRateEntry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.numberOfSamplesLabel = ttk.Label(self, text="Number of Samples")
        self.numberOfSamplesLabel.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.numberOfSamplesEntry = ttk.Entry(self)
        self.numberOfSamplesEntry.insert(0, "1000")
        self.numberOfSamplesEntry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.numberOfShownLabel = ttk.Label(self, text="Number of Shown")
        self.numberOfShownLabel.grid(row=4, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10,0))

        self.numberOfShownEntry = ttk.Entry(self)
        self.numberOfShownEntry.insert(0, "10000")
        self.numberOfShownEntry.grid(row=5, column=0, columnspan=2, sticky='ew', padx=self.xPadding)


        self.startButton = ttk.Button(self, text="Start Task", command=self.parent.startTask)
        self.startButton.grid(row=6, column=0, sticky='w', padx=self.xPadding, pady=(10,0))

        self.stopButton = ttk.Button(self, text="Stop Task", command=self.parent.stopTask)
        self.stopButton.grid(row=6, column=1, sticky='e', padx=self.xPadding, pady=(10,0))

class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        # self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(7,5))
        self.ax1 = self.fig.add_subplot(2,1,1)
        self.ax1.set_title("Current Data")
        self.ax1.set_ylabel("A")

        self.ax2 = self.fig.add_subplot(2,1,2)
        self.ax2.set_title("Voltage Data")
        self.ax2.set_ylabel("V")

        self.fig.tight_layout()
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

#Creates the tk class and primary application "voltageContinuousInput"
if __name__=="__main__":
    root = tk.Tk()
    app = voltageContinuousInput(root)

    #start the application
    app.mainloop()
