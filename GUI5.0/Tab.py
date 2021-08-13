import tkinter as tk 
from tkinter import ttk 
import GUI
import Plot 
import Parameters

def DataPass():
    global frame1 
    global frame2 
    frame2.Data=frame1.Data
    # frame2.pointer=frame1.pointer
    window.after(10,DataPass)

window=tk.Tk()
window.geometry("1150x700")


tab = ttk.Notebook(window)



# frame1 = tk.Frame(tab)
frame1 = GUI.voltageContinuousInput(tab)

tab1 = tab.add(frame1,text="I/V")

frame2 = Parameters.ParameterDisplay(tab)

tab2 = tab.add(frame2,text="Parameters")



frame3 = Plot.SOHorSOC(tab)
tab3 = tab.add(frame3,text="SOC/SOH")

tab.pack(expand=True,fill=tk.BOTH)

window.after(50,DataPass)
window.mainloop()