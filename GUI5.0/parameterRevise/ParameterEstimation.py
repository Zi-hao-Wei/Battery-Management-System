import numpy as np
import matplotlib.pyplot as plt
import math
import pandas
import DataShowing

def plot_graph(time, voltage, current,R,Ra,Ca,Rb,Cb):
    # Effects: Plot the values of 5 parameters based on the exponential algorithm
    fig = plt.figure()
    ax1 = fig.add_subplot(711)  # 2 x 1 canvas
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)
    # Label the graph
    ax1.set_title('')
    ax1.set_xlabel('Time/s')
    ax1.set_ylabel('Voltage / V')
    ax2.set_xlabel('Time/s')
    ax2.set_ylabel('Current / A')
    ax3.set_xlabel('Time/s')
    ax3.set_ylabel('R')
    ax4.set_xlabel('Time/s')
    ax4.set_ylabel('Ra')
    ax5.set_xlabel('Time/s')
    ax5.set_ylabel('Ca')
    ax6.set_xlabel('Time/s')
    ax6.set_ylabel('Rb')
    ax7.set_xlabel('Time/s')
    ax7.set_ylabel('Cb')


    # Plot the graph
    ax1.plot(time, voltage, label='Voltage')
    ax2.plot(time, current, label='Current')
    ax3.plot(time,R,label='R')
    ax4.plot(time,Ra,label = 'Ra')
    ax5.plot(time,Ca,label = 'Ca')
    ax6.plot(time,Rb,label = 'Rb')
    ax7.plot(time,Cb,label = 'Cb')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    plt.savefig("pic.png")
    plt.show()


class ParameterEstimation:
    def __init__(self):
        self.store_R = []
        self.store_Ra = []
        self.store_Ca = []
        self.store_Rb = []
        self.store_Cb = []

        #F和G的值都很小

    def pipeline(self,current,voltage):
        # self.time=Data["Step_Time(s)"].tolist()
        self.time=range(0,len(current))
        self.current=current
        self.voltage=voltage
        self.F = [0]
        self.G = [0]
        for i in range(0, len(self.time) - 1):
            self.F.append(self.F[i] + 0.5* (self.voltage[i+1] + self.voltage[i]) * (self.current[i+1] - self.current[i]))
        for i in range(0,len(self.time) - 1):
            self.G.append(self.G[i] + 0.5*(self.F[i+1] + self.F[i]) * (self.current[i+1] - self.current[i]))
        self.Estimation()

    def MatrixInit(self):
        # matrix 5 x 1
        self.a_11 = 0
        self.a_21 = 0
        self.a_31 = 0
        self.a_41 = 0
        self.a_51 = 0

        # matrix 5 x 5
        self.b_11 = 0
        self.b_12 = 0
        self.b_13 = 0
        self.b_14 = 0
        self.b_15 = 0
        self.b_21 = 0
        self.b_22 = 0
        self.b_23 = 0
        self.b_24 = 0
        self.b_25 = 0
        self.b_31 = 0
        self.b_32 = 0
        self.b_33 = 0
        self.b_34 = 0
        self.b_35 = 0
        self.b_41 = 0
        self.b_42 = 0
        self.b_43 = 0
        self.b_44 = 0
        self.b_45 = 0
        self.b_51 = 0
        self.b_52 = 0
        self.b_53 = 0
        self.b_54 = 0
        self.b_55 = len(self.time)

        # matrix 3 x 1
        self.c_11 = 0
        self.c_21 = 0
        self.c_31 = 0

        #matrix 3 x 3
        self.d_11 = len(self.time)
        self.d_12 = 0
        self.d_13 = 0
        self.d_21 = 0
        self.d_22 = 0
        self.d_23 = 0
        self.d_31 = 0
        self.d_32 = 0
        self.d_33 = 0


    def Estimation(self):

        # for window in range(0,len(self.time)-self.windowlength):
        self.MatrixInit()
        for i in range(0,len(self.time)):
            self.a_11 += self.G[i] * self.voltage[i]
            self.a_21 += self.F[i] * self.voltage[i]
            self.a_31 += self.current[i]*self.current[i]*self.voltage[i]
            self.a_41 += self.current[i] * self.voltage[i]
            self.a_51 += self.voltage[i]

            self.b_11 += self.G[i]*self.G[i]
            self.b_12 += self.G[i]*self.F[i]
            self.b_13 += self.G[i]*self.current[i]*self.current[i]
            self.b_14 += self.G[i]*self.current[i]
            self.b_15 += self.G[i]
            self.b_21 += self.G[i]*self.F[i]
            self.b_22 += self.F[i]*self.F[i]
            self.b_23 += self.F[i]*self.current[i]*self.current[i]
            self.b_24 += self.F[i]*self.current[i]
            self.b_25 += self.F[i]
            self.b_31 += self.G[i]*self.current[i]*self.current[i]
            self.b_32 += self.F[i]*self.current[i]*self.current[i]
            self.b_33 += math.pow(self.current[i],4)
            self.b_34 += math.pow(self.current[i],3)
            self.b_35 += math.pow(self.current[i],2)
            self.b_41 += self.G[i]*self.current[i]
            self.b_42 += self.F[i]*self.current[i]
            self.b_43 += math.pow(self.current[i],3)
            self.b_44 += math.pow(self.current[i],2)
            self.b_45 += self.current[i]
            self.b_51 += self.G[i]
            self.b_52 += self.F[i]
            self.b_53 += math.pow(self.current[i],2)
            self.b_54 += self.current[i]

        matrix_5_1 = np.mat([[self.a_11],[self.a_21],[self.a_31],[self.a_41],[self.a_51]])
        matrix_5_5 = np.mat([[self.b_11,self.b_12,self.b_13,self.b_14,self.b_15],[self.b_21,self.b_22,self.b_23,self.b_24,self.b_25],
                        [self.b_31,self.b_32,self.b_33,self.b_34,self.b_35],
                        [self.b_41,self.b_42,self.b_43,self.b_44,self.b_45],
                        [self.b_51,self.b_52,self.b_53,self.b_54,self.b_55]])
        matrix_55inv = np.linalg.pinv(matrix_5_5)
        result = matrix_55inv * matrix_5_1
        
        print(matrix_5_5)
        print(matrix_55inv)

        A = result[0,0]
        B = result[1,0]


        print(A)
        print(B)

        d = 0.5 * (B + math.sqrt(B*B + 4*A))
        f = 0.5 * (B - math.sqrt(B*B + 4*A))



        for i in range(0,len(self.time)):
            self.c_11 += self.voltage[i]
            self.c_21 += self.voltage[i] * math.exp(d * self.current[i])
            self.c_31 += self.voltage[i] * math.exp(f * self.current[i])

            # d_11 = 10
            self.d_12 += math.exp(d * self.current[i])
            self.d_13 += math.exp(f * self.current[i])
            self.d_21 += math.exp(d * self.current[i])
            self.d_22 += math.exp(2*d*self.current[i])
            self.d_23 += math.exp((d+f)*self.current[i])
            self.d_31 += math.exp(f * self.current[i])
            self.d_32 += math.exp((d+f) * self.current[i])
            self.d_33 += math.exp(2* f * self.current[i])


        matrix_3_1 = np.mat([[self.c_11],[self.c_21],[self.c_31]])
        matrix_3_3 = np.mat([[self.d_11,self.d_12,self.d_13],[self.d_21,self.d_22,self.d_23],[self.d_31,self.d_32,self.d_33]])
        inv_33 = np.linalg.pinv(matrix_3_3)

        result_abc = inv_33 * matrix_3_1
        a = result_abc[0,0]
        b = result_abc[1,0]
        c = result_abc[2,0]

        R = a
        if b>0:
            Ra = -b/-self.current[0]
        else:
            Ra = -b/self.current[0]
        if Ra* c < 0:
            Ca = -1/(Ra * c)
        else:
            Ca = 1/ (Ra * c)
        if d>0:
            Rb = -d / -self.current[0]
        else:
            Rb = -d / self.current[0]

        if Rb *f ==0:
            Cb = 0
        else:
            Cb = -1/(Rb*f)
        


        self.store_R.append(R)
        self.store_Ra.append(Ra)
        self.store_Cb.append(Cb)
        self.store_Ca.append(Ca)
        self.store_Rb.append(Rb)

if __name__=="__main__":
    AllData=DataShowing.getData()
    Split=DataShowing.DivideData(AllData)
    temp=ParameterEstimation()
    for PulseData in Split:
        temp.pipeline(PulseData)



