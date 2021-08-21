import torch.nn as nn
class lstm(nn.Module):
    def __init__(self,input_size=250,hidden_size=500,num_layers=2,output_size=20,dropout=0.2,batch_first=True):
        super(lstm,self).__init__()

        self.hidden_size=hidden_size


        self.input_size=input_size


        self.num_layers=num_layers
        self.output_size=output_size
        self.dropout=dropout
        self.batch_first=batch_first

        self.lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=self.batch_first,dropout=self.dropout)


        self.outLinear1=nn.Linear(self.hidden_size*5,self.hidden_size)
        self.outLinear2=nn.Linear(self.hidden_size,250)
        self.outLinear3=nn.Linear(250,1)


        self.relu=nn.ReLU(inplace=False)
        

        self.linear1=nn.Linear(2000,1000)
        self.linear2=nn.Linear(1000,500)
        self.linear3=nn.Linear(500,250)

        self.dropout=nn.Dropout(p=dropout)


    def forward(self,x):
     
        # print("Before Linear",x.shape)

        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.linear2(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.linear3(x)
        x=self.relu(x)
        x=self.dropout(x)

        # print("After Linear",x.shape)

        x,(_,_)=self.lstm(x)

        # print("After LSTM",x.shape)

        x=x.reshape(x.shape[0],-1)
        x=self.relu(x)

        # print("After Reshape",x.shape)

        x=self.outLinear1(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.outLinear2(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.outLinear3(x)
        return x
        # return out