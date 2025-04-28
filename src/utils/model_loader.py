import torch
import torch.nn as nn
from src.utils.config import MODEL_PATH
class MultiClassnn(nn.Module):
    def __init__(self , num_features,num_classes,hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_features , hidden_size)
        self.fc2  = nn.Linear(hidden_size , num_classes) 
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self , X):
        X = self.fc1(X)
        X = torch.relu(X) 
        X = self.fc2(X)
        X = self.logsoftmax(X)
        return X
    
def load_model():
    # Load the model architecture
    num_features = 4  
    num_classes = 3   
    hidden_size = 5 
    model = MultiClassnn(num_features, num_classes, hidden_size)

    # Load the model weights

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    return model
