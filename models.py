import torch
import torch.nn as nn
import os


MODELS_SAVE_DIR = 'models'
if not os.path.isdir(MODELS_SAVE_DIR):
    os.mkdir(MODELS_SAVE_DIR)


class ValueNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, lr,
                 model_name:str='untitled', load_saved_model=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        self.model_save_path = os.path.join(MODELS_SAVE_DIR, model_name+'.dat')

        if load_saved_model:
            if os.path.exists(self.model_save_path):
                self.load_state_dict(torch.load(self.model_save_path))
                print('Loaded model from', self.model_save_path)
            else:
                print('Model state dict not found.')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_on_datapoint(self, V_S, V_S_a_max):
        loss = self.loss_fn(V_S, V_S_a_max)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self):
        torch.save(self.state_dict(), self.model_save_path)
        print('Saved model to', self.model_save_path)

