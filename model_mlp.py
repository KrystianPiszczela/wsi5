import torch
import torch.nn as nn
from snake import Direction
import numpy as np
from model import check_if_body, check_if_bound, check_where_food
import pickle
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BCDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        # self.labels = {0: Directrion.UP, 1: Directrion.RIGHT, 2: Directrion.DOWN, 3: Directrion.LEFT}
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        # print(self.labels[idx])
        sample = {'data': torch.tensor(self.data[idx]), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample

class MLP(nn.Module):
    def __init__(self, hidden_size, num_of_layers):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4)
        self.num_of_layers = num_of_layers
        

    def forward(self, x):
        x = self.fc1(x.float())
        x = self.fc2(x.float())
        x = self.fc3(x.float())

        return x
    
def game_state_to_data_sample(game_state: dict, bounds, block_size):
    attributes = {
        'PD': False,
        'PG': False,
        'PL': False,
        'PP': False,
        'FG': False,
        'FD': False,
        'FL': False,
        'FP': False
    }

    head = game_state['snake_body'][-1]

    attributes = check_if_bound(head, bounds, attributes)

    attributes = check_if_body(head, game_state['snake_body'], attributes, block_size)

    attributes = check_where_food(head, game_state['food'], attributes)

    at = []

    for value in attributes.values():
        if value:
            at.append(1)
        else:
            at.append(0)

    return at
    
def prepare_data(file_path):
    with open(file_path, 'rb') as f:
        data_file = pickle.load(f)
    
    inputs = []
    outputs = []

    prev_len_body = 0
    
    for game_state in data_file['data']:

        len_body = len(game_state[0]['snake_body'])
        if len_body >= prev_len_body:
            prev_len_body = len_body
        else:
            inputs = inputs[:-1]
            outputs = outputs[:-1]
            prev_len_body = 0

        inputs.append(game_state_to_data_sample(game_state[0], data_file["bounds"], data_file["block_size"]))
        # outputs.append(game_state[1].value)
        out = [0, 0, 0, 0]
        out[game_state[1].value] = 1
        outputs.append(out)

    return inputs, outputs

if __name__ == "__main__":
    X, Y = prepare_data("data/snake.pickle")

    l = len(Y)

    ratio = 0.8

    train_inputs = X[:int(l*ratio)]
    train_outputs = Y[:int(l*ratio)]

    test_inputs = X[int(l*ratio):]
    test_outputs = Y[int(l*ratio):]

    # np.set_printoptions(threshold=np.inf)
    # print(train_inputs)

    train_ds = BCDataset(train_inputs, train_outputs)
    test_ds = BCDataset(test_inputs, test_outputs)

    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=64, shuffle=False)

    model = MLP(64, 5).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 3
    epochs = tqdm(range(num_epochs))

    for epoch in epochs:
        train_loss = []
        model.train()
        for batch in (train_dl):
            print(batch)
            optim.zero_grad()
            x = batch['data'].reshape(-1, 8).to(device)
            y = batch['label'].to(device)

            output = model(x)
            print(output)
            print(torch.argmax(output, dim=1))
            loss = criterion(torch.argmax(output, dim=1),y)

            loss.backward()
            optim.step()
            train_loss.append(loss.item())

        loss_now = np.mean(train_loss)
        epochs.set_postfix({'loss': loss_now})

    print(train_loss)