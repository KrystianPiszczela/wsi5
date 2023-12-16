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
    activation_funs = {'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU, 'identity': nn.Identity, 'sigmoid': nn.Sigmoid}

    def __init__(self, hidden_size, num_of_layers, activation_fun='ReLU'):
        super(MLP, self).__init__()
        self.num_of_layers = num_of_layers
        self.fc = nn.ModuleList([nn.Linear(8, hidden_size)])
        for _ in range(self.num_of_layers - 2):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size, 4))
        self.hidden_activation = MLP.activation_funs[activation_fun]()

    def forward(self, x):
        for i in range(self.num_of_layers - 1):
            x = self.fc[i](x.float())
            x = self.hidden_activation(x)
        x = self.fc[-1](x.float())
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
            predicted_classes = torch.argmax(output, dim=1)
            print(predicted_classes)

            loss = criterion(output, predicted_classes)

            loss.backward()
            optim.step()
            train_loss.append(loss.item())

        loss_now = np.mean(train_loss)
        epochs.set_postfix({'loss': loss_now})

    print(train_loss)
