import torch
import torch.nn as nn
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from model import check_if_body, check_if_bound, check_where_food
import pickle
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter()


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
        if num_of_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(8, 4)])
        elif num_of_layers == 2:
            self.fc = nn.ModuleList([nn.Linear(8, hidden_size)])
            self.fc.append(nn.Linear(hidden_size, 4))
        else:
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


def calculate_valid_accuracy(model, val_dl, criterion):
    model.eval()
    val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)
    val_loss = []

    with torch.no_grad():
        for val_batch in val_dl:
            val_x = val_batch['data'].reshape(-1, 8).to(device)
            val_y = val_batch['label'].to(device)
            val_output = model(val_x)
            val_loss_batch = criterion(val_output, torch.argmax(val_y, dim=1))
            val_loss.append(val_loss_batch.item())
            val_accuracy_metric(torch.argmax(val_output, dim=1), torch.argmax(val_y, dim=1))
    return val_loss, val_accuracy_metric


def log_valid_accuracy(val_loss, val_accuracy_metric, epoch):
    val_loss_now = np.mean(val_loss)
    logger.info(f'Validation Loss: {val_loss_now}')
    writer.add_scalar('Loss/Validation', val_loss_now, epoch)
    val_accuracy = val_accuracy_metric.compute()
    logger.info(f'Validation Accuracy: {val_accuracy}')
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)


def log_train_accuracy(train_loss, train_accuracy_metric, epoch, epochs):
    loss_now = np.mean(train_loss)
    logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_now}')
    writer.add_scalar('Loss/Train', loss_now, epoch)
    train_accuracy = train_accuracy_metric.compute()
    logger.info(f'Training Accuracy: {train_accuracy}')
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)


def prepare_MLP_model(hidden_size, num_of_layers, activ_fun):
    X, Y = prepare_data("data/snake.pickle")

    l = len(Y)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(l * train_ratio)
    val_size = int(l * val_ratio)
    test_size = int(l * test_ratio)

    train_inputs, val_test_inputs = X[:train_size], X[train_size:]
    train_outputs, val_test_outputs = Y[:train_size], Y[train_size:]

    val_inputs, test_inputs = val_test_inputs[:val_size], val_test_inputs[val_size:]
    val_outputs, test_outputs = val_test_outputs[:val_size], val_test_outputs[val_size:]

    # np.set_printoptions(threshold=np.inf)
    # print(train_inputs)

    train_ds = BCDataset(train_inputs, train_outputs)
    val_ds = BCDataset(val_inputs, val_outputs)
    test_ds = BCDataset(test_inputs, test_outputs)

    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=64, shuffle=True)
    val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=64, shuffle=False)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=64, shuffle=False)

    model = MLP(hidden_size, num_of_layers, activ_fun).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 100
    epochs = tqdm(range(num_epochs))
    train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)

    first_epoch = True
    for epoch in epochs:
        train_loss = []
        model.train()

        for batch in (train_dl):
            # print(batch)
            optim.zero_grad()
            x = batch['data'].reshape(-1, 8).to(device)
            y = batch['label'].to(device)

            output = model(x)

            # print(output)
            # print(y)
            # print(1)
            # print(output)
            predicted_classes = torch.argmax(output, dim=1)
            # print(2)
            # print(predicted_classes)
            y_classes = torch.argmax(y, dim=1)
            # print(2)
            # print(y_classes)

            loss = criterion(output, y_classes)
            # loss = criterion(predicted_classes, y_classes)
            # print(3)
            # print(loss)
            # loss.requires_grad = True
            loss.backward()
            optim.step()
            train_loss.append(loss.item())

            if first_epoch:
                for i, (name, param) in enumerate(model.named_parameters()):
                    if 'weight' in name:
                        grad_matrix_norm = torch.norm(param.grad, p='fro') / np.sqrt(param.grad.numel())
                        print(f'Layer: {name}, Gradient Matrix Norm: {grad_matrix_norm.item()}')

                first_epoch = False
            train_accuracy_metric(predicted_classes, y_classes)

        log_train_accuracy(train_loss, train_accuracy_metric, epoch, epochs)
        val_loss, val_accuracy_metric = calculate_valid_accuracy(model, val_dl, criterion)
        log_valid_accuracy(val_loss, val_accuracy_metric, epoch)

    writer.close()
    return model


if __name__ == "__main__":
    # 1, 2, 5, 30
    num_of_layers = 1
    activ_fun = "ReLU"
    # 'identity''ReLU''LeakyReLU''sigmoid'
    prepare_MLP_model(1024, num_of_layers, activ_fun)
