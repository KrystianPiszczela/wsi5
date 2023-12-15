import pickle
from snake import Snake, Direction
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score


"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""

def prepare_data(file_path):
    with open(file_path, 'rb') as f:
        data_file = pickle.load(f)
    
    inputs = np.empty(0)
    outputs = np.empty(0)

    prev_len_body = 0
    
    for game_state in data_file['data']:

        len_body = len(game_state[0]['snake_body'])
        if len_body >= prev_len_body:
            prev_len_body = len_body
        else:
            inputs = inputs[:-1]
            outputs = outputs[:-1]
            prev_len_body = 0

        inputs = np.append(inputs, game_state_to_data_sample(game_state[0], data_file["bounds"], data_file["block_size"]))
        outputs = np.append(outputs, game_state[1])

    return inputs, outputs


def check_if_bound(head, bounds, attributes):
    if head[0] == 0:
        attributes['PL'] = True
    if head[0] == bounds[0]:
        attributes['PP'] = True
    if head[1] == 0:
        attributes['PG'] = True
    if head[0] == bounds[0]:
        attributes['PD'] = True
    return attributes


def check_if_body(head, body, attributes, block_size):
    for part in body:
        if (part[0] + block_size, part[1]) == head:
            attributes['PL'] = True
        if (part[0] - block_size, part[1]) == head:
            attributes['PP'] = True
        if (part[0], part[1] + block_size) == head:
            attributes['PG'] = True
        if (part[0], part[1] - block_size) == head:
            attributes['PD'] = True
    return attributes  


def check_where_food(head, food, attributes):
    if head[0] > food[0]:
        attributes['FL'] = True
    if head[0] < food[0]:
        attributes['FP'] = True
    if head[1] > food[1]:
        attributes['FG'] = True
    if head[1] < food[1]:
        attributes['FD'] = True
    return attributes


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

    return attributes


class Node:
    def __init__(self, attribute=None, decision=None, child_node_true=None, child_node_false=None):
        self.attribute = attribute
        self.action = decision
        self.child_node_true = child_node_true
        self.child_node_false = child_node_false
    
    def decision(self, game_state):
        if self.action is not None:
            return self.action
        elif game_state[self.attribute] == True:
            return self.child_node_true.decision(game_state)
        elif game_state[self.attribute] == False:
            return self.child_node_false.decision(game_state)


def entropy(classes):

    return (-1) * sum((len(np.where(classes == output)[0])/len(classes))*math.log((len(np.where(classes == output)[0])/len(classes))) for output in np.unique(classes))


def inf_gain(data, classes, attribute):

    idx_true = [index for index, element in enumerate(data) if element[attribute]]
    idx_false = [index for index, element in enumerate(data) if not element[attribute]]

    return entropy(classes) - (((len(idx_true)/len(classes))*entropy(classes[idx_true])) + ((len(idx_false)/len(classes))*entropy(classes[idx_false])))


def get_best_attribute(data, classes, attributes):
    best_attribute = None
    best_gain = 0

    for attribute in attributes:

        gain = inf_gain(data, classes, attribute)

        if gain >= best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute


def build_id3_tree(data, classes, attributes):
    if len(np.unique(classes)) == 0:
        return Node(decision=Direction.DOWN) # tutaj tylko sprawdzam czy to zadziala
    
    if len(np.unique(classes)) == 1:
        return Node(decision=np.unique(classes)[0])
    
    if len(attributes) == 0:
        best_class = None
        best_count = 0
       
        for output in np.unique(classes):
            count = np.count_nonzero(classes == output)

            if count > best_count:
                best_class = output
                best_count = count

        return Node(decision=best_class)
  
    best_attribute = get_best_attribute(data, classes, attributes)

    remain_attributes = deepcopy(attributes)
    remain_attributes.remove(best_attribute)

    idx_true = [index for index, element in enumerate(data) if element[best_attribute]]
    idx_false = [index for index, element in enumerate(data) if not element[best_attribute]]

    child_node_true = build_id3_tree(data[idx_true], classes[idx_true], remain_attributes) 
    child_node_false = build_id3_tree(data[idx_false], classes[idx_false], remain_attributes)  

    return Node(attribute=best_attribute, child_node_true=child_node_true, child_node_false=child_node_false)


def prepare_id3_tree(path="data/snake.pickle"):
    
    inputs, outputs = prepare_data(path)

    l = len(outputs)

    train_inputs = inputs[:int(l*0.8)]
    train_outputs = outputs[:int(l*0.8)]

    return build_id3_tree(train_inputs, train_outputs, ['PP', 'PL', 'PG', 'PD', 'FP', 'FL', 'FG', 'FD'])


if __name__ == "__main__":
    """ Example of how to read a pickled file, feel free to remove this"""
    
    path = "data/snake.pickle"

    inputs, outputs = prepare_data(path)

    l = len(outputs)

    ratio = 0.1

    train_inputs = inputs[:int(l*ratio)]
    train_outputs = outputs[:int(l*ratio)]

    predicted_train_outputs = np.empty(0)

    test_inputs = inputs[int(l*ratio):]
    test_outputs = outputs[int(l*ratio):]

    predicted_test_outputs = np.empty(0)

    model = build_id3_tree(train_inputs, train_outputs, ['PP', 'PL', 'PG', 'PD', 'FP', 'FL', 'FG', 'FD'])

    for i in range(len(train_outputs)):
        predicted_train_outputs = np.append(predicted_train_outputs, model.decision(train_inputs[i]))

    for i in range(len(test_outputs)):
        predicted_test_outputs = np.append(predicted_test_outputs, model.decision(test_inputs[i]))

    predicted_test_outputs_int = [dir.value for dir in predicted_test_outputs]
    test_outputs_int = [dir.value for dir in test_outputs]
    predicted_train_outputs_int = [dir.value for dir in predicted_train_outputs]
    train_outputs_int = [dir.value for dir in train_outputs]

    train_accuracy = accuracy_score(train_outputs_int, predicted_train_outputs_int)

    test_accuracy = accuracy_score(test_outputs_int, predicted_test_outputs_int)

    print('Dokładność zbioru trenującego: ', train_accuracy)

    print('Dokładność zbioru testowego: ', test_accuracy)





