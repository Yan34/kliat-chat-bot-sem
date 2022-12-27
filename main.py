import random
import json
import re
from texttable import Texttable

import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

bot_name = "Федор"

# debug
# for intent in intents['intents']:
#     print(f"{bot_name}: {random.choice(intent['responses'])}")
#     if intent['output_type'] == "tab":
#         table = Texttable()
#         tab = intent['response_tab']
#         for s in tab:
#             if tab.index(s) == 0:
#                 table.header(s)
#             else:
#                 table.add_row(list(s))
#         print(table.draw())
# end debug

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("Давайте общаться! (напишите 'закончить', чтобы выйти)")

while True:
    # sentence = "do you use credit cards?"
    sentence = input("Вы: ")
    if re.match("(?i)^закончить$", sentence):
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                if intent['output_type'] == "tab":
                    table = Texttable()
                    tab = intent['response_tab']
                    for s in tab:
                        if tab.index(s) == 0:
                            table.header(s)
                        else:
                            table.add_row(list(s))
                    print(table.draw())
    else:
        print(f"{bot_name}: Я не понимаю...")
