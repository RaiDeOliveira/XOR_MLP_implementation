import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

# Dados de entrada para a porta XOR
entradas = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])

alvos = np.array(
    [[0],
     [1],
     [1],
     [0]])

# Convertendo os dados para tensores do PyTorch
entradas = torch.tensor(entradas, dtype=torch.float32)
alvos = torch.tensor(alvos, dtype=torch.float32)

# Inicialização do modelo MLP
tamanho_entrada = 2
tamanho_oculta = 2
tamanho_saida = 1
mlp = MLP(tamanho_entrada, tamanho_oculta, tamanho_saida)

# Definição do otimizador e da função de perda
otimizador = optim.SGD(mlp.parameters(), lr=0.1)
criterio = nn.MSELoss()

# Treinamento do modelo
epocas = 20000
for epoca in range(epocas):
    otimizador.zero_grad()
    saidas = mlp(entradas)
    perda = criterio(saidas, alvos)
    perda.backward()
    otimizador.step()
    
    if epoca % 1000 == 0:
        print(f"Época {epoca}; Perda: {perda.item()}")

# Testando o modelo
with torch.no_grad():
    for x in entradas:
        saida = mlp(x)
        previsao = 1 if saida.item() >= 0.5 else 0
        print(f"Input: {x.tolist()} Output: {previsao}")
