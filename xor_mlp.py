import numpy as np
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Inicializa os pesos e os biases para as camadas ocultas e de saída
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)
        self.activation_threshold = 0.5

    def _sigmoid(self, x):
        """
        Implementa a função sigmoide.
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Derivada da função sigmoide.
        Necessária para o cálculo do backpropagation.
        """
        return x * (1 - x)

    def _activation(self, x):
        """
        Implementação da função de ativação do perceptron.
        Escolha uma das funções de ativação possíveis.
        """
        return 1 if x >= self.activation_threshold else 0

    def forward_pass(self, inputs):
        """
        Implementa a etapa de inferência (feedforward) do MLP.
        Calcula as saídas da camada oculta e da camada de saída.
        """
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        return self.final_output

    def backward_pass(self, inputs, target_output, output):
        """
        Implementa a etapa de retropropagação (backpropagation) do MLP.
        Atualiza os pesos e biases com base nos erros calculados.
        """
        output_error = target_output - output
        output_delta = output_error * self._sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Atualiza os pesos e biases das camadas de saída e oculta
        self.weights_hidden_output += np.outer(self.hidden_output, output_delta) * self.learning_rate
        self.weights_input_hidden += np.outer(inputs, hidden_delta) * self.learning_rate
        self.bias_output += output_delta * self.learning_rate
        self.bias_hidden += hidden_delta * self.learning_rate

    def train(self, inputs, targets, epochs=20000):
        """
        Implementa o processo de treinamento utilizando gradiente descendente e backpropagation.
        Treina a rede ao longo de várias épocas.
        """
        for epoch in range(epochs):
            total_error = 0
            for x, y in zip(inputs, targets):
                output = self.forward_pass(x)
                self.backward_pass(x, y, output)
                total_error += np.sum((y - output) ** 2)

            # Imprime a taxa de erro a cada 1000 épocas
            if epoch % 1000 == 0:
                print(f"Época {epoch}; Taxa de erro: {total_error}")

    def predict(self, inputs):
        """
        Realiza a predição para novos dados de entrada.
        """
        output = self.forward_pass(inputs)
        return self._activation(output)


# Dados de entrada para a porta XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Inicialização do MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

# Treinamento do MLP
mlp.train(inputs, targets)

# Testando o MLP
for x in inputs:
    print(f"Input: {x} Output: {mlp.predict(x)}")
