import numpy as np

class RedeNeural:
    def __init__(self, tamanho_entrada, tamanho_oculta, tamanho_saida, taxa_aprendizado=0.1):
        # Inicializa os pesos e os biases para as camadas ocultas e de saída
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos_entrada_oculta = np.random.randn(tamanho_entrada, tamanho_oculta)
        self.pesos_oculta_saida = np.random.randn(tamanho_oculta, tamanho_saida)
        self.bias_oculta = np.random.randn(tamanho_oculta)
        self.bias_saida = np.random.randn(tamanho_saida)
        self.threshold_ativacao = 0.5

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
        return 1 if x >= self.threshold_ativacao else 0

    def forward_pass(self, inputs):
        """
        Implementa a etapa de inferência (feedforward) do MLP.
        Calcula as saídas da camada oculta e da camada de saída.
        """
        self.hidden_input = np.dot(inputs, self.pesos_entrada_oculta) + self.bias_oculta
        self.hidden_output = self._sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.pesos_oculta_saida) + self.bias_saida
        self.final_output = self._sigmoid(self.final_input)
        return self.final_output

    def backward_pass(self, inputs, target_output, output):
        """
        Implementa a etapa de retropropagação (backpropagation) do MLP.
        Atualiza os pesos e biases com base nos erros calculados.
        """
        output_error = target_output - output
        output_delta = output_error * self._sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.pesos_oculta_saida.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Atualiza os pesos e biases das camadas de saída e oculta
        self.pesos_oculta_saida += np.outer(self.hidden_output, output_delta) * self.taxa_aprendizado
        self.pesos_entrada_oculta += np.outer(inputs, hidden_delta) * self.taxa_aprendizado
        self.bias_saida += output_delta * self.taxa_aprendizado
        self.bias_oculta += hidden_delta * self.taxa_aprendizado

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
inputs = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])

targets = np.array(
    [[0],
     [1],
     [1],
     [0]])

# Inicialização da rede neural
rede_neural = RedeNeural(tamanho_entrada=2, tamanho_oculta=2, tamanho_saida=1, taxa_aprendizado=0.1)

# Treinamento da rede neural
rede_neural.train(inputs, targets)

# Testando a rede neural
for x in inputs:
    print(f"Input: {x} Output: {rede_neural.predict(x)}")
