import numpy as np
from scipy.signal import convolve2d as conv2
import random

class Estado:
    def __init__(self, matriz):
        self.matriz = matriz  # Matriz 3x3 do estado

    def __eq__(self, other):
        return np.array_equal(self.matriz, other.matriz)

    def mostrar(self):
        print(self.matriz)
        print()

def acoes_permitidas(estado):
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    blank = estado.matriz == 9
    mask = conv2(blank, adj, 'same')
    return estado.matriz[np.where(mask)]

def movimentar(estado, acao):
    matriz = estado.matriz.copy()
    pos_blank = np.where(matriz == 9)
    pos_acao = np.where(matriz == acao)
    matriz[pos_blank], matriz[pos_acao] = matriz[pos_acao], matriz[pos_blank]
    return Estado(matriz=matriz)

def manhattan(estado, objetivo):
    distancia = 0
    for i in range(1, 9):  # Ignora o espaço vazio (9)
        pos_estado = np.where(estado.matriz == i)
        pos_objetivo = np.where(objetivo.matriz == i)
        distancia += abs(pos_estado[0][0] - pos_objetivo[0][0]) + abs(pos_estado[1][0] - pos_objetivo[1][0])
    return distancia

def mcts_solver(estado_inicial, objetivo, n_clones=100, max_jogadas=20, max_iteracoes=1000):
    estado_atual = estado_inicial
    iteracao = 0

    while iteracao < max_iteracoes:
        if estado_atual == objetivo:
            return True, iteracao  # Objetivo alcançado

        # Passo 1: Gerar clones para cada ação possível
        acoes = acoes_permitidas(estado_atual)
        if acoes.size == 0:
            return False, iteracao  # Sem ações possíveis

        medias = {}
        for acao in acoes:
            heuristicas = []
            for _ in range(n_clones):
                # Passo 2: Simular jogadas aleatórias a partir da ação atual
                clone = movimentar(estado_atual, acao)
                for _ in range(max_jogadas):
                    acoes_clone = acoes_permitidas(clone)
                    if acoes_clone.size == 0:
                        break
                    acao_aleatoria = np.random.choice(acoes_clone)
                    clone = movimentar(clone, acao_aleatoria)
                # Passo 3: Calcular heurística do clone
                heuristica = manhattan(clone, objetivo)
                heuristicas.append(heuristica)
            # Passo 4: Média das heurísticas para a ação
            medias[acao] = np.mean(heuristicas)

        # Passo 5: Escolher a ação com a menor média de heurística
        melhor_acao = min(medias, key=lambda k: medias[k])

        # Passo 6: Atualizar estado atual
        estado_atual = movimentar(estado_atual, melhor_acao)
        iteracao += 1

        # Mostrar progresso
        print(f"Iteração {iteracao}: Melhor ação = {melhor_acao}, Média heurística = {medias[melhor_acao]:.2f}")
        estado_atual.mostrar()

    return False, iteracao  # Não encontrou solução

# Exemplo de uso
def gerar_estado_aleatorio():
    numeros = list(range(1, 10))
    random.shuffle(numeros)
    matriz = np.array(numeros).reshape(3, 3)
    return Estado(matriz=matriz)

def matriz_para_vetor(matriz):
    return matriz.flatten().tolist()

def is_solvable(estado):
   
    lista = matriz_para_vetor(estado.matriz)

    inversoes = 0
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                inversoes += 1

    # Se o número de inversões for par, o estado é solucionável
    return inversoes % 2 == 0 

while True:
    estado = gerar_estado_aleatorio()
    if is_solvable(estado):
        break

objetivo = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
estado_inicial = Estado(matriz=np.array([[1, 3, 2], [5, 9, 4], [7, 8, 6]]))

sucesso, iteracoes = mcts_solver(estado_inicial, objetivo, n_clones=50, max_jogadas=10)
print(f"Solução encontrada: {sucesso} em {iteracoes} iterações.")
# Parâmetros para geração dos dados
M = 1000  # Número de pontos de dados gerados

x = np.linspace(0, 2, M)  # Pontos espaçados uniformemente no intervalo [0, 1]
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, M)
# Adiciona ruído gaussiano (média 0, desvio padrão 0.15)

# Geração dos dados para a curva teórica
xr = np.linspace(0, 2, NR)  # Pontos espaçados uniformemente para a curva suave
yn = np.sin(2 * np.pi * xr)  # Valores da função seno sem ruído

line, = plt.plot(xr, yn, label='sin(2πx)')  # Plota a curva teórica
plt.scatter(x, y, label='Dados com ruído')  # Plota os pontos de dados com ruído
plt.legend(handles=[line])  # Adiciona a legenda ao gráfico
plt.xlabel('x')  # Rótulo do eixo x
plt.ylabel('y')  # Rótulo do eixo y
plt.title('Dados Gerados e Função Teórica')  # Título do gráfico
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.show()  # Exibe o gráfico
