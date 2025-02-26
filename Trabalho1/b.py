import numpy as np
import random
from queue import PriorityQueue

class Estado:
    def __init__(self, pai=None, matriz=None):
        self.pai = pai
        self.matriz = matriz
        self.d = 0  # Tamanho do caminho inicio - estado
        self.p = 0  # Prioridade
    
    def __eq__(self, other):
        return np.array_equal(self.matriz, other.matriz)
    
    def __hash__(self):
        return hash(tuple(self.matriz.flatten()))
    
    def mostrar(self):
        for linha in self.matriz:
            print(linha)
        print()

def acoes_permitidas(estado):
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    blank = estado.matriz == 9
    mask = np.convolve(blank.flatten(), adj.flatten(), 'same').reshape(3, 3)
    return estado.matriz[np.where(mask)]

def movimentar(estado, c):
    matriz = estado.matriz.copy()
    blank_pos = np.where(matriz == 9)
    c_pos = np.where(matriz == c)
    matriz[blank_pos], matriz[c_pos] = matriz[c_pos], matriz[blank_pos]
    return Estado(matriz=matriz)

class NodoMCTS:
    def __init__(self, estado, pai=None):
        self.estado = estado
        self.pai = pai
        self.filhos = []
        self.visitas = 0
        self.vitorias = 0

    def selecionar_filho(self):
        if not self.filhos:
            return self
        return max(self.filhos, key=lambda x: (x.vitorias / (x.visitas + 1)) + 1.41 * (2 * np.log(self.visitas + 1) / (x.visitas + 1)) ** 0.5)

    def expandir(self):
        acoes = acoes_permitidas(self.estado)
        for acao in acoes:
            novo_estado = movimentar(self.estado, acao)
            self.filhos.append(NodoMCTS(novo_estado, pai=self))
        return random.choice(self.filhos) if self.filhos else self

    def simular(self, objetivo):
        estado_simulacao = self.estado
        for _ in range(10):  # Simula até 10 movimentos aleatórios
            acoes = acoes_permitidas(estado_simulacao)
            if acoes.size == 0:
                break
            estado_simulacao = movimentar(estado_simulacao, random.choice(acoes))
            if np.array_equal(estado_simulacao.matriz, objetivo):
                return 1  # Vitória
        return 0  # Falha

    def retropropagar(self, resultado):
        self.visitas += 1
        self.vitorias += resultado
        if self.pai:
            self.pai.retropropagar(resultado)

def monte_carlo_tree_search(estado_inicial, objetivo, iteracoes=1000):
    raiz = NodoMCTS(estado_inicial)
    for _ in range(iteracoes):
        nodo = raiz.selecionar_filho()
        nodo = nodo.expandir()
        resultado = nodo.simular(objetivo)
        nodo.retropropagar(resultado)
    melhor_filho = max(raiz.filhos, key=lambda x: x.visitas)
    return melhor_filho.estado

# Teste do MCTS
o = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
s = Estado(matriz=np.array([[1, 3, 2], [5, 9, 4], [7, 8, 6]]))

melhor_estado = monte_carlo_tree_search(s, o.matriz, iteracoes=1000)
melhor_estado.mostrar()
