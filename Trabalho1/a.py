import numpy as np
from scipy.signal import convolve2d as conv2
import random

def acoes_permitidas(estado):
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    blank = estado == 9
    mask = conv2(blank, adj, 'same')
    return estado[np.where(mask)]

def movimentar(matriz, c):
    nova_matriz = matriz.copy()
    blank_pos = np.where(nova_matriz == 9)
    target_pos = np.where(nova_matriz == c)
    nova_matriz[blank_pos], nova_matriz[target_pos] = c, 9
    return nova_matriz

def dist(t1, t2):
    return abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])

def manhattan(estado, obj):
    return sum(dist(np.where(estado == i), np.where(obj == i)) for i in range(1, 9))

def mcts_solver(estado_inicial, objetivo, n_clones=100, max_jogadas=20, max_iteracoes=1000):
    estado_atual = estado_inicial
    iteracao = 0
    
    while iteracao < max_iteracoes:
        if np.array_equal(estado_atual, objetivo):
            return True, iteracao

        acoes = acoes_permitidas(estado_atual)
        if acoes.size == 0:
            return False, iteracao
        
        medias = {}
        for acao in acoes:
            heuristicas = []
            for _ in range(n_clones):
                clone = movimentar(estado_atual, acao)
                for _ in range(max_jogadas):
                    acoes_clone = acoes_permitidas(clone)
                    if acoes_clone.size == 0:
                        break
                    acao_aleatoria = random.choice(acoes_clone)
                    clone = movimentar(clone, acao_aleatoria)
                heuristica = manhattan(clone, objetivo)
                heuristicas.append(heuristica)
            medias[acao] = np.mean(heuristicas) + 0.8 * iteracao

        melhor_acao = min(medias, key=medias.get)
        estado_atual = movimentar(estado_atual, melhor_acao)
        iteracao += 1
    
    return False, iteracao

objetivo = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
estado_inicial = np.array([[1, 3, 2], [5, 9, 4], [7, 8, 6]])

sucesso, iteracoes = mcts_solver(estado_inicial, objetivo, n_clones=50, max_jogadas=10)
print(f"Solução encontrada: {sucesso} em {iteracoes} iterações.")
