import numpy as np
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import time
from queue import PriorityQueue

class Estado:
  def __init__(self, pai=None, matriz=None):
      self.pai = pai
      self.matriz = matriz

      self.d = 0 # tamanho do caminho inicio - estado
      self.c = 0 #
      self.p = 0 # prioridade

  def __eq__(self, other):
      return len(self.matriz[np.where(self.matriz!= other.matriz)]) == 0

  def __lt__(self, other):
    return self.p<other.p

  def mostrar(self):
    for i in self.matriz:
      print(i)
    print()

#estado = Estado(matriz=np.array([[4, 1, 3], [9, 2, 5],[7, 8, 6]]))
#estado.p = 1
#estado2 = Estado(matriz=np.array([[4, 1, 3], [9, 2, 5],[7, 8, 6]]))
#estado2.p = 2
#print(estado < estado2)
#estado.mostrar()

def acoes_permitidas(estado):

  adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
  blank = estado.matriz==9
  mask = conv2(blank, adj, 'same')

  return estado.matriz[np.where(mask)]

#s = Estado(matriz=np.array([[1, 2, 3], [4, 9, 5],[7, 8, 6]]))
#print("Acoes permitidas ", acoes_permitidas(s))

def movimentar(s, c):

  matriz = s.matriz.copy()

  matriz[np.where(s.matriz==9)] = c
  matriz[np.where(s.matriz==c)] = 9

  return Estado(matriz=matriz)

#s = Estado(matriz=np.array([[1, 2, 3], [4, 9, 5],[7, 8, 6]]))
#s.mostrar()
#for acao in acoes_permitidas(s):
  v = movimentar(s, acao)
  v.mostrar()

estado = Estado(matriz=np.array([[4, 6, 7], [9, 5, 8],[2, 1, 3]]))
obj = np.array([[1,2,3], [4,5,6], [7,8,9]])

def dist(t1, t2):
  return np.sum(list(map(lambda i, j: abs(i - j), t1, t2)))

def manhattan(estado, obj):
  return np.sum([dist(np.where(obj==i), np.where(estado.matriz==i)) for i in range(1,9)])

#manhattan(estado, obj)

def hamming(s):
  obj = np.array([[1,2,3], [4,5,6], [7,8,9]])
  qtde_fora_lugar = len(s.matriz[np.where(s.matriz != obj)])
  # 9 não pode entrar na conta
  return (qtde_fora_lugar-1 if qtde_fora_lugar > 0 else
          0)


def astar(s, f, o):


  Q = PriorityQueue()

  s.p = 0
  Q.put((s.p, s))


  while not Q.empty():
    v = Q.get()[1]

    if v==o:
      return v


    for a in acoes_permitidas(v):
      u = movimentar(v, a)

      u.d = v.d + 1
      u.pai = v
      u.p = f(u)+u.d
      Q.put((u.p, u))

  return s


def matriz_para_vetor(matriz):
    vetor = []  
    for linha in matriz:  
        for elemento in linha: 
            vetor.append(elemento)  
    return vetor


def is_solvable(estado):
   
    lista = matriz_para_vetor(estado.matriz)

    inversoes = 0
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                inversoes += 1

    # Se o número de inversões for par, o estado é solucionável
    return inversoes % 2 == 0



o = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]]))
s = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]]))

if is_solvable(s):
    print("O estado é solucionável. Buscando solução...")
    v = astar(s, hamming, o)
    v.mostrar()
    print(v.d)
else:
    print("O estado não tem solução.")

def gerar_estado_aleatorio():
    while True:
        numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.random.shuffle(numeros)
        estado = Estado(matriz=np.array(numeros).reshape(3, 3))
        if is_solvable(estado):  # Verifica se o estado é solucionável
            return estado

def monte_carlo_melhor_movimento(estado, objetivo, n=100, max_iteracoes=1000):
    iteracao = 0

    while iteracao < max_iteracoes:
        if estado == objetivo:
            return True, iteracao  # Solução encontrada

        # Gerar n movimentos aleatórios
        melhores_movimentos = []
        for _ in range(n):
            estado_temp = estado
            acoes = acoes_permitidas(estado_temp)
            if not acoes:
                break  # Nenhuma ação possível

            acao = np.random.choice(acoes)  # Escolhe uma ação aleatória
            estado_temp = movimentar(estado_temp, acao)

            # Avaliar o estado gerado usando a Distância de Manhattan
            distancia = manhattan(estado_temp, objetivo.matriz)
            melhores_movimentos.append((distancia, acao, estado_temp))

        if not melhores_movimentos:
            break  # Nenhum movimento possível

        # Escolher o movimento que leva ao estado mais próximo do objetivo
        melhor_distancia, melhor_acao, melhor_estado = min(melhores_movimentos, key=lambda x: x[0])

        # Atualizar o estado atual
        estado = melhor_estado
        iteracao += 1

    return False, iteracao  # Solução não encontrada

def monte_carlo(estado, objetivo, n=10, max_iteracoes=50): 
    iteracao = 0

    while iteracao < max_iteracoes:
        if estado == objetivo:
            return True, iteracao  # Solução encontrada

        acoes = acoes_permitidas(estado)
        if not acoes:
            break  # Nenhuma ação possível

        melhor_distancia = float("inf")
        melhor_acao = None
        melhor_estado = None

        # Testa cada ação inicial separadamente
        for acao in acoes:
            estado_temp = movimentar(estado, acao)  # Faz o primeiro movimento

            # Simula n movimentos aleatórios a partir desse estado
            for _ in range(n):
                estado_simulado = estado_temp

                for _ in range(n):  # Executa uma sequência de movimentos aleatórios
                    novas_acoes = acoes_permitidas(estado_simulado)
                    if not novas_acoes:
                        break
                    
                    acao_aleatoria = np.random.choice(novas_acoes)
                    estado_simulado = movimentar(estado_simulado, acao_aleatoria)

                # Avalia quão próximo chegou do objetivo
                distancia = manhattan(estado_simulado, objetivo.matriz)

                # Se for o melhor até agora, salva
                if distancia < melhor_distancia:
                    melhor_distancia = distancia
                    melhor_acao = acao
                    melhor_estado = estado_temp

        if melhor_acao is None:
            break  # Nenhuma melhora possível

        # Aplica a melhor ação real
        estado = melhor_estado
        iteracao += 1

    return False, iteracao  # Solução não encontrada
# Gerar um estado aleatório
estado = gerar_estado_aleatorio()
print("Estado inicial:")
estado.mostrar()
sucesso, iteracoes = monte_carlo_melhor_movimento(s, o)
print(f"Solução encontrada: {sucesso}, Iterações: {iteracoes}")
print("Estado final:")
s.mostrar()
sucesso, iteracoes = monte_carlo(s, o)
print(f"Solução encontrada: {sucesso}, Iterações: {iteracoes}")
print("Estado final:")
s.mostrar()