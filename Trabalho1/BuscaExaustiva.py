import numpy as np
from scipy.signal import convolve2d as conv2
import matplotlib
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

estado = Estado(matriz=np.array([[4, 1, 3], [9, 2, 5],[7, 8, 6]]))
estado.p = 1
estado2 = Estado(matriz=np.array([[4, 1, 3], [9, 2, 5],[7, 8, 6]]))
estado2.p = 2
print(estado < estado2)
estado.mostrar()

def acoes_permitidas(estado):

  adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
  blank = estado.matriz==9
  mask = conv2(blank, adj, 'same')

  return estado.matriz[np.where(mask)]

s = Estado(matriz=np.array([[1, 2, 3], [4, 9, 5],[7, 8, 6]]))
print("Acoes permitidas ", acoes_permitidas(s))

def movimentar(s, c):

  matriz = s.matriz.copy()

  matriz[np.where(s.matriz==9)] = c
  matriz[np.where(s.matriz==c)] = 9

  return Estado(matriz=matriz)

s = Estado(matriz=np.array([[1, 2, 3], [4, 9, 5],[7, 8, 6]]))
s.mostrar()
for acao in acoes_permitidas(s):
  v = movimentar(s, acao)
  v.mostrar()

estado = Estado(matriz=np.array([[4, 6, 7], [9, 5, 8],[2, 1, 3]]))
obj = np.array([[1,2,3], [4,5,6], [7,8,9]])

def dist(t1, t2):
  return np.sum(list(map(lambda i, j: abs(i - j), t1, t2)))

def manhattan(estado, obj):
  return np.sum([dist(np.where(obj==i), np.where(estado.matriz==i)) for i in range(1,9)])

manhattan(estado, obj)

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

o = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]]))
s = Estado(matriz=np.array([[1, 2, 3], [4, 9, 5],[7, 8, 6]]))
v = astar(s, hamming, o)
v.mostrar()
print(v.d)