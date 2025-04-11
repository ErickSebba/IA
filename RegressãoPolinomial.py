import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parâmetros para geração dos dados
M = 10  # Número de pontos de dados gerados
NR = 100  # Número de pontos usados para plotar a curva suave

# Geração dos dados de treinamento
x = np.linspace(0, 1, M)  # Pontos espaçados uniformemente no intervalo [0, 1]
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, M)  # Adiciona ruído gaussiano (média 0, desvio padrão 0.15)

# Geração dos dados para a curva teórica
xr = np.linspace(0, 1, NR)  # Pontos espaçados uniformemente para a curva suave
yn = np.sin(2 * np.pi * xr)  # Valores da função seno sem ruído

line, = plt.plot(xr, yn, label='sin(2πx)')  # Plota a curva teórica
plt.scatter(x, y, label='Dados com ruído')  # Plota os pontos de dados com ruído
plt.legend(handles=[line])  # Adiciona a legenda ao gráfico
plt.xlabel('x')  # Rótulo do eixo x
plt.ylabel('y')  # Rótulo do eixo y
plt.title('Dados Gerados e Função Teórica')  # Título do gráfico
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.show()  # Exibe o gráfico

# Parâmetros para geração da curva teórica
NR = 100  # Número de pontos para a curva suave
xr = np.linspace(0, 1, NR)  # Pontos espaçados uniformemente no intervalo [0, 1]
yr = np.sin(2 * np.pi * xr)  # Valores da função seno sem ruído

# Grau do polinômio para ajuste
K = 9  # Grau do polinômio usado na regressão polinomial

# Ajuste do modelo polinomial aos dados
polynomial_model = np.poly1d(np.polyfit(x, y, K))  # Cria um modelo polinomial de grau K

# Plotagem dos resultados
line1, = plt.plot(xr, polynomial_model(xr), label='Polinômio (K=9)')  # Plota a curva do polinômio ajustado
line2, = plt.plot(xr, yr, label='sin(2πx)')  # Plota a curva teórica da função seno
plt.scatter(x, y, label='Dados com ruído')  # Plota os pontos de dados com ruído

# Configurações do gráfico
plt.legend(handles=[line1, line2])  # Adiciona a legenda ao gráfico
plt.xlabel('x')  # Rótulo do eixo x
plt.ylabel('y')  # Rótulo do eixo y
plt.title('Ajuste Polinomial vs Função Teórica')  # Título do gráfico
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.show()  # Exibe o gráfico

import numpy as np
import matplotlib.pyplot as plt

def polyfit_with_regularization(x, y, degree, lambda_):
    """
    Ajusta um polinômio de grau especificado aos dados (x, y) com regularização L2.

    Parâmetros:
        x (array): Valores da variável independente.
        y (array): Valores da variável dependente.
        degree (int): Grau do polinômio.
        lambda_ (float): Parâmetro de regularização (termo L2).

    Retorna:
        theta (array): Coeficientes do polinômio ajustado.
    """
    # Cria a matriz de design para o polinômio de grau 'degree'
    X = np.vander(x, degree + 1, increasing=True)

    # Define a matriz de regularização (termo L2), ignorando o termo de intercepto
    L = lambda_ * np.eye(degree + 1)
    L[0, 0] = 0  # Não regularizamos o intercepto

    # Resolve o sistema linear (X^T X + L) theta = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.solve(XtX + L, Xty)

    return theta


# Geração dos dados
N = 10  # Número de pontos de dados
x = np.linspace(0, 1, N)  # Pontos espaçados uniformemente no intervalo [0, 1]
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, N)  # Adiciona ruído gaussiano

# Parâmetros do modelo
K = 9  # Grau do polinômio
lambda_ = 0.0001  # Parâmetro de regularização

# Ajuste do modelo polinomial com regularização
theta = polyfit_with_regularization(x, y, K, lambda_)

# Geração de pontos para a curva ajustada
xr = np.linspace(0, 1, 100)  # Pontos para a curva suave
Xr = np.vander(xr, K + 1, increasing=True)  # Matriz de design para os pontos xr
yr = Xr @ theta  # Valores previstos pelo modelo polinomial

# Geração da curva teórica (seno sem ruído)
yn = np.sin(2 * np.pi * xr)

# Plotagem dos resultados
line1, = plt.plot(xr, yr, label=f'Polinômio (K={K}) com regularização')  # Curva do polinômio ajustado
line2, = plt.plot(xr, yn, label='sin(2πx)')  # Curva teórica da função seno
plt.scatter(x, y, label='Dados com ruído')  # Pontos de dados com ruído

# Configurações do gráfico
plt.legend(handles=[line1, line2])  # Adiciona a legenda ao gráfico
plt.xlabel('x')  # Rótulo do eixo x
plt.ylabel('y')  # Rótulo do eixo y
plt.title('Ajuste Polinomial com Regularização vs Função Teórica')  # Título do gráfico
plt.grid(True)  # Adiciona uma grade ao gráfico
plt.show()  # Exibe o gráfico

N = 10
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.15, N)

NR = 100
xr = np.linspace(0, 1, NR)
yr = np.sin(2*np.pi*xr)


for K in range(0, 10):
  mymodel = np.poly1d(np.polyfit(x, y, K))


  line1, = plt.plot(xr, mymodel(xr), label='Regressão')
  line2, = plt.plot(xr, yr, label='Distribuição')

  plt.scatter(x, y)
  plt.legend(handles=[line1, line2])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(f'Polinômio de grau {K}')
  plt.show()

  def cost_function(theta, X, y, lambda_=0.1):
    """
    Calcula o custo e o gradiente para a regressão linear com regularização.

    Parâmetros:
    theta -- Parâmetros do modelo (vetor 1D)
    X -- Matriz de características (m x n)
    y -- Vetor de valores alvo (m x 1)
    lambda_ -- Parâmetro de regularização (default: 0.1)

    Retorna:
    J -- O valor do custo
    grad -- O vetor de gradiente (1D)
    """
    m = y.size
    h = X.dot(theta)                   

    # Custo com regularização (não regulariza o termo de bias theta[0])
    J = (1 / (2 * m)) * np.sum((h - y) ** 2) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    #J = (1 / (2 * m)) * np.sum((h - y) ** 2) + (lambda_) * np.sum(theta[1:])#L1
    # Gradiente com regularização
    grad = (1 / m) * X.T.dot(h - y)
    grad[1:] += (lambda_ / m) * theta[1:]

    return J, grad

def optimize_theta(X, y, initial_theta, lambda_=0.1):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y, lambda_), method='L-BFGS-B',
                               jac=True, options={'maxiter': 400})
    if not opt_results.success:
        raise RuntimeError("Otimização falhou: " + opt_results.message)
    return opt_results['x'], opt_results['fun']

def feature_normalize(X, mean=None, std=None):
    X = np.array(X)
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
    X = (X - mean) / std
    return X, mean, std

def extend_feature(X_ini, k):
    result = X_ini
    for i in range(2, k+1):
        result = np.hstack((result, np.power(X_ini, i)))
    return result

# Geração de dados
N = 30
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.5, N)

NR = 100
xr = np.linspace(0, 1, NR)
yr = np.sin(2*np.pi*xr)

m = y.size

# Preparação dos dados
k = 9
X_ini = x.copy()
X_ini = X_ini.reshape(-1, 1)
X = extend_feature(X_ini, k)
X, mean, std = feature_normalize(X)
ones = np.ones((m, 1))
print(X.shape)
print(ones.shape)
X = np.hstack([ones, X])

#X = np.hstack([np.ones((m, 1)), X])  # Correção aplicada aqui
theta = np.random.randn(k + 1)

# Otimização
opt_theta, cost = optimize_theta(X, y, theta)

# Previsão
xnew = np.linspace(0, 1, 50)
xnew = xnew.reshape(-1, 1)
X2 = extend_feature(xnew, k)
X2 = (X2 - mean) / std
X2 = np.hstack([np.ones((xnew.shape[0], 1)), X2])
h = np.dot(X2, opt_theta)

# Visualização
line1, = plt.plot(xnew, h, label='Regression')
line2, = plt.plot(xr, yr, label='True distribution')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression of Order 9')
plt.legend(handles=[line1, line2])
plt.show()