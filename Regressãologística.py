import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

df = pd.read_csv('data1.txt', sep=',', header=None)
df.columns = ['exame_1', 'exame_2', 'classe']

def sigmoid(z):
    z = np.array(z)
    return 1 / (1+np.exp(-z))

def cost_function(theta, X, y):
    m = y.shape[0]
    theta = theta[:, np.newaxis] #trick to make numpy minimize work
    h = sigmoid(X.dot(theta))
    J = (1/m) * (-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)))

    diff_hy = h - y
    grad = (1/m) * diff_hy.T.dot(X)

    return J, grad

m = df.shape[0]
X = np.hstack((np.ones((m,1)),df[['exame_1', 'exame_2']].values))
y = np.array(df.classe.values).reshape(-1,1)
initial_theta = np.zeros(shape=(X.shape[1]))

cost, grad = cost_function(initial_theta, X, y)
print('Custo inicial (com theta  valendo zero):', cost)
print('Valor esperado do custo : 0.693')
print('Gradiente para theta zero:')
print(grad.T)
print('Valor esperado para o gradinte:\n -0.1000\n -12.0092\n -11.2628')

test_theta = np.array([-24, 0.2, 0.2])
[cost, grad] = cost_function(test_theta, X, y)

print('Custo:', cost)
print('Custo Esperado: 0.218')
print('Gradiente:')
print(grad.T)
print('Gradiente Esperado:\n 0.043\n 2.566\n 2.647')


import scipy.optimize as opt
def optimize_theta(X, y, initial_theta):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y), method='TNC',
                               jac=True, options={'maxiter':400})
    return opt_results['x'], opt_results['fun']

opt_theta, cost = optimize_theta(X, y, initial_theta)

print('Custo para o theta encontrado pelo scipy:', cost)
print('Custo esperado: 0.203')
print('theta encontrado:\n', opt_theta.reshape(-1,1))
print('Theta esperado (approx):')
print(' -25.161\n 0.206\n 0.201')

plt.figure(figsize=(7,5))
ax = sns.scatterplot(x='exame_1', y='exame_2', hue='classe', data=df, style='classe', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['Doente', 'Saudável'])
plt.title('Dados de Treinamento com a superfície de Decisão')

plot_x = np.array(ax.get_xlim())
plot_y = (-1/opt_theta[2]*(opt_theta[1]*plot_x + opt_theta[0]))
plt.plot(plot_x, plot_y, '-', c="green")
plt.show(ax)

prob = sigmoid(np.array([1, 45, 85]).dot(opt_theta))
print('O paciente com exame_1 = 45 e exame_2 = 85, está saudável com probabilidade', prob)
print('Valor esperado: 0.775 +/- 0.002');


x1 = df.exame_1.values
x2 = df.exame_2.values
m = x1.shape[0]
y = df['classe'].values.reshape(-1, 1)  # agora com shape (118, 1)


""""X_vetor = np.column_stack([ #under fitting
    np.ones(m),       # bias
    x1,               # 1º grau
    x2,               # 1º grau
    x1**2,            # 2º grau
    x2**2,            # 2º grau
    x1*x2,            # interação 1-1
    (x1**2)*x2,       # interação 2-1
    x1*(x2**2),       # interação 1-2
    x1**3,            # 3º grau
    x2**3,            # 3º grau
    x1**4,            # 4º grau
    x2**4,            # 4º grau
    (x1**3)*x2,       # interação 3-1
    x1*(x2**3),       # interação 1-3
    (x1**2)*(x2**2)   # interação 2-2
])"""
"""X_vetor = np.column_stack([ #over fitting
    np.ones(m),          # bias (termo constante)
    x1,                  # 1º grau
    x2,                  # 1º grau
    x1**2,               # 2º grau
    x2**2,               # 2º grau
    x1 * x2,             # interação x1·x2 (1-1)
    (x1**2) * x2,        # interação x1²·x2 (2-1)
    x1 * (x2**2),        # interação x1·x2² (1-2)
    x1**3,               # 3º grau
    x2**3,               # 3º grau
    (x1**3) * x2,        # interação x1³·x2 (3-1)
    x1 * (x2**3),        # interação x1·x2³ (1-3)
    (x1**2) * (x2**2),   # interação x1²·x2² (2-2)
    x1**4,               # 4º grau
    x2**4,               # 4º grau
    (x1**4) * x2,        # interação x1⁴·x2 (4-1)
    x1 * (x2**4),        # interação x1·x2⁴ (1-4)
    (x1**3) * (x2**2),   # interação x1³·x2² (3-2)
    (x1**2) * (x2**3),   # interação x1²·x2³ (2-3)
    x1**5,               # 5º grau
    x2**5,               # 5º grau
    (x1**5) * x2,        # interação x1⁵·x2 (5-1)
    x1 * (x2**5),        # interação x1·x2⁵ (1-5)
    (x1**4) * (x2**2),   # interação x1⁴·x2² (4-2)
    (x1**2) * (x2**4),   # interação x1²·x2⁴ (2-4)
    (x1**3) * (x2**3)    # interação x1³·x2³ (3-3)
])"""

X_vetor = np.column_stack([
    np.ones(m),          # bias (termo constante)
    x1,                  # 1º grau
    x2,                  # 1º grau
    x1**2,               # 2º grau
    x2**2,               # 2º grau
    x1 * x2,             # interação x1·x2 (1-1)
    (x1**2) * x2,        # interação x1²·x2 (2-1)
    x1 * (x2**2),        # interação x1·x2² (1-2)
    x1**3,               # 3º grau
    x2**3,               # 3º grau
    (x1**3) * x2,        # interação x1³·x2 (3-1)
    x1 * (x2**3),        # interação x1·x2³ (1-3)
    (x1**2) * (x2**2),   # interação x1²·x2² (2-2)
    x1**4,               # 4º grau
    x2**4,               # 4º grau
    (x1**4) * x2,        # interação x1⁴·x2 (4-1)
    x1 * (x2**4),        # interação x1·x2⁴ (1-4)
    (x1**3) * (x2**2),   # interação x1³·x2² (3-2)
    (x1**2) * (x2**3),   # interação x1²·x2³ (2-3)
    x1**5,               # 5º grau
    x2**5,               # 5º grau
    (x1**5) * x2
])
initial_theta = np.zeros(X_vetor.shape[1])
opt_theta, cost = optimize_theta(X_vetor, y, initial_theta)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

for i in range(len(u)):
    for j in range(len(v)):
        x1 = u[i]
        x2 = v[j]
        """features = np.array([
            1, x1, x2, 
            x1**2, x2**2, 
            x1*x2,
            (x1**2)*x2,
            x1*(x2**2),
            x1**3, x2**3,
            x1**4, x2**4,
            (x1**3)*x2,
            x1*(x2**3),
            (x1**2)*(x2**2)
        ])"""
        """features = np.array([
            1, x1, x2,
            x1**2, x2**2,
            x1 * x2,
            (x1**2) * x2,
            x1 * (x2**2),
            x1**3, x2**3,
            (x1**3) * x2,
            x1 * (x2**3),
            (x1**2) * (x2**2),
            x1**4, x2**4,
            (x1**4) * x2,
            x1 * (x2**4),
            (x1**3) * (x2**2),
            (x1**2) * (x2**3),
            x1**5, x2**5,
            (x1**5) * x2,
            x1 * (x2**5),
            (x1**4) * (x2**2),
            (x1**2) * (x2**4),
            (x1**3) * (x2**3)
        ])"""
        features = np.array([
            1, x1, x2,
            x1**2, x2**2,
            x1 * x2,
            (x1**2) * x2,
            x1 * (x2**2),
            x1**3, x2**3,
            (x1**3) * x2,
            x1 * (x2**3),
            (x1**2) * (x2**2),
            x1**4, x2**4,
            (x1**4) * x2,
            x1 * (x2**4),
            (x1**3) * (x2**2),
            (x1**2) * (x2**3),
            x1**5, x2**5,
            (x1**5) * x2
        ])

        z[i, j] = features.dot(opt_theta)

z = z.T

plt.figure(figsize=(7,5))
sns.scatterplot(x='exame_1', y='exame_2', hue='classe', data=df, style='classe', s=80)
plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
plt.title('Superfície de decisão não-linear')
plt.xlabel('exame_1')
plt.ylabel('exame_2')
plt.legend(['Fronteira de decisão', 'Doente', 'Saudável'])
plt.show()


def map_feature(x1, x2, grau):

    out = [np.ones(x1.shape[0]).reshape(-1,1)]
    for i in range(1, grau+1):
        for j in range(i+1):
            term = (x1 ** (i - j)) * (x2 ** j)
            out.append(term.reshape(-1,1))
    return np.hstack(out)

x1 = df['exame_1'].values
x2 = df['exame_2'].values
grau = 7
X_mapped = map_feature(x1, x2, grau)  # X com features polinomiais

initial_theta = np.zeros(X_mapped.shape[1])
opt_theta, cost = optimize_theta(X_mapped, y, initial_theta)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

for i in range(len(u)):
    for j in range(len(v)):
        mapped = map_feature(np.array([u[i]]), np.array([v[j]]), grau)
        z[i,j] = mapped.dot(opt_theta)

z = z.T  # transpor para combinar com as dimensões do meshgrid

plt.figure(figsize=(7,5))
sns.scatterplot(x='exame_1', y='exame_2', hue='classe', data=df, style='classe', s=80)
plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
plt.title('Superfície de decisão não-linear')
plt.xlabel('exame_1')
plt.ylabel('exame_2')
plt.legend(['Fronteira de decisão', 'Doente', 'Saudável'])
plt.show()



