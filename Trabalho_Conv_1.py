import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros fixos do sistema
M = 0.5  # massa [kg]
B = 1.0  # coeficiente de atrito viscoso [kg/s]
L_prime = 46.8e-3  # indutância [H]
l_0 = 0.05  # posição natural da mola [m]
R = 5.0  # resistência [Ohm]
a = 2.5e-2  # parâmetro geométrico [m]
epsilon = 1e-6  # Pequena constante para evitar divisão por zero

# Função que define a tensão aplicada
def v_func(t):
    return 10.0 if t >= 0.5 else 0.0

# Função que define as equações diferenciais
def system_dynamics(t, y, K):
    x, x1, i = y  # y contém [x, x1, i]
    v_i = v_func(t)
    
    dx1_dt = (1/M) * (0.5 * L_prime * (a * i**2 / (a + x)**2) - B * x1 - K * (x - l_0))
    di_dt = ((a + x) / (x + epsilon)) * (v_i - i * R - L_prime * (a * i / (a + x + epsilon)**2) * x1)
    
    return [x1, dx1_dt, di_dt]

# Condições iniciais
t0 = 0.0
x0 = 0.05  # Posição inicial no ponto de equilíbrio [m]
x1_0 = 0.0  # Velocidade inicial [m/s]
i0 = 0.0  # Corrente inicial [A]
y0 = [x0, x1_0, i0]  # Condições iniciais
t_end = 4.0  # Tempo final de 4 segundos

# Diferentes valores de K
K_values = [10, 60, 100]

# Cores para os gráficos
colors = ['red', 'orange', 'green']

# Tempo de amostragem
t_eval = np.linspace(t0, t_end, 4000)

# Plotar os resultados para diferentes valores de K
plt.figure(figsize=(10, 8))

# Gráfico da posição x(t)
plt.subplot(2, 1, 1)
for K, color in zip(K_values, colors):
    sol = solve_ivp(system_dynamics, [t0, t_end], y0, method='RK45', t_eval=t_eval, args=(K,))
    x_values = sol.y[0] * 100  # Converter para cm
    plt.plot(sol.t, x_values, label=f'K = {K} N/m', color=color)
plt.xlabel('Tempo [s]')
plt.ylabel('x [cm]')
plt.grid(True)
plt.legend()
plt.ylim(5.0, 8.5)  # Ajuste do eixo y para facilitar a comparação

# Gráfico da corrente i(t)
plt.subplot(2, 1, 2)
for K, color in zip(K_values, colors):
    sol = solve_ivp(system_dynamics, [t0, t_end], y0, method='RK45', t_eval=t_eval, args=(K,))
    i_values = sol.y[2]
    plt.plot(sol.t, i_values, label=f'K = {K} N/m', color=color)
plt.xlabel('Tempo [s]')
plt.ylabel('i [A]')
plt.grid(True)
plt.legend()
plt.ylim(0, 2.5)  # Ajuste do eixo y para facilitar a comparação

plt.tight_layout()
plt.show()
