import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.linspace(-5, 5, 100)
nu, sigma = torch.tensor(0.), torch.tensor(1)
noise = torch.tensor([torch.normal(nu, sigma) for _ in range(len(x_data))])
y_data = 2 * x_data - 1 + noise

w = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)

lr = 0.01
episode_n = 100

for episode in range(episode_n):
    y_pred = w * x_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    print(f'episode: {episode}, loss = {loss}')
    loss.backward()
    w.data = w.data - lr * w.grad
    b.data = b.data - lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f'w = {w.data}, b = {b.data}')

plt.scatter(x_data.numpy(), y_data.numpy())
plt.plot(x_data.numpy(), y_pred.data.numpy(), 'r')
plt.show()
