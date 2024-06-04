import torch

n = 100
p = 5
X = torch.randn(n, p)
beta = torch.randn(p, 1)  # real beta
epsilon = torch.randn(n, 1)
Y = X @ beta + epsilon  # real Y

# classic least square
beta_ols = torch.inverse(X.T @ X) @ X.T @ Y

# least square on gradient descent
beta_gd0 = torch.zeros(p, 1)
alpha0 = 0.01
for i in range(100):
    gd = -X.T @ (Y - X @ beta_gd0)
    beta_gd0 = beta_gd0 - alpha0 * gd

# least square on automatic differentiation + gradient descent
beta_gd = torch.zeros(p, 1, requires_grad=True)
alpha = 0.01  # learning rate

for i in range(100):
    # forward
    Y_hat = X @ beta_gd
    loss = torch.mean((Y_hat - Y).pow(2))
    print(i, loss.data.numpy())

    # backward
    loss.backward()
    beta_gd.data = beta_gd.data - alpha * beta_gd.grad
    beta_gd.grad.fill_(0)

print(beta)
print(beta_ols)
print(beta_gd0)
print(beta_gd)
