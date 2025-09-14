import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 1000
learning_rate = 0.01

x_train_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=False, dtype=torch.float, device=device)
y_train_tensor = torch.tensor([2.0, 4.0, 6.0], requires_grad=False, dtype=torch.float, device=device)

a = torch.tensor(1.0, requires_grad=True, dtype=torch.float, device=device)
b = torch.tensor(2.0, requires_grad=True, dtype=torch.float, device=device)

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

    a.grad.zero_()
    b.grad.zero_()

print(f"Trained parameters: a = {a}, b = {b}")