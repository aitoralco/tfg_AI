import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N, D = 3, 4

x = torch.randn((N, D),requires_grad=True) # estos hacen que x sea un tensor
y = torch.randn((N, D),requires_grad=True)
z = torch.randn((N, D),requires_grad=True)
"""
    reqyuires_grad=True hace una variable tensor entrenable
    que permite calcular gradientes automaticamente

    false por defecto

    t.data --> accede a los datos del tensor sin la funcionalidad de autograd
    t.grad --> accede a los gradientes del tensor
"""

a = x * y
b = a + z
c = torch.sum(b).to(device)
"""
    to(device) mueve el tensor a la GPU si está disponible cuando device es 'cuda'
"""

c.backward()

print(x.grad)
print(x.data)