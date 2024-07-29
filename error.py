import torch

def add_tensors(a, b):
    c = a + b
    return c

a = torch.tensor([1, 2, 3]).to("cuda")
b = torch.tensor([1, 2, 3]).to("cuda")
c = add_tensors(a, b)

print(f"{a=}, {b=}, {c=}")

# Compile
add_tensors_compiled = torch.compile(add_tensors)

c_compiled = add_tensors_compiled(a, b)
print("{c_compiled=}")
