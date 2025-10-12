from value import Value
from vector import Vector
import numpy as np

def main():
    W = Vector(np.random.randn(3, 2), requires_grad=True)  
    x = Vector(np.array([1, 2, 3]), requires_grad=False)    
    
    for i in range(50):        
        z = x @ W
        
        loss = z @ z  # dot product gives scalar
        
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.data:.6f}, z: {z.data}")
        
        loss.backward()
        loss.grad_step(lr=0.01)
    
    print(f"\nFinal z: {z.data}")
    print(f"Final W:\n{W.data}")


if __name__ == "__main__":
    main()