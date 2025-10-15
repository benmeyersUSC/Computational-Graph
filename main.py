from value import Value
from vector import Vector
from perceptron import MultiLayerPerceptron
import numpy as np

def main():
    mlp = MultiLayerPerceptron(
        input_shape=(1, 512),
        layer_dims=[256, 128, 64, 32, 16, 8],
        activations=['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    )
    x = Vector(np.random.randn(512,))

    for i in range(100):
        output = mlp(x)

        # will create graph and fill each Vector with its gradient
        output.backward()

        mlp.grad_step(lr=0.01)

        if i % 10 == 0:
            print(f"Iteration {i}, Output sum: {output.data.sum()}")


if __name__ == "__main__":
    main()