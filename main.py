import numpy as np
import matplotlib.pyplot as plt

# Helper function to create a random gradient
def random_gradient():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])

# Helper function to compute dot product between distance and gradient vectors
def dot_grid_gradient(ix, iy, x, y, gradients):
    # Distance vectors
    dx, dy = x - ix, y - iy
    # Gradient vector from the grid
    gradient = gradients[ix, iy]
    # Dot product
    return dx * gradient[0] + dy * gradient[1]

# Smoothstep function
def smoothstep(t):
    return t * t * (3 - 2 * t)

# Perlin Noise function
def perlin_noise(x, y, gradients):
    # Determine grid cell coordinates
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    # Determine interpolation weights
    sx, sy = smoothstep(x - x0), smoothstep(y - y0)

    # Interpolate between grid point gradients
    n0 = dot_grid_gradient(x0, y0, x, y, gradients)
    n1 = dot_grid_gradient(x1, y0, x, y, gradients)
    ix0 = n0 + sx * (n1 - n0)

    n0 = dot_grid_gradient(x0, y1, x, y, gradients)
    n1 = dot_grid_gradient(x1, y1, x, y, gradients)
    ix1 = n0 + sx * (n1 - n0)

    return ix0 + sy * (ix1 - ix0)

# Generate Perlin Noise across a grid
def generate_perlin_noise(width, height, scale):
    # Create a grid of random gradient vectors
    gradients = np.array([[random_gradient() for _ in range(width + 1)] for _ in range(height + 1)])
    noise = np.zeros((width, height))

    # Compute the noise values
    for i in range(width):
        for j in range(height):
            x, y = i / scale, j / scale  # Scale coordinates
            noise[i, j] = perlin_noise(x, y, gradients)

    # Normalize to range [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

# Parameters
width, height = 100, 100
scale = 10

# Generate noise
perlin_noise_result = generate_perlin_noise(width, height, scale)

# Plotting the generated Perlin noise
plt.figure(figsize=(8, 8))
plt.imshow(perlin_noise_result, cmap='gray')
plt.title('Perlin Noise')
plt.colorbar()
plt.show()
