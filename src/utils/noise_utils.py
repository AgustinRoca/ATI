from src.utils.image_utils import normalize
from src.utils.random_generator import Generator
import matplotlib.pyplot as plt
import numpy as np

def plot_gen_hist(generator: Generator, name):
    numbers = generator.generate()

    plt.figure(1)
    plt.hist(numbers, 100, weights= np.zeros_like(numbers) + 1.0/ numbers.size)

    plt.xlabel("Random numbers weighted")
    plt.legend("Random Generated Numbers")
    plt.title("Histogram for " + name + " Generator")

    plt.show()

def apply_noise(image, generator: Generator, threshold, is_additive: bool):

    noise = generator.generate()
    noise = np.array(noise).reshape(image.shape)

    random_layer = np.random.uniform(low=0.0, high=1.0, size=image.shape)
    random_layer = np.where(random_layer < threshold, 1.0, 0.0)

    ones = np.ones(image.shape)

    random_layer_2 = ones - random_layer

    if is_additive:
        result = image + random_layer * noise
    else:
        result = image * random_layer * noise + image * random_layer_2

    result = normalize(result, min(result.flatten()), max(result.flatten()))
    
    return result


def apply_salt_pepper_noise(image, p0):
    if (p0 > 0.5):
        p1 = p0
        p0 = 1 - p1
    else:
        p1 = 1 - p0

    
    random_layer = np.random.uniform(low=0.0, high=1.0, size=image.shape)
    # Sets to 0 all pixels < p0
    random_layer_p0 = np.where(random_layer > p0, 1.0, 0.0)
    # Set to 0 all pixel > p1, so that they can be added later as 255
    random_layer_not_p1 = np.where(random_layer < p1, 1.0, 0.0)
    # Set to 255 all pixel > p1
    random_layer_p1 = np.where(random_layer > p1, 255.0, 0.0)

    result = image * random_layer_p0
    result = result * random_layer_not_p1 + random_layer_p1

    result = result.astype(np.uint8)

    return result