from PIL import Image, ImageDraw
import numpy as np

import matplotlib.pyplot as plt


def load(filename, w=None, h=None):
    if filename.lower().endswith('.raw'):
        img = np.fromfile(filename, dtype=np.uint8)
        img = img.reshape((h, w))
    else:
        img = np.asarray(Image.open(filename))
    return img


def save(image, name, mode=None):
    if name.lower().endswith('.raw'):
        arr = image.flatten()
        with open(name, 'wb') as f:
            f.write(arr.tobytes())

    else:
        im = Image.fromarray(image, mode=mode)
        im = im.convert(mode='RGB')
        im.save(name)


def get_pixel(image, x, y):
    return image[x][y]


def put_pixel(image, x, y, pixel):
    image[x][y] = pixel


# ul = upper-left
def paste_section(image_to_cut, x_start, y_start, x_end, y_end, new_image, new_ul_x, new_ul_y):
    image_to_cut = Image.fromarray(image_to_cut)
    new_image = Image.fromarray(new_image)
    section = image_to_cut.crop((x_start, y_start, x_end, y_end))
    new_image.paste(section, (new_ul_x, new_ul_y))
    return np.array(new_image)


def bin_circle(radius=50, img_height=200, img_width=200):
    img = Image.new("L", (img_width, img_height))
    draw = ImageDraw.Draw(img)
    ellipse_shape = (img_width / 2 - radius, img_height / 2 - radius, img_width / 2 + radius, img_height / 2 + radius)
    draw.ellipse(ellipse_shape, fill="white")
    return img


def bin_rectangle(hor_len=150, ver_len=150, img_height=200, img_width=200):
    img = Image.new("L", (img_width, img_height))
    draw = ImageDraw.Draw(img)
    rectangle_shape = ((img_width / 2 - hor_len / 2, img_height / 2 - ver_len / 2),
                       (img_width / 2 + hor_len / 2, img_height / 2 + ver_len / 2))
    draw.rectangle(rectangle_shape, fill="white")
    return img


def add(im1, im2):
    ans = np.add(im1.astype(np.uint32), im2.astype(np.uint32))
    ans = normalize(ans, 0, 255 + 255)
    return ans


def subtract(im1, im2):
    ans = np.subtract(im1.astype(np.int32), im2.astype(np.int32))
    ans = normalize(ans, -255, 255)
    return ans


def multiply(im1, im2):
    ans = np.multiply(im1.astype(np.uint32), im2.astype(np.uint32))
    ans = normalize(ans, 0, 255 * 255)
    return ans


def normalize(x, min_val, max_val):
    m = (255 / (max_val - min_val))
    b = -m * min_val
    ans = m * x + b
    ans = ans.astype(np.uint8)
    return ans


def rgb_to_hsv(img):
    rgb_image = Image.fromarray(img)
    hsv_image = rgb_image.convert(mode='HSV')
    return np.array(hsv_image)


def split_rgb(img):
    rgb_image = Image.fromarray(img)
    r, g, b = rgb_image.split()
    return np.array(r), np.array(g), np.array(b)


def split_hsv(img):
    hsv_image = Image.fromarray(img, mode='HSV')
    h, s, v = hsv_image.split()
    return np.array(h), np.array(s), np.array(v)


def pixels_info(pixels):
    w = pixels.shape[0]
    h = pixels.shape[1]
    count = w * h
    mean = np.mean(pixels, axis=(0, 1))
    return count, mean


def plot_hist_rgb(image):
    image = np.array(image).astype(float)
    if len(image.shape) > 2:
        color = ("red", "green", "blue")
        legends = ["Red Channel", "Green Channel", "Blue Channel", "Total"]
        plt.figure(1)

        for i in range(image.shape[2]):
            plt.figure(2 + i)
            channel = image[:, :, i]
            plt.hist(
                channel.ravel(),
                bins=256,
                weights=np.zeros_like(channel).ravel() + 1.0 / channel.size,
                color=color[i],
            )
            plt.xlabel("Intensity Value")
            plt.legend([legends[i]])
            plt.title("Histogram for " + legends[i])

            plt.figure(1)
            plt.hist(
                channel.ravel(),
                bins=256,
                weights=np.zeros_like(channel).ravel() + 1.0 / channel.size,
                color=color[i],
                alpha=0.35,
            )

        plt.xlabel("Intensity Value")
        plt.legend(legends)
        plt.show()
    else:
        plt.figure()
        plt.hist(
            image.ravel(),
            bins=256,
            weights=np.zeros_like(image).ravel() + 1.0 / image.size,
        )
        plt.xlabel("Intensity Value")
        plt.legend("Gray scale")
        plt.show()


def plot_hist_hsv(image):
    image = np.array(image).astype(float)
    if len(image.shape) > 2:
        color = ("red", "green", "blue")
        legends = ["Hue Channel", "Saturation Channel", "Value Channel", "Total"]
        plt.figure(5)

        for i in range(image.shape[2]):
            plt.figure(6 + i)
            channel = image[:, :, i]
            plt.hist(
                channel.ravel(),
                bins=256,
                weights=np.zeros_like(channel).ravel() + 1.0 / channel.size,
                color=color[i],
            )
            plt.xlabel("Intensity Value")
            plt.legend([legends[i]])
            plt.title("Histogram for " + legends[i])

            plt.figure(5)
            plt.hist(
                channel.ravel(),
                bins=256,
                weights=np.zeros_like(channel).ravel() + 1.0 / channel.size,
                color=color[i],
                alpha=0.35,
            )

        plt.xlabel("Intensity Value")
        plt.legend(legends)
        plt.show()
    else:
        plt.figure()
        plt.hist(
            image.ravel(),
            bins=256,
            weights=np.zeros_like(image).ravel() + 1.0 / image.size,
        )
        plt.xlabel("Intensity Value")
        plt.legend("Gray scale")
        plt.show()


def negative(img):
    neg = lambda x: - x + 255
    return neg(img)


def thresholding(threshold, img):
    new_img = np.copy(img)
    height = img.shape[1]
    width = img.shape[0]
    if len(img.shape) == 2:
        for w in range(0, width):
            for h in range(0, height):

                if new_img[w, h] < threshold:
                    new_img[w, h] = 0
                else:
                    new_img[w, h] = 255
    else:
        channels = img.shape[2]
        for c in range(0, channels):
            for w in range(0, width):
                for h in range(0, height):

                    if new_img[w, h, c] < threshold:
                        new_img[w, h, c] = 0
                    else:
                        new_img[w, h, c] = 255
    return new_img

def thresholding_color(threshold, img):
    new_img = np.copy(img)
    height = img.shape[1]
    width = img.shape[0]
    channels = img.shape[2]
    
    for c in range(0, channels):
        for w in range(0, width):
            for h in range(0, height):

                if new_img[w, h, c] < threshold[c]:
                    new_img[w, h, c] = 0
                else:
                    new_img[w, h, c] = 255
    return new_img

def power(img, gamma):
    max_pixel_value = 256
    c = (max_pixel_value - 1) ** (1 - gamma)
    function = lambda x: c * (x ** gamma)
    return function(np.copy(img)).astype(np.uint8)


def equalize(image):

    if len(image.shape) > 2:
        new_image = np.zeros(image.shape)
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            h = np.zeros(256)
            for j in range(0, len(channel)):
                for k in range(0, len(channel[0])):
                    h[channel[j][k]] += 1
            h /= len(channel) * len(channel[0])
            s = []
            accum = 0
            for j in range(0, len(h)):
                accum += h[j]
                s.append(accum)
            smin = s[0]
            equalized = []
            for k in range(0, len(s)):
                equalized.append(int(((s[k] - smin)*255)/(1-smin) + 0.5))

            for y in range(0, len(channel)):
                for x in range(0, len(channel[0])):
                    current_value = channel[y, x]
                    new_image[y, x, i] = equalized[current_value]
        return new_image.astype(np.uint8)
    else:
        h = np.zeros(256)
        for j in range(0, len(image)):
            for k in range(0, len(image[0])):
                h[image[j][k]] += 1
        h /= len(image) * len(image[0])
        s = []
        accum = 0
        for j in range(0, len(h)):
            accum += h[j]
            s.append(accum)
        smin = s[0]
        equalized = []
        for k in range(0, len(s)):
            equalized.append(int(((s[k] - smin) * 255) / (1 - smin) + 0.5))
        new_image = np.zeros(image.shape)
        for y in range(0, len(image)):
            for x in range(0, len(image[0])):
                current_value = image[y, x]
                new_image[y, x] = equalized[current_value]
        return new_image.astype(np.uint8)


def synthesis(img1, img2, normalize_out=True):
    img1 = img1.astype(np.int64)
    img2 = img2.astype(np.int64)
    new_img = np.sqrt(img1 ** 2 + img2 ** 2)
    if normalize_out:
        new_img = normalize(new_img, 0, 361)  # sqrt(255^2 + 255^2) = 361
    return new_img


def rotate(img, times_clockwise):
    for time in range(0, times_clockwise):
        top = 0
        bottom = len(img) - 1
        left = 0
        right = len(img[0]) - 1
        while left < right and top < bottom:
            # Store the first element of next row,
            # this element will replace first element of
            # current row
            prev = img[top + 1][left]

            # Move elements of top row one step right
            for i in range(left, right + 1):
                curr = img[top][i]
                img[top][i] = prev
                prev = curr

            top += 1

            # Move elements of rightmost column one step downwards
            for i in range(top, bottom + 1):
                curr = img[i][right]
                img[i][right] = prev
                prev = curr

            right -= 1

            # Move elements of bottom row one step left
            for i in range(right, left - 1, -1):
                curr = img[bottom][i]
                img[bottom][i] = prev
                prev = curr

            bottom -= 1

            # Move elements of leftmost column one step upwards
            for i in range(bottom, top - 1, -1):
                curr = img[i][left]
                img[i][left] = prev
                prev = curr

            left += 1

    return img
