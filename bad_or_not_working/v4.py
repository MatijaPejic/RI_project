from PIL import Image, ImageDraw
import cv2
import random
import numpy as np
from skimage import io
from copy import deepcopy

# TODO Test mutation
class Poly:
    def __init__(self, size, colors):
        self.colors = colors
        self.num_of_points = random.randrange(3, 4)
        self.size = size  # size of image [x, y]
        self.xy = [(random.randrange(self.size[0]), random.randrange(self.size[1])) for _ in range(self.num_of_points)]
        poly_colors = self.Color_select()
        self.color = tuple(poly_colors)

    def Color_select(self):
        selected_colors = []
        n, m = self.colors.shape
        for i in range(m):
            roll = random.randrange(0, n)
            selected_colors.append(self.colors[roll][i])
        return selected_colors

    def Slightly_change(self):
        # COLOR CHANGE
        n, _ = self.colors.shape
        rnd = self.colors[random.randrange(0, n)]
        tmp = [round(num) for num in rnd]
        self.color = tuple(tmp)

        # POINT CHANGE
        indx = random.randrange(0, len(self.xy))
        self.xy[indx] = (random.randrange(self.size[0]), random.randrange(self.size[1]))


class Approximation:
    def __init__(self, polys, size, target_image_path, colors):
        self.colors = colors
        self.target_image_path = target_image_path
        self.target_name = Image.open(target_image_path)
        self.size = size
        self.polys = polys
        bk_colors = self.Color_select()
        self.background_color = tuple(bk_colors)
        self.background = Image.new("RGB", self.size, self.background_color)

    def Color_select(self):
        selected_colors = []
        n, m = self.colors.shape
        for i in range(m - 1):
            roll = random.randrange(0, n)
            selected_colors.append(self.colors[roll][i])
        return selected_colors

    def Change_background_color(self):
        n, _ = self.colors.shape
        rnd = self.colors[random.randrange(0, n)]
        tmp = [round(num) for num in rnd]
        tmp = tmp[:-1]
        self.background_color = tuple(tmp)

    def Add_to_background(self):
        self.Clear_background()
        for poly in self.polys:
            to_draw = ImageDraw.Draw(self.background, 'RGBA')
            to_draw.polygon(poly.xy, fill=poly.color)

    def Clear_background(self):
        self.background = Image.new("RGB", self.size, self.background_color)

    def DrawImage_colored(self):
        self.background.show()

    def Fitness(self):
        width = self.size[1]
        height = self.size[0]
        fitness = 0
        for y in range(width):
            for x in range(height):
                c1 = self.target_name.getpixel((x, y))
                c2 = self.background.getpixel((x, y))

                red = c1[0] - c2[0]
                green = c1[1] - c2[1]
                blue = c1[2] - c2[2]

                fitness += red ** 2 + green ** 2 + blue ** 2

        return fitness


def Mutate(image, num_of_polys):
    copy = deepcopy(image)
    copy.Change_background_color()
    index = random.randrange(0, num_of_polys)
    copy.polys[index].Slightly_change()
    copy.Add_to_background()
    return copy


def Dominant_colors(filename, n_colors):
    img = io.imread(filename)[:, :, :-1]
    pixels = np.float32(img.reshape(-1, 4))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    return palette


def Gen_algorithm(filename):
    NUM_OF_POLYS = 50
    MAX_ITERS = 50000
    N_COLORS = 10
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    colors = Dominant_colors(filename, N_COLORS)  # [[r, g, b, a], [r, g, b, a], [...]]
    i = 0
    polygons = [Poly(SIZE, colors) for _ in range(NUM_OF_POLYS)]
    generated_image = Approximation(polygons, SIZE, filename, colors)
    generated_image.Add_to_background()
    score = generated_image.Fitness()

    best_ever = deepcopy(generated_image)
    best_ever_score = score

    while i < MAX_ITERS:
        new_generated_image = Mutate(generated_image, NUM_OF_POLYS)
        new_generated_image_score = new_generated_image.Fitness()
        if new_generated_image_score < score:
            generated_image = deepcopy(new_generated_image)
            score = new_generated_image_score
            print('GENERATION (CI)', i, ' ', score)
            print('Best ever ', best_ever_score)
            print('__________________________________')
            if score < best_ever_score:
                best_ever = deepcopy(generated_image)
                best_ever_score = score
                print('GENERATION (BE)', i, ' ', score)
                print('Best ever ', best_ever_score)
                print('__________________________________')
        else:
            p = 1.0 / (i + 1) ** 0.5
            q = random.uniform(0, 1)
            if q < p:
                score = new_generated_image_score
                generated_image = deepcopy(new_generated_image)
                print('GENERATION (SA)', i, ' ', score)
                print('Best ever ', best_ever_score)
                print('__________________________________')
        if i % 500 == 0:
            best_ever.background.save(
                'C:/Users/Matke/Desktop/images/GEN' + str(i) + '_' + str(best_ever_score) + '.jpeg', quality=100,
                subsampling=0)
        i += 1

    return best_ever


Gen_algorithm('monalisa.jpg')
