from PIL import Image, ImageDraw
import cv2
import random
import numpy as np
from skimage import io


# TODO create V4, throw 50 polygons on background, move them around until u get mona?

# TODO Num of points
class Poly:
    def __init__(self, size, colors):
        self.colors = colors
        self.num_of_points = random.randrange(3, 6)
        self.size = size  # size of image [x, y]
        self.xy = [(random.randrange(self.size[0]), random.randrange(self.size[1])) for _ in range(self.num_of_points)]
        self.color = tuple(self.Color_select())

    def Color_select(self):
        selected_colors = []
        n, m = self.colors.shape
        for i in range(m):
            roll = random.randrange(0, n)
            selected_colors.append(self.colors[roll][i])
        return selected_colors

    # TODO UNIFORM
    def Slightly_change(self):
        adj = [random.uniform(0.5, 1.5),  # r
               random.uniform(0.5, 1.5),  # g
               random.uniform(0.5, 1.5),  # b
               random.uniform(0.5, 1.5)  # a
               ]
        y = list(self.color)
        y = [round(y[i] * adj[i]) for i in range(len(y))]
        y = [y[i] if y[i] <= 255 else 255 for i in range(len(y))]
        self.color = tuple(y)

        new_points = []
        for t in self.xy:
            new_point = [x * random.uniform(0.5, 1.5) for x in list(t)]
            new_point = [round(x) for x in new_point]
            if new_point[0] > self.size[0]:
                new_point[0] = self.size[0]
            if new_point[1] > self.size[1]:
                new_point[1] = self.size[1]
            new_points.append(tuple(new_point))

        self.xy = new_points


class Approximation:
    def __init__(self, polys, size, target_image_path, colors):
        self.colors = colors
        self.target_image_path = target_image_path
        self.target_name = Image.open(target_image_path)
        self.size = size
        self.polys = polys
        self.background_color = tuple(self.Color_select())
        self.background = Image.new("RGB", self.size, self.background_color)  # PIL format

    def Color_select(self):
        selected_colors = []
        n, m = self.colors.shape
        for i in range(m - 1):
            roll = random.randrange(0, n)
            selected_colors.append(self.colors[roll][i])
        return selected_colors

    def Add_to_background(self):
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

    def Fitness_eu(self):
        # Converting approx to CV2 format
        im = self.background.convert('RGB')
        pil_image = np.array(im)
        pil_image = pil_image[:, :, ::-1].copy()  # Converted image

        image = cv2.imread(self.target_image_path)  # target image cv2 format
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray_image], [0],
                                 None, [256], [0, 256])

        gray_image1 = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
        histogram1 = cv2.calcHist([gray_image1], [0],
                                  None, [256], [0, 256])

        fit = 0
        i = 0
        while i < len(histogram) and i < len(histogram1):
            fit += (histogram[i] - histogram1[i]) ** 2
            i += 1
        fit = fit ** (1 / 2)

        return fit[0]


def Init_population(num_of_polys, image_size, filename, colors):
    polys1 = [Poly(image_size, colors) for _ in range(num_of_polys)]
    polys2 = [Poly(image_size, colors) for _ in range(num_of_polys)]
    approx1 = Approximation(polys1, image_size, filename, colors)
    approx2 = Approximation(polys2, image_size, filename, colors)
    approx1.Add_to_background()
    approx2.Add_to_background()

    return [approx1, approx2]


def Best_approximation(population):
    score1 = population[0].Fitness_eu()
    score2 = population[1].Fitness_eu()

    if score1 < score2:
        return [population[0], score1]

    return [population[1], score2]


def Mutation(rate, unit, size, num_of_polys, colors):
    ind = False
    for i in range(num_of_polys):
        roll = random.random()
        if roll < rate:
            ind = True
            unit.polys[i] = Poly(size, colors)

    if not ind:
        index = random.sample([i for i in range(num_of_polys)], 5)
        for i in index:
            unit.polys[i] = Poly(size, colors)

    return unit


# TODO p, brand_new, roll
def Crossover_uni(population, global_best, size, filename, num_of_polys, colors, iters, mutation_rate):
    # if (iters + 1) % 500 == 0:
    #     return [global_best, Mutation(mutation_rate, global_best, size, num_of_polys, colors)]

    p = 120
    brand_new = 40
    polys = []
    for i in range(num_of_polys):
        roll = random.randrange(0, 160)
        if roll < brand_new:
            polys.append(Poly(size, colors))
        elif brand_new <= roll < p:
            polys.append(global_best.polys[i])
        else:
            if global_best != population[1]:
                polys.append(population[1].polys[i])
            else:
                polys.append(population[0].polys[i])

    for i in range(num_of_polys):
        polys[i].Slightly_change()
    new_approx = Approximation(polys, size, filename, colors)
    q = random.random()
    if q < 0.8:
        new_approx.background_color = global_best.background_color
    new_approx.Add_to_background()

    return [global_best, new_approx]


# TODO n_colors
def Dominant_colors(filename):
    img = io.imread(filename)[:, :, :-1]
    pixels = np.float32(img.reshape(-1, 4))

    n_colors = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    return palette


def Gen_algorithm(filename):
    MUTATION_RATE = 0.3
    NUM_OF_POLYS = 50
    MAX_ITERS = 10000
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    colors = Dominant_colors(filename)
    population = Init_population(NUM_OF_POLYS, SIZE, filename, colors)
    [global_best, global_best_score] = Best_approximation(population)
    i = 0
    while i < MAX_ITERS:
        print('GENERATION ', i)
        print(global_best_score)
        new_population = Crossover_uni(population, global_best, SIZE, filename, NUM_OF_POLYS, colors, i, MUTATION_RATE)
        [new_best, new_best_score] = Best_approximation(new_population)
        if new_best_score < global_best_score:
            print('GENERATION ', i)
            print('Current best fitness ', global_best_score)
            print('This generations best ', new_best_score)
            print('Taking better')
            global_best = new_best
            global_best_score = new_best_score
        print('__________________________')
        # if i % 100 == 0:
        #     global_best.DrawImage_colored()
        population = new_population
        i += 1
    global_best.DrawImage_colored()


Gen_algorithm('apple.jpg')

