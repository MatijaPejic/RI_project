from PIL import Image, ImageDraw
import cv2
import random
import numpy as np
from skimage import io
from copy import deepcopy
import extcolors
import threading


# TODO Test mutation
total_fitness = 0
class Poly:
    def __init__(self, size, colors):
        self.colors = colors
        self.num_of_points = random.randrange(3, 4)
        self.size = size  # size of image [x, y]
        self.xy = [(random.randrange(self.size[0]), random.randrange(self.size[1])) for _ in range(self.num_of_points)]
        self.color = self.Color_select()

    def Color_select(self):
        index = random.randrange(0, len(self.colors))
        tmp = self.colors[index]
        tmp = list(tmp)
        tmp.append(125)
        tmp = tuple(tmp)
        return tmp

    def Slightly_change(self):
        # COLOR CHANGE
        index = random.randrange(0, len(self.colors))
        tmp = self.colors[index]
        tmp = list(tmp)
        tmp.append(random.randrange(0, 255))
        tmp = tuple(tmp)
        self.color = tmp

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
        self.background_color = self.Color_select()
        self.background = Image.new("RGB", self.size, self.background_color)

    def Color_select(self):
        index = random.randrange(0, len(self.colors))
        return self.colors[index]

    def Change_background_color(self):
        tmp = list(self.background_color)
        index = random.randrange(0, 3)
        new_value = random.randrange(0, 255)
        tmp[index] = new_value
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

    def Fitness(self, y):
        height = self.size[0]
        fitness = 0
        global total_fitness
        lock = threading.Lock()
        for x in range(height):
            c1 = self.target_name.getpixel((x, y))
            c2 = self.background.getpixel((x, y))

            red = c1[0] - c2[0]
            green = c1[1] - c2[1]
            blue = c1[2] - c2[2]
            fitness += red ** 2 + green ** 2 + blue ** 2
        lock.acquire()
        total_fitness += fitness
        lock.release()

    def calc_fit(self):
        y = self.size[1]
        threads = []
        global total_fitness
        for i in range(y):
            t = threading.Thread(target=self.Fitness, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        r = total_fitness
        total_fitness = 0
        return r


def Mutate(image, num_of_polys):
    copy = deepcopy(image)
    copy.Change_background_color()
    index = random.randrange(0, num_of_polys)
    copy.polys[index].Slightly_change()
    copy.Add_to_background()
    return copy


def Dominant_colors(filename):
    img = Image.open(filename)
    colors, pixel_count = extcolors.extract_from_image(img)
    dominant_colors = []
    for t in colors:
        dominant_colors.append(t[0])
    return dominant_colors


def Gen_algorithm(filename):
    NUM_OF_POLYS = 50
    MAX_ITERS = 50000
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    colors = Dominant_colors(filename)
    i = 0
    polygons = [Poly(SIZE, colors) for _ in range(NUM_OF_POLYS)]
    generated_image = Approximation(polygons, SIZE, filename, colors)
    generated_image.Add_to_background()
    score = generated_image.calc_fit()

    best_ever = deepcopy(generated_image)
    best_ever_score = score

    while i < MAX_ITERS:
        new_generated_image = Mutate(generated_image, NUM_OF_POLYS)
        new_generated_image_score = new_generated_image.calc_fit()
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
