from PIL import Image, ImageDraw
import cv2
import random
import numpy as np
from skimage import io
from copy import deepcopy
import extcolors


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
        tmp = list(tmp)  # random dominant color
        tmp.append(150)  # alpha
        tmp = tuple(tmp)
        self.color = tmp

        # POINT CHANGE
        indx = random.randrange(0, len(self.xy))
        self.xy[indx] = (random.randrange(self.size[0]), random.randrange(self.size[1]))


# TODO
class Approximation:
    def __init__(self, polys, size, target_image_path, colors, background_color):
        self.colors = colors
        self.target_image_path = target_image_path
        self.target_name = Image.open(target_image_path)
        self.size = size
        self.polys = polys
        # self.background_color = self.Color_select()
        self.background_color = background_color
        self.background = Image.new("RGB", self.size, self.background_color)

    def Color_select(self):
        index = random.randrange(0, len(self.colors))
        return self.colors[index]

    # def Change_background_color(self):
    #     index = random.randrange(0, len(self.colors))
    #     self.background_color = self.colors[index]

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


# TODO
def Mutate(image, num_of_polys):
    copy = deepcopy(image)
    # copy.Change_background_color()
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
    NUM_OF_POLYS = 200
    MAX_ITERS = 50000
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    STARTING_BACKGROUND_COLOR = 'black'
    colors = Dominant_colors(filename)
    i = 0
    polygons = [Poly(SIZE, colors) for _ in range(NUM_OF_POLYS)]
    generated_image = Approximation(polygons, SIZE, filename, colors, STARTING_BACKGROUND_COLOR)
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
                best_ever.background.save(
                    'C:/Users/Matke/Desktop/images/GEN' + str(i) + '_' + str(best_ever_score) + '.jpeg', quality=100,
                    subsampling=0)
        else:
            p = 1.0 / (i + 1) ** 0.5
            q = random.uniform(0, 1)
            if q < p:
                score = new_generated_image_score
                generated_image = deepcopy(new_generated_image)
                print('GENERATION (SA)', i, ' ', score)
                print('Best ever ', best_ever_score)
                print('__________________________________')
        # if i % 250 == 0:
        #     best_ever.background.save(
        #         'C:/Users/Matke/Desktop/images/GEN' + str(i) + '_' + str(best_ever_score) + '.jpeg', quality=100,
        #         subsampling=0)
        i += 1

    return best_ever


Gen_algorithm('monalisa.jpg')
