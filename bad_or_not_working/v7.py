from PIL import Image, ImageDraw
import cv2
import random
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


class Approximation:
    def __init__(self, polys, size, target_image_path, colors, background_color):
        self.colors = colors
        self.target_image_path = target_image_path
        self.target_name = Image.open(target_image_path)
        self.size = size
        self.polys = polys
        self.background_color = background_color
        self.background = Image.new("RGB", self.size, self.background_color)

    def Color_select(self):
        index = random.randrange(0, len(self.colors))
        return self.colors[index]

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


def Best_in_pop(population):
    score = float('inf')
    best = population[0]
    for approx in population:
        approx_fit = approx.Fitness()
        if approx_fit < score:
            best = approx
            score = approx_fit

    return deepcopy(best), score


def Gen_algorithm(filename):
    NUM_OF_POLYS = 200
    MAX_ITERS = 50000
    POP_SIZE = 10
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    STARTING_BACKGROUND_COLOR = 'black'
    colors = Dominant_colors(filename)
    i = 0
    population = []
    for _ in range(POP_SIZE):
        polys = [Poly(SIZE, colors) for _ in range(NUM_OF_POLYS)]
        population.append(Approximation(polys, SIZE, filename, colors, STARTING_BACKGROUND_COLOR))
    for x in population:
        x.Add_to_background()

    best_ever, best_ever_score = Best_in_pop(population)
    score = best_ever_score
    while True:
        new_population = [Mutate(image, NUM_OF_POLYS) for image in population]
        new_generated_image, new_generated_image_score = Best_in_pop(new_population)
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
                print('GENERATION (SA)', i, ' ', score)
                print('Best ever ', best_ever_score)
                print('__________________________________')

        population = [deepcopy(ap) for ap in new_population]
        i += 1


Gen_algorithm('monalisa.jpg')
