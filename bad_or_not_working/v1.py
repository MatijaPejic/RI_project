from PIL import Image, ImageDraw
import cv2
import numpy as np
import random
import imagehash
from skimage.measure._structural_similarity import structural_similarity as ssm


class Poly:
    def __init__(self, size):
        self.num_of_points = random.randrange(3, 10)
        self.size = size  # size of image [x, y]
        self.xy = [(random.randrange(self.size[0]), random.randrange(self.size[1])) for _ in range(self.num_of_points)]
        self.color = (random.randrange(255), random.randrange(255), random.randrange(255))


class Approximation:
    def __init__(self, polys, size, target_image_path):
        self.target_image_path = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2GRAY)
        self.target_name = Image.open(target_image_path)
        # self.target_image_hash = imagehash.average_hash(Image.open(target_image_path))
        self.size = size
        self.polys = polys
        self.background_color = (random.randrange(255), random.randrange(255), random.randrange(255))
        self.background = Image.new("RGB", self.size, self.background_color)  # PIL format

    def Add_to_background(self):
        for poly in self.polys:
            to_draw = ImageDraw.Draw(self.background)
            to_draw.polygon(poly.xy, outline="black", fill=poly.color)

    def Clear_background(self):
        self.background = Image.new("RGB", self.size, self.background_color)

    def DrawImage_colored(self):
        self.background.show()

    def DrawImage_grey(self):
        tmp = self.background.convert('L')
        background_cv2 = np.array(tmp)
        cv2.imshow("Grey image", background_cv2)
        cv2.waitKey(0)

    def Get_gray_image(self):
        tmp = self.background.convert('L')
        background_cv2 = np.array(tmp)
        return background_cv2

    def Compare_images(self):  # FITNESS, grt
        s = ssm(self.target_image_path, self.Get_gray_image())
        # hash = imagehash.average_hash(self.background)
        return s

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


def Init_population(population_size, num_of_polys, image_size, filename):
    init_pop = []
    for i in range(population_size):
        polys = [Poly(image_size) for _ in range(num_of_polys)]
        aprox = Approximation(polys, image_size, filename)
        aprox.Add_to_background()
        init_pop.append(aprox)
    return init_pop


def Best_approximation(population):
    best = population[0]
    best_score = best.Fitness()
    for approx in population:
        approx_score = approx.Fitness()
        if approx_score <= best_score:
            best = approx
            best_score = approx_score
    return best


def Tournament_selection(population, tournament_size):
    winners = []
    while len(winners) < len(population):
        selected_for_tour = random.sample(population, tournament_size)
        winners.append(Best_approximation(selected_for_tour))
    return winners


def Mutation(rate, population, size, num_of_polys):
    for i in range(len(population)):
        r = random.random()
        if r < rate:
            new_poly = Poly(size)
            index = random.randrange(0, num_of_polys - 1)
            (population[i]).polys[index] = new_poly
            population[i].Clear_background()
            population[i].Add_to_background()


def Generate_new_population_bp(winners, new_population_size, image_size, target_image_path, num_of_polys):
    new_population = []
    while len(new_population) < new_population_size:
        parents = random.sample(winners, 2)
        parent1 = parents[0]
        parent2 = parents[1]

        # 1 child = half of polys from each parent + the better parents background color
        poly_for_c1 = parent1.polys[:int(num_of_polys / 2)] + parent2.polys[int(num_of_polys / 2):]
        poly_for_c2 = parent1.polys[int(num_of_polys / 2):] + parent2.polys[:int(num_of_polys / 2)]

        if parent1.Fitness() <= parent2.Fitness():
            background_color_c = parent1.background_color
        else:
            background_color_c = parent2.background_color

        child1 = Approximation(poly_for_c1, image_size, target_image_path)
        child2 = Approximation(poly_for_c2, image_size, target_image_path)

        child1.background_color = background_color_c
        child2.background_color = background_color_c

        child1.background = Image.new("RGB", child1.size, child1.background_color)
        child2.background = Image.new("RGB", child2.size, child2.background_color)

        child1.Add_to_background()
        child2.Add_to_background()

        new_population.append(child1)
        new_population.append(child2)

    return new_population


def Generate_new_population_uni(winners, new_population_size, image_size, target_image_path, num_of_polys):
    new_population = []
    while len(new_population) < new_population_size:
        parents = random.sample(winners, 2)
        parent1 = parents[0]
        parent2 = parents[1]

        if parent1.Fitness() <= parent2.Fitness():
            background_color_c = parent1.background_color
            p = 0.6
        else:
            background_color_c = parent2.background_color
            p = 0.4

        poly_for_c1 = []
        for i in range(len(parent1.polys)):
            roll = random.random()
            if roll < p:
                poly_for_c1.append(parent1.polys[i])
            else:
                poly_for_c1.append(parent2.polys[i])

        child1 = Approximation(poly_for_c1, image_size, target_image_path)

        child1.background_color = background_color_c

        child1.background = Image.new("RGB", child1.size, child1.background_color)

        child1.Add_to_background()

        new_population.append(child1)

    return new_population


def Gen_algorithm(filename):
    # TODO COMPARE?
    MUTATION_RATE = 0.1
    TOURNAMENT_SIZE = 10
    POP_SIZE = 100
    NUM_OF_POLYS = 50
    MAX_ITERS = 10000
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]

    population = Init_population(POP_SIZE, NUM_OF_POLYS, SIZE, filename)
    best_approx = Best_approximation(population)
    best_approx_score = best_approx.Fitness()

    i = 0
    while i < MAX_ITERS:
        print('__________________________________________')
        print('GENERATION: ', i + 1)
        selected = Tournament_selection(population, TOURNAMENT_SIZE)  # returns POP_SIZE / 2 winners
        print('Selection completed successfully')
        new_population = Generate_new_population_uni(selected, POP_SIZE, SIZE, filename, NUM_OF_POLYS)
        print('New population generated')
        Mutation(MUTATION_RATE, new_population, SIZE, NUM_OF_POLYS)
        print('Mutation completed successfully')
        population = new_population
        current_best_approx = Best_approximation(population)
        current_best_approx_score = current_best_approx.Fitness()
        if current_best_approx_score <= best_approx_score:
            best_approx = current_best_approx
            best_approx_score = current_best_approx_score
        if i % 1000 == 0:
            best_approx.DrawImage_colored()
        print('Best approximation score: ', best_approx_score)
        i += 1


Gen_algorithm('dog.jpg')
