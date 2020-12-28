from PIL import Image, ImageDraw
import cv2
import random



class Poly:
    def __init__(self, size, colors):
        self.num_of_points = random.randrange(3, 4)
        self.size = size  # size of image [x, y]
        self.xy = [(random.randrange(self.size[0]), random.randrange(self.size[1])) for _ in range(self.num_of_points)]
        self.alpha = random.randrange(0,255)
        self.color = (random.randrange(colors[0]), random.randrange(colors[1]), random.randrange(colors[2]), self.alpha)


class Approximation:
    def __init__(self, polys, size, target_image_path, colors):
        self.target_name = Image.open(target_image_path)
        self.size = size
        self.polys = polys
        self.background_color = (random.randrange(colors[0]), random.randrange(colors[1]), random.randrange(colors[2]))
        self.background = Image.new("RGB", self.size, self.background_color)  # PIL format

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


def Init_population(pop_size, num_of_polys, image_size, filename, colors):
    pop = []
    for i in range(pop_size):
        polys = [Poly(image_size, colors) for _ in range(num_of_polys)]
        approx = Approximation(polys, image_size, filename, colors)
        approx.Add_to_background()
        pop.append(approx)
    return pop


def Best_approximation(population):
    best_approx = population[0]
    best_approx_score = best_approx.Fitness()
    for approx in population:
        score = approx.Fitness()
        if score < best_approx_score:
            best_approx = approx
            best_approx_score = score

    return [best_approx, best_approx_score]


def Mutation(pop_size, rate, population, size, num_of_polys, colors):
    for i in range(pop_size):
        r = random.random()
        if r < rate:
            new_poly = Poly(size, colors)
            index = random.randrange(0, num_of_polys - 1)
            (population[i]).polys[index] = new_poly
            population[i].Clear_background()
            population[i].Add_to_background()

        q = random.random()
        if q < rate:
            population[i].background_color = (
                random.randrange(colors[0]), random.randrange(colors[1]), random.randrange(colors[2]))
            population[i].Clear_background()
            population[i].Add_to_background()


def Crossover(population, global_best, size, filename, num_of_polys, colors):
    new_population = population
    while len(new_population) < len(population) * 2:
        parents = random.sample(population, 2)
        parent1 = parents[0]
        parent2 = parents[1]
        child_polys = random.sample(parent1.polys,int(num_of_polys/2)) + random.sample(parent2.polys, int(num_of_polys/2))
        random.shuffle(child_polys)

        child = Approximation(child_polys, size, filename, colors)
        roll = random.random()
        if roll < 0.5:
            child.background_color = parent1.background_color
        child.Add_to_background()
        new_population.append(child)
    return new_population


def Tournament_selection(population, tournament_size):
    winners = []
    while len(winners) < int(len(population)/2):
        selected_for_tour = random.sample(population, tournament_size)
        winners.append(Best_approximation(selected_for_tour)[0])
    return winners

def Colorscan(filename, size):
    im = Image.open(filename)
    max_r = 0
    max_g = 0
    max_b = 0
    width = size[1]
    height = size[0]
    for y in range(width):
        for x in range(height):
            px = im.getpixel((x, y))
            if px[0] > max_r:
                max_r = px[0]
            if px[1] > max_g:
                max_g = px[1]
            if px[2] > max_b:
                max_b = px[2]

    return [max_r, max_g, max_b]


def Gen_algorithm(filename):
    MUTATION_RATE = 0.05
    POP_SIZE = 4
    NUM_OF_POLYS = 50
    MAX_ITERS = 10000
    TOURNAMENT_SIZE = 2
    target_image_cv2 = cv2.imread(filename)  # for size params
    height, width, _ = target_image_cv2.shape
    SIZE = [width, height]
    colors = Colorscan(filename, SIZE)

    population = Init_population(POP_SIZE, NUM_OF_POLYS, SIZE, filename, colors)
    print('Initial population created compleated ...')
    [global_best, global_best_score] = Best_approximation(population)
    i = 0
    while i < MAX_ITERS:
        print('__________________________')
        selected = Tournament_selection(population, TOURNAMENT_SIZE)
        print('Selection compleated ...')
        population = Crossover(selected, global_best, SIZE, filename, NUM_OF_POLYS, colors)
        print('Crossover compleated ...')
        Mutation(POP_SIZE, MUTATION_RATE, population, SIZE, NUM_OF_POLYS, colors)
        print('Mutation compleated ...')
        [new_best, new_best_score] = Best_approximation(population)
        if new_best_score < global_best_score:
            global_best = new_best
            global_best_score = new_best_score

        if i % 100 == 0:
            global_best.DrawImage_colored()

        print('GENERATION ', i)
        print('Best fitness', global_best_score)
        i += 1


Gen_algorithm('tesla.jpeg')
