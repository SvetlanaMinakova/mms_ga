from low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from models.dnn_model.dnn import DNN
from low_memory.dp_by_parts import get_max_phases_per_layer # , eval_thr_loss, reset_phases
from low_memory.mms.ga_based.MMS_ga_eval import eval_chromosome_time_loss_ms, eval_dnn_buffers_size_mb
import random


class MMSga:
    """
        Genetic algorithm, that performs search of efficient maximum-memory-save (MMS) memory reuse.
        MMS reuses memory among and within layers of a CNN, thereby reducting the CNN memory cost.
        It employs and combines two memory reuse techniques, namely:
         1) DP (data processing by parts) where memory is reused among data parts processed by the same CNN layer.
         This memory reuse method can cause loss of throughput due to synchronization between data parts
         2) buffers reuse where memory is reused for storage of data, produced and accepted by different CNN layers
         This memory reuse method does not affect throughput (or any characteristics other than memory) of a CNN

        :param epochs: number of epochs to run GA
        standard GA parameters
        :param population_start_size: number of samples in the start population
        :param selection_percent: percent of population to be selected at every iteration
        :param mutation_probability: chance that a mutation will happen 0=<x<=1
        :param mutation_percent: percent of the samples in the current offspring to mutate
        :param max_no_improvement_epochs: max epochs to continue GA after exec. time have stopped improving
        :param eval_communication (flag): if True, communication between processors will be taken into account
        :param data_token_size: size of one data token (in Bytes)
        :param verbose: print details

        :return best found dp-by-parts list:
            list of boolean flags of len = len(self.layers_num)
            each i-th flag = True/False determines whether layer li of dnn
            processes data by parts (True) or not (False)
        """
    # TODO: implement
    # return pareto-front where every chromosome, representing DP within the CNN
    #         is annotated with loss of throughput (caused by processing data by parts, the smaller, the better)
    #         and buffer sizes (the smaller, the better)
    def __init__(self, dnn: DNN, epochs=10,
                 population_start_size=100, selection_percent=50, mutation_probability=0,
                 mutation_percent=10, max_no_improvement_epochs=10,
                 dp_by_parts_init_probability=0.5, data_token_size=4, verbose=True):
        self.dnn = dnn
        self.layers_num = len(self.dnn.get_layers())

        self.population_start_size = population_start_size
        self.epochs = epochs
        self.selection_percent = selection_percent
        self.mutation_probability = mutation_probability
        self.mutation_percent = mutation_percent
        self.max_no_improvement_epochs = max_no_improvement_epochs
        # specific
        self.dp_by_parts_init_probability = dp_by_parts_init_probability
        self.data_token_size = data_token_size

        # meta-data
        self.max_phases_per_layer = get_max_phases_per_layer(dnn)
        self.population = []
        self.selected_offspring = []

        # evals are built in the chromosome
        # self.time_evals = {}
        # self.time_evals_sorted = {}

        self.verbose = verbose

        # new
        # pareto front found across all epochs
        self.pareto_across_dse = []

    def init_with_random_population(self):
        self.population = []
        for i in range(0, self.population_start_size):
            random_chromosome = self.generate_random_chromosome()
            self.annotate_chromosome_with_fitness(random_chromosome)
            self.population.append(random_chromosome)

    def generate_random_chromosome(self):
        random_chromosome = MMSChromosome(self.layers_num)
        random_chromosome.init_random(self.dp_by_parts_init_probability)
        return random_chromosome

    def run(self):
        """ run GA"""
        if self.verbose:
            print("START GA, epochs = ", self.epochs, ", init_offspring: ", self.population_start_size,
                  ", selection:", self.selection_percent, "%", ", mutation probability:", self.mutation_probability)
        cur_epoch = 0
        cur_mem = 0
        # we are going to iteratively select top selection_percent chromosomes of current population ...
        chromosomes_to_select = int(len(self.population) * self.selection_percent / 100)

        # if 1. best time for population improves for >= no_improvement_epochs ...
        cur = self.population[0]
        self.annotate_chromosome_with_fitness(cur)
        cur_buf_size = cur.buf_size
        best = cur
        best_buf_size = cur_buf_size
        no_improvement_epochs = 0

        # ... and 2. there is something to select, and 3. done epochs < max_epochs,
        while chromosomes_to_select > 0 and cur_epoch < self.epochs:
            self.make_iteration(chromosomes_to_select)
            chromosomes_to_select = int(len(self.population) * self.selection_percent / 100)
            cur_epoch = cur_epoch + 1
            # population is annotated and sorted during selection so that
            # the first chromosome always corresponds to the best result
            cur = self.population[0]
            cur_buf_size = cur.buf_size
            best_buf_size = best.buf_size
            improved = False

            if cur_buf_size < best_buf_size:
                if self.verbose:
                    print("epoch", cur_epoch, " cur buffers size ", cur_buf_size, " < ", "best buffers size", best_buf_size, "best result reset to")
                best = cur
                best_buf_size = cur_buf_size
                if self.verbose:
                    best.print_short()
                improved = True

            if self.verbose:
                print("EPOCH: ", cur_epoch, "epoch best memory: ", cur_buf_size, "GA best memory: ", best_buf_size)
                print("population size", len(self.population))

            if improved:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs = no_improvement_epochs + 1

            if no_improvement_epochs == self.max_no_improvement_epochs:
                if self.verbose:
                    print("ALGORITHM FINISHED ON EPOCH", cur_epoch, " ,NO IMPROVEMENT FOR", self.max_no_improvement_epochs, "EPOCHS")

                if self.verbose:
                    best.print_short()
                return best.dp_by_parts
                # return partitions

            if chromosomes_to_select == 0:
                if self.verbose:
                    print("ALGORITHM FINISHED ON EPOCH", cur_epoch, " , POPULATION SIZE: ", self.population.__len__())

            if cur_epoch == self.epochs:
                if self.verbose:
                    print("ALGORITHM FINISHED, MAX EPOCHS: ", cur_epoch, " ACHIEVED: ")

        if self.verbose:
            print("ALGORITHM FINISHED WITH ACHIEVED BUFFERS SIZE", best_buf_size)
        if self.verbose:
            best.print_short()
            fin_eval = best.buf_size
            if self.verbose:
                print("fin buffers size: ", fin_eval)
        return best.dp_by_parts
        # return partitions

    def make_iteration(self, chromosomes_to_select):
        """
        Make a GA iteration:
       - select chromosomes_to_select from current population
       - crossover and get a child for every couple [x, x+1] in selected chromosomes
       - set current population = selected chromosomes + their children
       - mutate mutation_percent of population with probability = mutation_probability
        """

        # select top chromosomes_to_select from current population
        self.select(chromosomes_to_select)

        # crossover every couple [x, x+1] in selected offspring, and add children into population
        for i in range(0, int(self.selected_offspring.__len__()/2)):
            parent1 = self.selected_offspring[(2 * i)]
            parent2 = self.selected_offspring[(2 * i + 1)]
            child = self.crossover(parent1, parent2)
            self.selected_offspring.append(child)

        # set current offspring as selected offspring
        self.population = self.selected_offspring
        # make mutation_percent of our population more diverse with probability mutation_probability
        self.mutate()

    """
    GA operators
    """

    def select(self, chromosomes_to_select: int):
        """
        Selection: select top chromosomes_to_select from current population
        :param chromosomes_to_select top chromosomes to select
        """
        self.selected_offspring = []

        # evaluate current population in terms of  fitness function (time loss and buffers size)
        for chromosome in self.population:
            self.annotate_chromosome_with_fitness(chromosome)

        # sort chromosomes by memory cost (descending)
        self.population = sorted(self.population, key=lambda x: x.buf_size, reverse=False)

        # select top selection_percent chromosomes of current population
        for i in range(chromosomes_to_select):
            chromosome = self.population[i]
            self.selected_offspring.append(chromosome)

    def annotate_chromosome_with_fitness(self, chromosome):
        """ Annotate chromosome with fitness function: with time loss and buffers size"""
        # evaluate chromosome in terms of throughput loss and buffers size
        phases_per_layer = get_phases_per_layer(self.dnn, chromosome, self.max_phases_per_layer)
        time_loss_ms = eval_chromosome_time_loss_ms(phases_per_layer)
        buf_size_mb = eval_dnn_buffers_size_mb(self.dnn, phases_per_layer, self.data_token_size)

        # annotate chromosome
        chromosome.time_loss = time_loss_ms
        chromosome.buf_size = buf_size_mb

    def mutate(self):
        """
        Mutation: make our population more diverse with mutation
        """
        if self.mutation_probability == 0 or self.mutation_percent == 0:
            return

        # roll a dice: is there a mutation going to happen?
        random_chance = random.uniform(0, 1)  # Random float x, 0 <= x < 1
        # print("Mutation: random chance: ", random_chance)

        # mutate
        if random_chance <= self.mutation_probability:
            # at least one chromosome to mutate
            chromosomes_to_mutate = max(int(self.mutation_percent/100 * len(self.population)), 1)
            # print("Mutate ", chromosomes_to_mutate, "in current offspring of len", self.population.__len__())
            for i in range(chromosomes_to_mutate):
                random_chromosome_id = random.randint(0, self.population.__len__()-1)
                random_chromosome = self.population[random_chromosome_id]
                random_chromosome.mutate()

        # no mutation

    def crossover(self, chromosome1: MMSChromosome, chromosome2: MMSChromosome):
        """
        Crossover: exchange halves of parent chromosomes chromosome1 and chromosome2
        """
        child_chromosome = MMSChromosome(chromosome1.layers_num)
        # genes of first parent: range1 = [0, (layers_num / 2)]
        for layer_id in range(0, int(chromosome1.layers_num / 2)):
            child_chromosome.dp_by_parts[layer_id] = chromosome1.dp_by_parts[layer_id]
        # genes of second parent: range1 = [0, (layers_num / 2 - 1)]
        for layer_id in range(int(chromosome1.layers_num / 2), chromosome1.layers_num):
            child_chromosome.dp_by_parts[layer_id] = chromosome2.dp_by_parts[layer_id]
        return child_chromosome


def get_phases_per_layer(dnn: DNN, chromosome: MMSChromosome, max_phases_per_layer):
    """
    Determine number of phases performed by every DNN layer with respective MMS chromosome
    :param dnn: DNN
    :param chromosome: MMS chromosome
    :param max_phases_per_layer: dictionary where key = name of layer within the dnn,
    value = maximum number of phases that can be ever performed by this layer
    :return: phases_per_layer: dictionary where key = name of layer within the dnn,
    value =number of phases that are performed by this layer according to MMSChromosome
    """
    layers = dnn.get_layers()
    phases_per_layer = {}
    for layer_id in range(chromosome.layers_num):
        layer = layers[layer_id]
        dp_by_parts_flag = chromosome.dp_by_parts[layer_id]
        # data processing by parts
        if dp_by_parts_flag is True:
            phases_per_layer[layer.name] = max_phases_per_layer[layer.name]
        else:
            # no data processing by parts
            phases_per_layer[layer.name] = 1
    return phases_per_layer


