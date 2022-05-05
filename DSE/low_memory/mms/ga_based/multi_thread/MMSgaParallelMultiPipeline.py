import math

from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from models.dnn_model.dnn import DNN
from DSE.low_memory.dp_by_parts import get_max_phases_per_layer # , eval_thr_loss, reset_phases
from DSE.low_memory.mms.ga_based.MMS_ga_eval import eval_chromosome_time_loss_ms_multi_pipeline,\
    eval_dnn_buffers_size_multi_pipelined_mb
from DSE.low_memory.mms.ga_based.MMSParetoSelection import select_pareto, merge_pareto_fronts
import random
import copy
import time
from multiprocessing import Pool


class MMSgaParallelMultiPipeline:
    """
        This version performs the heaviest part of computation, i.e., DNN buffers size evaluation, in parallel
        Genetic algorithm, that performs search of efficient maximum-memory-save (MMS) memory reuse.
        MMS reuses memory among and within layers of a CNN, thereby reducing the CNN memory cost.
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
        :param return_pareto: (flag) return pareto-front of chromosomes, where every chromosome,
            representing DP within the CNN is annotated with loss of throughput (caused by processing data
            by parts, the smaller, the better) and buffer sizes (the smaller, the better)
            If this flag is False, the best chromosome from the pareto front is returned
        """
    def __init__(self, partitions_per_dnn: [], epochs=10,
                 population_start_size=100, selection_percent=50, mutation_probability=0,
                 mutation_percent=10, max_no_improvement_epochs=10,
                 dp_by_parts_init_probability=0.5, data_token_size=4,
                 parr_threads=1,
                 verbose=True,
                 return_pareto=True):

        # multi-dnn-specific
        self.partitions_per_dnn = partitions_per_dnn
        self.dnns_num = len(partitions_per_dnn)
        self.layers_num = 0
        for partitions in partitions_per_dnn:
            for partition in partitions:
                self.layers_num += len(partition.get_layers())

        # parallel processing
        self.parr_threads = parr_threads
        self.partitions_per_dnn_copies = [copy.deepcopy(self.partitions_per_dnn) for thr in range(self.parr_threads)]

        # standard GA parameters
        self.population_start_size = population_start_size
        self.epochs = epochs
        self.selection_percent = selection_percent
        self.mutation_probability = mutation_probability
        self.mutation_percent = mutation_percent
        self.max_no_improvement_epochs = max_no_improvement_epochs

        # specific GA parameters
        self.dp_by_parts_init_probability = dp_by_parts_init_probability
        self.data_token_size = data_token_size

        # meta-data
        # max phases in every layer of every partition of every dnn
        self.max_phases_per_layer_per_partition_per_dnn = []
        for dnn_id in range(self.dnns_num):
            max_ph_per_dnn = {}
            partitions = partitions_per_dnn[dnn_id]
            for partition in partitions:
                max_ph_per_dnn[partition.name] = get_max_phases_per_layer(partition)
            self.max_phases_per_layer_per_partition_per_dnn.append(max_ph_per_dnn)

        self.population = []
        self.selected_offspring = []

        self.verbose = verbose

        # new
        # pareto front found across all epochs
        self.return_pareto = return_pareto
        self.pareto_across_dse = []

    """ GA initialization"""

    def init_with_random_population(self):
        """ Generate random population"""
        # init-start timer
        init_start_time = time.time()

        # generate chromosomes
        self.population = []
        for i in range(0, self.population_start_size):
            random_chromosome = self.generate_random_chromosome()
            self.population.append(random_chromosome)

        # annotate chromosomes with fitness: buffer sizes and time loss
        self.annotate_chromosomes_with_fitness_parr(print_batches=True)
        # sort population by buffers size (descending)
        self.population = sorted(self.population, key=lambda x: x.buf_size, reverse=False)
        # update pareto-front
        self.update_pareto_front()

        # init-end timer
        init_end_time = time.time()
        if self.verbose:
            print("init time:", time_elapsed_str(init_start_time, init_end_time))

    def generate_random_chromosome(self):
        random_chromosome = MMSChromosome(self.layers_num)
        random_chromosome.init_random(self.dp_by_parts_init_probability)
        return random_chromosome

    """ GA execution"""

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
        cur.partitions_per_dnn = self.partitions_per_dnn
        # chromosome is already annotated
        # buf_size_mb, time_loss_ms = self.compute_fitness(cur)
        # annotate_chromosome_with_fitness(cur, buf_size_mb, time_loss_ms)
        cur_buf_size = cur.buf_size
        best = cur
        best_buf_size = cur_buf_size
        no_improvement_epochs = 0

        # ga-start timer
        ga_start_time = time.time()

        # ... and 2. there is something to select, and 3. done epochs < max_epochs,
        while chromosomes_to_select > 0 and cur_epoch < self.epochs:
            # epoch-start timer
            epoch_start_time = time.time()

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
                    print("epoch", cur_epoch, " cur buffers size ", cur_buf_size, " < ",
                          "best buffers size", best_buf_size, "best result reset to")
                best = cur
                best_buf_size = cur_buf_size
                if self.verbose:
                    best.print_short()
                improved = True

            # epoch-end timer
            epoch_end_time = time.time()

            if self.verbose:
                print("EPOCH: ", cur_epoch, "epoch best memory: ", cur_buf_size, "GA best memory: ", best_buf_size)
                print("population size", len(self.population))
                print("epoch time:", time_elapsed_str(epoch_start_time, epoch_end_time),
                      "; GA time:", time_elapsed_str(ga_start_time, epoch_end_time))

            if improved:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs = no_improvement_epochs + 1

            if no_improvement_epochs == self.max_no_improvement_epochs:
                if self.verbose:
                    print("ALGORITHM FINISHED ON EPOCH", cur_epoch, " ,NO IMPROVEMENT FOR", self.max_no_improvement_epochs, "EPOCHS")

                if self.verbose:
                    best.print_short()

                # return results
                if self.return_pareto:
                    return self.pareto_across_dse
                else:
                    return best

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
                print("fin best buffers size: ", fin_eval)
        # return results
        if self.return_pareto:
            return self.pareto_across_dse
        else:
            return best

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

        # make mutation_percent of the population more diverse by mutating them
        self.mutate()

        # estimate fitness of chromosomes
        self.annotate_chromosomes_with_fitness_parr()

        # sort chromosomes by memory cost (descending)
        self.population = sorted(self.population, key=lambda x: x.buf_size, reverse=False)

        # update pareto-front
        self.update_pareto_front()

    def update_pareto_front(self):
        current_pareto = select_pareto(self.population)
        self.pareto_across_dse = merge_pareto_fronts(self.pareto_across_dse, current_pareto)

    """
    Evaluation of chromosome in terms of fitness function (time loss and buffers size)
    """

    def annotate_chromosomes_with_fitness_parr(self, print_batches=False):
        """ Parallel evaluation: self.parr_threads (specified as GA input) are used to perform evaluation"""
        # split chromosomes in batches
        batch_size = self.parr_threads
        chromosomes_num = len(self.population)
        batches_num = math.ceil(chromosomes_num/batch_size)

        # chromosomes are processed in parallel batches
        batches = self.generate_chromosomes_batches(chromosomes_num, batches_num, batch_size)

        if print_batches:
            if self.verbose:
                print("eval ", chromosomes_num, "chromosomes in", len(batches), "parallel batches",
                      len(batches[0]), "chromosomes each (",
                      len(batches[-1]), " chromosomes in the last batch)")

        # evaluate fitness per batch (in parallel)
        fitness_per_batch = []
        for batch in batches:
            batch_fitness = self.compute_batch_fitness(batch)
            fitness_per_batch.append(batch_fitness)

        # annotate every chromosome with fitness
        chromosome_id = 0
        for batch_fitness in fitness_per_batch:
            for chromosome_fitness in batch_fitness:
                buf_size_mb, time_loss_ms = chromosome_fitness
                self.population[chromosome_id].buf_size = buf_size_mb
                self.population[chromosome_id].time_loss = time_loss_ms
                chromosome_id += 1

    def compute_batch_fitness(self, chromosomes_batch):
        # create parallel threads pool
        pool = Pool(processes=len(chromosomes_batch))
        # map the eval function to the batches (eval chromosomes per-batch in parallel)
        batch_fitness = pool.map(self.compute_fitness, chromosomes_batch)
        # close pool
        pool.close()
        # wait for threads pool to finish
        pool.join()

        return batch_fitness

    def generate_chromosomes_batches(self, chromosomes_num, batches_num, batch_size):
        batches = []

        # generate batches
        for i in range(batches_num):
            batch_start_id = i*batch_size
            batch_end_id = min((i+1)*batch_size, chromosomes_num)
            chromosomes_batch = self.population[batch_start_id:batch_end_id]
            batches.append(chromosomes_batch)

        # annotate chromosomes with respective dnn copy
        for batch in batches:
            for chromosome_id in range(len(batch)):
                batch[chromosome_id].partitions_per_dnn = self.partitions_per_dnn_copies[chromosome_id]

        return batches

    def compute_fitness(self, chromosome):
        """
        Evaluate chromosome in terms of throughput loss and buffers size
        :param chromosome: MMS chromosome to be evaluated, annotated with dnn
        :return: buf_size_mb, time_loss_ms where
            buf_size_mb (float) is the dnn buffers size (in megabytes)
            time_loss_ms (float) is the latency loss (delay) caused by data processing by parts
        """

        phases_per_layer_per_partition_per_dnn = get_phases_per_layer_per_partition_per_dnn(self.partitions_per_dnn,
                                                                                            chromosome,
                                                                                            self.max_phases_per_layer_per_partition_per_dnn)

        time_loss_ms = eval_chromosome_time_loss_ms_multi_pipeline(phases_per_layer_per_partition_per_dnn)
        buf_size_mb = eval_dnn_buffers_size_multi_pipelined_mb(self.partitions_per_dnn,
                                                               phases_per_layer_per_partition_per_dnn,
                                                               self.data_token_size)

        # return evaluation
        return buf_size_mb, time_loss_ms

    """
    GA operators
    """

    def select(self, chromosomes_to_select: int):
        """
        Selection: select top chromosomes_to_select from current population
        :param chromosomes_to_select top chromosomes to select
        """
        self.selected_offspring = []

        # select top selection_percent chromosomes of current population
        for i in range(chromosomes_to_select):
            chromosome = self.population[i]
            self.selected_offspring.append(chromosome)

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


def annotate_chromosome_with_fitness(chromosome, buf_size_mb, time_loss_ms):
    """ Annotate chromosome with fitness function: with time loss and buffers size"""
    chromosome.time_loss = time_loss_ms
    chromosome.buf_size = buf_size_mb
    print("chromosome buf size:", buf_size_mb, ", time loss: ", time_loss_ms)


def get_phases_per_layer_per_partition_per_dnn(partitions_per_dnn: [],
                                               chromosome: MMSChromosome,
                                               max_phases_per_layer_per_partition_per_dnn):
    """
    Determine number of phases performed by every DNN layer with respective MMS chromosome
    :param partitions_per_dnn: list of pipelined partitions per dnn
    :param chromosome: MMS chromosome
    :param max_phases_per_layer_per_partition_per_dnn:
    """
    phases_per_layer_per_partition_per_dnn = []
    dnns_num = len(partitions_per_dnn)
    layer_id_in_chromosome = 0
    for dnn_id in range(dnns_num):
        max_ph_per_dnn = max_phases_per_layer_per_partition_per_dnn[dnn_id]
        ph_per_dnn = {}
        partitions = partitions_per_dnn[dnn_id]
        for partition in partitions:
            max_ph_per_dnn_per_partition = max_ph_per_dnn[partition.name]
            ph_per_dnn_per_partition = {}
            for layer in partition.get_layers():
                dp_by_parts_flag = chromosome.dp_by_parts[layer_id_in_chromosome]
                # data processing by parts
                if dp_by_parts_flag is True:
                    ph_per_dnn_per_partition[layer.name] = max_ph_per_dnn_per_partition[layer.name]
                else:
                    # no data processing by parts
                    ph_per_dnn_per_partition[layer.name] = 1
                layer_id_in_chromosome += 1
            ph_per_dnn[partition.name] = ph_per_dnn_per_partition
        phases_per_layer_per_partition_per_dnn.append(ph_per_dnn)

    return phases_per_layer_per_partition_per_dnn


def time_elapsed(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds


def time_elapsed_str(start_time, end_time):
    hours, minutes, seconds = time_elapsed(start_time, end_time)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    return time_str
