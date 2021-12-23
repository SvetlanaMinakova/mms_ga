
def select_pareto(chromosomes):
    """ Select pareto-front from MMS chromosomes,
    :param chromosomes: list of MMS chromosomes
    """
    pareto = []
    for chromosome_id in range(len(chromosomes)):
        is_pareto = is_pareto_point(chromosome_id, chromosomes)
        if is_pareto:
            # avoid duplicates
            if not contains_point_with_the_same_fitness(pareto, chromosomes[chromosome_id]):
                pareto.append(chromosomes[chromosome_id])

    # print("pareto points num/points num:", len(pareto), "/", len(chromosomes))

    return pareto


def contains_point_with_the_same_fitness(pareto, point):
    """ Checks if pareto-front already contains a point with the same characteristics """
    for pareto_point in pareto:
        if pareto_point.buf_size == point.buf_size and pareto_point.time_loss == point.time_loss:
            return True
    return False


def merge_pareto_fronts(pareto1, pareto2):
    """
    Merge two pareto fronts of MMS chromosomes
    :param pareto1: first pareto front
    :param pareto2: second pareto front
    :return: merged pareto front that has pareto-optimal points from both fronts
    """
    concatenated_pareto_front = []
    for chromosome in pareto1:
        concatenated_pareto_front.append(chromosome)
    for chromosome in pareto2:
        concatenated_pareto_front.append(chromosome)

    merged_pareto = select_pareto(concatenated_pareto_front)
    return merged_pareto


def is_pareto_point(chromosome_id, chromosomes):
    """
    Check if chromosome is a pareto point, i.e., is not dominated
    by any other point
    :param chromosome_id: if of chromosome to check
    :param chromosomes: MMS chromosomes
    :return: True, if chromosome is a pareto point and False otherwise
    """
    chromosome = chromosomes[chromosome_id]

    for other_chromosome_id in range(len(chromosomes)):
        # do not compare to itself
        if other_chromosome_id != chromosome_id:
            other_chromosome = chromosomes[other_chromosome_id]
            # chromosome is dominated
            if other_chromosome.buf_size < chromosome.buf_size:
                if other_chromosome.time_loss < chromosome.time_loss:
                    return False
    # not dominated by any other chromosome
    return True
