"""
Describes unrolling of layer computational tensor over platform processors and resources
"""


class Limit:
    def __init__(self, resource_user_name: str, resource, dims: [], split_dims: [], load_dim, load):
        self.resource_user_name = resource_user_name
        self.resource = resource
        self.dims = dims
        self.split_dims = split_dims
        self.load_dim = load_dim
        self.load = load # load or store


class DimsUnrolling:
    """
    Set of UnrollingLines, describing full unrolling of layer computational tensor
     over (CUDA) computational block dimensions
       Unrolling line specifies part of unrolling of
       layer computational tensor over CUDA computational block dimensions
       Unrolling line can have one of following structures:
           1) one op. tensor. dim to one cuda block dim: direct unrolling.
               One op. tensor. dim uses all resources of one cuda block dim.
           2) one op. tensor. dim to several cuda block dims: distributed unrolling.
               One op. tensor dim uses all resources of several cuda block dims.
           3) several op. tensor dims to one cuda block dim: shared unrolling
               Resources of one cuda block dim are distributed over several op. tensor dims.

       NOTE: it is not allowed to unroll several op. tensor dims over several cuda block dims in
       one unrolling line! Such definition is ambiguous and is considered inconsistent!
       it is also not allowed to unroll zero op. tensor dums over one/several cuda block dims or
       other way around.

       """
    def __init__(self):
        self.comp_unrolling = []
        self.limits = []
        self.loops_order = []

    def add_limit(self, resource_user_name: str, resource, dims: [], split_dims: [], load_dim, load=True):
        limit = Limit(resource_user_name, resource, dims, split_dims, load_dim, load)
        self.limits.append(limit)

    def find_limit_by_resource_name(self, resource_name:str):
        for limit in self.limits:
            if limit.resource_user_name == resource_name:
                return limit
        return None

    def find_line_by_loop_dim_name(self, loop_dim_name: str):
        for line in self.comp_unrolling:
            if loop_dim_name in line[0]:
                return line
        return None

    def add_computations_unrolling(self, comp_dim_names, block_dim_ids: []):
        line = (comp_dim_names, block_dim_ids)
        self.comp_unrolling.append(line)

    def unrolled_comp_dim_names(self):
        names = []
        for line in self.comp_unrolling:
            for dim_name in line[0]:
                names.append(dim_name)
        return names

    def __str__(self):
        result = "["
        for line in self.comp_unrolling:
            result = result + "[" + str(line) + "]; "
        result = result + "]"
        return result

    # an elaborate printout
    def print_details(self, prefix=""):
        comp_unrolling = "["
        for line in self.comp_unrolling:
            comp_unrolling = comp_unrolling + "[" + str(line) + "]; "
        comp_unrolling = comp_unrolling + "]"
        print(prefix, "loops unrolling: ", comp_unrolling)
        print(prefix, "loops order: ", self.loops_order)
        print(prefix, "resource limits: ")
        for limit in self.limits:
            print(prefix, " -", "resource:", limit.resource.name, "; user: ",
                  limit.resource_user_name, "; split_dims:", limit.split_dims)

    def is_consistent(self, verbose=True):
        consistent = True
        for line in self.comp_unrolling:
            if not self.is_line_consistent(line, verbose):
                consistent = False

        if verbose:
            print("Unrolling is consistent:", consistent)

        return consistent

    def is_line_consistent(self, line, verbose=True):
        comp_tensor_dim_names = line[0]
        block_dim_ids = line[1]
        if len(block_dim_ids) == 0:
            if verbose:
                print("inconsistent unrolling line: ", self)
                print("zero block dimensions specified, while 1 or more expected")
            return False

        if len(comp_tensor_dim_names) == 0:
            if verbose:
                print("inconsistent unrolling line: ", self)
                print("zero comp. tensor dimensions specified, while 1 or more expected")
            return False

        if len(block_dim_ids) > 1 and len(comp_tensor_dim_names) > 1:
            if verbose:
                print("inconsistent unrolling line: ", self)
                print("Ambiguous definition: it is not allowed to unroll several op. "
                      "tensor dims over several cuda block dims in one unrolling line!")
            return False

        return True