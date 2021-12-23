import copy

class DNNTraverser:
    """
    DNN traverser: traverses DNN. Sets I/O data formats for CNN layers.
    Finds all alternative paths in the CNN (for multi-output adaptive CNNs)
    """
    def __init__(self, dnn):
        self.dnn = dnn
        self.__dnn_layers = dnn.get_layers()
        self.__dnn_output_layer_ids = []
        self.__dnn_input_layer_ids = []
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.multi_input_layers = {}
        self.multi_output_layers = {}
        # cache for found paths in the graph
        self.__paths = []

        for layer in self.dnn.get_layers():
            # process input_examples connections
            layer_input_connections = self.dnn.get_layer_input_connections(layer)
            self.layer_inputs[layer.id] = [connection.src.id for connection in layer_input_connections]
            if len(layer_input_connections) > 1:
                self.multi_input_layers[layer.id] = len(layer_input_connections)
            if len(layer_input_connections) == 0:
                self.__dnn_input_layer_ids.append(layer.id)
            # process output connections
            layer_output_connections = self.dnn.get_layer_output_connections(layer)
            self.layer_outputs[layer.id] = [connection.dst.id for connection in layer_output_connections]
            if len(layer_output_connections) > 1:
                self.multi_output_layers[layer.id] = len(layer_input_connections)
            if len(layer_output_connections) == 0:
                self.__dnn_output_layer_ids.append(layer.id)

    def get_output_layer_ids(self):
        return self.__dnn_output_layer_ids

    def get_input_layer_ids(self):
        return self.__dnn_input_layer_ids


    ##############################################################################
    ###    DFS-based find all paths reverse (from DNN outputs to DNN inputs)  ####
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''

    def _get_all_paths_reverse_util(self, u, d, visited, path):

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            path_copy = copy.deepcopy(path)
            self.__paths.append(path_copy)
            # print(path)
            # print(self.__paths)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.layer_inputs[u]:
                if visited[i] == False:
                    self._get_all_paths_reverse_util(i, d, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    # Prints all paths from 's' to 'd'
    def get_all_paths_reverse(self, s, d):

        # Mark all the vertices as not visited
        visited = []
        for i in range(len(self.dnn.get_connections())):
            visited.append(False)

        # clean paths cache
        self.__paths = []
        # Create an array to store paths
        path = []

        # Call the recursive helper function to print all paths
        self._get_all_paths_reverse_util(s, d, visited, path)

        return self.__paths

    ######################################################################
    ###    DFS-based find all paths (from DNN inputs to DNN outputs)  ####
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''

    def get_all_paths_util(self, u, d, visited, path):

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            path_copy = copy.deepcopy(path)
            self.__paths.append(path_copy)
            # print(path)
            # print(self.__paths)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.layer_outputs[u]:
                if visited[i] == False:
                    self.get_all_paths_util(i, d, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    # Prints all paths from 's' to 'd'
    def get_all_paths(self, s, d):

        # Mark all the vertices as not visited
        visited = []
        for i in range(len(self.dnn.get_connections())):
            visited.append(False)

        # clean paths cache
        self.__paths = []

        # Create an array to store paths
        path = []

        # Call the recursive helper function to print all paths
        self.get_all_paths_util(s, d, visited, path)
        return self.__paths

    ##################################################################
    ###### find early-exit subgraphs for DNNs with early exits #######

    def find_early_exit_subgraphs(self, src_id):
        subgraphs = []
        for dst_id in self.__dnn_output_layer_ids:
            all_paths_reverse = self.get_all_paths_reverse(dst_id, src_id)
            for path in all_paths_reverse:
                path.reverse()
            merged_path = self.merge_paths(all_paths_reverse)
            subgraphs.append(merged_path)
        return subgraphs

    #######################################
    ###### merge paths into a graph #######

    def merge_paths(self, paths):
        merged_path = []
        paths_copy = []
        for path in paths:
            path_copy = copy.deepcopy(path)
            paths_copy.append(path_copy)

        # remove empty paths
        for path in paths_copy:
            if len(path) == 0:
                paths_copy.remove(path)

        while len(paths_copy) > 0:
            candidates = []
            for path in paths_copy:
                if len(path) > 0:
                    path_candidate = path[0]
                    if path_candidate not in candidates:
                        candidates.append(path_candidate)
                else:
                    paths_copy.remove(path)

            next_layer_id = self.__get_next_layer_for_merged_path(candidates)

            # remove next layer from all paths (every layer can be only visited once)
            for path in paths_copy:
                if next_layer_id in path:
                    path.remove(next_layer_id)

            if next_layer_id == -1:
                if len(candidates) == 0:
                    break
                else:
                    raise Exception("paths merging interrupted at ", merged_path, ". next layer not found")

            merged_path.append(next_layer_id)

        return merged_path

    def __get_next_layer_for_merged_path(self, candidates):
        if not candidates:
            return -1
        best_candidate = candidates[0]
        for candidate in candidates:
            if candidate < best_candidate:
                best_candidate = candidate

        return best_candidate

    ##############################
    ###### print functions #######

    def print_layer_outputs(self):
        for layer in self.dnn.get_layers():
            output_ids = [output for output in self.layer_outputs[layer.id]]
            print("layer", layer.id, " output ids: ", output_ids)







