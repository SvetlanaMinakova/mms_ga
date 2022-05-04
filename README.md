# Max Memory Save (MMS) - genetic algorithm (GA)

GA-based algorithm that searches how to efficiently reuse memory of a target platform to execute a deep neural network (DNN) with minimum memory cost.

The code is based on scientific paper **Memory-Throughput Trade-off for CNN-based Applications at the Edge** by Svetlana Minakova and Todor Stefanov (available online at https://dl.acm.org/doi/10.1145/3527457), which combines and extends two existing DNN memory reduction techniques:
* Data processing by parts, proposed in scientific paper **Buffer Sizes Reduction for Memory-efficient CNN Inference on Mobile and Embedded Devices** by Svetlana Minakova and Todor Stefanov. In this work, input and output data of every DNN layer is split into parts, executed in a specific order. The memory of the target platform is reused among the data parts, thus reducing the memory cost of a CNN-based application
* Buffers reuse: a technique, widely used by existing Deep Learning frameworks, where target platform memory is reused among different layers of a CNN.

## Requirements
* python 3.6+

## Inputs and outputs
*Examples of the tool inputs are located in ./data folder*.
The inputs and outputs of the tool are specific per toolflow step (see section "toolflow" below).

## Toolflow
The tool flow consists of following subsequent steps:
* **GA-based search** (*run_mms_ga.py*). During this step, the tool uses a Genetic Algorithm (GA) to explore efficient manners to reduce the amount of target platform memory required to store intermediate computational results (buffers) of DNNs used by a DNN-based application.
* **Selection** (*run_mms_selection.py*). At this step, the best GA solution (manner of DNN-based application memory reduction) is chosen from the solutions, delivered by the GA-based search.

### GA-based search (run_mms_ga.py)

During this step, the tool uses a Genetic Algorithm (GA) to explore efficient manners to reduce the amount of target platform memory required to store intermediate computational results (buffers) of DNNs used by a DNN-based application.

#### Inputs and outputs
The GA-based search step accepts as **input** an application config in [JSON](https://www.json.org/json-en.html) format (see examples in ./data/mms_ga_configs), and number of parallel CPU threads to run GA on. An application config specifies:
* unique application name
* path to one or multiple DNNs, used by the application. Every DNN is represented as a [JSON](https://www.json.org/json-en.html) file of specific structure. *See example files in ./data/json_dnn*.
* path to GA config. GA config is a file in [JSON](https://www.json.org/json-en.html) format, which specifies standard GA parameters such as number of epochs, mutation probability etc. *See example files in ./data/mms_ga_configs*.

As **output**, the GA-based search delivers a set of pareto-optimal solutions, encoded as chromosomes: genetically encoded solutions, 
where every solution is a manner of CNN-based memory reduction. A chromosome is explained in details below.
Each pareto front is saved as a .json file (see examples in *./data/test/pareto*)


#### Chromosome
A chromosome is a genetically encoded solution. In MMS, a solution is a manner of CNN-based memory reduction, which involves 
utilization of data processing by parts and buffers reuse techniques, briefly mentioned above. A chromosome in MMS encodes the amount of data processing by parts, exploited by every layer of every DNN in the input DNN-based application.
It is defined as a string of *N* elements where, *N* is a total number of layers in every DNN, used by a DNN-based application. 
For example, if a DNN-based application uses two CNNs, CNN1 and CNN1, where CNN1 has 2 layers and CNN2 has 3 layers, then for this application *N=2+3=5*.

Every i-th, 0<i<N element in the chromosome encodes data processing by parts, exploited by i-th DNN layer in a DNN-based application. 
Layers are indexed in execution order (from input layer to output layer). If an application 
uses multiple DNNs, layers of the DNNs are concatenated in the order, in which DNNs are mentioned in the application definition. For example application, 
mentioned above, elements 0 and 1 of the chromosome correspond to layer1 and layer2 of CNN1, and elements 2, 3 and 4 of the chromosome 
correspond to layer1, layer2 and layer3 of CNN2.

Every i-th, 0<i<N element in the chromosome has value 0 or 1, where 0 specifies that the encoded i-th layer does not process data by parts, and 1 specifies that 
the encoded i-th layer processes data by parts. The number of data parts, processed by layer is computed using the hyper-parameters and i/o tensor shapes of the layer.

Additionally, every chromosome is characterized with (total) memory cost of the final application and (total) execution time (latency/throughput) reduction introduced by the memory reuse into the final application.

Example chromosomes can be found in *./output/example_app.json*.

#### Mutation and crossover
To perform GA-based search, algorithm uses standard two-parent crossover and a single-gene mutation as presented in **Genetic Algorithms. Springer US, Boston** by Kumara Sastry, David Goldberg, and Graham Kendall.

#### Example use
Run MMS for a CNN-based application, specified in ./data/app_configs/test_app_conf.json config file. Execute GA in parallel on 3 CPU cores.

    python run_mms_ga.py -c ./data/app_configs/test_app_conf.json -t 3

### Selection (run_mms_selection.py)
At this step, the best GA solution (manner of DNN-based application memory reduction) is chosen from the solutions, delivered by the GA-based search.

#### Inputs and outputs
As **inputs**, the selection step accepts:
* a pareto-front, delivered by the GA-based search and saved as a .json file (see examples in *./data/test/pareto* for the examples)
* [optionally] memory constraint (float) which specifies maximum size of platform memory, occupied by the DNN buffers
* [optionally] time loss constraint (float) which specifies maximum loss of time (in ms) which can be introduced into the application

As an **output**, this step prints into console the best chromosome selected from the pareto-front.
The selection goes as follows:
1) From the input list of chromosomes, the selection script forms a list of *filtered chromosomes* that meet memory and latency loss constrains.
2) If the list of *filtered chromosomes* is empty (i.e., the input list of chromosomes contains no chromosomes which meet memory and latency loss constrains), the selection script selects as best a chromosome from the input chromosomes list, such that the best chromosome is characterized with minimum buffer sizes among all input chromosomes. Otherwise, from the list of *filtered chromosomes*, the selection script chooses a chromosome characterized with minimum time loss.

#### Example use
Select best chromosome from a pareto-front located in ./data/test/pareto/single_dnn_app.json file. The seselected chromosome should occupy at most 0.03 MB of memory to store DNN buffers.

    python run_mms_selection.py -c ./output/single_dnn_app.json -m 0.03

Expected output:
best chromosome: {'layers_num': 4, 'dp_by_parts': [True, True, True, True], 'time_loss': 0.029, 'buf_size': 0.027008}


