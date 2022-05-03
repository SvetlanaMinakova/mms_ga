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
* **GA-based search**. During this step, the tool uses a Genetic Algorithm (GA) to explore efficient manners to reduce the amount of target platform memory required to store intermediate computational results (buffers) of a DNN-based application.
* **Selection**. At this step, the best solution (manner of DNN-based application memory reduction) is chosen from solutions, delivered by the GA.
#* [optional] **Simulation** At this step, the execution of a DNN-based application is simulated with the best solution, delivered by the previous step.

### GA-based search

#### Inputs and outputs
The GA-based search step accepts as **input** an application config in [JSON](https://www.json.org/json-en.html) format (see examples in ./data/mms_ga_configs). An application config specifies:
* unique application name
* path to one or multiple DNNs, used by the application. Every DNN is represented as a file of specific structure in [JSON](https://www.json.org/json-en.html). *See example files in ./data/json_dnn*.
* path to GA config. GA config is a file in [JSON](https://www.json.org/json-en.html) format, which specifies standard GA parameters such as number of epochs, mutation probability etc. *See example files in ./data/mms_ga_configs*.

As **output**, the GA-based search delivers a set of chromosomes: genetically encoded solutions, 
where every solution is a manner of CNN-based memory reduction. A chromosome is explained in details below.


#### Chromosome
A chromosome is a genetically encoded solution. In MMS, a solution is a manner of CNN-based memory reduction, which involves 
utilization of data processing by parts and buffers reuse techniques, briefly mentioned above.

In MMS, chromosome encodes the amount of data processing by parts, exploited by every layer of every DNN in the input DNN-based application. 


Additionally, every chromosome is characterized with (total) memory cost of the final application and (total) execution time (latency/throughput) reduction introduced by the memory reuse into the final application.
Example chromosomes are listed in *./output/example_app.json*.

#### Mutation and crossover

### Selection

