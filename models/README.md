# MODELS

Here the internal (designed within framework) models are located. Every model represents an entity, e.g., a DNN or a DNN-based application and serves certain purpose.

## DNN (analysis) model
DNN model, also referred as analysis DNN is a light-weighted and generic DNN model, used for DNN platform-aware characteristics estimation, and DNN executable code generation.

## System-level design models
For efficient partitioning, mapping and scheduling, DNN has to be represented as functionally equivalent a system-level model. We currently support following system-level models:
* TaskGraph: simple graph, used for high-throughput DNN partitioning and mapping.
* CSDF model: Cyclo-Static DataFlow model, used for low-memory DNN partitioning and scheduling.

## Platform models
Platform models provide high-level abstraction of target edge platform, used to execute DNN. They are used for the system-level design and simulation. Currently, the following platform models are supported:
* Architecture: main edge platform model, used for efficient partitioning, mapping and scheduling. Contains high-level abstractions of resources, available on the platform, such as computational resources (processors) and platform resources
* SimulationPlatform : edge platform model, used for simulation. This model is "heavier" than the Architecture, because it not only provides description of platform resources, but models behaviour (use) of these resources when the DNN execution is simulated.


