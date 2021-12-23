# CONVERTERS

Converters are designed to perform conversion between two representations of an entity.

## Conversion between DNN representations
Most of the converters are dedicated to change representation of a deep neural network (DNN) from an input representation to an output representation.
Currently, we support following third-party and internal (designed within framework) DNN representations:

* (third-party) ONNX DNN, i.e., DNN in Open Neural Network Exchange (ONNX) format. ONNX format is typically used to distribute the model among DL frameworks.
* (third-party) keras DNN, i.e., DNN represented as a DNN model from Keras DL framework. We use this format for DNN training and validation, required to estimate DNN accuracy. We also use this format to access DNN models zoo of Keras DL framework.
* (third-party) tflite DNN, i.e., DNN represented as a DNN model from TensorflowLite DL framework. We use this format for DNN quantization.
* (internal) DNN model, also referred as analysis DNN is a light-weighted and generic DNN model, used for DNN platform-aware characteristics estimation, and DNN executable code generation.

For specified dnn representations we support following internal and third-party converters.
* (third-party) ONNX DNN -> keras DNN through onnx2keras library
* (third-party) keras DNN -> ONNX DNN through keras2onnx library
* (internal) keras DNN -> analysis DNN
* (internal) ONNX DNN -> analysis DNN

## DNN to system-level representations
For efficient partitioning, mapping and scheduling, DNN has to be represented as a system-level model. We currently support following system-level models:
* (internal) TaskGraph: simple graph, used for high-throughput DNN partitioning and mapping.
* (internal) CSDF model: Cyclo-Static DataFlow model, used for low-memory DNN partitioning and scheduling.

We support conversion of analysis DNN to all system-level representations.



