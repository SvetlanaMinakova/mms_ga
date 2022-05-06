# SCHEDULING

Scheduling determines how execution of an application is spread in time. For DNNs, we support following types of scheduling:

* **SEQUENTIAL**: layers of the DNN are executed one-by-one
* **PIPELINE**: layers of the DNN are executed in a pipelined manner, as soon as possible
* **CUSTOM**: layers of the DNN are executed in a custom, explicitly specified execution order