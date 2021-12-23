"""
This module parses a .json description of an embedded platform and creates corresponding object of 
EmbeddedPlatform class, defined in platform_spec.platform_spec.py
"""
from models.edge_platform.aloha_eval_platform_spec.platform_spec import EmbeddedPlatform, Processor, Memory, \
    DataTransferChannel, SharedResource
from models.edge_platform.aloha_eval_platform_spec.comp_model import DimsUnrolling, Limit
from converters.json_converters.json_util import extract_or_default
from fileworkers.json_fw import read_json


def read_and_parse_json_platform(json_platform_path):
    json_platform = read_json(json_platform_path)
    embedded_platform = parse_json_platform(json_platform)
    return embedded_platform


def parse_json_platform(json_platform):
    name = extract_or_default(json_platform, "name", "platform")
    embedded_platform = EmbeddedPlatform(name)
    embedded_platform.data_bytes = extract_or_default(json_platform, "data_bytes", 4)
    embedded_platform.total_bandwidth = extract_or_default(json_platform, "total_bandwidth", 0)
    embedded_platform.memory_partitions = extract_or_default(json_platform, "memory_partitions", 1)
    embedded_platform.idle_power = extract_or_default(json_platform, "idle_power", 0)

    # memories:
    str_memories = extract_or_default(json_platform, "memories", [])
    for str_memory in str_memories:
        memory = parse_memory(str_memory)
        embedded_platform.memories.append(memory)

    # channels:
    str_channels = extract_or_default(json_platform, "channels", [])
    for str_channel in str_channels:
        channel = parse_channel(str_channel)
        embedded_platform.channels.append(channel)

    # additional shared resources:
    str_shared_resources = extract_or_default(json_platform, "shared_resources", [])
    for str_shared_resource in str_shared_resources:
        resource = parse_shared_resource(str_shared_resource)
        embedded_platform.shared_resources.append(resource)

    # processors:
    str_processors = extract_or_default(json_platform, "processors", [])
    for str_processor in str_processors:
        processor = parse_processor(str_processor, embedded_platform)
        embedded_platform.processors.append(processor)

    return embedded_platform


def parse_processor(json_processor, platform):
    name = extract_or_default(json_processor, "name", "processor")
    type = extract_or_default(json_processor, "type", "CPU")
    processor = Processor(name, type)
    # set common properties
    processor.frequency = extract_or_default(json_processor, "frequency", 0)
    processor.max_perf = extract_or_default(json_processor, "max_perf", 0)
    processor.max_power = extract_or_default(json_processor, "max_power", 0)
    processor.parallel_comp_matrix = extract_or_default(json_processor, "parallel_comp_matrix", [1])
    processor.data_on_memory_mapping = extract_or_default(json_processor, "data_on_memory_mapping", {})
    processor.unrolling = parse_comp_model(json_processor, platform)
    return processor


def parse_comp_model(json_processor, platform):
    str_comp_model = extract_or_default(json_processor, "unrolling", "")
    comp_model = DimsUnrolling()

    if str_comp_model == "":
        return comp_model

    comp_unrolling = extract_or_default(str_comp_model, "comp_unrolling", [])
    comp_model.comp_unrolling = comp_unrolling

    loops_order = extract_or_default(str_comp_model, "loops_order", [])
    comp_model.loops_order = loops_order

    str_resource_limits = extract_or_default(str_comp_model, "limits", [])
    for str_resource_limit in str_resource_limits:
        limit = parse_resource_limit(str_resource_limit, platform)
        comp_model.limits.append(limit)

    return comp_model


def parse_resource_limit(json_resource_limit, platform):
    resource_user_name = extract_or_default(json_resource_limit, "resource_user_name", "")
    str_resource = extract_or_default(json_resource_limit, "resource", None)
    resource = None
    if str_resource is not None:
        resource_name = extract_or_default(str_resource, "name", "")
        resource = platform.get_memory(resource_name)
        if resource is None:
            resource = platform.get_resource(resource_name)

    dims = extract_or_default(json_resource_limit, "dims", [])
    split_dims = extract_or_default(json_resource_limit, "split_dims", [])
    load_dim = extract_or_default(json_resource_limit, "load_dim", "")
    load = extract_or_default(json_resource_limit, "load", True)

    limit = Limit(resource_user_name, resource, dims, split_dims, load_dim, load)
    return limit


def parse_memory(json_memory):
    name = extract_or_default(json_memory, "name", "memory")
    size = extract_or_default(json_memory, "size", 0)
    bank_size = extract_or_default(json_memory, "bank_size", 0)
    memory = Memory(name, size, bank_size)
    return memory


def parse_shared_resource(json_resource):
    name = extract_or_default(json_resource, "name", "resource")
    size = extract_or_default(json_resource, "size", 0)
    resource = SharedResource(name, size)
    return resource


def parse_channel(json_channel):
    src = extract_or_default(json_channel, "src", "none")
    dst = extract_or_default(json_channel, "dst", "none")
    bandwidth = extract_or_default(json_channel, "bandwidth", 0)
    channel = DataTransferChannel(src, dst, bandwidth)
    return channel
