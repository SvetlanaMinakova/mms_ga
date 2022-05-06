from models.data_buffers import DataBuffer, DNNDataBuffer, CSDFGDataBuffer
from models.dnn_model.dnn import DNN
import traceback
from util import print_to_stderr

"""
Transforms buffer of one type into buffer of another type
"""


def csdf_buf_to_generic_buf(csdf_buf: CSDFGDataBuffer) -> DataBuffer:
    """
    Transforms a CSDF buffer into a more generic DataBuffer by omitting CSDF-specific information
    :param csdf_buf: CSDF buffer
    :return: DataBuffer, obtained from the CSDF buffer
    """
    data_buf = DataBuffer(csdf_buf.name, csdf_buf.size)
    csdf_buf_specifies_csdf_models_per_channel = len(csdf_buf.channels) == len(csdf_buf.csdf_model_name_per_channel)
    csdf_channel_id = 0
    for csdf_channel in csdf_buf.channels:
        src_name = csdf_channel.src.name
        dst_name = csdf_channel.dst.name
        csdf_model_name = ""
        if csdf_buf_specifies_csdf_models_per_channel:
            csdf_model_name = csdf_buf.csdf_model_name_per_channel[csdf_channel_id]
        user_desc = (csdf_model_name, src_name, dst_name)
        data_buf.assign(user_desc)
        csdf_channel_id += 1
    return data_buf


def dnn_buf_to_generic_buf(dnn_buf: DNNDataBuffer) -> DataBuffer:
    """
    Transforms a CSDF buffer into a more generic DataBuffer by omitting CSDF-specific information
    :param dnn_buf: CSDF buffer
    :return: DataBuffer, obtained from the CSDF buffer
    """
    data_buf = DataBuffer(dnn_buf.name, dnn_buf.size)
    for dnn_name in dnn_buf.connections_per_dnn.keys():
        for dnn_connection in dnn_buf.connections_per_dnn[dnn_name]:
            src_name = dnn_connection.src.name
            dst_name = dnn_connection.dst.name
            user_desc = (dnn_name, src_name, dst_name)
            data_buf.assign(user_desc)
    return data_buf


def csdf_reuse_buf_to_generic_dnn_buf(csdf_reuse_buffers: [CSDFGDataBuffer], dnns: [DNN]) -> [DataBuffer]:
    """
    Convert CSDF reuse buffers, generated for a set of DNNs into a functionally equivalent set of
     DataBuffers, used by the DNNs
    :param csdf_reuse_buffers: CSDF reuse buffers
    :param dnns: dnns, using the CSDF reuse buffers
    :return: DataBuffers, used by the DNNs
    """
    generic_csdf_buffers = [csdf_buf_to_generic_buf(buffer) for buffer in csdf_reuse_buffers]
    generic_dnn_buffers = [generic_csdf_buf_to_generic_dnn_buf(buffer, dnns) for buffer in generic_csdf_buffers]
    return generic_dnn_buffers


def generic_csdf_buf_to_generic_dnn_buf(generic_csdf_buffer: DataBuffer, dnns: [DNN])->DataBuffer:
    """
    Convert DataBuffer describing a CSDF buffer into a DataBuffer describing a DNN buffer
        for functionally equivalent CSDF and DNN
    :param generic_csdf_buffer:  generic csdf buffer, generated for a dnn model
    :param dnns: dnns, that (potentially) use the generic buffer
    """
    error_prefix = "CSDF-to-DNN (generic) buffers conversion error: "
    generic_dnn_buffer = DataBuffer(generic_csdf_buffer.name, generic_csdf_buffer.size)
    for csdf_buf_user in generic_csdf_buffer.users:
        dnn_name, src_actor_name, dst_actor_name = csdf_buf_user
        dnn = get_dnn_by_name(dnns, dnn_name)
        if dnn is None:
            print_to_stderr(error_prefix + "dnn model " + dnn_name + " not found!")
            raise Exception("CSDF-to-DNN (generic) buffers conversion error")
        src_layer_name = actor_name_to_layer_name(dnn, src_actor_name)
        if src_layer_name is None:
            raise Exception(error_prefix + "null src actor name!")
        dst_layer_name = actor_name_to_layer_name(dnn, dst_actor_name)
        if dst_layer_name is None:
            raise Exception(error_prefix + "null dst actor name!")
        dnn_buf_user = (dnn_name, src_layer_name, dst_layer_name)
        generic_dnn_buffer.assign(dnn_buf_user)
    return generic_dnn_buffer


def get_dnn_by_name(dnns: [DNN], name: str):
    for dnn in dnns:
        if dnn.name == name:
            return dnn
    return None


def actor_name_to_layer_name(dnn: DNN, actor_name: str):
    try:
        actor_id = int(actor_name.replace("a", ""))
        layer_id = actor_id
        layer_name = dnn.get_layers()[layer_id].name
        return layer_name
    except Exception as e:
        print("Actor name to layer name conversion error: ")
        traceback.print_tb(e.__traceback__)
        return None

