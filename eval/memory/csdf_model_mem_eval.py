from util import mega


def eval_csdf_buffers_memory_mb(csdf_buffers, token_size=4):
    """
    Evaluate memory (in MegaBytes) occupied by CSDFG buffers
    :param csdf_buffers: list of CSDFG buffers
    :param token_size: size of one token in bytes
    :return: (float) memory (in MegaBytes) occipied by CSDFG buffers
    """
    mem = 0.0
    for buf in csdf_buffers:
        buffer_mb = (buf.size * token_size) / float(mega())
        mem += buffer_mb
    return mem

