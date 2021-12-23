from models.csdf_model.csdf import CSDFGraph, CSDFActor, CSDFFIFOChannel
from models.dnn_model.dnn import DNN, layer_has_null_or_empty_pads

"""
CNN-to-CSDF conversion
"""


def dnn_to_csfd_one_to_one(dnn: DNN, fuse_self_loops=True):
    """
    One-to-one DNN-to-CSDF conversion
    :param dnn: dnn
    :param fuse_self_loops: (flag) if True, every self-loop channel which stores reused data in CSDFG
    with respective  "main" data source channel which produces the overlapping data
    :return: CSDF model, functionally equivalent to the DNN model
    """
    def __create_actors():
        actors = []
        for layer in dnn.get_layers():
            actor = __create_actor(layer)
            csdf.add_actor(actor)
        return actors

    def __create_actor(layer):
        actor_id = layer.id
        exec_seq = [layer.subop for i in range(layer.phases)]
        actor = CSDFActor("a" + str(actor_id), exec_seq)
        actor.time_per_phase = [layer.time_eval/max(float(layer.phases), 1.0) for phase in range(layer.phases)]
        return actor

    def __create_data_transfer_channels():
        for edge in dnn.get_connections():
            src_id = edge.src.id
            dst_id = edge.dst.id
            # no communication happens between any op-> built-in op
            if edge.dst.built_in:
                prod_seq = [0 for phase in range(edge.src.phases)]
                cons_seq = [0 for phase in range(edge.dst.phases)]
            else:
                prod_seq = __compute_production_sequence(edge)
                cons_seq = __compute_consumption_sequence(edge)

            if edge.dst.subop == "mul":
                to_consume = sum(cons_seq)
                to_produce = sum(prod_seq)
                if to_consume != to_produce:
                    # broadcast
                    # print("BROADCAST", "to prod:", to_produce, "to", to_consume)
                    # print("SRC PHASES: ", edge.src, "DST:", edge.dst)
                    # print("SRC PHASES: ", edge.src.phases, "DST PHASES:", edge.dst.phases)
                    cons_seq = [to_produce]
                    for i in range(1, edge.dst.phases):
                        cons_seq.append(0)

            csdf.connect_actors_by_ids(src_id, dst_id, prod_seq, cons_seq)

    def __compute_production_sequence(edge):
        prod_seq = []
        src_layer = edge.src
        phase_oh = src_layer.oh
        if src_layer.phases > 1:
            phase_oh = 1 # max(layer.oh/layer.phases, 1)
        for phase in range(src_layer.phases):
            phase_rate = int(src_layer.ow * phase_oh * src_layer.ofm)
            prod_seq.append(phase_rate)
        return prod_seq

    def __compute_consumption_sequence(edge):
        dst_layer = edge.dst
        # process multi-input_examples
        layer_inputs = dnn.get_layer_input_connections(dst_layer)
        if len(layer_inputs) > 1:
            cons_seq = __compute_multi_input_cons_seq(edge)
            return cons_seq

        # process single-input_examples
        cons_seq = []
        total_lines = dst_layer.ih
        lines_consumed = 0
        for phase_id in range(dst_layer.phases):
            phase_ih = dst_layer.ih
            if dst_layer.phases > 1:
                phase_ih = dst_layer.fs
                if __layer_reuses_inp_data(dst_layer) and phase_id > 0: # 0 < phase_id < (layer.phases-1)
                    phase_ih = dst_layer.stride
                # TODO: take into account
                """
                # pads < 0
                if layer_has_null_or_empty_pads(layer) and layer.stride < layer.fs and layer.oh < layer.ih:
                    h_crop = (layer.ih-1)*layer.stride - layer.ih + layer.fs
                    if phase_id == 0:
                        phase_ih += int(h_crop/2)
                    if phase_id == layer.phases-1:
                        phase_ih -= int(h_crop/2)
                
                # pads > 0
                if not layer_has_null_or_empty_pads(layer):
                    h_extension = layer.pads[1] + layer.pads[3]
                    if phase_id == 0:
                        phase_ih -= layer.pads[1]
                    if phase_id == layer.phases-1:
                        phase_ih -= layer.pads[3]
                """
            # adjust phases in case of padding etc. (over-consumption)
            if (lines_consumed + phase_ih) > total_lines:
                phase_ih = max(total_lines - lines_consumed, 0)

            # last phase
            # adjust phases in case of padding etc. (under-consumption)
            if phase_id == dst_layer.phases - 1:
                if (lines_consumed + phase_ih) < total_lines:
                    phase_ih = max(total_lines - lines_consumed, 0)

            lines_consumed += phase_ih
            phase_rate = int(dst_layer.iw * phase_ih * dst_layer.ifm)
            cons_seq.append(phase_rate)
        return cons_seq

    def __compute_multi_input_cons_seq(edge):
        cons_seq = []
        src_layer = edge.src
        dst_layer = edge.dst

        for phase_id in range(dst_layer.phases):
            phase_ih = max(int(src_layer.oh/dst_layer.phases), 1)
            phase_rate = int(src_layer.ow * phase_ih * src_layer.ofm)
            cons_seq.append(phase_rate)

        return cons_seq

    def __create_self_loops(fused_self_loops: bool):
        """
        Create self-loops that store data, overlapping between
        subsequent exec. steps
        """
        for layer in dnn.get_layers():
            # data overlaps in layer
            if __layer_reuses_inp_data(layer):
                actor_id = layer.id
                prod_seq = __self_loop_prod_seq(layer)
                cons_seq = __self_loop_cons_seq(layer)

                # create self-loop as a separate channel
                if not fused_self_loops:
                    csdf.connect_actors_by_ids(actor_id, actor_id, prod_seq, cons_seq)
                # reuse main data source channel to store overlapping data
                # i.e. fuse self-loop channel with main data source channel
                else:
                    pass
                    # __create_fused_self_loop(actor_id, prod_seq, cons_seq)

    def __create_fused_self_loop(actor_id, prod_seq, cons_seq):
        """
        Create self-loop channel, fused into the main data source channel
        :param actor_id: id of actor with self-loop
        :param prod_seq: production sequence of self-loop
        :param cons_seq: consumption sequence of self-loop
        """
        actor = csdf.get_actors()[actor_id]
        input_channels = csdf.get_input_channels(actor)

        # actor with self-loop should only have one input
        if len(input_channels) != 1:
            error_reason = "Actor a" + str(actor_id) + "has " + str(len(input_channels)) + \
                           " input channels, while 1 expected."
            raise FusedSelfLoopCreationError(actor_id, error_reason)

        main_data_source_channel = input_channels[0]
        main_prod_seq = main_data_source_channel.prod_seq
        main_cons_seq = main_data_source_channel.cons_seq

        # length of production-consumption sequences should merge
        if len(main_prod_seq) != len(prod_seq) or len(main_cons_seq) != len(cons_seq):
            # actor with self-loop should only have one input
            if len(input_channels) != 1:
                error_reason = "Production-consumption sequences length mismatch. " + \
                               "Main data source prod-len:" + str(len(main_prod_seq)) + \
                               ", cons-len: " + str(len(main_cons_seq)) + \
                               "Self-loop prod-len:" + str(len(prod_seq)) + \
                               ", cons-len: " + str(len(cons_seq))
                raise FusedSelfLoopCreationError(actor_id, error_reason)

        # update main data source production rate
        # update main data source consumption rate

    def __layer_reuses_inp_data(layer):
        if layer.op not in ["conv", "pool"]:
            return False
        if layer.phases == 1:
            return False
        if layer.stride >= layer.fs:
            return False
        return True

    def __self_loop_prod_seq(layer):
        prod_seq = []
        for phase in range(layer.phases-1):
            reuse_h = layer.fs - layer.stride
            reuse_rate = int(layer.iw * reuse_h * layer.ifm)
            prod_seq.append(reuse_rate)
        # reuse is not needed at the last phase
        # thus, rate at the last phase is 0
        prod_seq.append(0)
        return prod_seq

    def __self_loop_cons_seq(layer):
        # nothing to reuse at the first phase
        # thus, rate at the first phase is 0
        cons_seq = [0]
        for phase in range(layer.phases-1):
            reuse_h = layer.fs - layer.stride
            reuse_rate = int(layer.iw * reuse_h * layer.ifm)
            cons_seq.append(reuse_rate)
        return cons_seq

    # main script
    csdf = CSDFGraph(dnn.name)
    __create_actors()
    __create_data_transfer_channels()
    __create_self_loops(fuse_self_loops)

    return csdf


class CSDFCreationError(Exception):
    """ CSDF creation error"""

    def __init__(self, message):
        self.message = "CSDF CREATION ERROR: " + message
        super().__init__(self.message)


class FusedSelfLoopCreationError(CSDFCreationError):
    """ CSDF creation error"""
    def __init__(self, actor_id: int, reason: str):
        self_loop_description = "a" + str(actor_id) + "_a" + str(actor_id)
        error_msg = "Fused self-loop " + self_loop_description + " cannot be created. " + reason
        super().__init__(error_msg)

