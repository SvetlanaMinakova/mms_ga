from models.app_model.MMSDNNInferenceModel import MMSDNNInferenceModel
from models.dnn_model.dnn import DNN
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor
from converters.json_converters.dnn_to_json import DNNJSONNestedClassVisitor
from DSE.scheduling.mms_dnn_inf_model_schedule import MMSDNNInfModelSchedule
from models.data_buffers import DataBuffer


def mms_app_to_json(app: MMSDNNInferenceModel, filepath: str):
    """
    Convert MMS final application model into a JSON File
    :param app: MMS final application model
    :param filepath: path to target .json file
    """
    visitor = MMSDNNInfModelJSONNestedClassVisitor(app, filepath)
    visitor.run()


class MMSDNNInfModelJSONNestedClassVisitor(JSONNestedClassVisitor):
    """
    JSON visitor. Used to save custom classes as a .json file.
    Attributes:
        :param app: MMS final application model
        filepath: path to target .json file
    """
    def __init__(self, app: MMSDNNInferenceModel, filepath):
        super(MMSDNNInfModelJSONNestedClassVisitor, self).__init__(app, filepath)
        # represent object of a dnn class as dictionary of fields
        self.app_as_dict = self._root.__dict__
        # number of fields in the dnn class object represented as a dictionary
        self.app_fields_num = len(self.app_as_dict.items())
        self.cur_dnn_field_id = 0

    def visit_object(self, obj):
        """
        Recursive object visitor
        :param obj: object to visit
        """
        """
        special-type objects:
                 partitions_per_dnn: [[DNN]],
                 schedule: MMSDNNInfModelSchedule,
                 data_buffers: [DataBuffer]
        """

        if isinstance(obj, DNN):
            self._visit_dnn(obj)
            return

        if isinstance(obj, MMSDNNInfModelSchedule):
            self._visit_schedule(obj)
            return

        if isinstance(obj, DataBuffer):
            self._visit_data_buffer(obj)
            return

        super().visit_object(obj)

    def _visit_dnn(self, dnn: DNN):
        dnn_visitor = DNNJSONNestedClassVisitor(dnn, None)
        dnn_visitor.prefix = self._prefix
        dnn_visitor.prefix_inc()
        dnn_visitor.run_in_file(self._file)
        dnn_visitor.prefix_dec()

    def _visit_schedule(self, schedule: MMSDNNInfModelSchedule):
        schedule_visitor = JSONNestedClassVisitor(schedule, None)
        schedule_visitor._prefix = self._prefix
        schedule_visitor.prefix_inc()
        schedule_visitor.run_in_file(self._file)
        schedule_visitor.prefix_dec()

    def _visit_data_buffer(self, data_buffer: DataBuffer):
        db_visitor = JSONNestedClassVisitor(data_buffer, None)
        db_visitor._prefix = self._prefix
        db_visitor.prefix_inc()
        db_visitor.run_in_file(self._file)
        db_visitor.prefix_dec()

