from Classes.Data import DataFactory, Creation
from Classes.Factories import DataInfoFactory


class Manager():

    def __init__(self, experiment_name: str):

        self.experiment_name = experiment_name

        fac = DataInfoFactory()
        self.dataInfo = fac.get_Datainfo(name=self.experiment_name)
        self.take_names = fac.get_take_names(name=experiment_name)

        self.name_file = "original"

    def generate_original_data(self, name):

        self.dataInfo.set_data_info(name_take=name)
        data_dict = self.dataInfo.get_data_dict(name_file=self.name_file)
        data_manager = DataFactory(data_dict=data_dict, data_info=self.dataInfo)
        original_df = data_manager.create()

        return original_df, data_dict

    def all_creation(self):

        for name in self.take_names:

            original_df, data_dict = self.generate_original_data(name=name)
            creation = Creation(df_original=original_df, data_dict=data_dict)
            creation.get_CSV()


experiment_names = ["experiment_1", "experiment_2", "experiment_3"]
experiment_name = experiment_names[2]

manager = Manager(experiment_name=experiment_name)
manager.all_creation()