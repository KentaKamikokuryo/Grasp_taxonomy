from Classes.Info import IDataInfo, DataInfo1, DataInfo2, DataInfo3, DataInfo4, DataInfo5
from typing import List

class DataInfoFactory():

    data_strategy: IDataInfo
    take_names: List[str]

    def get_Datainfo(self, name: str) -> IDataInfo:

        if name == "experiment_1":

            data_strategy = DataInfo1()

        elif name == "experiment_2":

            data_strategy = DataInfo2()

        elif name == "experiment_3":

            data_strategy = DataInfo3()

        elif name == "experiment_4":

            data_strategy = DataInfo4()

        elif name == "experiment_5":

            data_strategy = DataInfo5()

        return data_strategy


    def get_take_names(self, name: str) -> List[str]:

        if name == "experiment_1":

            take_names = ["take1", "take1-1", "take1-1_001",  # ~2
                          "take-A", "take-A-2", "take-A-3", "take-A-4", "take-A-5", "take-A-6", "take-A-7", "take-A-8",  # ~10
                          "take-B-1-1", "take-B-1-2", "take-B-1-3", "take-B-1-4", "take-B-1-5",  # ~15
                          "take-B-2-1", "take-B-2-1-1", "take-B-2-2", "take-B-2-3", "take-B-2-4", "take-B-2-5",  # ~21
                          "take-B-3-1", "take-B-3-2", "take-B-3-3",  # ~24
                          "takeB-4-1-1", "takeB-4-2", "takeB4-3", "takeB4-4"]  # ~28

        elif name == "experiment_2":

            take_names = ["IST-M-2", "IST-T-1", "IST-T-2",  # ~2
                          "PRG-M-1", "PRG-M-2", "PRG-T-1", "PRG-T-2",  # ~6
                          "WRK-M-A-1", "WRK-M-A-2", "WRK-M-A-3", "WRK-M-A-4", "WRK-M-A-5",  # ~11
                          "WRK-M-A-6", "WRK-M-A-7", "WRK-M-A-8", "WRK-M-A-9", "WRK-M-A-10",  # ~16
                          "WRK-M-A-11", "WRK-M-A-12", "WRK-M-A-13",  # ~19
                          "WRK-M-B-1", "WRK-M-B-2", "WRK-M-B-3", "WRK-M-B-4", "WRK-M-B-5",  # ~24
                          "WRK-M-B-6", "WRK-M-B-7", "WRK-M-B-8", "WRK-M-B-9", "WRK-M-B-10",  # ~29
                          "WRK-M-B-11",  # ~30
                          "WRK-T-A-1", "WRK-T-A-2", "WRK-T-A-3", "WRK-T-A-4", "WRK-T-A-5",  # ~35
                          "WRK-T-A-6", "WRK-T-A-7", "WRK-T-A-8", "WRK-T-A-9", "WRK-T-A-10",  # ~40
                          "WRK-T-A-11", "WRK-T-A-12", "WRK-T-A-13", "WRK-T-A-14", "WRK-T-A-15",  # ~45
                          "WRK-T-B-1", "WRK-T-B-2", "WRK-T-B-3", "WRK-T-B-4", "WRK-T-B-5",  # ~50
                          "WRK-T-B-6", "WRK-T-B-7", "WRK-T-B-8", "WRK-T-B-9", "WRK-T-B-10",  # ~55
                          "WRK-T-B-11", "WRK-T-B-12", "WRK-T-B-13"]  # ~58

        elif name == "experiment_3":

            take_names = ["e_junbi", "e_p01", "e_p02", "e_p03", "e_p04", "e_p05",  # ~5
                          "e_p06", "e_p07", "e_p08", "e_p09", "e_p10"]  # ~10

        elif name == "experiment_4":

            take_names = ["test"]

        elif name == "experiment_5":

            take_names = ["test"]

        return take_names


class RetrievingSystemFactory():
    pass


