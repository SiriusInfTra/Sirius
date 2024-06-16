# cython: c_string_type=unicode, c_string_encoding=utf8

cdef extern from "<common/sm_partition.h>" namespace "colserve":
    cdef cppclass SMPartitioner:
        @staticmethod
        void Init(int device, int cleanup, int observe)

        @staticmethod
        int GetInferRequiredTpcNum()

        @staticmethod
        int GetTrainAvailTpcNum()

        @staticmethod
        unsigned long long GetTrainAvailTpcMask()


def monitor_sm_partition(interval: float):
    import sys, time

    SMPartitioner.Init(0, 0, 1)

    fmt = "Infer TPC Num: {}, Train TPC Num: {}, Train Avail TPC Mask: {}"
    while True:
        print(fmt.format(
            SMPartitioner.GetInferRequiredTpcNum(),
            SMPartitioner.GetTrainAvailTpcNum(),
            hex(SMPartitioner.GetTrainAvailTpcMask())
        ))
        time.sleep(interval)