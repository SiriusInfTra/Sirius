# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
from libcpp.string cimport string


######################
#  MARK: Dist Train  #
######################

cdef extern from "<torch_col/csrc/dist_train_sync.h>" namespace "torch_col":
    cdef cppclass DistTrainSync:
        @staticmethod
        void WaitBarrier()
        @staticmethod
        void Send(int dst, const string &msg)
        @staticmethod
        string Recv(int src)


def wait_barrier():
    DistTrainSync.WaitBarrier()


def send_msg(dst_rank: int, msg: str):
    DistTrainSync.Send(dst_rank, msg)


def recv_msg(src_rank: int):
    return DistTrainSync.Recv(src_rank)
