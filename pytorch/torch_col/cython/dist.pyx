# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.optional cimport optional, make_optional
from libcpp.pair cimport pair
from libcpp cimport bool

from typing import List, Tuple, Optional

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


#########################
#  MARK: Dyanmic Batch  #
#########################

cdef extern from "<torch_col/csrc/dynamic_batch.h>" namespace "torch_col":
    cdef cppclass DynamicBatchDistirbutor:
        ctypedef pair[int, int] batch_range_t
        ctypedef vector[pair[int, int]] batch_range_vec_t

        @staticmethod
        void Init(int batch_size, int global_batch_size)
        @staticmethod
        pair[batch_range_vec_t, bool] GetBatch(int batch_size)
        @staticmethod
        int QueryNextBatchSize()
        @staticmethod
        void FinishBatch(batch_range_vec_t batch_range_vec)
        @staticmethod
        void AbortBatch(batch_range_vec_t batch_range_vec)
        @staticmethod
        void DistributeBatch(bool check_num_unproced_samples)
        @staticmethod
        void NextEpoch()
        @staticmethod
        void NextGlobalBatch()


class _DynamicBatchDistirbutor:
    @staticmethod
    def get_batch(batch_size: int) -> Tuple[List[Tuple[int, int]], bool]:
        return DynamicBatchDistirbutor.GetBatch(batch_size)

    @staticmethod
    def query_next_batch_size() -> int:
        return DynamicBatchDistirbutor.QueryNextBatchSize()

    @staticmethod
    def finish_batch(batch_range_vec: List[Tuple[int, int]]):
        DynamicBatchDistirbutor.FinishBatch(batch_range_vec)

    @staticmethod
    def abort_batch(batch_range_vec: List[Tuple[int, int]]):
        DynamicBatchDistirbutor.AbortBatch(batch_range_vec)

    @staticmethod
    def distribute_batch(check_num_unproced_samples: bool):
        DynamicBatchDistirbutor.DistributeBatch(check_num_unproced_samples)

    @staticmethod
    def next_epoch():
        DynamicBatchDistirbutor.NextEpoch()

    @staticmethod
    def next_global_batch():
        DynamicBatchDistirbutor.NextGlobalBatch()


def init_dynamic_batch_distributor(batch_size: int, 
                                   global_batch_size: int):
    DynamicBatchDistirbutor.Init(batch_size, 
                                 global_batch_size)



