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
        void Init(int dataset_size, 
                  int input_batch_size, 
                  int global_batch_size)
        @staticmethod
        pair[batch_range_vec_t, bool] GetBatch(int batch_size)
        @staticmethod
        int QueryNextBatchSize(int batch_size)
        @staticmethod
        void FinishBatch(batch_range_vec_t batch_range_vec,
                         bool end_of_global_batch)
        @staticmethod
        void AbortBatch(batch_range_vec_t batch_range_vec)
        @staticmethod
        void DistributeBatch(bool check_num_unproced_samples)
        @staticmethod
        void NextEpoch()
        @staticmethod
        void NextGlobalBatch()
        @staticmethod
        int GetGlobalBatchSize()
        @staticmethod
        int GetNumGlobalBatchPerEpoch()


class _DynamicBatchDistirbutor:
    @staticmethod
    def get_batch(batch_size: int) -> Tuple[List[Tuple[int, int]], bool]:
        return DynamicBatchDistirbutor.GetBatch(batch_size)

    @staticmethod
    def query_next_batch_size(batch_size: int) -> int:
        return DynamicBatchDistirbutor.QueryNextBatchSize(batch_size)

    @staticmethod
    def finish_batch(batch_range_vec: List[Tuple[int, int]],
                     end_of_global_batch: bool):
        DynamicBatchDistirbutor.FinishBatch(batch_range_vec, 
                                            end_of_global_batch)

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

    @staticmethod
    def get_global_batch_size() -> int:
        return DynamicBatchDistirbutor.GetGlobalBatchSize()

    @staticmethod
    def get_num_global_batch_per_epoch() -> int:
        return DynamicBatchDistirbutor.GetNumGlobalBatchPerEpoch()


def init_dynamic_batch_distributor(dataset_size: int, 
                                   input_batch_size: int,
                                   global_batch_size: int):
    DynamicBatchDistirbutor.Init(dataset_size, 
                                 input_batch_size,
                                 global_batch_size)



