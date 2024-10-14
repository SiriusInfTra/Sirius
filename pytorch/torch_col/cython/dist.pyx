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
                  int global_batch_size,
                  bool lazy_distributing,
                  string distribute_policy)
        @staticmethod
        pair[batch_range_vec_t, bool] GetBatch(int batch_size)
        @staticmethod
        int QueryNextBatchSize(int batch_size)
        @staticmethod
        void FinishBatch(batch_range_vec_t batch_range_vec,
                         bool end_of_global_batch)
        @staticmethod
        void AbortBatch(batch_range_vec_t batch_range_vec, 
                        bool end_of_global_batch)
        @staticmethod
        bool VoteFinishLastMicroBatch()
        @staticmethod
        void VoteAbortLastMicroBatch()
        @staticmethod
        void ResetLastMicroBatchFinishVote()
        @staticmethod
        void DistributeBatch(bool check_num_unproced_samples,
                             bool distribute_to_all,
                             bool at_global_batch_begin)
        @staticmethod
        int NextEpoch()
        @staticmethod
        void NextGlobalBatch()
        @staticmethod
        int GetGlobalBatchSize()
        @staticmethod
        int GetNumGlobalBatchPerEpoch()
        @staticmethod
        int GetNumProcedGlobalBatch()


class _DynamicBatchDistirbutor:
    _lazy_distributing: bool = False

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
    def abort_batch(batch_range_vec: List[Tuple[int, int]], 
                    end_of_global_batch: bool):
        DynamicBatchDistirbutor.AbortBatch(batch_range_vec, 
                                           end_of_global_batch)

    @staticmethod
    def vote_finish_last_micro_batch():
        return DynamicBatchDistirbutor.VoteFinishLastMicroBatch()

    @staticmethod
    def vote_abort_last_micro_batch():
        DynamicBatchDistirbutor.VoteAbortLastMicroBatch()

    @staticmethod
    def reset_last_micro_batch_finish_vote():
        DynamicBatchDistirbutor.ResetLastMicroBatchFinishVote()

    @staticmethod
    def distribute_batch(check_num_unproced_samples: bool,
                         distribute_to_all: bool,
                         at_global_batch_begin: bool):
        DynamicBatchDistirbutor.DistributeBatch(
            check_num_unproced_samples,
            distribute_to_all,
            at_global_batch_begin)

    @staticmethod
    def next_epoch():
        return DynamicBatchDistirbutor.NextEpoch()

    @staticmethod
    def next_global_batch():
        DynamicBatchDistirbutor.NextGlobalBatch()

    @staticmethod
    def get_global_batch_size() -> int:
        return DynamicBatchDistirbutor.GetGlobalBatchSize()

    @staticmethod
    def get_num_global_batch_per_epoch() -> int:
        return DynamicBatchDistirbutor.GetNumGlobalBatchPerEpoch()

    @staticmethod
    def get_num_proced_global_batch() -> int:
        return DynamicBatchDistirbutor.GetNumProcedGlobalBatch()
        

def init_dynamic_batch_distributor(dataset_size: int, 
                                   input_batch_size: int,
                                   global_batch_size: int,
                                   lazy_distributing: bool,
                                   distribute_policy: str):
    _DynamicBatchDistirbutor._lazy_distributing = lazy_distributing
    DynamicBatchDistirbutor.Init(dataset_size, 
                                 input_batch_size,
                                 global_batch_size,
                                 lazy_distributing,
                                 distribute_policy)


####################
# MARK: Perf Model #
####################

cdef extern from "<torch_col/csrc/perf_model.h>" namespace "torch_col":
    cdef cppclass PerfModel:
        @staticmethod
        void Init()
        @staticmethod
        void RecordThpt(int batch_size, double batch_time_ms)
        @staticmethod
        double GetThpt(int batch_size)
        @staticmethod
        vector[double] GetThptVec(const vector[int] &batch_sizes)


def init_train_performance_model():
    PerfModel.Init()


class _PerfModel:
    @staticmethod
    def record_thpt(batch_size: int, batch_time_ms: float):
        PerfModel.RecordThpt(batch_size, batch_time_ms)

    @staticmethod
    def get_thpt(batch_size: int) -> float:
        return PerfModel.GetThpt(batch_size)

    @staticmethod
    def get_thpt_vec(batch_sizes: List[int]) -> List[float]:
        return PerfModel.GetThptVec(batch_sizes)

