#ifndef COLSERVE_CONTROLLING_H
#define COLSERVE_CONTROLLING_H

#include <iostream>
#include <cstdint>

namespace colserve {
namespace ctrl {

struct CtrlMsgEntry {
  uint64_t id;
  int event;
  int value;
};

enum class CtrlEvent {
    // status event
    kTrainStart,
    kTrainEnd,
    kInterruptTrainDone,
    kResumeTrainDone,
    kColocateAdjustL1Done,
    kColocateAdjustL2Done,
    
    kReportBatchSize,

    // cmd event: switch mode
    kInterruptTrain,
    kResumeTrain,
    // cmd event: colocate mode
    kColocateAdjustL1,
    kColocateAdjustL2,
    kInferExit, // train adjust back

    kInferenceWorkloadDone,

    kNumEvent,
  };

}
}

#endif