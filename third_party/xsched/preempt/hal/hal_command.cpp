#include "utils/xassert.h"
#include "preempt/xqueue/xqueue.h"
#include "preempt/hal/hal_command.h"

using namespace xsched::preempt;

void HalCommand::BeforeHalSubmit()
{
    // nothing to do, just for override
}

void HalCommand::Synchronize()
{
    if (xqueue_) {
        xqueue_->Synchronize(
            std::static_pointer_cast<HalCommand>(shared_from_this()));
        return;
    }

    XASSERT(this->HalSynchronizable(),
            "The HalCommand being synchronized should either be "
            "HalSynchronizable or have been submitted to an XQueue.");
    this->HalSynchronize();
}
