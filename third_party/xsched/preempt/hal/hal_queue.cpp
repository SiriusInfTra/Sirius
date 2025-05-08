#include "utils/common.h"
#include "preempt/hal/hal_queue.h"

using namespace xsched::preempt;

void HalQueue::OnInitialize()
{
    // nothing to do, just for overriding
}

void HalQueue::OnSubmit(std::shared_ptr<HalCommand>)
{
    // nothing to do, just for overriding
}

void HalQueue::Deactivate()
{
    // nothing to do, just for overriding
}

void HalQueue::Reactivate(const CommandLog &)
{
    // nothing to do, just for overriding
}

void HalQueue::Interrupt()
{
    // nothing to do, just for overriding
}
