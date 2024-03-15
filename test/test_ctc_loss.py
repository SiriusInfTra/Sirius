import os
os.environ['USE_SHARED_TENSOR'] = "1"
os.environ['SHARED_TENSOR_POOL_GB'] = "12"
os.environ['HAS_SHARED_TENSOR_SERVER'] = "0"
os.environ['GLOG_logtostderr'] = "1"

import torch
import torch_col

input_length = 4000
vocab_size = 3
batch_size = 4
target_length = 1200

log_probs = torch.randn(input_length, batch_size, vocab_size).log_softmax(2).requires_grad_()
targets = torch.randint(low=1, high=vocab_size - 1, size=(batch_size, target_length), dtype=torch.long)
input_lengths = batch_size * [input_length]
target_lengths = batch_size * [target_length]

res_cpu = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                        reduction='sum', zero_infinity=True)
grad_out = torch.randn_like(res_cpu)
grad_cpu, = torch.autograd.grad(res_cpu, log_probs, grad_out)

with torch.backends.cudnn.flags(enabled=False):
    res_gpu = torch.nn.functional.ctc_loss(log_probs.cuda(), targets.cuda(), input_lengths, target_lengths,
                                            reduction='sum', zero_infinity=True)
    grad_gpu, = torch.autograd.grad(res_gpu, log_probs, grad_out.cuda())

print((res_cpu - res_gpu).abs().max())
print((grad_cpu - grad_gpu).abs().max())

# self.assertEqual(res_cpu, res_gpu, atol=1e-4, rtol=0)
# self.assertEqual(grad_cpu, grad_gpu, atol=1e-4, rtol=0)