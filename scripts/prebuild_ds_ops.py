"""Build and smoke-test the DeepSpeed CPUAdam extension once per environment."""

from __future__ import annotations

import time

import torch


def main() -> None:
    print("[cpu_adam] importing DeepSpeed builder", flush=True)
    from deepspeed.ops.op_builder import CPUAdamBuilder

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA compute node is required for this smoke test.")

    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"devices={[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
        flush=True,
    )
    builder = CPUAdamBuilder()
    print(f"[cpu_adam] compatible={builder.is_compatible(verbose=True)}", flush=True)
    start = time.perf_counter()
    builder.load()
    print(f"cpu_adam built/loaded in {time.perf_counter() - start:.1f}s", flush=True)

    from deepspeed.ops.adam import DeepSpeedCPUAdam

    parameter = torch.nn.Parameter(torch.randn(1024, dtype=torch.float32))
    parameter.grad = torch.randn_like(parameter)
    optimizer = DeepSpeedCPUAdam(
        [parameter],
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-2,
        adamw_mode=True,
    )
    before = parameter.detach().clone()
    optimizer.step()
    if torch.equal(before, parameter.detach()):
        raise RuntimeError("DeepSpeedCPUAdam.step() did not update the parameter.")
    print("DeepSpeedCPUAdam step ok (adamw_mode=True)", flush=True)


if __name__ == "__main__":
    main()
