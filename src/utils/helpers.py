import os
import time
import datetime
import random
import subprocess
import numpy as np
import torch

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=True, sleep_time=10, run_from_python = False):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    # def _check():
    #     # Get the list of GPUs via nvidia-smi
    #     smi_query_result = subprocess.check_output(
    #         "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    #     )
    #     # Extract the usage information
    #     gpu_info = smi_query_result.decode("utf-8").split("\n")
    #     gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    #     gpu_info = [
    #         int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
    #     ]  # Remove garbage
    #     # Keep gpus under threshold only
    #     free_gpus = [
    #         str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
    #     ]
    #     free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
    #     gpus_to_use = ",".join(free_gpus)
    #     return gpus_to_use

    def run_cmd(cmd):
        out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
        return out

    def _check():
        out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
        # print(out)
        out = (out.split('\n'))[1:]
        out = [l for l in out if '--' not in l]

        total_gpu_num = int(len(out)/5)
        gpu_bus_ids = []
        for i in range(total_gpu_num):
            gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])
        
        # print(run_cmd('ps | grep python'))
        # pid = os.getpid()
        # print(f"pid {pid}")
        out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
        # print(out)
        gpu_bus_ids_in_use = (out.split('\n'))[1:]
        gpu_ids_in_use = []

        for bus_id in gpu_bus_ids_in_use:
            gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

        p_pos, pid = _check_pid_pos()

        pid_queueing = False

        # plus = 1 if run_from_python else 0

        if p_pos >= total_gpu_num:
            # print(f"pid {pid} || p pos {p_pos}")
            pid_queueing = True
            return [], pid_queueing, p_pos

        return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use], pid_queueing, p_pos

    def _check_pid_pos():
        plus = 1 if run_from_python else 0

        pid = os.getpid()
        all_pid = run_cmd('ps | grep python')
        
        process_infos = all_pid.split('\n')
        all_pid = [int(info.split()[0]) for info in process_infos]

        all_pid = sorted(all_pid)

        p_pos = all_pid.index(pid)

        p_pos = p_pos - plus

        return p_pos, pid

    # t = random.randint(1,5)
    # time.sleep(t)

    while True:
        gpus_to_use, pid_queueing, p_pos = _check()
        if gpus_to_use or not wait:
            break
        
        if pid_queueing:
            print(f"Too many python processes, retrying in {sleep_time}s")
        else:
            print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")

    # avoid conflicts of adjcant two processes
    gpu_idx = p_pos if p_pos < len(gpus_to_use) else -1
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_to_use[gpu_idx])
    print(f"Using GPU(s): {gpus_to_use[gpu_idx]}")

    return gpus_to_use[gpu_idx]