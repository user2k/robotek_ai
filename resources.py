"""Odczyty CPU/RAM i NVIDIA poza wątkiem Tkinter, bez wpływu na trening."""
import os
import subprocess
from queue import Empty, Full, Queue
from threading import Event, Thread

try:
    import psutil
except ImportError:
    psutil = None


def gpu_sample():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=1.5, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        values = result.stdout.strip().split(',')
        return dict(zip(('gpu_percent', 'vram_used_mib', 'vram_total_mib'), map(float, values)))
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


class ResourceMonitor:
    def __init__(self):
        self.samples = Queue(maxsize=1)
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        process = psutil.Process() if psutil else None
        if process:
            psutil.cpu_percent()
            process.cpu_percent()
        while not self.stop.is_set():
            sample = {}
            if process:
                try:
                    memory = psutil.virtual_memory()
                    sample.update(cpu_percent=psutil.cpu_percent(),
                                  app_cpu_percent=process.cpu_percent() / (psutil.cpu_count() or 1),
                                  ram_used=memory.total - memory.available, ram_total=memory.total,
                                  app_ram=process.memory_info().rss)
                except psutil.Error:
                    pass
            sample.update(gpu_sample())
            try:
                self.samples.get_nowait()
            except Empty:
                pass
            try:
                self.samples.put_nowait(sample)
            except Full:
                pass
            self.stop.wait(1.)

    def close(self):
        self.stop.set()


def format_resources(sample):
    gib = 1024 ** 3
    cpu = (f"CPU: {sample['cpu_percent']:.0f}% · aplikacja: {sample['app_cpu_percent']:.0f}%"
           if 'cpu_percent' in sample else 'CPU: brak odczytu (psutil)')
    ram = (f"RAM: {sample['ram_used']/gib:.1f}/{sample['ram_total']/gib:.1f} GiB\n"
           f"RAM aplikacji: {sample['app_ram']/gib:.2f} GiB" if 'ram_total' in sample else 'RAM: brak odczytu')
    gpu = (f"GPU 0: {sample['gpu_percent']:.0f}%\n"
           f"VRAM: {sample['vram_used_mib']/1024:.2f}/{sample['vram_total_mib']/1024:.2f} GiB"
           if 'vram_total_mib' in sample else 'GPU / VRAM: brak odczytu NVIDIA')
    return cpu + '\n' + ram + '\n' + gpu
