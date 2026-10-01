# Host setup for the B70

What the reference machine (Arc Pro B70, Ryzen 7 5700, 32 GB, Linux 7.1 with the xe driver, oneAPI 2026.0)
has besides the repository's build.

- **oneAPI Base Toolkit** in `/opt/intel/oneapi` (icx/icpx, SYCL runtime, Level Zero). Every build and run
  script sources `setvars.sh`; a process launched through `sudo` loses `LD_LIBRARY_PATH`, so pass it on or
  source oneAPI again, or the server finds no GPU ("No device of requested type available").
- **oneDNN with SYCL runtimes** in `/opt/onednn-sycl`: `build-onednn-sycl.sh`. `scripts/b70/build-llama-sycl.sh`
  takes it from `DNNL_DIR`.
- **Locked memory**: `QWFN_LOCK_HOST=1` needs the memlock limit raised, or it logs "NOT locked" and runs
  unlocked (`ulimit -l` is 8 MB by default). Under systemd: `LimitMEMLOCK=infinity` in the service; by hand:
  `sudo prlimit --memlock=unlimited setpriv --reuid=$USER --regid=$USER --init-groups <command>`. Check the log for
  `locked (VmLck +... MB), registered with the GPU driver`.
- **Power cap**: `b70-power-cap.sh` + `.service` + `.timer` into `/usr/local/sbin` and `/etc/systemd/system`,
  then `systemctl enable --now b70-power-cap.timer`. Set the cap in the service's `B70_CAP_W`. Every number in
  the docs states the cap it was measured at: 160 W unless noted.
- **Memory headroom**: `--ram` (the RAM expert tier) + `--prefix-cache` + ~4 GB for the system must fit the
  machine's RAM; the server refuses a prefix cache that would not.
