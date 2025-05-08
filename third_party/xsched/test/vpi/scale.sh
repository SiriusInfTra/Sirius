#!/bin/bash

# A script to set CPU and GPU frequency on the Jetson Orin

function usage() {
  echo "Usage:"
  echo "-p|--print-cpu         Print current CPU frequency"
  echo "-s|--set-cpu <freq>    Set CPU frequency to <freq>"
  echo "-d|--print-gpu         Print current GPU frequency"
  echo "-g|--set-gpu <freq>    Set GPU frequency to <freq>"
  echo "-h|--help              Display help"
}

function print_cpu_freq() {
  num_cpus=$(nproc)
  for i in $(seq 0 $(($num_cpus - 1))); do
    echo "CPU $i frequency is at $(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_cur_freq)"
    echo "CPU $i min frequency is $(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq)"
    echo "CPU $i max frequency is $(cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq)"
    echo ""
  done
}

function set_cpu_freq() {
  freq=$1
  num_cpus=$(nproc)
  for i in $(seq 0 $(($num_cpus - 1))); do
    echo $freq > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq
    echo $freq > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_min_freq
    echo "CPU $i frequency is set to $freq"
  done
}

function print_gpu_freq() {
  echo "GPU frequency is at $(cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq)"
  echo "GPU min frequency is $(cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq)"
  echo "GPU max frequency is $(cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq)"
}

function set_gpu_freq() {
  freq=$1
  echo $freq > /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/max_freq
  echo $freq > /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq
  echo "GPU frequency is set to $freq"
}


# parse options
OPTS=$(getopt -o ps:dg:h --long print-cpu,set-cpu:,print-gpu,set-gpu:,help -- "$@")
if [ $? != 0 ]; then
  echo "Failed to parse options...exiting" >&2;
  exit 1;
fi
eval set -- "$OPTS"

while true; do
  case "$1" in
    -p|--print-cpu) print_cpu_freq; shift;;
    -s|--set-cpu) set_cpu_freq $2; shift; shift;;
    -d|--print-gpu) print_gpu_freq; shift;;
    -g|--set-gpu) set_gpu_freq $2; shift; shift;;
    -h|--help) usage; shift; exit 0;;
    --) shift; break;;
    *) echo "Invalid option...exiting" >&2; exit 1;;
  esac
done

