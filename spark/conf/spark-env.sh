#!/usr/bin/env bash

export SPARK_LOCAL_DIRS="/media/root/spark"

# Standalone cluster options
export SPARK_EXECUTOR_INSTANCES="1"
export SPARK_EXECUTOR_CORES="$(($(nproc) / 1))"
export SPARK_WORKER_CORES="$(nproc)"

export SPARK_MASTER_HOST="ip-172-31-6-65.ec2.internal"

# TODO: Make this dependent on HDFS install.
export HADOOP_CONF_DIR="$HOME/hadoop/conf"

# TODO: Make this non-EC2-specific.
# Bind Spark's web UIs to this machine's public EC2 hostname
export SPARK_PUBLIC_DNS="$(curl --silent http://169.254.169.254/latest/meta-data/public-hostname)"

# TODO: Set a high ulimit for large shuffles
# Need to find a way to do this, since "sudo ulimit..." doesn't fly.
# Probably need to edit some Linux config file.
# ulimit -n 1000000

# Should this be made part of a Python service somehow?
export PYSPARK_PYTHON="python3"

