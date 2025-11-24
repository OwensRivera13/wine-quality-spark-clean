# ==========================================================
# WINE QUALITY PREDICTION CONTAINER  (By Owens)
# ==========================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Using an official lightweight Spark image as base
# This comes with Java & Python pre-installed, saves setup time
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#
#FROM bitnami/spark:3.5.1
#FROM apache/spark-py:v3.5.1
#FROM docker.io/bitnami/spark:3.3.2
FROM docker.io/apache/spark:3.5.0

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Make a working directory inside container
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
WORKDIR /app

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Copy all project files into the container
# (training, validation data, and scripts)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
COPY . /app

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Install Python dependencies used by the scripts
# (pyspark is already in base image, but numpy etc. might not be)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#RUN pip install numpy pandas
USER root
RUN pip install --no-cache-dir numpy pandas
USER spark

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Default command for container startup
# We’ll just make it print help message by default
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
CMD ["bash"]

ENV PATH="/opt/spark/bin:${PATH}"
# Automatically run prediction script when container starts
ENTRYPOINT ["spark-submit", "wine_test_cls.py"]

