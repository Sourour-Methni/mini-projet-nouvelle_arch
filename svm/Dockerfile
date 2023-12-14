# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the SVM service files
COPY app/svm_service.py .
COPY svm_model.pkl .
COPY pca_model.pkl .


# Update the package repository mirror and install ffmpeg
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y ffmpeg

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 510

# Define environment variable
ENV NAME World

# Run svm_service.py when the container launches
CMD ["python", "svm_service.py"]
