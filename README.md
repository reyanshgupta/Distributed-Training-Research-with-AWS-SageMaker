# Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Research Findings](#research-findings)
- [Future Work](#future-work)
- [References](#references)

# Overview

This project is based on the research paper "Comparative Analysis of Image Recognition Techniques in Traditional and Distributed Computing Environments" by Reyansh Gupta, Khushi Kothari, Bhavya Chopra, Yuvika Singh, Hittanshu Upadhyay, and Aditya Kasar from SVKM’s NMIMS, STME Navi Mumbai. The research compares the performance of facial recognition algorithms using single instance machine learning algorithm training versus distributed training with parameter servers, implemented using AWS SageMaker. The metrics for comparison include processing time, accuracy, resource utilization, and cost savings using spot training on AWS SageMaker.

## Project Structure

The codebase includes two Python Jupyter notebooks:

- `Single_Instance_Face_Recognition_Script.ipynb`: Implementation for single instance machine learning algorithm training.
- `Distributed_Instance_Face_Recognition_Script.ipynb`: Implementation for distributed training with parameter servers.

## Dependencies

- Python 3.x
- TensorFlow
- AWS SageMaker
- Any other dependencies are listed in the respective notebooks.

## Getting Started

To run these notebooks, you will need access to AWS SageMaker. Follow these steps to set up your environment:

1. **AWS Account**: Ensure you have an AWS account set up.
2. **AWS SageMaker**: Navigate to the AWS SageMaker console and create a notebook instance.
3. **Upload Notebooks**: Upload the `Single_Instance_Face_Recognition_Script.ipynb` and `Distributed_Instance_Face_Recognition_Script.ipynb` notebooks to your SageMaker instance.
4. **Install Dependencies**: Ensure all the required libraries are installed in your SageMaker environment.
5. **Run Notebooks**: You can now run each notebook cell by cell, following the instructions within.

## Research Findings

The research found that while the distributed system trained faster and was more cost-effective, the single instance system achieved slightly higher accuracy. This highlights a trade-off between training speed/cost and model accuracy, which is crucial for real-world applications where accuracy is of utmost importance.

## Future Work

Future studies could explore the impact of different CNN architectures, dataset sizes, platforms, and frameworks on performance within both single-instance and distributed environments. Additionally, further hyperparameter tuning and scalability considerations in distributed systems would provide valuable insights for real-world image recognition applications.

## References

- Amazon SageMaker Official Documentation: [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html)
- AWS Distributed Computing Introduction: [Distributed Computing on AWS](https://aws.amazon.com/what-is/distributed-computing/)
- TensorFlow on Amazon SageMaker: [TensorFlow with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html)
- AWS Spot Training with SageMaker: [Spot Instances for SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
- For a detailed list of references, including datasets, frameworks, and tools used in this research, please refer to the research paper "Comparative Analysis of Image Recognition Techniques in Traditional and Distributed Computing Environments".
