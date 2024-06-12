# Comparing simulation technologies across MATLAB, C, Python and Java

**Table of Contents**

- [[#Introduction|Introduction]]
- [[#Rationale Behind the Study|Rationale Behind the Study]]
- [[#Overview of Simulation Implementations|Overview of Simulation Implementations]]
	- [[#Overview of Simulation Implementations#Naive Earthquake Damage Calculation Model|Naive Earthquake Damage Calculation Model]]
	- [[#Overview of Simulation Implementations#Simulation Parameters|Simulation Parameters]]
	- [[#Overview of Simulation Implementations#Technological Implementation|Technological Implementation]]
- [[#Benchmarking and Performance Analysis|Benchmarking and Performance Analysis]]
	- [[#Benchmarking and Performance Analysis#Hardware and Environment Setup|Hardware and Environment Setup]]
	- [[#Benchmarking and Performance Analysis#Benchmarking Results|Benchmarking Results]]
	- [[#Benchmarking and Performance Analysis#Interpretation of Results|Interpretation of Results]]
	- [[#Benchmarking and Performance Analysis#Source Code Availability|Source Code Availability]]
- [[#Implications for Production Systems|Implications for Production Systems]]
- [[#Conclusion|Conclusion]]
- [[#Appendix: How to run models|Appendix: How to run models]]

**Abstract**

This paper offers a comparative analysis of natural catastrophe (natcat) simulation technologies across multiple programming environments. The study assesses the performance, efficiency, and ease of transitioning from MATLAB-based model development to production-ready systems on GPU-enabled servers using languages and technologies like Python/PyTorch and Java/TornadoVM. Key findings reveal significant performance differences, with GPU implementations markedly outperforming CPU-based counterparts. Insights from the study suggest strategic technology selections for insurance firms, aiming to enhance the speed and accuracy of risk assessments, which could potentially revolutionize natcat modeling and insurance product tailoring.

## Introduction

In the intricate world of insurance, understanding and predicting natural catastrophes through simulations is not just about crunching numbers—it's about making sense of chaos. The stakes are exceptionally high; accurate simulations can lead to better risk assessments, more tailored insurance products, and ultimately, more resilient communities. But achieving this accuracy is no small feat. It requires sophisticated computational models that can process vast amounts of data quickly and efficiently.

In this paper we venture into a comparative study of various simulation techniques for natural catastrophe (natcat) modeling, employed across different programming environments and technologies. From the raw power of C with CUDA to the high-level abstractions of Python and Java, each approach offers unique insights and presents distinct challenges. As we explore these diverse methodologies, we'll uncover not only their computational nuances but also how these technical details could shape the future architecture of insurance risk assessment tools.

Our journey through code and concept aims to highlight the common threads that tie these simulations together, as well as the pivotal differences that could influence scalability and performance in real-world applications. 

## Rationale Behind the Study

In the ever-evolving landscape of insurance and reinsurance, natural catastrophe (natcat) modeling serves as a cornerstone, influencing decisions from policy pricing to risk management strategies. Traditionally, these models are crafted in environments like MATLAB or R—platforms renowned for their robust analytical capabilities and ease of use during the development phase. However, translating these models into production-ready systems often involves porting them to Java and deploying on CPU-based clusters. This transition not only introduces complexity but can also limit the models' efficiency and scalability.

The primary goal of this study was to explore alternative frameworks that could support direct implementation and execution of natcat models on GPU-enabled servers. By investigating implementations across several programming languages and technologies, including Python with PyCUDA and PyTorch, and Java with JCUDA, Aparapi and TornadoVM, this research aimed to gauge the effort required to adapt MATLAB-developed models for GPU-accelerated environments.

Understanding these dynamics is crucial, as GPU acceleration promises significant enhancements in processing speeds and computational efficiency. This could lead to faster, more accurate risk assessments, ultimately enabling insurers to respond more agilely to evolving risk landscapes. The exploration thus provides insights into not only the technical feasibility of such adaptations but also their practical implications in the broader context of natcat risk modeling and insurance analytics.

## Overview of Simulation Implementations

To effectively analyze the impact of different programming technologies on the execution of natural catastrophe (natcat) models, a simplified earthquake model was adopted across various implementations. This section outlines the model used and the technologies leveraged to understand their capabilities and limitations.

### Naive Earthquake Damage Calculation Model

The model is based on the following assumptions:
- All insured properties are located on the earthquake fault line and are evenly distributed.
- Each property is characterized by specific parameters: location, value, construction type, and soil amplification.
- Each earthquake event is defined by its epicenter location and magnitude.

The core of the model is the damage function, which calculates the potential loss for each property given an earthquake event. The function is expressed as:

`damage = max(0, (1.0 - 0.01 * distance_from_epicenter) * (0.3 + 0.2 * construction_type + 0.02 * magnitude + 0.1 * amplification))`

### Simulation Parameters

The model takes two primary inputs:
- `numEvents`: The number of earthquake events to simulate.
- `numInsuredProperties`: The total number of assets in the insurance portfolio to be assessed for damage with each event.

During each simulation, random values are generated for the magnitudes and epicenter locations of earthquake events, alongside the properties' specific parameters. The model then computes the damage for all properties for each event, simulating the potential impact across the portfolio.

### Technological Implementation

To explore the potential of various technologies, the model was implemented across several programming environments:
- **C/CUDA**: Utilized for its close-to-hardware operation capability, offering granular control over GPU computations and memory management.
- **Python/PyCUDA and PyTorch**: Chosen for their ease of use and extensive libraries, which significantly speed up development time and offer seamless GPU integration.
- **Java/JCUDA, Aparapi and TornadoVM**: Investigated for their robustness in enterprise environments, where Java is widely used. Particularly, TornadoVM shows promise for enabling Java-based solutions to leverage GPU resources effectively, making it a strong candidate for future architectural upgrades.
- **MATLAB**: Employed in preliminary tests to set a baseline for performance and accuracy, given its widespread use in initial model development phases in the industry.

Each technology was assessed not only for its computational performance but also for how intuitively it could integrate the simulation's requirements, aiming to provide insights into their practical applicability in production settings. This comparative approach helps highlight each framework's strengths and weaknesses in handling natcat simulations efficiently, with a special focus on TornadoVM for its potential to enhance Java's capabilities in processing-intensive environments.

## Benchmarking and Performance Analysis

In this section, we delve into the benchmarking results of the various simulation implementations evaluated on a standardized hardware setup. These benchmarks provide insights into the performance and efficiency of each programming and computational approach under identical conditions, shedding light on potential optimizations and suitable choices for different operational scenarios.

### Hardware and Environment Setup

The benchmarking tests were conducted on the following hardware and software environment, ensuring a consistent and controlled platform for all simulations:

- **Hardware Model:** Intel NUC (NUC10i7FNH)
- **RAM:** 32GB
- **Processor:** Intel® Core™ i7-10710U CPU @ 1.10GHz × 12
- **GPU:** NVIDIA GeForce RTX 3090/PCIe/SSE2 by NVIDIA Corporation GA102
- **VRAM:** 24GB
- **Operating System:** Pop!_OS 22.04 LTS

### Benchmarking Results

The results from the benchmarking are summarized in the table below, showcasing the execution times across various technologies and computation platforms:

| Language | Library   | Computation Platform | Time (ms) |
| -------- | --------- | -------------------- | --------- |
| Python   | PyTorch   | GPU                  | 0.195     |
| Java     | TornadoVM | GPU                  | 0.833     |
| C        | C/CUDA    | GPU                  | 3.717     |
| Java     | JCUDA     | GPU                  | 3.763     |
| Python   | PyCUDA    | GPU                  | 4.039     |
| Java     | Aparapi   | GPU                  | 15.863    |
| Matlab   | MATLAB    | GPU                  | 22.038    |
| Java     | TornadoVM | CPU (parallel)       | 116.713   |
| Java     | TornadoVM | CPU (single thread)  | 486.523   |
| Matlab   | MATLAB    | CPU (parallel)       | 693.858   |
| Python   | PyTorch   | CPU (parallel)       | 859.893   |
| Matlab   | MATLAB    | CPU (single thread)  | 6387.674  |

### Interpretation of Results

The data reveals significant variations in execution times, which are influenced by the choice of programming language, library, and whether the computations were carried out on GPU or CPU. Python's PyTorch implementation on GPU stands out as the fastest, demonstrating the effectiveness of optimized GPU kernels in high-level languages for complex mathematical computations. On the other end, MATLAB's single-threaded CPU performance underscores the potential bottlenecks when traditional methods are applied without parallelization or hardware acceleration.

These results underscore the importance of selecting the right technology stack based on specific operational requirements and computational constraints. They also highlight the evolving nature of computational tools in handling intensive simulations like those required in natural catastrophe modeling.

### Source Code Availability

For transparency and further exploration, the source code for all simulations is available on GitHub: https://github.com/vlebedev/eqsim. 

## Implications for Production Systems

The benchmarking results bring to light the significant performance benefits of using GPU-accelerated technologies such as Python/PyTorch and Java/TornadoVM in natural catastrophe (natcat) simulations. PyTorch excels in rapid prototyping and agile development due to its dynamic nature and extensive ecosystem, making it ideal for scaling up complex simulations that require frequent updates and iterations. TornadoVM, on the other hand, offers a unique advantage with its ability to seamlessly switch computation between GPUs and multicore CPUs, enhancing scalability across diverse production environments.

In terms of technology selection, Python/PyTorch should be considered for environments where speed of development and model iteration are paramount. Its extensive libraries and community support streamline the implementation of advanced machine learning algorithms that can enhance simulation accuracy. Java/TornadoVM is recommended for organizations that require robust integration with existing Java-based infrastructure and can benefit from TornadoVM's distinctive ability to optimize resource usage dynamically across CPUs and GPUs.

These frameworks not only differ in performance but also in maintenance and developer-related costs. Python generally requires less boilerplate code than Java, potentially reducing development time and costs. However, Java's strong type system and mature ecosystem might reduce long-term maintenance costs by catching errors early in the development cycle.

In conclusion, selecting the right technology stack depends on specific organizational needs, including existing infrastructure compatibility, developer expertise, and the specific requirements of natcat simulation tasks. For firms looking to balance rapid development with system robustness, a hybrid approach using both Python for model development and Java for production could combine the strengths of both ecosystems.

## Conclusion

This exploration into various simulation technologies has highlighted key insights and differences in performance between platforms like Python/PyTorch and Java/TornadoVM. These findings underscore the potential for enhancing natural catastrophe simulation efficiency and scalability, particularly through GPU-accelerated computations.

**Future Work:** There remains a lot of room for further research, especially in optimizing existing models and exploring new computational frameworks that could further reduce computation times and enhance data handling capabilities. Investigating hybrid models that leverage both CPU and GPU resources effectively could also yield significant benefits.

# Appendix: How to run models

The easiest way to run all implementations (except for MATLAB-based ones) is by using the Visual Studio Code devcontainers feature. Each project has a corresponding `.devcontainers` directory with a settings file and Dockerfile that defines a suitable container for running each simulation. After opening a project in VS Code, re-open it in a devcontainer. Then follow the instructions given in comments at the top of each source file.

One important note about the host machine setup: As all simulations are GPU-based, your host machine's GPU must be accessible to the devcontainers. To enable this, NVIDIA Container Toolkit needs to be installed on the host machine prior to running the simulations. Please find detailed installation instructions here: [NVIDIA Docker Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

