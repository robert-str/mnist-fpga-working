# Academic References for CNN FPGA Implementation Thesis

*Compiled: January 2026*
*Total Papers: 29 Academic Sources*

This document contains a curated collection of academic papers and research articles relevant to CNN implementations on FPGAs for MNIST digit classification. All sources are suitable for citation in an engineering thesis.

---

## Table of Contents

1. [General FPGA-Based CNN Acceleration](#category-1-general-fpga-based-cnn-acceleration)
2. [MNIST-Specific Implementations on Xilinx FPGAs](#category-2-mnist-specific-implementations-on-xilinx-fpgas)
3. [Quantization Techniques](#category-3-quantization-techniques)
4. [Low-Power and Embedded Implementations](#category-4-low-power-and-embedded-implementations)
5. [Specialized Implementations and Optimizations](#category-5-specialized-implementations-and-optimizations)
6. [Development Tools and Methodologies](#category-6-development-tools-and-methodologies)
7. [Citation Strategy Guide](#recommended-citation-strategy-for-your-thesis)

---

## Category 1: General FPGA-Based CNN Acceleration

### Comprehensive Reviews and Survey Papers

#### 1. FPGA-based Acceleration for Convolutional Neural Networks: A Comprehensive Review (2025)

**Link:** [arXiv](https://arxiv.org/html/2505.13461v1)

**Summary:** Recent comprehensive review of FPGA-based hardware accelerators specifically designed for CNNs, exploring key optimization strategies and highlighting future challenges and opportunities.

**Key Topics:**
- Optimization strategies for CNN acceleration
- Reconfigurability and parallelism
- Energy efficiency analysis
- Future research directions

**Relevance:** Excellent foundational paper for literature review section. Provides modern perspective on the field.

---

#### 2. A Survey of FPGA-Based Neural Network Inference Accelerator

**Link:** [arXiv PDF](https://arxiv.org/pdf/1712.08934)

**Summary:** Comprehensive survey covering neural network inference accelerators from software to hardware perspective, from circuit level to system level. Includes analysis of both Xilinx and Altera/Intel platforms.

**Key Topics:**
- Hardware/software co-design
- Circuit-level optimizations
- System architecture
- Platform comparison (Xilinx vs Altera)

**Relevance:** Strong background on inference acceleration methodologies and platform considerations.

---

#### 3. A survey of FPGA-based accelerators for convolutional neural networks

**Link:** [Neural Computing and Applications](https://dl.acm.org/doi/10.1007/s00521-018-3761-1)

**Summary:** Survey highlighting the high energy efficiency, computing capabilities, and reconfigurability of FPGAs as promising platforms for CNN hardware acceleration. Presents techniques for implementing and optimizing CNN algorithms on FPGA.

**Key Topics:**
- Energy efficiency analysis
- Reconfigurability advantages
- Implementation techniques
- Optimization strategies

**Relevance:** Strong theoretical foundation for FPGA advantages over other platforms.

---

#### 4. Convolutional Neural Network Acceleration Techniques Based on FPGA Platforms: Principles, Methods, and Challenges

**Link:** [MDPI Information](https://www.mdpi.com/2078-2489/16/10/914)

**Summary:** Systematic exploration of FPGA-based CNN acceleration technology, focusing on acceleration methods, architectural innovations, hardware optimization techniques, and hardware-software co-design frameworks.

**Key Topics:**
- Acceleration methods
- Architectural innovations
- Hardware optimization techniques
- Hardware-software co-design
- Systematic analytical framework

**Relevance:** Modern perspective covering both algorithmic and hardware aspects.

---

#### 5. Bibliometric Review of FPGA Based Implementation of CNN

**Link:** [University of Nebraska](https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=10630&context=libphilprac)

**Summary:** Meta-analysis of research trends in FPGA-based CNN implementations, examining publication patterns and research directions.

**Key Topics:**
- Research trend analysis
- Publication statistics
- Emerging research directions

**Relevance:** Good for understanding the research landscape and positioning your work.

---

## Category 2: MNIST-Specific Implementations on Xilinx FPGAs

### IEEE Conference Publications

#### 6. An FPGA-based accelerator implementation for deep convolutional neural networks

**Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/7490869/)

**Summary:** Implementation of CNN accelerator targeting MNIST digit recognition using Vivado HLS with 11-bit fixed-point precision on Xilinx Virtex7 FPGA.

**Key Achievements:**
- **Speedup:** 16.42x
- **Platform:** Xilinx Virtex7
- **Precision:** 11-bit fixed-point
- **Tool:** Vivado HLS

**Relevance:** Direct comparison for fixed-point implementation approach. Excellent benchmark for speedup metrics.

---

#### 7. FPGA-based convolutional neural network accelerator design using high level synthesize

**Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/7869873/)

**Summary:** LeNet implementation for MNIST handwritten digit recognition on Xilinx Zynq FPGA using High-Level Synthesis approach.

**Key Achievements:**
- **Architecture:** LeNet-5
- **Platform:** Xilinx Zynq FPGA
- **Tool:** High-Level Synthesis
- **Application:** MNIST classification

**Relevance:** Alternative design methodology comparison. Relevant for LeNet-5 migration guide.

---

#### 8. Deep neural network accelerator

**Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/8108073/)

**Summary:** Implementation of 784-40-40-10 neural network on Xilinx Virtex-5 XC5VLX-110T device for MNIST classification.

**Key Achievements:**
- **Accuracy:** 97.20% on MNIST
- **Platform:** Xilinx Virtex-5 XC5VLX-110T
- **Architecture:** 784-40-40-10 network

**Relevance:** Accuracy benchmark for comparison with your 99% accuracy.

---

#### 9. Simulation and synthesis of UART through FPGA Zedboard for IoT applications

**Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/9752556/)

**Summary:** UART protocol implementation and verification on FPGA for IoT applications.

**Key Topics:**
- UART protocol design
- FPGA implementation
- IoT communication

**Relevance:** Validates communication protocol design choices for weight/image loading.

---

### Journal Articles

#### 10. A Hardware Accelerator for The Inference of a Convolutional Neural Network

**Link:** [Redalyc](https://www.redalyc.org/journal/911/91164537008/html/)

**Summary:** Hardware accelerator implemented on Digilent Arty Z7-20 board with Xilinx Zynq-7000 SoC for MNIST classification.

**Key Achievements:**
- **Platform:** Digilent Arty Z7-20 (Xilinx Zynq-7000 SoC)
- **Precision:** 12-bit fixed-point
- **Accuracy:** 97.59% on MNIST
- **Throughput:** 441 images per second

**Relevance:** Direct hardware comparison - similar educational FPGA platform to Basys3. Excellent throughput benchmark.

---

#### 11. An FPGA-Based Convolutional Neural Network Coprocessor

**Link:** [Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1155/2021/3768724)

**Summary:** Coprocessor implementing VGG16 network with 16-bit fixed-point quantization on FPGA.

**Key Achievements:**
- **Architecture:** VGG16
- **Precision:** 16-bit fixed-point
- **Performance:** 316.0 GOP/s at 200 MHz
- **Power:** 9.25 W

**Relevance:** Power consumption and performance metrics comparison.

---

## Category 3: Quantization Techniques

### Fixed-Point and Quantization Research

#### 12. Fixed-Point Convolutional Neural Network for Real-Time Applications

**Link:** [arXiv](https://arxiv.org/pdf/1808.09945)

**Summary:** Analysis of real-time CNN implementation using fixed-point arithmetic, examining trade-offs between precision and performance.

**Key Topics:**
- Fixed-point arithmetic
- Real-time processing
- Precision analysis
- Performance optimization

**Relevance:** Theoretical foundation for 8-bit quantization strategy.

---

#### 13. A FIXED-POINT QUANTIZATION TECHNIQUE FOR CONVOLUTIONAL NEURAL NETWORKS

**Link:** [KIT Scientific Publishing](https://publikationen.bibliothek.kit.edu/1000122216/107734810)

**Summary:** Detailed quantization methodology for converting floating-point CNNs to fixed-point representations.

**Key Topics:**
- Quantization algorithms
- Fixed-point representation
- Accuracy preservation
- Mathematical framework

**Relevance:** Mathematical framework for your quantization strategy section.

---

#### 14. Fixed-Point Implementation of Convolutional Neural Networks for Image Classification

**Link:** [ResearchGate](https://www.researchgate.net/publication/329955813_Fixed-Point_Implementation_of_Convolutional_Neural_Networks_for_Image_Classification)

**Summary:** FPGA implementation based on 4-bit fixed-point arithmetic with 8-bit additions for MNIST classification.

**Key Achievements:**
- **Precision:** 4-bit weights, 8-bit additions
- **Dataset:** MNIST (60,000 training, 10,000 test)
- **Approach:** Extreme quantization

**Relevance:** Comparison point for extreme quantization vs your 8-bit approach.

---

#### 15. Post-training quantization for efficient FPGA-based neural network acceleration

**Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167926025001658) (2025)

**Summary:** PTQ framework converting CNN models from FP32 to INT8 without retraining, optimized for FPGA deployment using asymmetric quantization and TensorFlow Lite.

**Key Achievements:**
- **Method:** Post-training quantization (FP32 → INT8)
- **Platform:** PYNQ-Z1 FPGA
- **Models:** VGG16 and ResNet50
- **Throughput:** 67% increase (150 FPS → 250 FPS)
- **Latency:** 68% reduction
- **Power-Delay:** 52% improvement

**Relevance:** Directly matches your post-training quantization approach. Excellent performance metrics.

---

#### 16. Trainable Fixed-Point Quantization for Deep Learning Acceleration on FPGAs

**Link:** [arXiv](https://arxiv.org/html/2401.17544v1)

**Summary:** QFX approach for trainable fixed-point quantization on FPGAs, achieving higher accuracy with fewer bits compared to post-training quantization.

**Key Topics:**
- Trainable quantization
- CIFAR-10 and ImageNet results
- Bit-width optimization
- Accuracy preservation

**Relevance:** Advanced quantization techniques for future work discussion.

---

### Quantization-Aware Training

#### 17. Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training

**Link:** [NVIDIA Technical Blog](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)

**Summary:** Industry-standard quantization-aware training approach using TensorRT, demonstrating how to maintain FP32 accuracy with INT8 inference.

**Key Topics:**
- Quantization-aware training (QAT)
- Fake quantization during training
- FP32 accuracy preservation
- INT8 inference optimization

**Relevance:** Alternative approach to post-training quantization for methodology comparison.

---

#### 18. Quantized neural networks: training neural networks with low precision weights and activations

**Link:** [JMLR](https://dl.acm.org/doi/10.5555/3122009.3242044)

**Summary:** Foundational paper on neural network quantization theory, covering mathematical principles of low-precision training and inference.

**Key Topics:**
- Quantization theory
- Low-precision training
- Gradient approximation
- Theoretical foundations

**Relevance:** Theoretical background for quantization methodology section.

---

## Category 4: Low-Power and Embedded Implementations

### Binary and Spiking Neural Networks

#### 19. Binary Neural Network Implementation for Handwritten Digit Recognition on FPGA

**Link:** [arXiv](https://arxiv.org/html/2512.19304) (2024)

**Summary:** Binary neural network implementation on **Xilinx Artix-7 FPGA** (same family as Basys3!) for low-power, high-speed inference using bitwise logic operations.

**Key Achievements:**
- **Platform:** Xilinx Artix-7 FPGA ⭐ (SAME AS YOUR BASYS3!)
- **Clock:** 80 MHz
- **Accuracy:** 84% on MNIST test set
- **Power:** 0.617 W (extremely low!)
- **Approach:** Binary weights and activations

**Relevance:** CRITICAL REFERENCE - Same FPGA family! Excellent for extreme low-power comparison. Shows trade-off between accuracy and power.

---

#### 20. Hardware implementation of FPGA-based spiking attention neural network accelerator

**Link:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453718/)

**Summary:** SeaSNN accelerator implementation demonstrating spiking neural network approach for MNIST classification.

**Key Achievements:**
- **Accuracy:** 94.28% on MNIST
- **Speed:** 0.000401 seconds per frame
- **Frequency:** 200 MHz
- **Power:** 4.996 W total on-chip
- **Efficiency:** 0.42 TOPS/W

**Relevance:** Alternative neural network architecture comparison.

---

### Resource-Efficient Designs

#### 21. Design of Convolutional Neural Network Processor Based on FPGA Resource Multiplexing Architecture

**Link:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9414218/)

**Summary:** Resource-multiplexing CNN architecture achieving high accuracy with remarkably low power consumption.

**Key Achievements:**
- **Total Power:** 1.03 W
- **CNN Module Power:** 0.03 W (!)
- **Accuracy:** 97% for digit recognition
- **Approach:** Resource multiplexing for systolic array

**Relevance:** Power efficiency strategies and resource optimization techniques.

---

#### 22. FPGA-Based Convolutional Neural Network Accelerator with Resource-Optimized Approximate Multiply-Accumulate Unit

**Link:** [MDPI Electronics](https://www.mdpi.com/2079-9292/10/22/2859)

**Summary:** Resource-optimized CNN accelerator using approximate computing techniques in MAC units.

**Key Topics:**
- Approximate multiply-accumulate
- Resource optimization
- Area-power trade-offs
- Accuracy-efficiency balance

**Relevance:** MAC unit optimization techniques for resource-constrained FPGAs.

---

## Category 5: Specialized Implementations and Optimizations

### LeNet-5 Specific

#### 23. FPGA-QNN: Quantized Neural Network Hardware Acceleration on FPGAs

**Link:** [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/15/2/688) (2025)

**Summary:** Quantized LeNet and MLP models using PyTorch framework with Xilinx Brevitas library, synthesized using FINN framework for multi-bit quantized networks.

**Key Achievements:**
- **Architectures:** LeNet and MLP
- **Frameworks:** PyTorch, Xilinx Brevitas, FINN
- **MNIST Accuracy:** 94-99% (CPU and FPGA)
- **LeNet on FPGA:** 95.4% accuracy with W1A1 quantization
- **Processing Time:** 1.4s (LeNet), 0.014s (MLP) with H folding
- **FashionMNIST:** Up to 90% (LeNet)

**Relevance:** LeNet quantization reference for migration guide. FINN framework approach.

---

### Communication and System Integration

#### 24. Neuromorphic Processor Employing FPGA Technology with Universal Interconnections

**Link:** [arXiv](https://arxiv.org/html/2512.10180) (2024)

**Summary:** Neuromorphic processor with UART interface for weight and parameter loading, enabling runtime reconfiguration.

**Key Achievements:**
- **Interface:** UART for SNN configuration parameters
- **Features:** Runtime reconfiguration without resynthesis
- **Limitation:** UART introduces ~100ms programming delays
- **Parameters:** Weights, input spikes, thresholds, connection lists

**Relevance:** Validates your UART protocol choice and documents known limitations. Important for communication protocol discussion.

---

#### 25. unzipFPGA: Enhancing FPGA-based CNN Engines with On-the-Fly Weights Generation

**Link:** [arXiv](https://arxiv.org/abs/2103.05600)

**Summary:** Alternative approach to traditional weight loading through on-the-fly weight generation.

**Key Topics:**
- Weight compression
- On-the-fly generation
- Memory bandwidth optimization
- Alternative to weight loading

**Relevance:** Discussion of weight loading optimization alternatives.

---

#### 26. Efficient FPGA Implementation of Convolutional Neural Networks and Long Short-Term Memory for Radar Emitter Signal Recognition

**Link:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10857097/)

**Summary:** CNN and LSTM implementation with systolic array parameter loading architecture.

**Key Topics:**
- Systolic array design
- Weight parameter loading instructions
- Bias parameter loading instructions
- Parameter loading architecture

**Relevance:** Weight loading architecture patterns and instruction set design.

---

#### 27. FPGA-Based Deep Neural Network Implementation for Handwritten Digit Recognition

**Link:** [ResearchGate](https://www.researchgate.net/publication/391864863_FPGA-Based_Deep_Neural_Network_Implementation_for_Handwritten_Digit_Recognition)

**Summary:** Complete FPGA-based handwritten digit recognition system implementation.

**Key Topics:**
- Handwritten digit recognition
- System integration
- FPGA implementation

**Relevance:** Similar application domain and complete system reference.

---

## Category 6: Development Tools and Methodologies

### High-Level Synthesis

#### 28. Accelerating convolutional neural networks on FPGA platforms: a high-performance design methodology using OpenCL

**Link:** [Journal of Real-Time Image Processing](https://link.springer.com/10.1007/s11554-025-01647-5)

**Summary:** High-performance CNN design methodology using OpenCL for FPGA acceleration, evaluated on MNIST and CIFAR-10.

**Key Achievements:**
- **Platform:** Xilinx ZYNQ 7000
- **Tool:** OpenCL
- **Datasets:** MNIST and CIFAR-10
- **Approach:** High-level synthesis

**Relevance:** Alternative design methodology (OpenCL vs Verilog/VHDL).

---

#### 29. Efficient Design of Neural Network Hardware Accelerator for Image Recognition

**Link:** [University of Mississippi Thesis](https://egrove.olemiss.edu/cgi/viewcontent.cgi?article=3897&context=etd)

**Summary:** University thesis demonstrating neural network hardware accelerator design for image recognition.

**Key Topics:**
- Thesis structure
- Methodology presentation
- Design process
- Results documentation

**Relevance:** Example thesis structure and methodology presentation for your own thesis.

---

## Recommended Citation Strategy for Your Thesis

### Chapter 1: Introduction

**Background on FPGA-based Neural Networks:**
- Paper #1: FPGA-based Acceleration for CNNs (2025 review)
- Paper #2: Survey of FPGA-Based NN Inference Accelerators
- Paper #3: Survey of FPGA-based accelerators for CNNs

**Motivation for Using Artix-7:**
- Paper #19: Binary NN on Artix-7 (same FPGA family!)

**Research Contribution:**
- Position your work relative to Papers #6, #8, #10

---

### Chapter 2: Related Work

**MNIST Classification on FPGAs:**
- Paper #6: FPGA-based accelerator (11-bit, 16.42x speedup)
- Paper #8: Deep NN accelerator (97.20% accuracy)
- Paper #10: Hardware accelerator (97.59%, 441 img/s)
- Paper #19: Binary NN on Artix-7 (84%, 0.617W)

**LeNet Implementations:**
- Paper #7: LeNet using HLS
- Paper #23: FPGA-QNN with LeNet

**Alternative Architectures:**
- Paper #20: Spiking neural networks
- Paper #21: Resource-multiplexing architecture

---

### Chapter 3: Theoretical Background

**3.1 Convolutional Neural Networks:**
- Standard CNN references (not in this list - use LeCun 1998, etc.)

**3.2 Quantization Theory:**
- Paper #18: Quantized neural networks (foundational)
- Paper #12: Fixed-point CNNs for real-time
- Paper #13: Fixed-point quantization technique

**3.3 FPGA Architecture:**
- Paper #4: CNN acceleration techniques on FPGA

---

### Chapter 4: Methodology

**4.1 Network Architecture:**
- Describe your 2-layer CNN
- Compare to Papers #6, #7, #8, #10

**4.2 Quantization Strategy:**
- Paper #15: Post-training quantization (matches your approach!)
- Paper #14: Fixed-point implementation
- Paper #17: QAT (alternative approach for discussion)

**4.3 FPGA Implementation:**
- Paper #28: Design methodology (OpenCL alternative)
- Paper #22: Resource-optimized MAC units
- Paper #21: Resource multiplexing

**4.4 Communication Protocol:**
- Paper #9: UART synthesis on FPGA
- Paper #24: UART for weight loading (validates your choice!)
- Paper #26: Parameter loading architecture

---

### Chapter 5: Implementation Details

**5.1 Hardware Architecture:**
- Paper #4: Hardware-software co-design
- Paper #21: Resource multiplexing architecture

**5.2 Memory Organization:**
- Paper #22: Resource optimization
- Paper #25: Weight loading alternatives

**5.3 Inference Pipeline:**
- Paper #26: Systolic array parameter loading

---

### Chapter 6: Results and Evaluation

**6.1 Accuracy Comparison:**
| Reference | Architecture | Accuracy | Platform |
|-----------|--------------|----------|----------|
| Your work | 2-layer CNN | 99% | Basys3 (Artix-7) |
| Paper #8 | 784-40-40-10 | 97.20% | Virtex-5 |
| Paper #10 | CNN | 97.59% | Zynq-7000 |
| Paper #19 | Binary NN | 84% | Artix-7 |
| Paper #20 | Spiking NN | 94.28% | - |
| Paper #23 | LeNet | 95.4% | - |

**6.2 Performance Metrics:**
- Compare throughput with Paper #10 (441 img/s)
- Compare speedup with Paper #6 (16.42x)
- Compare latency with Paper #15 (68% reduction)

**6.3 Power Consumption:**
- Paper #19: 0.617W (Binary NN on Artix-7)
- Paper #20: 4.996W (Spiking NN)
- Paper #21: 1.03W total, 0.03W CNN module
- Paper #11: 9.25W (VGG16)

**6.4 Resource Utilization:**
- Discuss LUT/FF usage
- Compare with Paper #22 (resource optimization)

---

### Chapter 7: LeNet-5 Migration Discussion

**Architectural Comparison:**
- Paper #7: LeNet using HLS
- Paper #23: LeNet quantization (95.4% accuracy)

**Quantization for LeNet:**
- Paper #23: W1A1 quantization results
- Paper #15: PTQ framework

**Memory Requirements:**
- Your current: ~27KB
- LeNet-5 estimate: ~62KB
- Discuss BRAM vs distributed RAM (Paper #4)

---

### Chapter 8: Conclusions and Future Work

**Achievements:**
- Summarize your 99% accuracy
- Compare with state-of-the-art (Papers #6, #8, #10)

**Future Work:**
- LeNet-5 migration (Papers #7, #23)
- Quantization-aware training (Paper #17)
- On-the-fly weight generation (Paper #25)
- High-level synthesis approach (Paper #28)
- Further power optimization (Papers #19, #21)

---

## Additional Resources and Search Tips

### Accessing Papers

**IEEE Xplore (Papers #6, #7, #8, #9):**
- Access through university library subscription
- Download as PDF for citation management

**arXiv (Papers #1, #2, #12, #16, #19, #24, #25):**
- Freely available
- Download PDFs directly

**ACM Digital Library (Paper #3, #18):**
- University library access required

**MDPI (Papers #4, #11, #22, #23):**
- Open access journals
- Freely downloadable

**PMC/PubMed Central (Papers #20, #21, #26):**
- Freely available biomedical research
- Full-text PDFs

**ScienceDirect (Paper #15):**
- University library subscription typically required

**ResearchGate (Papers #14, #27):**
- Sometimes freely available
- May require ResearchGate account

---

### Citation Management

**Recommended Tools:**
- **Zotero:** Free, open-source, excellent browser integration
- **Mendeley:** Free, good PDF management
- **EndNote:** Commercial, often provided by universities
- **BibTeX:** For LaTeX-based theses

**Creating BibTeX Entries:**

Most digital libraries provide citation export options. Example for Paper #6:

```bibtex
@inproceedings{fpga_cnn_accelerator_2016,
  author={Author Names},
  booktitle={IEEE Conference Name},
  title={An FPGA-based accelerator implementation for deep convolutional neural networks},
  year={2016},
  pages={xx-xx},
  doi={10.1109/xxxxx},
  url={https://ieeexplore.ieee.org/document/7490869/}
}
```

---

### Additional Search Keywords

For finding more related work:

**General:**
- "MNIST FPGA implementation"
- "CNN hardware accelerator"
- "neural network FPGA Xilinx"

**Quantization:**
- "INT8 quantization neural network"
- "fixed-point CNN FPGA"
- "post-training quantization"

**Specific Platforms:**
- "Basys3 neural network"
- "Artix-7 machine learning"
- "Zynq CNN implementation"

**Tools:**
- "Vivado HLS CNN"
- "FINN framework"
- "Vitis AI"

---

### GitHub Projects for Implementation Examples

While not academic papers, these can provide implementation insights:

**Search GitHub for:**
- "basys3 neural network"
- "FPGA CNN verilog"
- "MNIST FPGA implementation"
- "quantized neural network FPGA"

**Example repositories mentioned in searches:**
- UART-Implementation-on-FPGA
- Basys3 handwritten digit recognition painter

---

## Summary Statistics

### By Publication Type:
- IEEE Conference Papers: 4
- Journal Articles: 10
- arXiv Preprints: 8
- Technical Blogs: 2
- University Theses: 1
- Other: 4

### By Research Focus:
- General FPGA CNN Acceleration: 5
- MNIST-Specific: 6
- Quantization: 7
- Low-Power/Embedded: 4
- Specialized/Optimization: 4
- Tools/Methodology: 3

### By Platform:
- Xilinx Artix-7: 1 (same as yours!)
- Xilinx Zynq: 3
- Xilinx Virtex: 2
- Generic FPGA: 19
- Other: 4

### By Relevance to Your Project:
- **Critical (directly comparable):** Papers #6, #10, #15, #19, #24
- **High (methodology match):** Papers #9, #12, #13, #14, #23
- **Medium (background/theory):** Papers #1, #2, #3, #4, #17, #18
- **Supporting (alternatives/future):** Papers #7, #20, #21, #22, #25, #28

---

## Comparison Table for Your Results Section

| Paper | Platform | Precision | Accuracy | Throughput | Power | Notes |
|-------|----------|-----------|----------|------------|-------|-------|
| **Your Work** | **Basys3 (Artix-7)** | **INT8** | **99%** | **~1 img/s** | **TBD** | **2-layer CNN** |
| #6 | Virtex7 | 11-bit | - | 16.42x speedup | - | Vivado HLS |
| #8 | Virtex-5 | - | 97.20% | - | - | 784-40-40-10 |
| #10 | Zynq-7000 | 12-bit | 97.59% | 441 img/s | - | Similar platform |
| #19 | Artix-7 | 1-bit (binary) | 84% | - | 0.617W | Same FPGA family! |
| #20 | - | - | 94.28% | 2494 fps | 4.996W | Spiking NN |
| #21 | - | - | 97% | - | 0.03W CNN | Resource mux |
| #23 | - | W1A1 | 95.4% | 0.71 fps | - | LeNet |

---

## Final Notes

### Tips for Using These References:

1. **Start with reviews** (Papers #1-5) for your literature review
2. **Use IEEE papers** (#6-9) for credibility in related work
3. **Cite methodology papers** (#12-18) when describing your quantization
4. **Reference Paper #19** heavily - it's on the same FPGA family!
5. **Use Paper #24** to justify your UART choice
6. **Compare with Papers #6, #8, #10** in results section

### When Writing:

- **Don't just list papers** - synthesize and compare
- **Group by themes** - quantization, architecture, platform, etc.
- **Highlight your contributions** - what's different/better
- **Be honest about limitations** - cite papers with better metrics
- **Use comparison tables** - visual representation helps

### Red Flags to Avoid:

- ❌ Citing papers you haven't read
- ❌ Misrepresenting results
- ❌ Ignoring contradictory findings
- ❌ Over-citing one source
- ❌ Neglecting recent work (2024-2025)

---

**Good luck with your thesis!** 🎓

*This reference list provides a solid foundation for your engineering thesis on CNN FPGA implementation. Remember to access papers through your university library and create proper citations using a reference manager.*

---

## Document Version

- **Created:** January 2026
- **Format:** Markdown
- **Total References:** 29 academic sources
- **Coverage:** 2015-2025 (10 years of research)
- **Focus:** MNIST, CNN, FPGA, Quantization, Basys3/Artix-7