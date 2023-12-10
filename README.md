# CLIP-dissect Reproduction and Activation Image

This repository is a reproduction and modification of the [Trustworthy Machine Learning Lab's CLIP-dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect). The primary modification is in the activation image part of the project.

This is a repository to reproduce CLIP-dissect and try to further analyze its functionality, specifically its activating images k. 

To analyze rank stability in a way that is meaningful, you should compare the entire ordered list of top-5 activations to the first 5 in the top-10, and similarly compare the first 5 in the top-10 to the first 5 in the top-16. This way, you're checking if any new labels have emerged in the higher ranked positions that were not present in the smaller set. In this updated version, you compare the set of top-5 labels from the top-5 activations to the first 5 labels from the top-10 and top-16 activations, which allows you to observe if any labels have been replaced by others as k increases, indicating a change in the neuron's preference ranking for different concepts.

### Hardware Set up

CPU: i5-13400F (16CPU), ~ 2.5GHz

RAM: 16GB

GPU: GTX3050

## Setup

### Prerequisites

- **Python 3.10**: Download from [Python's official website](https://www.python.org/downloads/).
- **PyTorch 1.12.0 or 2.0** and **Torchvision >= 0.13**: Follow the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Trustworthy-ML-Lab/CLIP-dissect
   
### Install Python Dependencies
To install the necessary Python dependencies, run the following command:
  ```bash
  pip install -r requirements.txt
  ```

Download the Broden dataset (images only):
  ```bash
  bash dlbroden.sh
  ```
## Citations

- **CLIP by OpenAI**: [GitHub Repository](https://github.com/openai/CLIP)
- **Broden Dataset Download Script**: Based on [NetDissect-Lite by CSAILVision](https://github.com/CSAILVision/NetDissect-Lite)
- **Original CLIP-dissect Project by the Trustworthy Machine Learning Lab**: [GitHub Repository](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)
