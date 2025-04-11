 # 👨🏻‍🔬👩🏼‍🔬👨🏽‍🔬👩🏾‍🔬👨🏿‍🔬RouterKT: Mixture-of-Experts for Knowledge Tracing 

RouterKT is a novel knowledge tracing framework that leverages Mixture-of-Experts (MoE) architecture to enhance the modeling of student learning processes. This repository contains implementations of three variants: RouterAKT, RouterCL4KT, and RouterSimpleKT. 

**For detailed logs and experiment results, please refer to the [logs](./logs) folder.**


## Overview

Knowledge Tracing (KT) aims to model students' knowledge states and predict their future performance. RouterKT introduces a dynamic routing mechanism that adaptively selects the most relevant experts for different learning patterns, leading to more accurate and interpretable predictions.

### Key Features

- **Dynamic Expert Selection**: Implements a Mixture-of-Experts architecture with dynamic routing
- **Multiple Model Variants**:
  - RouterAKT: Advanced Knowledge Tracing with dynamic routing
  - RouterCL4KT: Contrastive Learning-based KT with expert routing
  - RouterSimpleKT: Simplified KT model with routing mechanism
- **Flexible Architecture**: Supports various configurations of attention heads and routing modes
- **Interpretable Predictions**: Provides insights into expert selection patterns

## Installation

```bash
# Clone the repository
git clone git@github.com:RingoTC/RouterKT.git
cd RouterKT

# Install dependencies
pip install -r requirements.txt
```

## Usage
```
python main.py --model_name routerakt --data_name algebra05
python main.py --model_name routerakt --data_name bridge06
python main.py --model_name routerakt --data_name assistments09
python main.py --model_name routerakt --data_name ednet
python main.py --model_name routerakt --data_name slepemapy
python main.py --model_name routerakt --data_name linux
python main.py --model_name routerakt --data_name database

python main.py --model_name routersimplekt --data_name algebra05
python main.py --model_name routersimplekt --data_name bridge06
python main.py --model_name routersimplekt --data_name assistments09
python main.py --model_name routersimplekt --data_name ednet
python main.py --model_name routersimplekt --data_name slepemapy
python main.py --model_name routersimplekt --data_name linux
python main.py --model_name routersimplekt --data_name database

python main.py --model_name routercl4kt --data_name algebra05
python main.py --model_name routercl4kt --data_name bridge06
python main.py --model_name routercl4kt --data_name assistments09
python main.py --model_name routercl4kt --data_name ednet
python main.py --model_name routercl4kt --data_name slepemapy
python main.py --model_name routercl4kt --data_name linux
python main.py --model_name routercl4kt --data_name database
```

## Model Architecture

RouterKT consists of three main components:

1. **Input Embedding Layer**: Encodes questions, skills, and responses
2. **Mixture-of-Experts Transformer**: 
   - Multiple expert networks
   - Dynamic routing mechanism
   - Shared and selected attention heads
3. **Prediction Layer**: Outputs probability of correct response

## Citation

If you use RouterKT in your research, please cite:

```bibtex
@article{your_paper,
  title={RouterKT: Mixture-of-Experts for Knowledge Tracing},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue or contact [your email].
