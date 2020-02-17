# Adaptive-Low-Rank-Tensor-Representation

Matlab implementation of TNNLS2019 paper: "Accurate Tensor Completion via Adaptive Low-Rank Representation"

## Dependencies
  - Matlab 2017
  - tensor_toolbox 2.6
## Preparation
1. We provide the library tensor_toolbox_2.6, please decompress the file 'tensor_toolbox_2.6.tar.gz' using the following command line
```
cd ./Adaptive-Low-Rank-Tensor-Representation/
tar -zxvf tensor_toolbox_2.6.tar.gz
```
2. Run the matlab file './script/CompileFile.m' to mex compile the CPP file in the sampling method
### Usage
Run the matlab file 'main_Inpaiting.m' to see the demo on image inpainting.
  
 ### Reference
If you find our work useful in your research or publication, please cite our work:<br>
[1] Lei Zhang, Wei Wei, Qinfeng Shi, Chunhua Shen, Anton van den Hengel, and Yanning Zhang. "Accurate Tensor Completion via Adaptive Low-Rank Representation." IEEE Transactions on Neural Networks and Learning Systems (2019).</i>[[PDF](https://ieeexplore.ieee.org/abstract/document/8945165)]
```
@article{zhang2019accurate,
  title={Accurate Tensor Completion via Adaptive Low-Rank Representation},
  author={Zhang, Lei and Wei, Wei and Shi, Qinfeng and Shen, Chunhua and van den Hengel, Anton and Zhang, Yanning},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2019},
  publisher={IEEE}
}

```

