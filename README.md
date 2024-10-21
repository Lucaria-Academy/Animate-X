<p align="center">

  <h2 align="center">Animate-X: Universal Character Image Animation with Enhanced Motion Representation</h2>
  <p align="center">
    <a href=""><strong>Shuai Tan</strong></a> · <a href="https://scholar.google.com/citations?user=BwdpTiQAAAAJ"><strong>Biao Gong</strong></a> · <a href="https://scholar.google.com.hk/citations?user=cQbXvkcAAAAJ&hl=zh-CN&oi=sra"><strong>Xiang Wang</strong></a> · <a href="https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ&hl=zh-CN&oi=sra"><strong>Shiwei Zhang</strong></a>
    <br>
    <a href="#"><strong>Dandan Zheng</strong></a> · <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=S8FmqTUAAAAJ&view_op=list_works&sortby=pubdate"><strong>Ruobing Zheng</strong></a> · <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=hMDQifQAAAAJ&view_op=list_works&sortby=pubdate"><strong>Kecheng Zheng</strong></a> · <a href="#"><strong>Jingdong Chen</strong></a> · <a href="#"><strong>Ming Yang</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2311.17002"><img src='https://img.shields.io/badge/arXiv-Animate_X-red' alt='Paper PDF'></a>
        <a href='https://lucaria-academy.github.io/Animate-X/'><img src='https://img.shields.io/badge/Project_Page-Animate_X-blue' alt='Project Page'></a>
    <br>
    <b> Ant Group&nbsp; | &nbsp; Alibaba Group  </b>
  </p>

  <!-- This repository is the official implementation of CVPR 2024 paper "Ranni: Taming Text-to-Image Diffusion for Accurate Instruction Following". It contains two main components: 1) a LLM-based planning model that maps text instructions into visual elements in image, 2) a diffusion-based painting model that draws image following the visual elements in first stage. Ranni achieves better semantic understanding thanks to the powerful ability of LLM. Currently, we release the model weights including a LoRA-finetuned LLaMa-2-7B, and a fully-finetuned SDv2.1 model. -->
  
  <table align="center">
    <tr>
    <td>
      <img src="https://img.alicdn.com/imgextra/i3/O1CN01QZs1bU1LoW68dhIb2_!!6000000001346-0-tps-1783-856.jpg">
    </td>
    </tr>
  </table>

## News
- **2024 10.21**: Thank you all for your interest in Animate-X. The main reasons we haven't made the code public yet are: 1. The company needs to go through certain public approval processes, which take time. 2. We are currently cleaning up the code and preparing for open source in various aspects. We promise to release the code and the models by mid to late November. We appreciate your patience.


## TODO List
- [ ] Release model, checkpoint and demo code.



<!-- ## Installation
Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate ranni
```


## Download Checkpoints
Download Ranni [checkpoints](https://modelscope.cn/models/yutong/Ranni/files) and put all files in `model` dir, which should be like:
```
models/
  llama2_7b_lora_bbox.pth
  llama2_7b_lora_element.pth
  ranni_sdv21_v1.pth
```

## Gradio demo 
We present the interactivate image generation by running the gradio demo:

```bash
python demo_gradio.py
```

It should look like the UI shown below:

<table align="center">
  <tr>
  <td>
    <img src="assets/Figures/Gradio.png">
  </td>
  </tr>
</table>

### Tutorial for image generation
Simply type in the image prompt. Click the button `text-to-panel` for generate semantic panel, then click the button `panel-to-image` for generate corresponding image:

> prompt: A black dog and a white cat
<table align="center">
  <tr>
  <td>
    <img src="assets/Figures/demo_gradio_generation.png">
  </td>
  </tr>
</table>


### Tutorial for continuous editing
After generating an image, you could modify the box answer to adjust the panel (modify the prompt if needed). Click button `refresh` to refresh the condition. Enable the checkbox `with memory` after the `panel-to-image`, then generate the modified image:

> prompt: A black dog and a white cat
> modification: black dog -> white dog

<table align="center">
  <tr>
  <td>
    <img src="assets/Figures/demo_gradio_editing.png">
  </td>
  </tr>
</table>

By operating on the boxes and prompts, you could achieve multiple editing operations in following types:
<table align="center">
  <tr>
  <td>
    <img src="assets/Figures/demo_gradio_ops.png">
  </td>
  </tr>
</table> -->

<!-- ## Acknowledgement
This repository is based on the following codebases:
* https://github.com/Stability-AI/stablediffusion
* https://github.com/lllyasviel/ControlNet/ -->

## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{tan2024animate-x,
  title={Animate-X: Universal Character Image Animation with Enhanced Motion Representation},
  author={Tan, Shuai and Gong, Biao and Wang, Xiang and Zhang, Shiwei and Zheng, Dandan and Zheng, Ruobin and Zheng, Kecheng and Chen, Jingdong and Yang, Ming},
  journal={arXiv preprint arXiv:2410.10306},
  year={2024}
}
```
