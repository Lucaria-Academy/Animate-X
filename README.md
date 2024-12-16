
<p align="center">

  <h2 align="center">Animate-X: Universal Character Image Animation with Enhanced Motion Representation</h2>
  <p align="center">
    <a href=""><strong>Shuai Tan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=BwdpTiQAAAAJ"><strong>Biao Gong</strong></a><sup>†</sup>
    ·
    <a href="https://scholar.google.com/citations?user=cQbXvkcAAAAJ"><strong>Xiang Wang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=ZO3OQ-8AAAAJ"><strong>Shiwei Zhang</strong></a>
    <br>
    <a href="https://openreview.net/profile?id=~DanDan_Zheng1"><strong>Dandan Zheng</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=S8FmqTUAAAAJ"><strong>Ruobing Zheng</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=hMDQifQAAAAJ"><strong>Kecheng Zheng</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Jingdong_Chen1"><strong>Jingdong Chen</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Ming_Yang2"><strong>Ming Yang</strong></a>            
    <br>
    <br>
        <a href="https://arxiv.org/abs/2410.10306"><img src='https://img.shields.io/badge/arXiv-Animate--X-red' alt='Paper PDF'></a>
        <a href='https://lucaria-academy.github.io/Animate-X/'><img src='https://img.shields.io/badge/Project_Page-Animate--X-blue' alt='Project Page'></a>
        <a href='https://mp.weixin.qq.com/s/vDR4kPLqnCUwfPiBNKKV9A'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
        <a href='https://huggingface.co/Shuaishuai0219/Animate-X'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <br>
    <b></a>Ant Group &nbsp; | &nbsp; </a>Tongyi Lab  </b>
    <br>
  </p>
</p>

This repository is the official implementation of paper "Animate-X: Universal Character Image Animation with Enhanced Motion Representation". Animate-X is a universal animation framework based on latent diffusion models for various character types (collectively named X), including anthropomorphic characters.
  <table align="center">
    <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/fb2f4396-341f-4206-8d70-44d8b034f810">
    </td>
    </tr>
  </table>


## &#x1F4CC; Updates
- **2024 12.16** 🔥 We release our inference codes at [Animate-X](https://github.com/antgroup/animate-x).
- **2024 11.25**: Thank you for your patience. Although the code is ready, due to the company's latest open-source policy, this project can only be open-sourced on https://github.com/antgroup, with an expected release in the spring of 2025. We sincerely apologize for the delay. Open-sourcing is **never easy**, and we kindly ask for your understanding.
- **2024 10.21**: Thank you all for your interest in Animate-X. The main reasons we haven't made the code public yet are: 1. The company needs to go through certain public approval processes, which take time. 2. We are currently cleaning up the code and preparing for open source in various aspects. We promise to release the code and the models by mid to late November. We appreciate your patience.
- **2024 10.1**: The project page and Arxiv paper of Animate-X are released.


## TODO List
- [x] Release model, checkpoint and demo code.



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
