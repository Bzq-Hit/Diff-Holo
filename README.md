# Diff-Holo: A Residual Diffusion Model with Complex Transformer for Rapid Single-Frame Hologram Reconstruction (IEEE TCI 2025)

Ziqi Bai, [Xianming Liu](https://homepage.hit.edu.cn/xmliu), [Cheng Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=D_jtz9sAAAAJ&view_op=list_works), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Junjun Jiang](https://homepage.hit.edu.cn/jiangjunjun?lang=zh), [Xiangyang Ji](https://www.au.tsinghua.edu.cn/info/1111/1524.htm)

---

Paper link: https://ieeexplore.ieee.org/document/10966195

accepted by IEEE Transactions on Computational Imaging (IEEE TCI)

---

*Deep learning approaches have gained significant traction in holographic imaging, with diffusion models—an emerging class of deep generative models—showing particular promise in hologram reconstruction. Unlike conventional neural networks that directly generate outputs, diffusion models gradually add noise to data and train neural networks to remove it, enabling them to learn implicit priors of the underlying data distribution. However, current diffusion-based hologram reconstruction methods often require hundreds or even thousands of iterations to achieve high-fidelity results, leading to processing times of several minutes or more—falling short of the fast imaging demands of holographic systems. To address this, we propose **Diff-Holo**, a residual diffusion model integrated with a complex transformer, designed for rapid and high-quality single-frame hologram reconstruction. Specifically, we create a shorter and more efficient Markov chain by controlling the residuals between clean images and those degraded by twin-image artifacts. Additionally, we incorporate complex-valued priors into the network by using a complex window-based transformer as the backbone, enhancing the network's ability to process complex-valued data in the reverse reconstruction process. Experimental results demonstrate that Diff-Holo achieves high-quality single-frame reconstructions in as few as 15 sampling steps, reducing reconstruction time from minutes to under 2.2 seconds.*

![Image text](https://github.com/Bzq-Hit/Diff-Holo/blob/main/fig/fig1.png)

---

## Dependencies

For dependencies, you can install them by

```
pip install -r requirements.txt
```

---

## Data

To train Diff-Holo, you need to prepare your own hologram dataset, as we do not open-source the dataset used in this research paper. For detailed information about the dataset, please refer to Section Ⅳ.A of the research paper.

Once you have your dataset ready, you can choose to use our provided dataloader by placing the path of your dataset in the appropriate location in `holo.yaml`. This allows you to start training Diff-Holo from scratch. Of course, you are also free to develop your own dataloader.

---

## Training

To train Diff-Holo, you can begin the training by:

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 main.py --cfg_path configs/holo.yaml --save_dir [Your Logging Folder]
```


---

## Citation

If you find HoloFormer useful in your research, please consider citing:

```
@ARTICLE{10966195,
  author={Bai, Ziqi and Liu, Xianming and Guo, Cheng and Jiang, Kui and Jiang, Junjun and Ji, Xiangyang},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Diff-Holo: A Residual Diffusion Model With Complex Transformer for Rapid Single-Frame Hologram Reconstruction}, 
  year={2025},
  volume={11},
  number={},
  pages={689-703},
  keywords={Image reconstruction;Diffusion models;Transformers;Imaging;Holography;Training;Iterative methods;Reconstruction algorithms;Deep learning;Loss measurement;Single-frame hologram reconstruction;residual diffusion model;transformer;complex-valued deep neural network;rapid imaging},
  doi={10.1109/TCI.2025.3561683}}

```

---

## Contact

If you have any questions or suggestions regarding this project or the research paper, please feel free to contact the author, Ziqi Bai, at 21B951029@stu.hit.edu.cn.

---
