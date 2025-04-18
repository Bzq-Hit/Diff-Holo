# Diff-Holo: A Residual Diffusion Model with Complex Transformer for Rapid Single-Frame Hologram Reconstruction (IEEE TCI 2024)

Ziqi Bai, [Xianming Liu](https://homepage.hit.edu.cn/xmliu), [Cheng Guo](https://scholar.google.com.hk/citations?hl=zh-CN&user=D_jtz9sAAAAJ&view_op=list_works), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui?lang=zh), [Junjun Jiang](https://homepage.hit.edu.cn/jiangjunjun?lang=zh), [Xiangyang Ji](https://www.au.tsinghua.edu.cn/info/1111/1524.htm)

---

Paper link: https://ieeexplore.ieee.org/document/10966195

---

*Deep learning approaches have gained significant traction in holographic imaging, with diffusion models—an emerging class of deep generative models—showing particular promise in hologram reconstruction. Unlike conventional neural networks that directly generate outputs, diffusion models gradually add noise to data and train neural networks to remove it, enabling them to learn implicit priors of the underlying data distribution. However, current diffusion-based hologram reconstruction methods often require hundreds or even thousands of iterations to achieve high-fidelity results, leading to processing times of several minutes or more—falling short of the fast imaging demands of holographic systems. To address this, we propose **Diff-Holo**, a residual diffusion model integrated with a complex transformer, designed for rapid and high-quality single-frame hologram reconstruction. Specifically, we create a shorter and more efficient Markov chain by controlling the residuals between clean images and those degraded by twin-image artifacts. Additionally, we incorporate complex-valued priors into the network by using a complex window-based transformer as the backbone, enhancing the network's ability to process complex-valued data in the reverse reconstruction process. Experimental results demonstrate that Diff-Holo achieves high-quality single-frame reconstructions in as few as 15 sampling steps, reducing reconstruction time from minutes to under 2.2 seconds.*

![Image text](https://github.com/Bzq-Hit/Diff-Holo/blob/main/fig/fig1.png)

---

## To-Do List

We are currently organizing and preparing to open-source the code related to Diff-Holo's training, algorithm architecture, and other engineering aspects.

---

## Contact

If you have any questions or suggestions regarding this project or the research paper, please feel free to contact the author, Ziqi Bai, at 21B951029@stu.hit.edu.cn.

---
