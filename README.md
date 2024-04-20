# CRNet

PyTorch implementation of CRNet. Our model achievied third place in track 1 of the Bracketing Image Restoration and Enhancement Challenge.
## 1. Abstract

It is challenging but highly desired to acquire high-quality photos with clear content in low-light environments. Although multi-image processing methods (using burst, dual-exposure, or multi-exposure images) have made significant progress in addressing this issue, they typically focus exclusively on specific restoration or enhancement tasks, being insufficient in exploiting multi-image. Motivated by that multi-exposure images are complementary in denoising, deblurring, high dynamic range imaging, and super-resolution, we propose to utilize bracketing photography to unify restoration and enhancement tasks in this work. Due to the difficulty in collecting real-world pairs, we suggest a solution that first pre-trains the model with synthetic paired data and then adapts it to real-world unlabeled images. In particular, a temporally modulated recurrent network (TMRNet) and self-supervised adaptation method are proposed. Moreover, we construct a data simulation pipeline to synthesize pairs and collect real-world images from 200 nighttime scenarios. Experiments on both datasets show that our method performs favorably against the state-of-the-art multi-image processing ones.

## 2. Overview of CRNet

<p align="center"><img src="imgs/Overview of CRNet.png" width="95%"></p>

## 3. Comparison of other methods in track 1 of the Bracketing Image Restoration and Enhancement Challenge.

<p align="center"><img src="imgs/multi_pro.png" width="95%"></p>

## 4. Expample Result

<p align="center"><img src="imgs/out1.png" width="95%"></p>
<p align="center"><img src="imgs/out2.png" width="95%"></p>

## 5. Checkpoint

https://pan.baidu.com/s/10GnNm1XJ9e6srXXnjctTyQ?pwd=ra81 
