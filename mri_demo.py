import numpy as np
import matplotlib.pyplot as plt

# 1) 生成 64x64 的二维图像，并在中心放一个亮方块
img = np.zeros((64, 64), dtype=float)
img[24:40, 24:40] = 1.0

# 2) 对图像做二维傅立叶变换，得到 k-space
k_space = np.fft.fftshift(np.fft.fft2(img))

# 3) 从 k-space 做逆变换重建图像
recon = np.fft.ifft2(np.fft.ifftshift(k_space)).real

# 4) 显示原始图像、k-space 对数幅度图、重建图像
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(np.log1p(np.abs(k_space)), cmap="gray")
axes[1].set_title("k-space (log magnitude)")
axes[1].axis("off")

axes[2].imshow(recon, cmap="gray")
axes[2].set_title("Reconstructed Image")
axes[2].axis("off")

plt.tight_layout()
plt.show()
