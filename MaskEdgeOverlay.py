import numpy as np
import cv2
import torch

class MaskEdgeOverlay:
    """
    输入：
        image: 原始图像 (numpy.ndarray 或 torch.Tensor，支持批量 [B,H,W,3]，范围0..1)
        mask: 遮罩图像 (numpy.ndarray 或 torch.Tensor，支持批量 [B,H,W] 或 [B,H,W,1]，范围0..1)
        edge_color: 边缘颜色 (HEX字符串, 如'#fb0404')
        edge_width: 边缘宽度 (int, 单位像素)
        mask_dilate: 遮罩外扩 (int, 单位像素)
        edge_style: 边缘样式（'solid'/'dashed'，下拉菜单）
    输出：
        result: 边缘叠加效果图 (torch.Tensor，形状 [B,H,W,3]，范围0..1)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("MASK", {}),
                "edge_color": ("STRING", {"default": "#ff0000"}),
                "edge_width": ("INT", {"default": 3, "min": 1, "max": 50}),
                "mask_dilate": ("INT", {"default": 0, "min": 0, "max": 50}),
                "edge_style": (["实线", "虚线"], {"default": "实线"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "overlay_edge"
    CATEGORY = "image/overlay"  # 实际分组由__init__.py动态设置

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError('颜色格式错误，需为#RRGGBB')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)

    def _to_numpy_image_batch(self, image):
        """将ComfyUI IMAGE输入标准化为numpy批量数组 [B,H,W,3]，uint8 0..255。"""
        if isinstance(image, dict) and "image" in image:
            image = image["image"]
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            image_np = np.array(image)

        # 允许 [H,W,3] 或 [B,H,W,3]
        if image_np.ndim == 3:
            image_np = image_np[None, ...]  # [1,H,W,3]
        elif image_np.ndim != 4:
            raise ValueError("image的维度应为[H,W,3]或[B,H,W,3]")

        # 归一化到uint8
        if image_np.dtype != np.uint8:
            max_val = float(image_np.max()) if image_np.size > 0 else 1.0
            if max_val <= 1.0:
                image_np = (np.clip(image_np, 0, 1) * 255.0).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        return image_np

    def _normalize_mask_batch(self, mask, expected_batch, expected_h, expected_w):
        """将MASK标准化为 [B,H,W] 的uint8 0..255。
        尽量对各种可能形状进行健壮处理。
        """
        if isinstance(mask, dict) and "mask" in mask:
            mask = mask["mask"]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = np.array(mask)

        # 统一为float或uint8的numpy
        if mask_np.ndim == 2:
            # [H,W]
            mask_np = mask_np[None, ...]  # [1,H,W]
        elif mask_np.ndim == 3:
            # 可能是 [B,H,W] 或 [H,W,1]
            if mask_np.shape[0] == expected_batch and mask_np.shape[1] == expected_h and mask_np.shape[2] == expected_w:
                # [B,H,W]
                pass
            elif mask_np.shape[0] == expected_h and mask_np.shape[1] == expected_w and mask_np.shape[2] == 1:
                # [H,W,1]
                mask_np = mask_np[..., 0][None, ...]  # -> [1,H,W]
            elif mask_np.shape[0] == 1 and mask_np.shape[1] == expected_h and mask_np.shape[2] == expected_w:
                # [1,H,W]
                pass
            else:
                # 尝试将其视为 [H,W,B] 这类错误维度，降维取首通道
                mask_np = mask_np[..., 0]
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, ...]
        elif mask_np.ndim == 4:
            # [B,H,W,1] 或其他，尽量压到 [B,H,W]
            if mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            else:
                mask_np = mask_np[..., 0]  # 取首通道
        else:
            raise ValueError("mask的维度应为[H,W]、[H,W,1]、[B,H,W]或[B,H,W,1]")

        # 若批量为1而图像为多张，进行广播
        if mask_np.shape[0] == 1 and expected_batch > 1:
            mask_np = np.repeat(mask_np, expected_batch, axis=0)
        elif mask_np.shape[0] != expected_batch:
            raise ValueError("image与mask的批量大小不一致")

        # 对齐尺寸（防御性，通常ComfyUI已对齐）
        if mask_np.shape[1] != expected_h or mask_np.shape[2] != expected_w:
            mask_resized = []
            for i in range(mask_np.shape[0]):
                mask_resized.append(cv2.resize(mask_np[i].astype(np.float32), (expected_w, expected_h), interpolation=cv2.INTER_NEAREST))
            mask_np = np.stack(mask_resized, axis=0)

        # 转为0/255的uint8二值
        if mask_np.dtype != np.uint8:
            mask_np = (mask_np > 0).astype(np.uint8) * 255
        return mask_np

    def _dilate(self, binary_img, ksize):
        if ksize <= 0:
            return binary_img
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(binary_img, kernel, iterations=1)

    def overlay_edge(self, image, mask, edge_color, edge_width, mask_dilate, edge_style):
        # 标准化输入为批量
        image_b = self._to_numpy_image_batch(image)  # [B,H,W,3], uint8
        B, H, W, _ = image_b.shape
        mask_b = self._normalize_mask_batch(mask, expected_batch=B, expected_h=H, expected_w=W)  # [B,H,W], uint8

        # 颜色：直接使用RGB，避免BGR混淆
        rgb_color = self.hex_to_rgb(edge_color)

        results = []
        for i in range(B):
            img = image_b[i].copy()           # [H,W,3] uint8 (RGB)
            mask_i = mask_b[i]               # [H,W] uint8 0/255

            # 遮罩外扩
            if mask_dilate > 0:
                mask_i = self._dilate(mask_i, mask_dilate)

            # 计算边缘并加粗
            edges = cv2.Canny(mask_i, 10, 50)
            if edge_width > 1:
                edges = self._dilate(edges, edge_width)

            if edge_style == "实线":
                edge_mask = edges > 0
                img[edge_mask] = rgb_color
            else:
                # 在单通道掩码上绘制虚线，再应用到图像
                dashed = np.zeros((H, W), dtype=np.uint8)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                dash_len = 8
                gap_len = 8
                for cnt in contours:
                    pts = cnt[:, 0, :]
                    cur_len = 0.0
                    for j in range(len(pts) - 1):
                        pt1 = tuple(int(v) for v in pts[j])
                        pt2 = tuple(int(v) for v in pts[j + 1])
                        seg_len = float(np.linalg.norm(np.array(pt2) - np.array(pt1)))
                        if int(cur_len // (dash_len + gap_len)) % 2 == 0:
                            cv2.line(dashed, pt1, pt2, 255, thickness=edge_width)
                        cur_len += seg_len
                img[dashed > 0] = rgb_color

            # 转回float32 0..1
            results.append(torch.from_numpy(img).float() / 255.0)

        # 组装为[B,H,W,3]
        result_tensor = torch.stack(results, dim=0)
        return (result_tensor,) 