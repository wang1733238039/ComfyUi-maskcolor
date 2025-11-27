from .MaskEdgeOverlay import MaskEdgeOverlay

# 统一分组到'WZF'
MaskEdgeOverlay.CATEGORY = "WZF"

NODE_CLASS_MAPPINGS = {
    "MaskEdgeOverlay": MaskEdgeOverlay
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskEdgeOverlay": "遮罩边缘叠加"
} 