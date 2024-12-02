# WeedsDetectNet
Accurate weed detection is crucial for precision agriculture and green crop protection. Existing methods often struggle with high recognition errors, particularly in complex environments where weeds share similar colors and shapes. This paper presents WeedsDetectNet, an improved YOLOv5n model for weed detection. WeedsDetectNet introduces a green attention module to enhance focus on green weed regions while suppressing non-target areas. An adaptive joint feature fusion method is proposed to combine low-level details such as weed color and texture with high-level semantic information, enabling better extraction of weed-specific features. Additionally, a decoupled head design utilizes dynamic attention to separately handle classification and localization tasks, improving detection accuracy. The model is evaluated on the CottonWeedDet12 and 4WEED DATASET, as well as a self-constructed dataset. Experimental results demonstrate that WeedsDetectNet outperforms existing methods, achieving higher mean average precision, lower misidentification rates, and more accurate bounding box regression. This lightweight model exhibits strong generalization and robustness, making it suitable for real-world weed detection tasks. 
