# WBC Classification

This repository contains a **simple baseline** for white blood cell (WBC) image classification using
**Histogram of Oriented Gradients (HoG)** features + **template matching by MSE**.

The pipeline:

1. **Preprocess** each image (center crop → grayscale → Sobel magnitude).
2. **Extract HoG** features.
3. **Build class templates** by averaging HoG vectors over the *Train* set.
4. **Classify** a test image by comparing its HoG to every class template (lowest **MSE** wins).
5. **Evaluate** on a held-out *Test* set and plot the **confusion matrix**.

---

## ✨ Classes

`Basophil`, `Eosinophil`, `Lymphocyte`, `Monocyte`, `Neutrophil`
