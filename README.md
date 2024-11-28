# 📘 Image Processing Course

This repository contains summarized notes and key concepts from the Image Processing course at Polytech Paris-Saclay, focusing on topics such as filtering, 3D reconstruction, and motion analysis.

---

## 🔍 Linear Filtering

- **Hough Transform**: Used for line detection with parameters \(r \in [0, r_{\text{max}}]\), \(\theta \in [0, 2\pi]\).  
- **Laplacian Operator**:  
  \[
  \Delta f = \nabla \cdot \nabla f
  \]
  Employed for edge detection and diffusion processes.

---

## 🧩 3D Point Representation

- **Euclidean Transformations**: \(p' = [R|t]p\) for \(R\) as rotation matrix and \(t\) as translation vector.  
- **Camera Model**: Central projection mapping 3D points to 2D.  
- **Stereo Vision**: Depth estimation via disparity:
  \[
  Z = \frac{fb}{d}
  \]

---

## 📐 Robust Estimation

Techniques to handle outliers in datasets:
- **Least Median of Squares (LMedS)**  
- **RANSAC**: Random sampling for model fitting.  

Applications include target tracking and system state estimation.

---

## 📊 Mathematical Morphology

- **Operations**:
  - **Dilation**: Expands object boundaries.
  - **Erosion**: Shrinks object boundaries.
  - **Opening**: Smooths object contours.
  - **Closing**: Fills small holes.  

---

## 🎥 Motion Analysis

- **Optical Flow**: Estimation of motion between frames.  
- **Applications**:
  - Dense pixel tracking.
  - 2D to 3D motion estimation.

---

## 📚 Resources

- Recommended reading:
  - Z. Zhang, "A flexible new technique for camera calibration" (2000)
  - M. Pollefeys et al., "Self-calibration and metric reconstruction" (1999)
