\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{listings}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single
}

\title{\textbf{Avatar Creation}}
\author{Portrait to Cartoon Transformation}
\date{}

\begin{document}
\maketitle

\section*{Overview}
Transform portrait photos into cartoon avatars using OpenCV. Fast, CPU-only, no training required.

\section*{Features}
\begin{itemize}
    \item K-means color quantization (9 colors)
    \item Adaptive threshold edge detection
    \item Bilateral filtering for smoothing
    \item 50-100ms processing time
    \item No GPU required
\end{itemize}

\section*{Installation}
\begin{lstlisting}
pip install opencv-python numpy
\end{lstlisting}

\section*{Usage}
\begin{lstlisting}[language=Python]
import cv2
import numpy as np

def reduce_colors(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS + 
                cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, _, center = cv2.kmeans(data, k, None, criteria, 
                              10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[_.flatten()].reshape(img.shape)
    return result

def get_edges(img, line_size, blur_val):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_val)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, line_size, blur

