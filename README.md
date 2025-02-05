# Solving bivariate kinetic equations for polymer diffusion using deep learning

Authors: Heng Wang (220220934161@lzu.edu.cn) and Weihua Deng (dengwh@lzu.edu.cn)

Address: School of Mathematics and Statistics, Gansu Key Laboratory of Applied Mathematics and Complex Systems, Lanzhou University, Lanzhou 730000, China.

# Environment configuration

python==3.7.15 torch==1.13.1

# Introduction

We extend the deep BSDE method to a new class of binary semilinear parabolic partial differential equations. The equation can be represented as

$$
\frac{\partial}{\partial t}u(n,x,t) + \mathcal{T}_nu(n,x,t) + \frac{1}{2}Tr\left(\sigma\sigma^T(n,x,t)(Hess_x)u(n,x,t)\right) + \nabla u(n,x,t)\cdot \mu(n,x,t) + f\left(t,n,x,u(n,x,t),\left(\sigma^T\nabla u\right)(n,x,t)\right) = 0
$$

with some specified terminal condition $u(n,x,T) = g(n,x)$. Here $\mathcal{T}_n$ represents the operator with respect to the discrete variable $n$, defined as

$$
\mathcal{T}_nf(n) = 
    \begin{cases}
      \alpha(n)(f(n+1) - f(n)) + \beta(n)(f(n-1) - f(n)),& n\geq 1,\\
      \alpha(0)\left(f(1)-f(0)\right),& n=0.
    \end{cases}
$$

# How to use?
We provide an invocation example in the Python Script "./example.py", where you can define and modify any of the components in the sample for testing.

# Cite
@article{JML-3-2,<br>
&ensp;author = {Wang, Heng and Weihua, Deng},<br>
&ensp;title = {Solving Bivariate Kinetic Equations for Polymer Diffusion Using Deep Learning},<br>
&ensp;journal = {Journal of Machine Learning},<br>
&ensp;year = {2024},<br>
&ensp;volume = {3},<br>
&ensp;number = {2},<br>
&ensp;pages = {215--244},<br>
&ensp;issn = {2790-2048},<br>
&ensp;doi = {https://doi.org/10.4208/jml.240124}<br>
}