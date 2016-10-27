# GPU based Design Space Exploration for Image Convolution

This project realizes a 2D image gaussian blurring algorithm on GPU. As a commonly-used image processing algorithm, gaussian blurring can be implemented onto GPU. This project aims to explore the various optimizations available on GPU to achieve high throughput for gaussian blurring algorithm. 


**Kernel 1: Image Seperation**

Each pixel in the RGB image will be divided into three different values. Each value presents the one of the color components of the pixel, i.e., Red (R), Green (G), and Blue (B). These color components can be retrieved as follows:

Red=Image[Pixel index].x; 

Green=Image[Pixel index].y; 

Blue=Image[Pixel index].z;


**Kernel 2: Filtering**

After separating the image into the color components, the next step is to blur the image by multiplying it by the corresponding elements in the filter array. A square array of weight values will be used. 


**Optimizations**

1. shared memory

2. data reuse and locality

3. Additional algorithmic and architectural optimizations


**Conclusion**

By identifying and utilizing suitable optimizations on GPUs as well as algorithm-specific optimizations, we can significantly reduce the amount of execution time required to perform the 2D image gaussian blurring algorithm. Using optimizations including memory types and block size, we can achieve up to 2.65× improvement on the execution time of the algorithm. 


 
