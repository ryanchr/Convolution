

#include "reference_calc.cpp"
#include "utils.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y BLOCKSIZE_X

__constant__ float myFilter[9*9];


__global__
void gaussian_blur2(const uchar4* const inputChannel,
                   uchar4* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
   // Copied from recombineChannels
   const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
   
   // Temp vars for pixels
   int2 pixel_2D_pos;
   int pixel_1D_pos;
   
   int rowFilter=0; // x coordinate of the position we're working on
   int colFilter=0; // y coordinate of the position we're working on
   float tempOutRed = 0.0f; // Temp result
   float tempOutGreen = 0.0f; // Temp result
   float tempOutBlue = 0.0f; // Temp result
   int filter_pos; // Used to find the filter weight
   int filterD2 = filterWidth/2;  // Save calculations later
   
   int maxColClamp =numCols-1;
   int maxRowClamp =numRows-1;
   
   if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows ||
      thread_2D_pos.x < 0 || thread_2D_pos.y < 0)
      return;
   
   for (rowFilter=-filterD2;rowFilter<=filterD2;rowFilter++)
   {
      for(colFilter=-filterD2;colFilter<=filterD2;colFilter++)
      {
         filter_pos=(filterD2+rowFilter)*filterWidth+filterD2+colFilter;
         pixel_2D_pos=make_int2(max(min(thread_2D_pos.x+colFilter,maxColClamp),0),
                                max(min(thread_2D_pos.y+rowFilter,maxRowClamp),0));
         pixel_1D_pos=pixel_2D_pos.y*numCols+pixel_2D_pos.x;
         
         tempOutRed+=static_cast<float>(inputChannel[pixel_1D_pos].x)*myFilter[filter_pos];
         tempOutGreen+=static_cast<float>(inputChannel[pixel_1D_pos].y)*myFilter[filter_pos];
         tempOutBlue+=static_cast<float>(inputChannel[pixel_1D_pos].z)*myFilter[filter_pos];
      }
   }
   
   uchar4 outPixel = make_uchar4(static_cast<unsigned char>(tempOutRed),
                                 static_cast<unsigned char>(tempOutGreen),
                                 static_cast<unsigned char>(tempOutBlue),
                                 255);
   outputChannel[thread_1D_pos]=outPixel;
}



__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
   // TODO
   
   // NOTE: Be sure to compute any intermediate results in floating point
   // before storing the final result as unsigned char.
   
   // NOTE: Be careful not to try to access memory that is outside the bounds of
   // the image. You'll want code that performs the following check before accessing
   // GPU memory:
   //
   // if ( absolute_image_position_x >= numCols ||
   //      absolute_image_position_y >= numRows )
   // {
   //     return;
   // }
   
   // Copied from recombineChannels
   const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
   
   // Temp vars for pixels
   int2 pixel_2D_pos;
   int pixel_1D_pos;
   
   int rowFilter=0; // x coordinate of the position we're working on
   int colFilter=0; // y coordinate of the position we're working on
   float tempOut = 0.0f; // Temp result
   int filter_pos; // Used to find the filter weight
   int filterD2 = filterWidth/2;  // Save calculations later
   
   if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows ||
      thread_2D_pos.x < 0 || thread_2D_pos.y < 0)
      return;
   
   for (rowFilter=-filterD2;rowFilter<=filterD2;rowFilter++)
   {
      for(colFilter=-filterD2;colFilter<=filterD2;colFilter++)
      {
         filter_pos=(filterD2+rowFilter)*filterWidth+filterD2+colFilter;
         pixel_2D_pos=make_int2(max(min(thread_2D_pos.x+colFilter,numCols-1),0),
                                max(min(thread_2D_pos.y+rowFilter,numRows-1),0));
         pixel_1D_pos=pixel_2D_pos.y*numCols+pixel_2D_pos.x;
         
         tempOut+=static_cast<float>(inputChannel[pixel_1D_pos])*myFilter[filter_pos];
      }
   }
   
   outputChannel[thread_1D_pos]=static_cast<unsigned char>(tempOut);
   
   // NOTE: If a thread's absolute position 2D position is within the image, but some of
   // its neighbors are outside the image, then you will need to be extra careful. Instead
   // of trying to read such a neighbor value from GPU memory (which won't work because
   // the value is out of bounds), you should explicitly clamp the neighbor values you read
   // to be within the bounds of the image. If this is not clear to you, then please refer
   // to sequential reference solution for the exact clamping semantics you should follow.
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
   // DONE
   //
   // NOTE: Be careful not to try to access memory that is outside the bounds of
   // the image. You'll want code that performs the following check before accessing
   // GPU memory:
   //
   // if ( absolute_image_position_x >= numCols ||
   //      absolute_image_position_y >= numRows )
   // {
   //     return;
   // }
   
   // Copied from recombineChannels
   const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
   
   //  Check boundaries
   if(thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows ||
      thread_2D_pos.x < 0 || thread_2D_pos.y < 0)
      return;
   
   // Read input only once instead of three times
   uchar4 temp = inputImageRGBA[thread_1D_pos];
   blueChannel[thread_1D_pos] = temp.z;
   greenChannel[thread_1D_pos] = temp.y;
   redChannel[thread_1D_pos] = temp.x;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
   const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
   
   const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
   
   //make sure we don't try and access memory outside the image
   //by having any threads mapped there return early
   if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;
   
   unsigned char red   = redChannel[thread_1D_pos];
   unsigned char green = greenChannel[thread_1D_pos];
   unsigned char blue  = blueChannel[thread_1D_pos];
   
   //Alpha should be 255 for no transparency
   uchar4 outputPixel = make_uchar4(red, green, blue, 255);
   
   outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
   
   //allocate memory for the three different channels
   //original
//   checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
//   checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
//   checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
   
   //DONE:
   //Allocate memory for the filter on the GPU
   //Use the pointer d_filter that we have already declared for you
   //You need to allocate memory for the filter with cudaMalloc
   //be sure to use checkCudaErrors like the above examples to
   //be able to tell if anything goes wrong
   //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
   //   checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));
   
   //DONE:
   //Copy the filter on the host (h_filter) to the memory you just allocated
   //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
   //Remember to use checkCudaErrors!
   //   checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filterWidth*filterWidth, cudaMemcpyHostToDevice));
   
   // Using constant memory to save access time to global memory
   checkCudaErrors(cudaMemcpyToSymbol(myFilter,h_filter,sizeof(float)*filterWidth*filterWidth));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
   //DONE: Set reasonable block size (i.e., number of threads per block)
   // const dim3 blockSize(numCols*numRows,1,1);
   const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y,1); // Tested and found 16x16 to be optimal
   
   //DONE:
   //Compute correct grid size (i.e., number of blocks per kernel launch)
   //from the image size and and block size.
   size_t gridCols = (numCols + blockSize.x - 1) / blockSize.x; // Add block size to round up
   size_t gridRows = (numRows + blockSize.y - 1) / blockSize.y; // Add block size to round up
   const dim3 gridSize( gridCols, gridRows, 1 );
   
   /*
   //DONE: Launch a kernel for separating the RGBA image into different color channels
   separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA,numRows,numCols,d_red,d_green,d_blue);
   
   // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
   // launching your kernel to make sure that you didn't make any mistakes.
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
   //DONE: Call your convolution kernel here 3 times, once for each color channel.
   gaussian_blur<<<gridSize,blockSize>>> (d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
   gaussian_blur<<<gridSize,blockSize>>> (d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
   gaussian_blur<<<gridSize,blockSize>>> (d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
   
   // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
   // launching your kernel to make sure that you didn't make any mistakes.
   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
   // Now we recombine your results. We take care of launching this kernel for you.
   //
   // NOTE: This kernel launch depends on the gridSize and blockSize variables,
   // which you must set yourself.
   recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                              d_greenBlurred,
                                              d_blueBlurred,
                                              d_outputImageRGBA,
                                              numRows,
                                              numCols);
    */
   gaussian_blur2<<<gridSize,blockSize>>> (d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, d_filter, filterWidth);

   cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//DONE: make sure you free any arrays that you allocated
void cleanup() {
//   checkCudaErrors(cudaFree(d_red));
//   checkCudaErrors(cudaFree(d_green));
//   checkCudaErrors(cudaFree(d_blue));
   checkCudaErrors(cudaFree(d_filter));
}
