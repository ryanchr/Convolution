

#include "reference_calc.cpp"
#include "utils.h"

#define BLOCK_SIZE 16

__constant__ float myFilter[9*9];


__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO
  //assert(filterWidth % 2 == 1);
    
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.
  // make sure shared buffer size is (num_loc_row+)(num_loc_col+2)
  __shared__ unsigned char temp[(BLOCK_SIZE+8)*(BLOCK_SIZE+8)];  //£¨16+8£©*£¨16+8£©
  __shared__ float filter_tmp[9*9];
  //__shared__ unsigned char temp_left[512];
  //__shared__ unsigned char temp_right[512];
    
  //int lindex = threadIdx.x + BLOCK_ROW + BLOCK_COL;
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  int loc_image_position_x = threadIdx.x;  //x:col idx, y:row idx
  int loc_image_position_y = threadIdx.y;
 
  int ab_image_position_x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int ab_image_position_y = (blockIdx.y * blockDim.y) + threadIdx.y;   
    
  int image_c = min(max(ab_image_position_x, 0), (numCols - 1));
  int image_r = min(max(ab_image_position_y, 0), (numRows - 1));
    
  temp[(loc_image_position_y+4)*(blockDim.x+8)+loc_image_position_x+4] = 
            inputChannel[image_r*numCols+image_c]; 
    
  if ( ab_image_position_x >= numCols  ||
        ab_image_position_y >= numRows )
    {
        return; 
    }
    else
    {
        //temp_left[ab_image_position_y] = inputChannel[ab_image_position_y*numCols];
        //temp_right[ab_image_position_y] = inputChannel[ab_image_position_y*numCols+numCols-1];
    }
    
  //if ( ab_image_position_x >= numCols  ||
  //      ab_image_position_y >= numRows )
  //   {
  //      return; 
  //      temp[(loc_image_position_y+4)*(blockDim.x+8)+loc_image_position_x+4] = 
  //          inputChannel[ab_image_position_y*numCols+ab_image_position_x]; 
  //   }
  //else
  //   {
       //int image_c = min(max(ab_image_position_x, 0), (numCols - 1));
       //int image_r = min(max(ab_image_position_y + offset_y, 0), (numRows - 1));       
 
  //   }
    
    __syncthreads();
    
   //top
   if(loc_image_position_y < 4)
     //for(int offset_y = 0; offset_y<4; offset_y++)
         {
         int ima_ca = min(max(ab_image_position_x, 0), (numCols - 1));
         int ima_ra = min(max(ab_image_position_y-4, 0), (numRows - 1));
       
         temp[(loc_image_position_y+4-4)*(blockDim.x+8)+loc_image_position_x+4] = 
         //   ((ab_image_position_y-4+loc_image_position_y < 0)
         //    ? inputChannel[ab_image_position_x] : 
            inputChannel[ima_ra*numCols+ima_ca];
         }
     
   //bottom
   if(loc_image_position_y > blockDim.y-5)
     //for(int offset_y = 0; offset_y<4; offset_y++)
       {
         int ima_cb = min(max(ab_image_position_x, 0), (numCols - 1));
         int ima_rb = min(max(ab_image_position_y+4, 0), (numRows - 1));
       
         temp[(loc_image_position_y+4+4)*(blockDim.x+8)+loc_image_position_x+4] = 
         //   ((ab_image_position_y-4+loc_image_position_y < 0)
         //    ? inputChannel[ab_image_position_x] : 
         inputChannel[ima_rb*numCols+ima_cb];
       } 
    
   //left
     if(loc_image_position_x < 4)
          //for(int offset_x = 0; offset_x<4; offset_x++)
         {
         int ima_cc = min(max(ab_image_position_x-4, 0), (numCols - 1));
         int ima_rc = min(max(ab_image_position_y, 0), (numRows - 1));
       
         temp[(loc_image_position_y+4)*(blockDim.x+8)+loc_image_position_x+4-4] = 
         //   ((ab_image_position_y-4+loc_image_position_y < 0)
         //    ? inputChannel[ab_image_position_x] : 
         inputChannel[ima_rc*numCols+ima_cc];
         }
         
   //right
     if(loc_image_position_x > blockDim.x-5)
          //for(int offset_x = 0; offset_x<4; offset_x++)
         {
         int ima_cd = min(max(ab_image_position_x+4, 0), (numCols - 1));
         int ima_rd = min(max(ab_image_position_y, 0), (numRows - 1));
       
         temp[(loc_image_position_y+4)*(blockDim.x+8)+loc_image_position_x+4+4] = 
         //   ((ab_image_position_y-4+loc_image_position_y < 0)
         //    ? inputChannel[ab_image_position_x] : 
         inputChannel[ima_rd*numCols+ima_cd];
          }
    
    
     int numBlock = (numRows+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    //left top
    if(blockIdx.x < numBlock - 1 && blockIdx.y < numBlock - 1 ){
     if(loc_image_position_x < 4 && loc_image_position_y < 4)
          //for(int offset_x = 0; offset_x<4; offset_x++)
          {
            
            int ima_c1 = min(max(ab_image_position_x-4, 0), (numCols - 1));
            int ima_r1 = min(max(ab_image_position_y-4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4-4)*(blockDim.x+8)+ loc_image_position_x+4-4] = 
            //  ((blockIdx.x == 0 || blockIdx.y == 0)
            // ? inputChannel[(ab_image_position_y-4)*numCols+ab_image_position_x-4] : 
            inputChannel[(ima_r1)*numCols+ima_c1];
          }
    }else
    {
         {
            int ima_c1_ = min(max(ab_image_position_x-4, 0), (numCols - 1));
            int ima_r1_ = min(max(ab_image_position_y-4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4-4)*(blockDim.x+8)+ loc_image_position_x+4-4] = 
            //  ((blockIdx.x == 0 || blockIdx.y == 0)
            // ? inputChannel[(ab_image_position_y-4)*numCols+ab_image_position_x-4] : 
            inputChannel[(ima_r1_)*numCols+ima_c1_];
         }
    
    }
    
    //right top
    if(blockIdx.x < numBlock - 1 && blockIdx.y < numBlock - 1 )
    {
       if(loc_image_position_x > blockDim.x - 5 && loc_image_position_y < 4)
         {      
            int ima_c2 = min(max(ab_image_position_x+4, 0), (numCols - 1));
            int ima_r2 = min(max(ab_image_position_y-4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4-4)*(blockDim.x+8)+loc_image_position_x+4+4] = 
            inputChannel[(ima_r2)*numCols+ima_c2];
         }
     }else
     {
         //if(loc_image_position_x >= 0 && //(numCols-(numBlock-1)*blockDim.x) - 5 && 
            //loc_image_position_x <= (numCols-(numBlock-1)*blockDim.x) - 1  &&
         //  loc_image_position_y < 4)
         {
            int ima_c2_ = min(max(ab_image_position_x+4, 0), (numCols - 1));
            int ima_r2_ = min(max(ab_image_position_y-4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4-4)*(blockDim.x+8)+loc_image_position_x+4+4] = 
            inputChannel[(ima_r2_)*numCols+ima_c2_];
         }
     }
         
         //left bottom
    if(blockIdx.x < numBlock - 1 && blockIdx.y < numBlock - 1 ){
      if(loc_image_position_x < 4 && loc_image_position_y > blockDim.y - 5)
         {
            int ima_c3 = min(max(ab_image_position_x-4, 0), (numCols - 1));
            int ima_r3 = min(max(ab_image_position_y+4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4+4)*(blockDim.x+8)+loc_image_position_x+4-4] = 
            inputChannel[(ima_r3)*numCols+ima_c3];
         }
    }else
    {
        //if(loc_image_position_x < 4 && 
        // loc_image_position_y >= 0) // (numRows-(numBlock-1)*blockDim.y) - 5)
         {
            int ima_c3_ = min(max(ab_image_position_x-4, 0), (numCols - 1));
            int ima_r3_ = min(max(ab_image_position_y+4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4+4)*(blockDim.x+8)+loc_image_position_x+4-4] = 
            inputChannel[(ima_r3_)*numCols+ima_c3_];
         }
    }
         
    
    //right bottom
    if(blockIdx.x < numBlock - 1 && blockIdx.y < numBlock - 1 ){
      if(loc_image_position_x > blockDim.x - 5 && loc_image_position_y > blockDim.y - 5)
         {
            int ima_c4 = min(max(ab_image_position_x+4, 0), (numCols - 1));
            int ima_r4 = min(max(ab_image_position_y+4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4+4)*(blockDim.x+8)+loc_image_position_x+4+4] = 
            inputChannel[(ima_r4)*numCols+ima_c4];
         }
    }else 
    {
      //if(loc_image_position_x >= 0 && //  (numCols-(numBlock-1)*blockDim.x) - 5 && 
      //   loc_image_position_y >=0 ) // (numRows-(numBlock-1)*blockDim.y) - 5)
         {
            int ima_c4_ = min(max(ab_image_position_x+4, 0), (numCols - 1));
            int ima_r4_ = min(max(ab_image_position_y+4, 0), (numRows - 1));
         
            temp[(loc_image_position_y+4+4)*(blockDim.x+8)+loc_image_position_x+4+4] = 
            inputChannel[(ima_r4_)*numCols+ima_c4_];
         }
    }
    
    
       //if(loc_image_position_y == 0)
       //int image_c = min(max(ab_image_position_x + offset_x, 0), (numCols - 1));
       //int image_r = min(max(ab_image_position_y + offset_y, 0), (numRows - 1));
      
   // if(loc_image_position_x < 9 && loc_image_position_y < 9)
   //    filter_tmp[(loc_image_position_y*9+loc_image_position_x)] = 
   //     filter[(loc_image_position_y*9+loc_image_position_x)];
 
    __syncthreads();
    
    float result = 0.0;
    //float image_val = 0.0;
    //cout<<"filterWidth"<<filterWidth<<endl;
    for(int filter_y = -filterWidth/2; filter_y <= filterWidth/2; filter_y++)
       for(int filter_x = -filterWidth/2; filter_x <= filterWidth/2; filter_x++)
          //result += 
          //temp[(loc_image_position_x+1+filter_x)*blockDim.y+loc_image_position_y+1+filter_y]
          // *filter[(filter_x+3/2)*3+filter_y+3/2];

    { 
        result += (float)temp[(loc_image_position_y+4+filter_y)*(blockDim.x+8)
                              +loc_image_position_x+4+filter_x] 
                  * myFilter[(filter_y+filterWidth/2)*filterWidth+filter_x+filterWidth/2];
        //else
        //result += (float)inputChannel[image_r*numCols+image_c]
        //           * filter[(filter_y+filterWidth/2)*filterWidth+filter_x+filterWidth/2];
         //if(ab_image_position_x+filter_x < 0 ||
       //   ab_image_position_x+filter_x >= numCols ||
       //   ab_image_position_y+filter_y < 0 ||
       //   ab_image_position_y+filter_y >= numRows
       //   )
       //  temp[(loc_image_position_y+4+filter_y)*(blockDim.x+8)+loc_image_position_x+4+filter_x] 
       //    = inputChannel[image_r*numCols+image_c];
       // else
       //     return;
     
       // result += (float)inputChannel[ (image_r) * numCols + image_c]
       //           *  filter[(filter_y+filterWidth/2)*filterWidth+filter_x+filterWidth/2];;
    }
        
  if ( ab_image_position_x >= numCols ||
        ab_image_position_y >= numRows )
        return;                    
  else
        outputChannel[ab_image_position_y*numCols + ab_image_position_x] 
        = (unsigned char)result;

        
  // NOTE: If a thread's ab position 2D position is within the image, but some of
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
  // TODO
  //
  int idx_col = blockIdx.x * blockDim.x + threadIdx.x;  
  int idx_row = blockIdx.y * blockDim.y + threadIdx.y;
    
  if(idx_row >= numRows || idx_col >= numCols){
      return;
  }
  else
  {
    uchar4 rgba = inputImageRGBA[idx_row*numCols + idx_col];
    redChannel[idx_row*numCols + idx_col] = rgba.x;
    greenChannel[idx_row*numCols + idx_col] = rgba.y;
    blueChannel[idx_row*numCols + idx_col] = rgba.z;
  }
  
 // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( ab_image_position_x >= numCols ||
  //      ab_image_position_y >= numRows )
  // {
  //     return;
  // }
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
   //const int2 thread_2D_pos = make_int2( 
   //                                    ,
   //                                   );

  int thread_pos_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_pos_y = blockIdx.y * blockDim.y + threadIdx.y;
  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_pos_x >= numCols || thread_pos_y >= numRows)
    return;
      
  const int thread_1D_pos = thread_pos_y * numCols + thread_pos_x;

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
  checkCudaErrors(cudaMalloc((void **)&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc((void **)&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc((void **)&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float)*filterWidth*filterWidth));
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
   checkCudaErrors(cudaMemcpyToSymbol(myFilter,h_filter,sizeof(float)*filterWidth*filterWidth));
}

                         
void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);

  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  //const int num_blocks_row = (int) sqrt(numRows * numCols / (blockSize*blockSize));
  const dim3 gridSize((numCols+BLOCK_SIZE-1)/BLOCK_SIZE,(numRows+BLOCK_SIZE-1)/BLOCK_SIZE);
    
  //TODO: Launch a kernel for separating the RGBA image into different color channels      
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                   numRows,
                   numCols,
                     d_red,
                   d_green,
                    d_blue);
    
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //int bufSize = 18*18;
  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows
                                         , numCols, d_filter, filterWidth);
//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows
                                         , numCols, d_filter, filterWidth);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows
                                         , numCols, d_filter, filterWidth);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  const dim3 gridSize2((numCols+BLOCK_SIZE-1)/BLOCK_SIZE, (numRows+BLOCK_SIZE-1)/BLOCK_SIZE);
  recombineChannels<<<gridSize2, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}


