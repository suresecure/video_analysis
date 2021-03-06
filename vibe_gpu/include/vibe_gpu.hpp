/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef HEADER_VIBE_GPU
#define HEADER_VIBE_GPU

//#include "opencv2/opencv_modules.hpp"

#include "opencv2/gpu/gpu.hpp"

namespace vibe_gpu{

/*!
 * The class implements the following algorithm:
 * "ViBe: A universal background subtraction algorithm for video sequences"
 * O. Barnich and M. Van D Roogenbroeck
 * IEEE Transactions on Image Processing, 20(6) :1709-1724, June 2011
 */
class VIBE_GPU
{
public:
    //! the default constructor
    explicit VIBE_GPU(unsigned long rngSeed = 1234567);

    //! re-initiaization method
    void initialize(const GpuMat& firstFrame, Stream& stream = Stream::Null());

    //! the update operator
    void operator()(const GpuMat& frame, GpuMat& fgmask, Stream& stream = Stream::Null());

    //! releases all inner buffers
    void release();

    int nbSamples;         // number of samples per pixel
    int reqMatches;        // #_min
    int radius;            // R
    int subsamplingFactor; // amount of random subsampling

private:
    Size frameSize_;

    unsigned long rngSeed_;
    GpuMat randStates_;

    GpuMat samples_;
};

} // namespace vibe_gpu

#endif // __OPENCV_NONFREE_GPU_HPP__
