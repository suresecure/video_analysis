//#include "precomp.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/stream_accessor.hpp"
#include "opencv2/gpu/device/common.hpp"

namespace vibe_cpu {
void loadConstants(int nbSamples, int reqMatches, int radius,
                   int subsamplingFactor);

void init_gpu(PtrStepSzb frame, int cn, PtrStepSzb samples,
              PtrStepSz<unsigned int> randStates, cudaStream_t stream);

void update_gpu(PtrStepSzb frame, int cn, PtrStepSzb fgmask, PtrStepSzb samples,
                PtrStepSz<unsigned int> randStates, cudaStream_t stream);
}

namespace {
const int defaultNbSamples = 20;
const int defaultReqMatches = 2;
const int defaultRadius = 20;
const int defaultSubsamplingFactor = 16;
}

vibe_gpu::VIBE_GPU::VIBE_GPU(unsigned long rngSeed)
    : frameSize_(0, 0), rngSeed_(rngSeed) {
  nbSamples = defaultNbSamples;
  reqMatches = defaultReqMatches;
  radius = defaultRadius;
  subsamplingFactor = defaultSubsamplingFactor;
}

void vibe_gpu::VIBE_GPU::initialize(const GpuMat &firstFrame, Stream &s) {
  using namespace vibe_gpu::device::vibe;

  CV_Assert(firstFrame.type() == CV_8UC1 || firstFrame.type() == CV_8UC3 ||
            firstFrame.type() == CV_8UC4);

  cudaStream_t stream = StreamAccessor::getStream(s);

  loadConstants(nbSamples, reqMatches, radius, subsamplingFactor);

  frameSize_ = firstFrame.size();

  if (randStates_.size() != frameSize_) {
    cv::RNG rng(rngSeed_);
    cv::Mat h_randStates(frameSize_, CV_8UC4);
    rng.fill(h_randStates, cv::RNG::UNIFORM, 0, 255);
    randStates_.upload(h_randStates);
  }

  int ch = firstFrame.channels();
  int sample_ch = ch == 1 ? 1 : 4;

  samples_.create(nbSamples * frameSize_.height, frameSize_.width,
                  CV_8UC(sample_ch));

  init_gpu(firstFrame, ch, samples_, randStates_, stream);
}

void vibe_gpu::VIBE_GPU::operator()(const GpuMat &frame, GpuMat &fgmask,
                                    Stream &s) {
  using namespace vibe_gpu::device::vibe;

  CV_Assert(frame.depth() == CV_8U);

  int ch = frame.channels();
  int sample_ch = ch == 1 ? 1 : 4;

  if (frame.size() != frameSize_ || sample_ch != samples_.channels())
    initialize(frame);

  fgmask.create(frameSize_, CV_8UC1);

  update_gpu(frame, ch, fgmask, samples_, randStates_,
             StreamAccessor::getStream(s));
}

void vibe_gpu::VIBE_GPU::release() {
  frameSize_ = Size(0, 0);

  randStates_.release();

  samples_.release();
}
