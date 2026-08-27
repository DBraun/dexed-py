/*
 * Copyright 2014 Pascal Gauthier.
 * Copyright 2012 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENGINEMKI_H_INCLUDED
#define ENGINEMKI_H_INCLUDED

#include "msfa/synth.h"
#include "msfa/aligned_buf.h"
#include "msfa/fm_op_kernel.h"
#include "msfa/controllers.h"
#include "msfa/fm_core.h"


class EngineMkI : public FmCore {
public:
    EngineMkI();
    
    void render(int32_t *output, FmOpParams *params, int algorithm, int32_t *fb_buf, int32_t feedback_shift) override;
    
    void compute(int32_t *output, const int32_t *input, int32_t phase0, int32_t freq, int32_t gain1, int32_t gain2,
                    bool add);
    
	void compute_pure(int32_t *output, int32_t phase0, int32_t freq, int32_t gain1, int32_t gain2,
                         bool add);
    
    void compute_fb(int32_t *output, int32_t phase0, int32_t freq, int32_t gain1, int32_t gain2,
                       int32_t *fb_buf, int fb_gain, bool add);
    
    // stage0/stage1 optionally receive the output of each operator inside the
    // collapsed feedback chain; `output` always receives the last one. Dexed
    // itself never needs them, but capturing per-operator audio does.
    void compute_fb2(int32_t *output, FmOpParams *params, int32_t gain01, int32_t gain02, int32_t *fb_buf, int fb_shift,
                     int32_t *stage0 = nullptr);
    
    void compute_fb3(int32_t *output, FmOpParams *params, int32_t gain01, int32_t gain02, int32_t *fb_buf, int fb_shift,
                     int32_t *stage0 = nullptr, int32_t *stage1 = nullptr);
};


#endif  // ENGINEMKI_H_INCLUDED
