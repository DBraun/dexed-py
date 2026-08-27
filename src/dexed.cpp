// dexed.cpp: Nanobind wrapper for Dexed synthesizer
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <algorithm>

#include "msfa/controllers.h"
#include "msfa/tuning.h"
#include "msfa/lfo.h"
#include "msfa/dx7note.h"
#include "msfa/freqlut.h"
#include "msfa/exp2.h"
#include "msfa/sin.h"
#include "msfa/env.h"
#include "msfa/pitchenv.h"
#include "msfa/fm_core.h"
#include "EngineMkI.h"

namespace nb = nanobind;

// Acquire `m` with the Python GIL released.
//
// The render loops mutate per-instance state while holding no GIL, so a thread
// that blocked on the mutex *while* holding the GIL would stop the rendering
// thread from ever reacquiring it, deadlocking the interpreter. Dropping the
// GIL before waiting keeps that from happening.
static std::unique_lock<std::mutex> lock_without_gil(std::mutex &m) {
    std::unique_lock<std::mutex> lock(m, std::defer_lock);
    {
        nb::gil_scoped_release release;
        lock.lock();
    }
    return lock;
}

// Freqlut, Env, PitchEnv and Lfo keep process-wide lookup tables built for one
// sample rate. Whichever synth initialized them last owns them, so without this
// bookkeeping a synth created at a second sample rate silently retunes every
// synth already alive -- by a semitone and a half between 44.1k and 48k.
//
// Renders re-point the tables at their own sample rate and hold a claim on them
// until they finish. Concurrent renders at one rate share the tables freely;
// concurrent renders at *different* rates cannot both be right, so the second
// one raises rather than returning quietly detuned audio.
static std::mutex g_table_mutex;
static double g_table_sample_rate = 0.0;
static int g_table_claims = 0;

class GlobalTableClaim {
public:
    explicit GlobalTableClaim(double sample_rate) {
        std::lock_guard<std::mutex> guard(g_table_mutex);
        if (g_table_sample_rate != sample_rate) {
            if (g_table_claims > 0) {
                throw std::runtime_error(
                    "Cannot render at two different sample rates at the same "
                    "time: the DX7 frequency, envelope and LFO tables are "
                    "shared process-wide. Render sequentially, or use one "
                    "sample rate per process.");
            }
            Freqlut::init(sample_rate);
            Env::init_sr(sample_rate);
            PitchEnv::init(sample_rate);
            Lfo::init(sample_rate);
            g_table_sample_rate = sample_rate;
        }
        g_table_claims++;
    }

    ~GlobalTableClaim() {
        std::lock_guard<std::mutex> guard(g_table_mutex);
        g_table_claims--;
    }

    GlobalTableClaim(const GlobalTableClaim &) = delete;
    GlobalTableClaim &operator=(const GlobalTableClaim &) = delete;
};

// EngineMkI envelope constants (defined in EngineMkI.cpp, mirrored here)
static const uint16_t ENV_BITDEPTH = 14;
static const uint16_t ENV_MAX = 1 << ENV_BITDEPTH;

// Custom engine class that captures individual operator outputs
class EngineWithCapture : public EngineMkI {
public:
    // Storage for individual operator outputs
    int32_t op_outputs[6][N];

    // When true, use consistent feedback scaling across all algorithms
    // When false (default), use Dexed-authentic behavior where algorithms 4, 6, 32 have reduced feedback
    bool normalize_feedback = false;
    
    // Override the virtual render method to capture individual operator outputs
    void render(int32_t *output, FmOpParams *params, int algorithm,
                int32_t *fb_buf, int32_t feedback_shift) override {
        // Clear operator output buffers
        for (int op = 0; op < 6; ++op) {
            std::memset(op_outputs[op], 0, N * sizeof(int32_t));
        }

        // Clear internal engine buffers to prevent state leakage
        std::memset(buf_[0].get(), 0, N * sizeof(int32_t));
        std::memset(buf_[1].get(), 0, N * sizeof(int32_t));
        
        // Use parent class algorithm definitions
        const FmAlgorithm alg = algorithms[algorithm];
        bool has_contents[3] = { true, false, false };
        bool fb_on = feedback_shift < 16;
        
        // Special handling for algorithms with feedback
        FmAlgorithm alg_copy = alg;
        switch(algorithm) {
            case 3: case 5:
                if (fb_on)
                    alg_copy.ops[0] = 0xc4;
                break;
        }
        
        const int kLevelThresh = ENV_MAX - 100;
        
        for (int op = 0; op < 6; op++) {
            int flags = alg_copy.ops[op];
            bool add = (flags & OUT_BUS_ADD) != 0;
            FmOpParams &param = params[op];
            int inbus = (flags >> 4) & 3;
            int outbus = flags & 3;
            int32_t *outptr = (outbus == 0) ? output : buf_[outbus - 1].get();
            
            // Store pointer to capture this operator's output
            int32_t *capture_ptr = op_outputs[op];
            
            int32_t gain1 = param.gain_out == 0 ? (ENV_MAX-1) : param.gain_out;
            int32_t gain2 = ENV_MAX-(param.level_in >> (28-ENV_BITDEPTH));
            param.gain_out = gain2;
            
            if (gain1 <= kLevelThresh || gain2 <= kLevelThresh) {
                if (!has_contents[outbus]) {
                    add = false;
                }
                
                if (inbus == 0 || !has_contents[inbus]) {
                    // Handle feedback operators
                    if ((flags & 0xc0) == 0xc0 && fb_on) {
                        // Compute effective feedback shift
                        // When normalize_feedback is true, use feedback_shift directly for all algorithms
                        // When false (Dexed-authentic), algorithms 4, 6, 32 get reduced feedback (+2)
                        int32_t fb_shift_3op = normalize_feedback ? feedback_shift : min((feedback_shift+2), 16);
                        int32_t fb_shift_2op = normalize_feedback ? feedback_shift : min((feedback_shift+2), 16);
                        int32_t fb_shift_1op_special = normalize_feedback ? feedback_shift : min((feedback_shift+2), 16);

                        switch (algorithm) {
                            case 3:  // Algorithm 4 - special 3-op feedback
                                compute_fb3(capture_ptr, params, gain1, gain2, fb_buf, fb_shift_3op);
                                // Copy to output buffer
                                if (add) {
                                    for (int i = 0; i < N; ++i) {
                                        outptr[i] += capture_ptr[i];
                                    }
                                } else {
                                    std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                                }
                                // Skip next two operators as they were processed
                                params[1].phase += params[1].freq << LG_N;
                                params[2].phase += params[2].freq << LG_N;
                                op += 2;
                                break;

                            case 5:  // Algorithm 6 - special 2-op feedback
                                compute_fb2(capture_ptr, params, gain1, gain2, fb_buf, fb_shift_2op);
                                // Copy to output buffer
                                if (add) {
                                    for (int i = 0; i < N; ++i) {
                                        outptr[i] += capture_ptr[i];
                                    }
                                } else {
                                    std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                                }
                                // Skip next operator as it was processed
                                params[1].phase += params[1].freq << LG_N;
                                op++;
                                break;

                            case 31:  // Algorithm 32 - single op feedback (Dexed uses reduced feedback here too)
                                compute_fb(capture_ptr, param.phase, param.freq, gain1, gain2,
                                         fb_buf, fb_shift_1op_special, false);
                                // Copy to output buffer
                                if (add) {
                                    for (int i = 0; i < N; ++i) {
                                        outptr[i] += capture_ptr[i];
                                    }
                                } else {
                                    std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                                }
                                break;

                            default:  // Single operator feedback - normal process
                                compute_fb(capture_ptr, param.phase, param.freq, gain1, gain2,
                                         fb_buf, feedback_shift, false);
                                // Copy to output buffer
                                if (add) {
                                    for (int i = 0; i < N; ++i) {
                                        outptr[i] += capture_ptr[i];
                                    }
                                } else {
                                    std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                                }
                                break;
                        }
                    } else {
                        // Pure operator (no modulation input)
                        compute_pure(capture_ptr, param.phase, param.freq, gain1, gain2, false);
                        // Copy to output buffer
                        if (add) {
                            for (int i = 0; i < N; ++i) {
                                outptr[i] += capture_ptr[i];
                            }
                        } else {
                            std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                        }
                    }
                } else {
                    // Operator with modulation input
                    compute(capture_ptr, buf_[inbus - 1].get(), param.phase, param.freq, 
                           gain1, gain2, false);
                    // Copy to output buffer
                    if (add) {
                        for (int i = 0; i < N; ++i) {
                            outptr[i] += capture_ptr[i];
                        }
                    } else {
                        std::memcpy(outptr, capture_ptr, N * sizeof(int32_t));
                    }
                }
                
                has_contents[outbus] = true;
            } else if (!add) {
                has_contents[outbus] = false;
            }
            
            param.phase += param.freq << LG_N;
        }
    }
};

// Forward declaration
class DexedSynth;

// Standard tuning implementation
struct StdTuning : public TuningState {
    int32_t midinote_to_logfreq(int midinote) override {
        const double base_d = (1ULL << 24) * (std::log(440.0) / std::log(2.0) - 69.0 / 12.0);
        const int32_t base = static_cast<int32_t>(base_d);
        const int32_t step = (1 << 24) / 12;
        return base + step * midinote;
    }
};

class DexedSynth {
private:
    EngineWithCapture engine;  // Use custom engine with capture capability
    Controllers controllers;
    std::shared_ptr<StdTuning> tuning;
    double sample_rate;
    int algorithm;  // DX7 algorithm (1-32)
    bool initialized;
    bool params_loaded;
    
    // Store DX7 parameters and synthesis state
    uint8_t dx7_params[156];
    std::unique_ptr<Dx7Note> voice;
    std::unique_ptr<Lfo> lfo;
    
    // Pre-allocated buffer for render_all_ops
    std::vector<int32_t> mixed_buffer;

    // Serializes every method that touches the state above, so that sharing one
    // synth between Python threads is safe rather than a use-after-free.
    std::mutex state_mutex;
    static constexpr float INT32_TO_FLOAT_SCALE = 1.0f / (1L << 25);
    
    // Convert normalized [0,1] to DX7 parameter range
    static uint8_t denormalize_param(float value, int min_val, int max_val) {
        int range = max_val - min_val;
        int result = static_cast<int>(value * range + 0.5f) + min_val;
        return static_cast<uint8_t>(std::clamp(result, min_val, max_val));
    }
    
    // Convert one-hot categorical to index
    static uint8_t categorical_to_index(const float* one_hot, int num_classes) {
        int max_idx = 0;
        float max_val = one_hot[0];
        for (int i = 1; i < num_classes; i++) {
            if (one_hot[i] > max_val) {
                max_val = one_hot[i];
                max_idx = i;
            }
        }
        return static_cast<uint8_t>(max_idx);
    }
    
    static const uint8_t pitchmodsenstab[8];
    
    void convert_params_to_dx7(const float* params_185, uint8_t* dx7_params) {
        // Initialize to zeros
        std::memset(dx7_params, 0, 156);
        
        // Parameter mapping based on the order from show_parameter_order.py
        // First 131 parameters are continuous values, then 54 categorical (one-hot encoded)
        
        // Global LFO parameters (indices 0-5 in params_185)
        dx7_params[140] = denormalize_param(params_185[0], 0, 99);   // AMD
        dx7_params[138] = denormalize_param(params_185[1], 0, 99);   // Delay
        dx7_params[143] = denormalize_param(params_185[2], 0, 7);    // P Mod Sens
        dx7_params[139] = denormalize_param(params_185[3], 0, 99);   // PMD
        dx7_params[137] = denormalize_param(params_185[4], 0, 99);   // Speed
        dx7_params[141] = denormalize_param(params_185[5], 0, 1);    // Sync
        
        // Global Main parameters (indices 6-8 in params_185)
        dx7_params[135] = denormalize_param(params_185[6], 0, 7);    // Feedback
        dx7_params[136] = denormalize_param(params_185[7], 0, 1);    // Osc Key Sync
        dx7_params[144] = denormalize_param(params_185[8], 0, 48);   // Transpose
        
        // Global Pitch EG (indices 9-16 in params_185)
        // Pitch EG Levels (indices 9-12)
        for (int i = 0; i < 4; i++) {
            dx7_params[130 + i] = denormalize_param(params_185[9 + i], 0, 99);
        }
        // Pitch EG Rates (indices 13-16)
        for (int i = 0; i < 4; i++) {
            dx7_params[126 + i] = denormalize_param(params_185[13 + i], 0, 99);
        }
        
        // Operator parameters (6 operators, each with 19 continuous params)
        // Operators are indexed 1-6 in the input, but stored in reverse order in DX7 sysex
        // DX7 stores Op6 first (positions 0-20), then Op5, etc., with Op1 last
        for (int op = 0; op < 6; op++) {
            int input_base = 17 + (op * 19);  // Start index in params_185 (Op1, Op2, ...)
            int op_base = (5 - op) * 21;  // Reverse mapping: Op1→pos[105-125], Op6→pos[0-20]
            
            // Amp Env Generator Levels (L1-L4)
            for (int i = 0; i < 4; i++) {
                dx7_params[op_base + 4 + i] = denormalize_param(params_185[input_base + i], 0, 99);
            }
            
            // Amp Env Generator Rates (R1-R4)
            for (int i = 0; i < 4; i++) {
                dx7_params[op_base + i] = denormalize_param(params_185[input_base + 4 + i], 0, 99);
            }
            
            // Breakpoint parameters
            dx7_params[op_base + 8] = denormalize_param(params_185[input_base + 8], 0, 99);   // Breakpoint
            dx7_params[op_base + 9] = denormalize_param(params_185[input_base + 9], 0, 99);   // L Depth
            dx7_params[op_base + 10] = denormalize_param(params_185[input_base + 10], 0, 99); // R Depth
            
            // Other parameters
            dx7_params[op_base + 14] = denormalize_param(params_185[input_base + 11], 0, 3);  // A Mod Sens
            dx7_params[op_base + 17] = denormalize_param(params_185[input_base + 12], 0, 1);  // Freq Mode
            dx7_params[op_base + 15] = denormalize_param(params_185[input_base + 13], 0, 7);  // Key Vel
            dx7_params[op_base + 16] = denormalize_param(params_185[input_base + 14], 0, 99); // Level
            dx7_params[op_base + 13] = denormalize_param(params_185[input_base + 15], 0, 7);  // Rate Scaling
            
            // Tone parameters
            dx7_params[op_base + 18] = denormalize_param(params_185[input_base + 16], 0, 31); // Coarse
            dx7_params[op_base + 19] = denormalize_param(params_185[input_base + 17], 0, 99); // Fine
            dx7_params[op_base + 20] = denormalize_param(params_185[input_base + 18], 0, 14); // Tune (Detune)
        }
        
        // Algorithm: stored internally as 0-31, matching the DX7 sysex byte directly
        dx7_params[134] = static_cast<uint8_t>(algorithm);
        
        // Handle categorical parameters (one-hot encoded, starting at index 131)
        int cat_idx = 131;
        
        // LFO Wave (indices 131-137, 6 classes)
        dx7_params[142] = categorical_to_index(&params_185[cat_idx], 6);
        cat_idx += 6;  // Now at 137
        
        // Operator L/R Curves (4 classes each)
        // The order in params_185 is Op1 L Curve, Op1 R Curve, Op2 L Curve, etc.
        // But DX7 stores operators in reverse order, so we need to reverse the mapping
        for (int op = 0; op < 6; op++) {
            int op_base = (5 - op) * 21;  // Reverse mapping to match operator ordering
            
            // L Curve (4 classes)
            dx7_params[op_base + 11] = categorical_to_index(&params_185[cat_idx], 4);
            cat_idx += 4;
            
            // R Curve (4 classes)
            dx7_params[op_base + 12] = categorical_to_index(&params_185[cat_idx], 4);
            cat_idx += 4;
        }
        
        // Voice name (10 characters) - set default
        for (int i = 0; i < 10; i++) {
            dx7_params[145 + i] = ' ';
        }
        
        // Operator enable switches (byte 155) - enable all operators by default
        // Bit 0 = Op1, Bit 1 = Op2, ... Bit 5 = Op6
        // 0x3F = binary 111111 = all operators enabled
        dx7_params[155] = 0x3F;
    }

public:
    DexedSynth(double sr = 44100.0, int alg = 0)
        : sample_rate(sr), algorithm(0), initialized(false), params_loaded(false),
          mixed_buffer(1 << LG_N) {  // Pre-allocate mixed buffer
        // Algorithm is 0-31, matching the DX7 sysex byte directly
        if (alg < 0 || alg > 31) {
            throw std::runtime_error("Algorithm must be between 0 and 31");
        }
        algorithm = alg;
        // Delay all initialization to first render call
        initialized = false;
        params_loaded = false;
    }
    
    void ensure_initialized() {
        if (initialized) return;
        
        // Rate-independent tables; the rate-dependent ones are claimed per
        // render by GlobalTableClaim.
        Exp2::init();
        Sin::init();

        // Setup tuning
        tuning = std::make_shared<StdTuning>();
        
        // Initialize controllers
        controllers.core = &engine;
        controllers.values_[kControllerPitch] = 0x2000;
        controllers.values_[kControllerPitchRangeUp] = 3;
        controllers.values_[kControllerPitchRangeDn] = 3;
        controllers.values_[kControllerPitchStep] = 0;
        controllers.masterTune = 0;
        controllers.modwheel_cc = 0;
        controllers.foot_cc = 0;
        controllers.breath_cc = 0;
        controllers.aftertouch_cc = 0;
        controllers.mpeEnabled = false;
        controllers.refresh();
        
        initialized = true;
    }
    
    void load_params(nb::ndarray<nb::numpy, float, nb::shape<185>, nb::c_contig> params) {
        auto lock = lock_without_gil(state_mutex);
        ensure_initialized();
        
        // Get raw pointer to parameters
        const float* params_ptr = static_cast<const float*>(params.data());
        
        // Convert parameters to DX7 format
        convert_params_to_dx7(params_ptr, dx7_params);
        
        params_loaded = true;
    }
    
    nb::ndarray<float, nb::shape<-1>, nb::numpy> render(
        int midi_note = 60,
        int velocity = 100,
        float note_duration = 3.0,
        float render_duration = 4.0
    ) {
        auto lock = lock_without_gil(state_mutex);
        ensure_initialized();
        
        // Check that parameters have been loaded
        if (!params_loaded) {
            throw std::runtime_error("Parameters must be loaded before rendering. Call load_params() first.");
        }

        // Point the shared DX7 tables at this synth's sample rate.
        GlobalTableClaim tables(sample_rate);

        // Create fresh LFO and voice objects to ensure complete state reset
        lfo = std::make_unique<Lfo>();
        lfo->reset(&dx7_params[137]);
        lfo->keydown();

        voice = std::make_unique<Dx7Note>(tuning, &engine);

        // Apply transpose: DX7 stores 0-48 with 24 meaning no shift (C3)
        int transposed_note = midi_note + (dx7_params[144] - 24);
        transposed_note = std::max(0, std::min(127, transposed_note));

        // Initialize voice for this note
        voice->init(dx7_params, transposed_note, velocity, 1, &controllers);

        // Apply osc sync if enabled
        if (dx7_params[136] == 1) {
            voice->oscSync();
        }
        
        // Calculate sample counts
        const int N = 1 << LG_N;  // Block size (typically 64)
        int total_samples = static_cast<int>(render_duration * sample_rate + 0.5);
        int note_samples = static_cast<int>(note_duration * sample_rate + 0.5);
        int total_blocks = (total_samples + N - 1) / N;
        int note_blocks = (note_samples + N - 1) / N;
        
        // Allocate output buffer directly (avoid double allocation)
        float* output_data = new float[total_samples]();
        std::vector<int32_t> block_buf(N);
        
        // Render audio blocks with GIL released for multicore processing
        {
            nb::gil_scoped_release release;

            for (int b = 0; b < total_blocks; ++b) {
                // Release note at specified time
                if (b == note_blocks) {
                    voice->keyup();
                }

                // Get LFO values
                int32_t lfoval = lfo->getsample();
                int32_t lfodel = lfo->getdelay();

                // Clear block buffer
                std::fill(block_buf.begin(), block_buf.end(), 0);

                // Compute samples
                voice->compute(block_buf.data(), lfoval, lfodel, &controllers);

                // Convert to float and store
                for (int i = 0; i < N; ++i) {
                    int idx_out = b * N + i;
                    if (idx_out >= total_samples) break;

                    // Convert from int32 to float without clipping
                    // Use scaling that matches typical DX7 output levels
                    int32_t val = block_buf[i];

                    // Scale to float range without clipping
                    // Using 1<<25 gives reasonable headroom while maintaining good levels
                    float sample = static_cast<float>(val) / (float)(1L << 25);
                    output_data[idx_out] = sample;
                }
            }
        }
        
        // Create capsule with custom deleter to manage memory
        nb::capsule owner(output_data, [](void *p) noexcept { delete[] (float*)p; });
        
        // Return the data as a NumPy array with shape (T,)
        return nb::ndarray<float, nb::shape<-1>, nb::numpy>(
            output_data, {static_cast<size_t>(total_samples)}, owner);
    }
    
    nb::ndarray<float, nb::shape<-1, -1>, nb::numpy> render_all_ops(
        int midi_note = 60,
        int velocity = 100,
        float note_duration = 3.0,
        float render_duration = 4.0
    ) {
        auto lock = lock_without_gil(state_mutex);
        ensure_initialized();
        
        // Check that parameters have been loaded
        if (!params_loaded) {
            throw std::runtime_error("Parameters must be loaded before rendering. Call load_params() first.");
        }

        // Point the shared DX7 tables at this synth's sample rate.
        GlobalTableClaim tables(sample_rate);

        // Create fresh LFO and voice objects to ensure complete state reset
        lfo = std::make_unique<Lfo>();
        lfo->reset(&dx7_params[137]);
        lfo->keydown();

        voice = std::make_unique<Dx7Note>(tuning, &engine);

        // Apply transpose: DX7 stores 0-48 with 24 meaning no shift (C3)
        int transposed_note = midi_note + (dx7_params[144] - 24);
        transposed_note = std::max(0, std::min(127, transposed_note));

        // Initialize voice for this note
        voice->init(dx7_params, transposed_note, velocity, 1, &controllers);

        // Apply osc sync if enabled
        if (dx7_params[136] == 1) {
            voice->oscSync();
        }

        // Calculate sample counts
        const int N = 1 << LG_N;  // Block size (typically 64)
        int total_samples = static_cast<int>(render_duration * sample_rate + 0.5);
        int note_samples = static_cast<int>(note_duration * sample_rate + 0.5);
        int total_blocks = (total_samples + N - 1) / N;
        int note_blocks = (note_samples + N - 1) / N;
        
        // Allocate output buffer for 6 operators + 1 mixed output [7, T]
        float* output_data = new float[7 * total_samples]();
        
        // Render audio blocks with GIL released for multicore processing
        {
            nb::gil_scoped_release release;

            for (int b = 0; b < total_blocks; ++b) {
                // Release note at specified time
                if (b == note_blocks) {
                    voice->keyup();
                }

                // Get LFO values
                int32_t lfoval = lfo->getsample();
                int32_t lfodel = lfo->getdelay();

                // Clear mixed buffer
                std::fill(mixed_buffer.begin(), mixed_buffer.end(), 0);

                // Compute using the standard voice compute (which calls our custom engine)
                // This will fill engine.op_outputs with individual operator data
                // and mixed_buffer with the final mixed output
                voice->compute(mixed_buffer.data(), lfoval, lfodel, &controllers);

                // Convert and store all channels for better cache locality
                int block_start = b * N;
                int block_end = std::min(block_start + N, total_samples);

                for (int i = 0; i < N && (block_start + i) < total_samples; ++i) {
                    int idx_out = block_start + i;

                    // Convert and store all 6 operators
                    // Note: DX7 stores operators in reverse order, so we need to reverse the mapping
                    // engine.op_outputs[0] = Op6, [1] = Op5, ..., [5] = Op1
                    // We want output channels: [0] = Op1, [1] = Op2, ..., [5] = Op6
                    for (int op = 0; op < 6; ++op) {
                        // Reverse the operator index to get correct ordering
                        int engine_op_idx = 5 - op;
                        // Convert from int32 to float using multiplication
                        float sample = static_cast<float>(engine.op_outputs[engine_op_idx][i]) * INT32_TO_FLOAT_SCALE;
                        output_data[op * total_samples + idx_out] = sample;
                    }

                    // Convert and store mixed output (channel 7)
                    float mixed_sample = static_cast<float>(mixed_buffer[i]) * INT32_TO_FLOAT_SCALE;
                    output_data[6 * total_samples + idx_out] = mixed_sample;
                }
            }
        }
        
        // Create capsule with custom deleter to manage memory
        nb::capsule owner(output_data, [](void *p) noexcept { delete[] (float*)p; });
        
        // Return the data as a NumPy array with shape [7, T]
        return nb::ndarray<float, nb::shape<-1, -1>, nb::numpy>(
            output_data, {7, static_cast<size_t>(total_samples)}, owner);
    }
    
    double get_sample_rate() const {
        return sample_rate;
    }
    
    int get_algorithm() const {
        return algorithm;  // 0-31
    }

    void set_algorithm(int alg) {
        auto lock = lock_without_gil(state_mutex);
        if (alg < 0 || alg > 31) {
            throw std::runtime_error("Algorithm must be between 0 and 31");
        }
        algorithm = alg;

        if (params_loaded) {
            dx7_params[134] = static_cast<uint8_t>(algorithm);
        }
    }

    void load_sysex(nb::bytes data) {
        auto lock = lock_without_gil(state_mutex);
        ensure_initialized();
        if (data.size() < 156) {
            throw std::runtime_error("Sysex data must be at least 156 bytes");
        }
        std::memcpy(dx7_params, data.data(), 156);
        algorithm = dx7_params[134] & 0x1F;  // Sync from sysex byte (0-31)
        params_loaded = true;
    }

    bool get_normalize_feedback() const {
        return engine.normalize_feedback;
    }

    void set_normalize_feedback(bool normalize) {
        auto lock = lock_without_gil(state_mutex);
        engine.normalize_feedback = normalize;
    }

    // Pickle support using getstate/setstate to avoid reference leaks
    nb::tuple __getstate__() const {
        return nb::make_tuple(sample_rate, algorithm, engine.normalize_feedback);
    }
};

// Define static member outside the class
const uint8_t DexedSynth::pitchmodsenstab[8] = {
    0, 10, 20, 33, 55, 92, 153, 255
};

NB_MODULE(_dexed, m) {
    m.doc() = "Dexed DX7 synthesizer Python bindings";
    
    nb::class_<DexedSynth>(m, "DexedSynth")
        .def(nb::init<double, int>(),
             nb::arg("sample_rate") = 44100.0,
             nb::arg("algorithm") = 0,
             "Initialize DexedSynth with given sample rate and algorithm\n\n"
             "Args:\n"
             "    sample_rate: Audio sample rate in Hz (default 44100)\n"
             "    algorithm: DX7 algorithm number 0-31 (default 0)")
        .def("load_sysex", &DexedSynth::load_sysex,
             nb::arg("data"),
             "Load synthesizer parameters from 156-byte DX7 sysex voice data\n\n"
             "Args:\n"
             "    data: bytes of length >= 156 (unpacked DX7 VCED format)")
        .def("load_params", &DexedSynth::load_params,
             nb::arg("params"),
             "Load synthesizer parameters (legacy 185-element float array)\n\n"
             "Args:\n"
             "    params: numpy array of shape [185] with normalized parameters\n"
             "            (131 continuous + 54 categorical one-hot encoded)")
        .def("render", &DexedSynth::render,
             nb::arg("midi_note") = 60,
             nb::arg("velocity") = 100,
             nb::arg("note_duration") = 3.0f,
             nb::arg("render_duration") = 4.0f,
             "Render audio with current parameters\n\n"
             "Args:\n"
             "    midi_note: MIDI note number (0-127)\n"
             "    velocity: Note velocity (0-127)\n"
             "    note_duration: Duration of note in seconds\n"
             "    render_duration: Total duration to render in seconds\n\n"
             "Returns:\n"
             "    numpy array of shape [T] with audio samples\n\n"
             "Note: load_params() must be called before render()")
        .def("render_all_ops", &DexedSynth::render_all_ops,
             nb::arg("midi_note") = 60,
             nb::arg("velocity") = 100,
             nb::arg("note_duration") = 3.0f,
             nb::arg("render_duration") = 4.0f,
             "Render individual operator outputs with current parameters\n\n"
             "Args:\n"
             "    midi_note: MIDI note number (0-127)\n"
             "    velocity: Note velocity (0-127)\n"
             "    note_duration: Duration of note in seconds\n"
             "    render_duration: Total duration to render in seconds\n\n"
             "Returns:\n"
             "    numpy array of shape [7, T] with individual operator outputs\n"
             "    Channel 0: Operator 1\n"
             "    Channel 1: Operator 2\n"
             "    Channel 2: Operator 3\n"
             "    Channel 3: Operator 4\n"
             "    Channel 4: Operator 5\n"
             "    Channel 5: Operator 6\n"
             "    Channel 6: Mixed output (sum of carriers per algorithm)\n\n"
             "Note: load_params() must be called before render_all_ops()")
        .def_prop_ro("sample_rate", &DexedSynth::get_sample_rate,
                     "Get the sample rate")
        .def_prop_rw("algorithm",
                     &DexedSynth::get_algorithm,
                     &DexedSynth::set_algorithm,
                     "Get or set the DX7 algorithm (0-31)")
        .def_prop_rw("normalize_feedback",
                     &DexedSynth::get_normalize_feedback,
                     &DexedSynth::set_normalize_feedback,
                     "When True, use consistent feedback scaling across all algorithms.\n"
                     "When False (default), use Dexed-authentic behavior where algorithms 4, 6, 32\n"
                     "have reduced feedback strength compared to other algorithms.")
        .def("__getstate__", &DexedSynth::__getstate__,
             "Get state for pickle serialization")
        .def("__setstate__", [](DexedSynth &synth, nb::tuple state) {
            if (state.size() < 2 || state.size() > 3) {
                throw std::runtime_error("Invalid state for unpickling");
            }

            // Use placement new to reconstruct the object properly
            double sr = nb::cast<double>(state[0]);
            int alg = nb::cast<int>(state[1]);
            new (&synth) DexedSynth(sr, alg);

            // Handle normalize_feedback if present (backward compatibility)
            if (state.size() == 3) {
                synth.set_normalize_feedback(nb::cast<bool>(state[2]));
            }
        }, "Set state for pickle deserialization");
}