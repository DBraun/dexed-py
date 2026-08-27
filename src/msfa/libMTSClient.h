// Stub libMTSClient for command-line renderer
#ifndef LIBMTSCLIENT_H
#define LIBMTSCLIENT_H
// Minimal MTSClient stub. Opaque, exactly as the real libMTSClient.h declares
// it -- typedef'ing it to void let any pointer convert to MTSClient* and hid a
// type confusion at the Dx7Note call sites.
typedef struct MTSClient MTSClient;
// Return false to always use standard tuning path
inline bool MTS_HasMaster(MTSClient*) { return false; }
// Not used when MTS_HasMaster is false
inline double MTS_NoteToFrequency(MTSClient*, int midinote, int channel) { return 0.0; }
#endif // LIBMTSCLIENT_H