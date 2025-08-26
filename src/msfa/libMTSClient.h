// Stub libMTSClient for command-line renderer
#ifndef LIBMTSCLIENT_H
#define LIBMTSCLIENT_H
// Minimal MTSClient stub
typedef void MTSClient;
// Return false to always use standard tuning path
inline bool MTS_HasMaster(MTSClient*) { return false; }
// Not used when MTS_HasMaster is false
inline double MTS_NoteToFrequency(MTSClient*, int midinote, int channel) { return 0.0; }
#endif // LIBMTSCLIENT_H