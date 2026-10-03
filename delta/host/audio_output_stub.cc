// Audio output where SDL is not linked (the Android builds): every port open
// fails and output is swallowed, so libSceAudioOut runs silently.

#include "host/audio_output.h"

namespace host {

int OpenAudioPort(u32, u32, int) {
  return -1;
}
int QueueAudio(int, const void*, u32 frames) {
  return static_cast<int>(frames);
}
void SetAudioPortVolume(int, float) {}
void CloseAudioPort(int) {}

void CloseAllAudioPorts() {}

void SetAudioPaused(bool) {}
}  // namespace host
