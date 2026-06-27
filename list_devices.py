import sounddevice as sd
import sys

devices = sd.query_devices()
print("=== AUDIO DEVICES ===")
for i, d in enumerate(devices):
    inp = d["max_input_channels"]
    out = d["max_output_channels"]
    marker = ""
    if i == sd.default.device[0]: marker += " <-- DEFAULT IN"
    if i == sd.default.device[1]: marker += " <-- DEFAULT OUT"
    print(f"  [{i:2d}] {d['name']:50s}  in={inp}  out={out}{marker}")

print(f"\nDefault input device : {sd.default.device[0]}")
print(f"Default output device: {sd.default.device[1]}")
print(f"Default sample rate  : {sd.default.samplerate}")
