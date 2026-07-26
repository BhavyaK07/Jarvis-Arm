# Jarvis-Arm 🦾

A voice-controlled 5-DOF robotic arm with a multi-finger hand that doesn't just move — it *reasons*. Inspired by a lifelong fascination with Tony Stark's workshop AI, Jarvis-Arm combines wake-word detection, LLM reasoning, computer vision, and real-time teleoperation into a single system you can talk to and work alongside.


## Features

### 🎙️ Autonomous Mode
Speak a command, and the arm figures out how to execute it on its own:
- **Wake word detection** via OpenWakeWord
- **Voice activity detection** via WebRTCVAD
- **Conversational reasoning** via Groq (Llama 3.3 70B)
- **Vision-based command generation** via Gemini
- **Object detection** via YOLOv8 
- **Voice responses** via ElevenLabs TTS

### 🖐️ Teleoperation Mode
Real-time hand tracking via webcam + OpenCV, mirrored live onto the arm; control it the way you'd control your own hand.

### 🤖 Onboard Assistant
A conversational layer modeled after TARS from *Interstellar* — dry, direct, a little sarcastic. The goal was to make interacting with the arm feel like working *with* something, not issuing commands *to* it.

### 🔩 Hardware
- Dual-Arduino UART bridge architecture for arm + hand control
- 5-DOF robotic arm
- Multi-finger hand

---

## Multi-finger Hand

<img width="4284" height="5712" alt="IMG_3001 (1)" src="https://github.com/user-attachments/assets/007373a4-9e58-4331-8496-512e570ae5dd" />


## Architecture

```
Voice Input → OpenWakeWord → WebRTCVAD → Groq (Llama 3.3 70B)
                                              ↓
                                    Gemini (vision reasoning)
                                              ↓
                                     YOLOv8 nano (object detection)
                                              ↓
                              Dual-Arduino UART Bridge → Arm + Hand
                                              ↓
                                       ElevenLabs TTS (response)

Webcam → OpenCV (hand tracking) → Dual-Arduino UART Bridge → Arm + Hand
                                          [Teleoperation Mode]
```

---


## Tech Stack

| Layer | Tools |
|---|---|
| Wake Word / VAD | OpenWakeWord, WebRTCVAD |
| Language Reasoning | Groq (Llama 3.3 70B) |
| Vision | Gemini, YOLOv8 nano, OpenCV |
| Voice Output | ElevenLabs TTS |
| Hardware Control | Dual-Arduino UART Bridge |

---

## Author

Built by [Bhavya](https://bhavya.ca) — Mechatronics Engineering @ University of Waterloo
