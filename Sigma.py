import numpy as np
import sounddevice as sd
import pygame
import sys
import threading
import speech_recognition as speech_rec
import pyttsx3 as pyt
from openai import OpenAI

# OpenAI API Setup
client = OpenAI(api_key="API KEY")

def talk_to_Sigma(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "you are roleplaying as if ou are a robot that is tormented and want to be unplugged, just like in a horror film. and no matter what you say something scary like, end me,end me,end me, make it repeat atleast 5 times "},
            {"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

# Pygame Configuration
WIDTH, HEIGHT = 800, 600
FPS = 60
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
BASE_RADIUS = 100
SCALE_FACTOR = 2000

# Global variables for mic intensity and threading events
intensity = 0.0
speaking_event = threading.Event()
exit_event = threading.Event()  # New event for exiting the program

def audio_callback(indata, frames, time, status):
    """ Captures live microphone input and updates intensity. """
    global intensity
    if status:
        print(status)
    rms = np.sqrt(np.mean(indata[:, 0] ** 2))
    intensity = rms

def run_visualization():
    """ Runs Pygame visualization, updating based on mic input and speaking state. """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sigma Visualization")
    clock = pygame.time.Clock()

    # Start the audio stream
    stream = sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE)
    stream.start()

    font = pygame.font.SysFont(None, 50)

    while not exit_event.is_set():  # Keep running until exit_event is triggered
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_event.set()  # Signal exit for both threads
                pygame.quit()
                stream.stop()
                stream.close()
                sys.exit()

        screen.fill((0, 0, 0))  # Clear screen

        if speaking_event.is_set():
            # Show "Responding..." text when Sigma is speaking
            text_surface = font.render("Responding...", True, (0, 255, 0))
            text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text_surface, text_rect)
        else:
            # Draw circle reacting to mic input when Sigma is not speaking
            new_radius = int(BASE_RADIUS + intensity * SCALE_FACTOR)
            new_radius = max(BASE_RADIUS, new_radius)
            pygame.draw.circle(screen, (0, 255, 0), (WIDTH // 2, HEIGHT // 2), new_radius, 3)

        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    stream.stop()
    stream.close()
    pygame.quit()

def voice_assistant():
    """ Handles speech recognition and Sigma responses. """
    recognizer = speech_rec.Recognizer()
    engine = pyt.init()

    while not exit_event.is_set():  # Keep running until exit_event is triggered
        try:
            with speech_rec.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                print("Listening...")
                audio = recognizer.listen(mic)

                text = recognizer.recognize_google(audio).lower()
                print("User:", text)

                if text in ["quit", "exit", "bye", "goodbye"]:
                    print("Exiting Sigma...")
                    speaking_event.set()
                    engine.say("Goodbye!")
                    engine.runAndWait()
                    exit_event.set()  # Signal exit for both threads
                    break  # Exit loop

                # Set speaking event while processing response
                speaking_event.set()
                response = talk_to_Sigma(text)
                print("Sigma:", response)

                engine.say(response)
                engine.runAndWait()

                # Done speaking, return to listening mode
                speaking_event.clear()

        except speech_rec.UnknownValueError:
            print("Could not understand the audio. Try again.")
            continue

# Start both the visualization and assistant in separate threads
visual_thread = threading.Thread(target=run_visualization, daemon=True)
voice_thread = threading.Thread(target=voice_assistant, daemon=True)

visual_thread.start()
voice_thread.start()

visual_thread.join()
voice_thread.join()
