# Function should be called:
# python generate.py --file
import argparse
import data
import audio
from StoryTime import StorytimeArchitecture

def readModel(filename):
    model = StorytimeArchitecture()
    model.load_weights(filename)

def generateAudio(text, model, outputFilename):
    modelInput = data.text_to_sequence(text))
    generatedSpectrogram = model.predict(modelInput)

    wavfile.write(outputFilename, config.audio_sample_rate, spectrogram2wav(generatedSpectrogram))


if (__file__ == "__main__"):
    parser = argparse.ArgumentParser(description='Generates a new sound file from the given text')

    parser.add_argument('--model', dest='modelFilename', required=True, help='Saved model file path to use')
    parser.add_argument('--output', dest='outputFilename', required=True, help='Location to save file')
    parser.add_argument('--text', dest='text', required=True, help='Text to generate')

    args = parser.parse_args()

    # Generate the desired audio
    model = readModel(args["modelFilename"])
    generateAudio(args["text"], args["outputFilename"])
