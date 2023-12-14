import base64

# Read your audio file
with open('C:\\Users\\pc\\Desktop\\nouvelle_arch\\Data\\genres_original\\rock\\rock.00007.wav', 'rb') as audio_file:
    audio_binary = audio_file.read()
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')

# Save the audio_base64 data to a file
with open('audio1_base64.txt', 'w') as output_file:
    output_file.write(audio_base64)



