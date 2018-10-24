import argparse
import youtube_audioset

#Define the sounds
EXPLOSION_SOUNDS = [
    'Fireworks',
    'Burst, pop',
    'Eruption',
    'Gunshot, gunfire',
    'Explosion',
    'Boom',
    'Fire'
]

MOTOR_SOUNDS = [
    'Chainsaw',
    'Medium engine (mid frequency)',
    'Light engine (high frequency)',
    'Heavy engine (low frequency)',
    'Engine starting',
    'Engine',
    'Motor vehicle (road)',
    'Vehicle'
]

WOOD_SOUNDS = [
    'Wood',
    'Chop',
    'Splinter',
    'Crack'
]

HUMAN_SOUNDS = [
    'Chatter',
    'Conversation',
    'Speech',
    'Narration, monologue',
    'Babbling',
    'Whispering',
    'Laughter',
    'Chatter',
    'Crowd',
    'Hubbub, speech noise, speech babble',
    'Children playing',
    'Whack, thwack',
    'Smash, crash',
    'Breaking',
    'Crushing',
    'Tearing',
    'Run',
    'Walk, footsteps',
    'Clapping'

]


DOMESTIC_SOUNDS = [
    'Dog',
    'Bark',
    'Howl',
    'Bow-wow',
    'Growling',
    'Bay',
    'Livestock, farm animals, working animals',
    'Yip',
    'Cattle, bovinae',
    'Moo',
    'Cowbell',
    'Goat',
    'Bleat',
    'Sheep',
    'Squawk',
    'Domestic animals, pets'

]


TOOLS_SOUNDS = [
    'Jackhammer',
    'Sawing',
    'Tools',
    'Hammer',
    'Filing (rasp)',
    'Sanding',
    'Power tool'
]


WILD_ANIMALS = [
    'Roaring cats (lions, tigers)',
    'Roar',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Squawk',
    'Pigeon, dove',
    'Chirp, tweet',
    'Coo',
    'Crow',
    'Caw',
    'Owl',
    'Hoot',
    'Gull, seagull',
    'Bird flight, flapping wings',
    'Canidae, dogs, wolves',
    'Rodents, rats, mice',
    'Mouse',
    'Chipmunk',
    'Patter',
    'Insect',
    'Cricket',
    'Mosquito',
    'Fly, housefly',
    'Buzz',
    'Bee, wasp, etc.',
    'Frog',
    'Croak',
    'Snake',
    'Rattle'
]

NATURE_SOUNDS = [
    "Silence",
    "Stream",
    "Wind noise (microphone)",
    "Wind",
    "Rustling leaves",
    "Howl",
    "Raindrop",
    "Rain on surface",
    "Rain",
    "Thunderstorm",
    "Thunder",
    'Crow',
    'Caw',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Chirp, tweet',
    'Owl',
    'Hoot'

]

#Defining Ambient and Impact sounds as to what sounds it must comprise of.
AMBIENT_SOUNDS = NATURE_SOUNDS
IMPACT_SOUNDS = EXPLOSION_SOUNDS + WOOD_SOUNDS + MOTOR_SOUNDS + HUMAN_SOUNDS + TOOLS_SOUNDS + DOMESTIC_SOUNDS

#create a dictionary of sounds
SOUNDS_DICT = {'explosion_sounds': EXPLOSION_SOUNDS, 'wood_sounds': WOOD_SOUNDS,
               'nature_sounds': NATURE_SOUNDS, 'motor_sounds': MOTOR_SOUNDS,
               'human_sounds': HUMAN_SOUNDS, 'tools': TOOLS_SOUNDS, 'domestic_sounds': DOMESTIC_SOUNDS,
               'Wild_animals':WILD_ANIMALS}

#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description='Input one of these sounds : explosion_sounds , wood_sounds , motor_sounds, human_sounds, tools ,domestic_sounds, Wild_animals, nature_sounds')
PARSER.add_argument('-target_sounds', '--target_sounds', action='store', help='Input the target sounds. It should be one of the listed sounds', default='explosion_sounds')
PARSER.add_argument('-target_path', '--target_path', action='store', help='Input the path', default='sounds/explosion_sounds/')
RESULT = PARSER.parse_args()


#call the function to dowload the target or sounds of Interest. Set the target
if __name__ == '__main__':
    youtube_audioset.download_data(SOUNDS_DICT[RESULT.target_sounds], RESULT.target_path)
