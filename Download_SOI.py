import youtube_audioset

#Define the sounds

explosion_sounds = [
'Fireworks',
'Burst, pop',
'Eruption',
'Gunshot, gunfire',
'Explosion',
'Boom',
"Fire"
]

motor_sounds = [
'Chainsaw',
'Medium engine (mid frequency)',
'Light engine (high frequency)',
'Heavy engine (low frequency)',
'Engine starting',
'Engine',
'Motor vehicle (road)',
'Vehicle'
]

wood_sounds = [
'Wood',
'Chop',
'Splinter',
'Crack'
]

human_sounds = [
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


domestic_sounds=[
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


tools=[
'Jackhammer',
'Sawing',
'Tools',
'Hammer',
'Filing (rasp)',
'Sanding',
'Power tool'
]


Wild_animals=[
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

nature_sounds = [
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
ambient_sounds = nature_sounds
impact_sounds = explosion_sounds + wood_sounds + motor_sounds + human_sounds + tools  + domestic_sounds

#SELECT the sounds that must be downloaded and set the path to download . For example we are taking explosion sounds
target_sounds = explosion_sounds
target_path = 'sounds/explosion_sounds/'

#call the function to dowload the target or sounds of Interest. Set the target
if __name__=='__main__':
    youtube_audioset.download_data(target_sounds,target_path)
