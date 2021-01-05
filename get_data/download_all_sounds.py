"""
Downloads all the Impact and Ambient sounds
"""
from .download_soi import download_data, AMBIENT_SOUNDS, IMPACT_SOUNDS

########################################################################
			# Main Function
########################################################################
if __name__ == "__main__":

    TARGET_SOUNDS = AMBIENT_SOUNDS + IMPACT_SOUNDS
    TARGET_PATH = 'sounds/ambient_impact/'
    download_data(TARGET_SOUNDS, TARGET_PATH)
