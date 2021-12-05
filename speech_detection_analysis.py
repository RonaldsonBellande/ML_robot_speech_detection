from header_imports import *
from speech_model_training import *

if __name__ == "__main__":
    
    # Begin analysis
    if len(sys.argv) != 1:

        # Build the model
        if sys.argv[1] == "model_building":
            speech__analysis_obj = speech_building(model_type = sys.argv[2])

        # Classify the speech
        if sys.argv[1] == "model_training":
            speech_analysis_obj = speech_training(model_type = sys.argv[2])
        
        # Indentify and speech with model
        if sys.argv[1] == "model_compare":
            pass

